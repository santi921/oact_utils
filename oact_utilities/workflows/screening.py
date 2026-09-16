"""Optional pre-run screening of structures with a trained classifier.

Scores structures before they are written into a workflow DB, so a campaign can skip
calculations that a model expects to be thrown away. All entry points are no-ops when
screening is not requested, so callers never need to guard with ``if SCREENING_AVAILABLE``.

Usage in a converter::

    from oact_utilities.workflows.screening import (
        add_screening_args,
        apply_screening,
        screening_extra_columns,
    )

    # In argparse setup:
    add_screening_args(parser)

    # In build_workflow(), just before create_workflow_db():
    df, stats = apply_screening(
        df, args,
        geometry_column=GEOMETRY_COLUMN,
        charge_column=CHARGE_COLUMN,
        spin_column=SPIN_COLUMN,
    )
    create_workflow_db(
        csv_path=df,
        ...,
        extra_columns={**EXTRA_COLUMNS, **screening_extra_columns(args)},
    )

APPLICABILITY IS NOT OPTIONAL. A screener declares the domain it was trained on, and rows
outside it are marked ``screen_in_domain = 0`` and never dropped. This matters because
``sklearn``'s ``OneHotEncoder(handle_unknown="ignore")`` emits an all-zero encoding for an
unseen category without complaint, so an out-of-domain structure still receives a
confident-looking probability. The ``filter_risk`` model, for example, has seen no
lanthanide metal centre at all.

ADDING A SCREENER. Implement :class:`StructureScreener` and add it to :data:`SCREENERS`.
A future job-outcome model (fail / complete / timeout) plugs in that way and would declare
its own columns, e.g. ``screen_p_fail`` / ``screen_p_timeout``; no converter changes.
"""

from __future__ import annotations

import argparse
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


def _deps_installed() -> bool:
    """Probe for the optional ML stack without importing it.

    ``import xgboost`` costs over a second, and a converter that is not screening should
    not pay it, so availability is checked with ``find_spec`` and the real imports happen
    only inside the screener.
    """
    try:
        return all(
            find_spec(name) is not None for name in ("joblib", "sklearn", "xgboost")
        )
    except (ImportError, ValueError):
        return False


SCREENING_AVAILABLE = _deps_installed()

MODE_ANNOTATE = "annotate"
MODE_FILTER = "filter"

ACTION_KEPT = "kept"
ACTION_DROPPED = "dropped"
ACTION_OUT_OF_DOMAIN = "skipped_out_of_domain"
ACTION_UNSCORED = "unscored"

# Columns every screener writes, whatever it predicts.
SHARED_COLUMNS: dict[str, str] = {
    "screen_model": "TEXT",
    "screen_in_domain": "INTEGER",
    "screen_action": "TEXT",
}

DEFAULT_TARGET_PRECISION = 0.91


class StructureScreener(Protocol):
    """Scores structures before they are submitted.

    Implementations own their feature construction and their own applicability rules, so
    this module never needs to know what a particular model looks at.
    """

    name: str

    @classmethod
    def declared_columns(cls) -> dict[str, str]:
        """Column name -> SQLite type for everything this screener writes.

        A classmethod because the converter needs the schema before a bundle is loaded,
        and loading a bundle requires the optional ML dependencies.
        """

    def extra_columns(self) -> dict[str, str]:
        """Instance view of :meth:`declared_columns`."""

    def score(
        self,
        df: pd.DataFrame,
        *,
        geometry_column: str,
        charge_column: str,
        spin_column: str,
        metal_column: str | None = None,
    ) -> pd.DataFrame:
        """Score rows, preserving ``df.index``.

        Must return a frame carrying ``probability`` and ``in_domain`` plus any
        model-specific columns declared by :meth:`extra_columns`.
        """

    def resolve_threshold(
        self, threshold: float | None, target_precision: float | None
    ) -> tuple[float, dict]:
        """Pick the probability cut, returning it with its recorded operating point."""


class FilterRiskScreener:
    """The v4 model-dev filter classifier.

    Predicts the probability that the quality/energy filter would discard the result of
    running a structure. Takes a bundle of fitted artifacts; the feature code is
    :mod:`oact_utilities.utils.filter_features`, shared with the training notebook, so
    featurization here is identical to training by construction.
    """

    name = "filter_risk"
    probability_column = "screen_p_filtered"

    def __init__(self, bundle_dir: Path) -> None:
        # Imported here, not at module scope, so the converters load without xgboost.
        from .filter_model import FilterModel

        self.bundle_dir = Path(bundle_dir).resolve()
        self._model = FilterModel(self.bundle_dir)

    @property
    def version(self) -> str:
        return str(self._model.version)

    @property
    def calibrated(self) -> bool:
        return bool(self._model.calibrated)

    @classmethod
    def declared_columns(cls) -> dict[str, str]:
        return {**SHARED_COLUMNS, cls.probability_column: "REAL"}

    def extra_columns(self) -> dict[str, str]:
        return self.declared_columns()

    def score(
        self,
        df: pd.DataFrame,
        *,
        geometry_column: str,
        charge_column: str,
        spin_column: str,
        metal_column: str | None = None,
    ) -> pd.DataFrame:
        scores = self._model.score_frame(
            df,
            geometry_column=geometry_column,
            charge_column=charge_column,
            spin_column=spin_column,
            metal_column=metal_column,
        )
        return pd.DataFrame(
            {
                "probability": scores["p_filtered"],
                "in_domain": scores["in_domain"].astype(bool),
                self.probability_column: scores["p_filtered"],
            },
            index=df.index,
        )

    def resolve_threshold(
        self, threshold: float | None, target_precision: float | None
    ) -> tuple[float, dict]:
        if threshold is not None:
            return float(threshold), {}
        cut, operating_point = self._model.threshold_for_precision(
            target_precision
            if target_precision is not None
            else DEFAULT_TARGET_PRECISION
        )
        return float(cut), dict(operating_point)


SCREENERS: dict[str, Any] = {FilterRiskScreener.name: FilterRiskScreener}


def add_screening_args(parser: argparse.ArgumentParser) -> None:
    """Add screening CLI arguments to a converter's parser.

    Screening is entirely off unless ``--screen-with`` is given.
    """
    group = parser.add_argument_group("Screening Options")
    group.add_argument(
        "--screen-with",
        default=None,
        choices=sorted(SCREENERS),
        help="Score structures with this model before writing them (default: no screening)",
    )
    group.add_argument(
        "--screen-bundle",
        type=Path,
        default=None,
        metavar="PATH",
        help="Directory holding the model bundle (required with --screen-with)",
    )
    group.add_argument(
        "--screen-mode",
        default=MODE_ANNOTATE,
        choices=(MODE_ANNOTATE, MODE_FILTER),
        help="annotate: record scores only. filter: also drop rows above the threshold",
    )
    group.add_argument(
        "--screen-target-precision",
        type=float,
        default=DEFAULT_TARGET_PRECISION,
        metavar="P",
        help=(
            "Choose the threshold whose recorded precision meets P. Stable across a "
            f"recalibration or retrain, unlike a raw cut (default: {DEFAULT_TARGET_PRECISION})"
        ),
    )
    group.add_argument(
        "--screen-threshold",
        type=float,
        default=None,
        metavar="P",
        help="Raw probability cut, overriding --screen-target-precision",
    )


def screening_enabled(args: argparse.Namespace) -> bool:
    """Whether the caller asked for screening."""
    return getattr(args, "screen_with", None) is not None


def screening_extra_columns(args: argparse.Namespace) -> dict[str, str]:
    """Extra DB columns this run will write. Empty when screening is off."""
    if not screening_enabled(args):
        return {}
    return dict(SCREENERS[args.screen_with].declared_columns())


def build_screener(args: argparse.Namespace) -> StructureScreener:
    """Instantiate the requested screener.

    Raises:
        ValueError: if ``--screen-bundle`` was not supplied.
        ImportError: if the ML dependencies are not installed.
    """
    if args.screen_bundle is None:
        raise ValueError("--screen-with requires --screen-bundle PATH")
    if not SCREENING_AVAILABLE:
        raise ImportError(
            "screening needs xgboost, scikit-learn and joblib. "
            'Install with: pip install -e ".[classifier]"'
        )
    screener: StructureScreener = SCREENERS[args.screen_with](args.screen_bundle)
    return screener


def apply_screening(
    df: pd.DataFrame,
    args: argparse.Namespace,
    *,
    geometry_column: str,
    charge_column: str,
    spin_column: str,
    metal_column: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Score ``df`` and, in filter mode, drop the rows the model rejects.

    The DataFrame index is never reset, so ``orig_index`` stays a stable position in the
    source stream, matching ``filter_by_natoms`` in the LMDB converter.

    Out-of-domain rows are annotated and kept, never dropped: a structure the model was not
    trained on gets a meaningless probability, and acting on it would discard real work for
    no reason.

    Returns:
        The (possibly shorter) DataFrame and a stats dict. When screening is off, ``df`` is
        returned unchanged with an empty stats dict.

    Raises:
        ImportError: in filter mode when the ML dependencies are missing. Annotate mode
            warns and continues instead, since nothing is lost by skipping annotation.
    """
    if not screening_enabled(args):
        return df, {}

    if not SCREENING_AVAILABLE:
        message = (
            "screening requested but xgboost/scikit-learn/joblib are not installed; "
            'install with: pip install -e ".[classifier]"'
        )
        if args.screen_mode == MODE_FILTER:
            raise ImportError(message)
        print(f"Warning: {message} -- continuing without screening")
        return df, {}

    screener = build_screener(args)
    threshold, operating_point = screener.resolve_threshold(
        args.screen_threshold, args.screen_target_precision
    )

    scores = screener.score(
        df,
        geometry_column=geometry_column,
        charge_column=charge_column,
        spin_column=spin_column,
        metal_column=metal_column,
    )

    in_domain = scores["in_domain"].fillna(False).astype(bool)
    scored = scores["probability"].notna()
    above = scored & (scores["probability"] >= threshold)
    # Only in-domain rows are ever rejected: a probability from outside the training
    # distribution is an extrapolation, and acting on it would destroy real work.
    reject = above & in_domain

    annotated = df.copy()
    for column in screener.extra_columns():
        if column in scores.columns:
            annotated[column] = scores[column]
    annotated["screen_model"] = (
        f"{screener.name}@{getattr(screener, 'version', 'unknown')}"
    )
    annotated["screen_in_domain"] = in_domain.astype(int)
    annotated["screen_action"] = ACTION_KEPT
    annotated.loc[~scored, "screen_action"] = ACTION_UNSCORED
    annotated.loc[scored & ~in_domain, "screen_action"] = ACTION_OUT_OF_DOMAIN
    if args.screen_mode == MODE_FILTER:
        annotated.loc[reject, "screen_action"] = ACTION_DROPPED

    stats = {
        "screener": screener.name,
        "calibrated": bool(getattr(screener, "calibrated", False)),
        "mode": args.screen_mode,
        "threshold": threshold,
        "rows": len(df),
        "scored": int(scored.sum()),
        "unscored": int((~scored).sum()),
        "in_domain": int((scored & in_domain).sum()),
        "out_of_domain": int((scored & ~in_domain).sum()),
        "above_threshold": int(above.sum()),
        "rejectable": int(reject.sum()),
        "dropped": int(reject.sum()) if args.screen_mode == MODE_FILTER else 0,
        "operating_point": operating_point,
    }

    if args.screen_mode == MODE_FILTER:
        annotated = annotated[~reject]

    _print_report(stats)
    return annotated, stats


def _print_report(stats: dict) -> None:
    """Per-reason breakdown, so a coverage gap is never silent."""
    print(f"\nScreening with {stats['screener']} (mode: {stats['mode']})")
    if not stats["calibrated"]:
        print(
            "  WARNING: bundle is not calibrated; probabilities are ranking scores, "
            "not frequencies"
        )
    point = stats["operating_point"]
    if point:
        print(
            f"  threshold {stats['threshold']:.3f} -> recorded precision "
            f"{point.get('precision', float('nan')):.3f}, recall "
            f"{point.get('recall', float('nan')):.3f}, "
            f"{point.get('good_lost_pct', float('nan')):.2f}% of viable structures lost"
        )
    else:
        print(
            f"  threshold {stats['threshold']:.3f} (raw cut, no recorded operating point)"
        )

    print(f"  rows            {stats['rows']:>9,}")
    print(f"  scored          {stats['scored']:>9,}")
    if stats["unscored"]:
        print(
            f"  unscored        {stats['unscored']:>9,}  (geometry could not be parsed)"
        )
    print(f"  in domain       {stats['in_domain']:>9,}")
    if stats["out_of_domain"]:
        print(
            f"  out of domain   {stats['out_of_domain']:>9,}  "
            "(annotated, never dropped -- outside the training distribution)"
        )
    print(f"  above threshold {stats['above_threshold']:>9,}")
    if stats["rejectable"] != stats["above_threshold"]:
        print(
            f"    of which in domain {stats['rejectable']:>5,}  "
            "(only these are eligible to be dropped)"
        )
    if stats["mode"] == MODE_FILTER:
        print(f"  DROPPED         {stats['dropped']:>9,}")
        print(f"  written         {stats['rows'] - stats['dropped']:>9,}")
    else:
        print(f"  written         {stats['rows']:>9,}  (annotate mode drops nothing)")
