"""Load and apply a trained v4-filter classifier.

Predicts, from an input structure alone, the probability that the v4 model-dev quality and
energy filters would discard the resulting calculation. No ORCA run is involved, so this is
usable as a screen before spending core-hours. ``workflows/screening.py`` wraps it for the
converters; use this module directly for ad-hoc scoring.

Feature construction lives in :mod:`oact_utilities.utils.filter_features`, shared with the
training notebook (``notebooks/classifier_v4_filter.ipynb``), so the serving path and the
training path cannot drift apart.

The fitted artifacts are not part of the package -- they are weights, not source. A bundle
directory holds:

===============================  ==================================================
``v4_filter_manifest.json``      feature order, domain bounds, thresholds, metrics
``v4_filter_model.json``         XGBoost booster in its native format
``v4_filter_preprocessor.joblib``  fitted ColumnTransformer
``v4_filter_calibrator.joblib``  fitted isotonic map (optional)
===============================  ==================================================

Requires the ``classifier`` extra (xgboost, scikit-learn, joblib)::

    from oact_utilities.workflows.filter_model import FilterModel

    model = FilterModel("data/v4_model_dev")
    p = model.predict_proba(xyz_text, charge=-2, spin=3)
    threshold, operating_point = model.threshold_for_precision(0.91)

APPLICABILITY. The shipped model has seen 19 metal centres -- the actinides Ac-Lr plus Po,
At, Fr and Ra -- and structures of 5 to 170 atoms. It has seen no lanthanide centre at all.
``OneHotEncoder(handle_unknown="ignore")`` emits an all-zero encoding for an unseen centre
without complaint and the model still returns a confident-looking number, so callers must
consult :meth:`FilterModel.in_domain` rather than trusting the probability alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.filter_features import (
    build_features,
    geometric_features,
    infer_metal,
    parse_xyz,
)

try:
    from ..utils.basis import count_basis_functions
except ImportError:  # pragma: no cover - basis table is part of the package
    count_basis_functions = None  # type: ignore[assignment]

MANIFEST_NAME = "v4_filter_manifest.json"
BOOSTER_NAME = "v4_filter_model.json"
PREPROCESSOR_NAME = "v4_filter_preprocessor.joblib"
CALIBRATOR_NAME = "v4_filter_calibrator.joblib"


def pick_threshold(
    thresholds: pd.DataFrame, target_precision: float
) -> tuple[float, dict]:
    """Lowest recorded threshold whose precision meets ``target_precision``.

    Selecting by precision rather than by a raw number keeps an operating point stable
    across a recalibration or a retrain, either of which moves the probability scale
    underneath a hardcoded cut.

    Args:
        thresholds: operating points indexed by threshold, with a ``precision`` column.
        target_precision: the precision the caller wants to achieve.

    Returns:
        The threshold and the full recorded row at that threshold.

    Raises:
        ValueError: if no recorded threshold reaches ``target_precision``.
    """
    table = thresholds.sort_index()
    eligible = table[table["precision"] >= target_precision]
    if eligible.empty:
        raise ValueError(
            f"no recorded threshold reaches precision {target_precision:.3f}; "
            f"the best available is {float(table['precision'].max()):.3f}"
        )
    return float(eligible.index[0]), dict(eligible.iloc[0])


class FilterModel:
    """Load a bundle once, score many structures."""

    def __init__(self, bundle_dir: str | Path) -> None:
        import joblib
        import xgboost as xgb

        directory = Path(bundle_dir).resolve()
        manifest_path = directory / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{directory} is not a filter bundle: {MANIFEST_NAME} is missing"
            )

        self.bundle_dir = directory
        self.manifest = json.loads(manifest_path.read_text())
        self.preprocessor = joblib.load(directory / PREPROCESSOR_NAME)
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(directory / BOOSTER_NAME))

        calibrator_path = directory / CALIBRATOR_NAME
        self.calibrator = (
            joblib.load(calibrator_path) if calibrator_path.exists() else None
        )

        self.feature_order: list[str] = self.manifest["feature_order"]
        self.ligand_elems: list[str] = self.manifest["ligand_elems"]
        self.trained_metals: set[str] = set(self.manifest.get("trained_metals", []))
        natoms_range = self.manifest.get("natoms_range")
        self.natoms_range = tuple(natoms_range) if natoms_range else None

    @property
    def calibrated(self) -> bool:
        """Whether probabilities are calibrated frequencies rather than ranking scores."""
        return self.calibrator is not None

    @property
    def version(self) -> str:
        """Short identifier recorded alongside every score."""
        trained = self.manifest.get("trained_on", {}).get("structures", "?")
        return f"{self.manifest['name']}@{trained}"

    def in_domain(self, metal: str, natoms: int) -> bool:
        """Whether this structure falls inside the training distribution.

        False means the probability is an extrapolation and must not be acted on.
        """
        if self.trained_metals and metal not in self.trained_metals:
            return False
        if self.natoms_range and not (
            self.natoms_range[0] <= natoms <= self.natoms_range[1]
        ):
            return False
        return True

    def _row(
        self, xyz_text: str, charge: int, spin: int, metal: str | None, n_basis: Any
    ) -> dict:
        symbols, coords = parse_xyz(xyz_text)
        if (
            metal is None
            or metal == ""
            or (isinstance(metal, float) and np.isnan(metal))
        ):
            metal = infer_metal(symbols)
        if n_basis is None:
            n_basis = np.nan
            if count_basis_functions is not None:
                n_basis = count_basis_functions(symbols, strict=False) or np.nan
        return {
            "elements": ";".join(symbols),
            "natoms": len(symbols),
            "charge": charge,
            "spin": spin,
            "metal": metal,
            "n_basis": n_basis,
            **geometric_features(symbols, coords, metal),
        }

    def _predict(self, features: pd.DataFrame) -> np.ndarray:
        encoded = np.asarray(self.preprocessor.transform(features[self.feature_order]))
        raw = self.model.predict_proba(encoded)[:, 1]
        if self.calibrator is not None:
            raw = self.calibrator.predict(raw)
        return np.asarray(raw, dtype=float)

    def predict_proba(
        self,
        xyz_text: str,
        charge: int,
        spin: int,
        metal: str | None = None,
        n_basis: int | None = None,
    ) -> float:
        """Probability in [0, 1] that the v4 filter removes this structure."""
        row = self._row(xyz_text, charge, spin, metal, n_basis)
        features = build_features(pd.DataFrame([row]), self.ligand_elems)
        return float(self._predict(features)[0])

    def score_frame(
        self,
        df: pd.DataFrame,
        *,
        geometry_column: str,
        charge_column: str,
        spin_column: str,
        metal_column: str | None = None,
    ) -> pd.DataFrame:
        """Score a whole DataFrame, preserving its index.

        Returns a frame with ``p_filtered``, ``in_domain`` and the ``metal`` actually used.
        Rows whose geometry cannot be parsed come back with ``p_filtered`` NaN and
        ``in_domain`` False rather than failing the whole batch.
        """
        rows: list[dict] = []
        ok_index: list[Any] = []
        for idx, record in df.iterrows():
            metal = (
                record[metal_column]
                if metal_column and metal_column in df.columns
                else None
            )
            try:
                rows.append(
                    self._row(
                        record[geometry_column],
                        int(record[charge_column]),
                        int(record[spin_column]),
                        metal,
                        None,
                    )
                )
                ok_index.append(idx)
            except (ValueError, KeyError, TypeError, IndexError):
                continue

        result = pd.DataFrame(
            {"p_filtered": np.nan, "in_domain": False, "metal": None}, index=df.index
        )
        if rows:
            frame = pd.DataFrame(rows, index=pd.Index(ok_index))
            features = build_features(frame, self.ligand_elems)
            result.loc[ok_index, "p_filtered"] = self._predict(features)
            result.loc[ok_index, "metal"] = frame["metal"].to_numpy()
            result.loc[ok_index, "in_domain"] = [
                self.in_domain(m, n) for m, n in zip(frame["metal"], frame["natoms"])
            ]
        return result

    def thresholds(self) -> pd.DataFrame:
        """Held-out operating points recorded at training time."""
        return pd.DataFrame(self.manifest["thresholds"]).set_index("threshold")

    def threshold_for_precision(self, target: float) -> tuple[float, dict]:
        """Lowest threshold whose recorded precision meets ``target``."""
        return pick_threshold(self.thresholds(), target)
