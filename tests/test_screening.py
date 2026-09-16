"""Tests for pre-run structure screening.

The riskiest behaviour here is not the scoring, it is the gating: dropping a structure is
irreversible work lost, so out-of-domain rows must survive, screening must be off unless
asked for, and the DataFrame index must never be renumbered (``orig_index`` is derived
from it).
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oact_utilities.utils.architector import create_workflow_db, validate_extra_columns
from oact_utilities.workflows.filter_model import pick_threshold
from oact_utilities.workflows.screening import (
    ACTION_DROPPED,
    ACTION_KEPT,
    ACTION_OUT_OF_DOMAIN,
    MODE_ANNOTATE,
    MODE_FILTER,
    SCREENERS,
    SHARED_COLUMNS,
    FilterRiskScreener,
    add_screening_args,
    apply_screening,
    build_screener,
    screening_enabled,
    screening_extra_columns,
)

GEOMETRY = "Np 0.0 0.0 0.0\nO 1.80 0.0 0.0\nO -1.80 0.0 0.0"
LANTHANIDE = "Gd 0.0 0.0 0.0\nO 2.00 0.0 0.0\nO -2.00 0.0 0.0"


class StubScreener:
    """Deterministic stand-in: probability comes from the row's ``want`` column.

    Keeps the tests independent of a fitted bundle while exercising the real gating code.
    """

    name = "stub"
    probability_column = "screen_p_filtered"

    def __init__(self, bundle_dir=None):
        self.bundle_dir = bundle_dir
        self.calibrated = True
        self.version = "stub@1"

    @classmethod
    def declared_columns(cls):
        return {**SHARED_COLUMNS, cls.probability_column: "REAL"}

    def extra_columns(self):
        return self.declared_columns()

    def score(
        self, df, *, geometry_column, charge_column, spin_column, metal_column=None
    ):
        probability = df["want"].astype(float)
        in_domain = df["metal"] != "Gd"  # the model never saw a lanthanide centre
        return pd.DataFrame(
            {
                "probability": probability,
                "in_domain": in_domain,
                self.probability_column: probability,
            },
            index=df.index,
        )

    def resolve_threshold(self, threshold, target_precision):
        return (float(threshold) if threshold is not None else 0.8), {
            "precision": 0.91,
            "recall": 0.65,
            "good_lost_pct": 1.78,
        }


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setitem(SCREENERS, StubScreener.name, StubScreener)
    return StubScreener


def _args(**overrides) -> argparse.Namespace:
    base = {
        "screen_with": None,
        "screen_bundle": None,
        "screen_mode": MODE_ANNOTATE,
        "screen_target_precision": 0.91,
        "screen_threshold": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _frame() -> pd.DataFrame:
    """Four rows: two safe, one rejectable, one lanthanide that is also rejectable."""
    rows = [
        {"structure": GEOMETRY, "charge": 0, "spin": 1, "metal": "Np", "want": 0.10},
        {"structure": GEOMETRY, "charge": 0, "spin": 1, "metal": "Np", "want": 0.50},
        {"structure": GEOMETRY, "charge": 0, "spin": 1, "metal": "Np", "want": 0.95},
        {"structure": LANTHANIDE, "charge": 0, "spin": 1, "metal": "Gd", "want": 0.99},
    ]
    return pd.DataFrame(rows, index=pd.Index([10, 11, 12, 13]))


def _screen(df, **overrides):
    return apply_screening(
        df,
        _args(screen_with="stub", screen_bundle=Path("."), **overrides),
        geometry_column="structure",
        charge_column="charge",
        spin_column="spinmult" if "spinmult" in df.columns else "spin",
        metal_column="metal",
    )


class TestDisabled:
    def test_none_namespace_is_a_no_op(self):
        df = _frame()
        result, stats = apply_screening(
            df,
            None,
            geometry_column="structure",
            charge_column="charge",
            spin_column="spin",
        )
        assert result is df
        assert stats == {}

    def test_absent_flag_is_a_no_op(self):
        df = _frame()
        result, stats = apply_screening(
            df,
            _args(),
            geometry_column="structure",
            charge_column="charge",
            spin_column="spin",
        )
        assert list(result.columns) == list(df.columns)
        assert stats == {}

    def test_no_extra_columns_when_off(self):
        assert screening_extra_columns(_args()) == {}
        assert screening_extra_columns(None) == {}
        assert not screening_enabled(_args())


class TestDeclaredColumns:
    def test_pass_the_db_validator(self, stub):
        """Column names and types must survive architector's whitelist."""
        columns = screening_extra_columns(_args(screen_with="stub"))
        assert validate_extra_columns(columns) == {
            k: v.upper() for k, v in columns.items()
        }

    def test_real_screener_declares_without_a_bundle(self):
        """The converter needs the schema before the optional ML deps are touched."""
        columns = FilterRiskScreener.declared_columns()
        assert columns["screen_p_filtered"] == "REAL"
        assert set(SHARED_COLUMNS) <= set(columns)


class TestAnnotateMode:
    def test_drops_nothing(self, stub):
        df = _frame()
        result, stats = _screen(df)
        assert len(result) == len(df)
        assert stats["dropped"] == 0
        # Both 0.95 and the lanthanide's 0.99 clear the cut, but only the in-domain one
        # is eligible to be dropped. The two counts are reported separately so the
        # report cannot imply that out-of-domain rows were considered for removal.
        assert stats["above_threshold"] == 2
        assert stats["rejectable"] == 1

    def test_records_scores_and_domain(self, stub):
        result, _ = _screen(_frame())
        assert result.loc[12, "screen_p_filtered"] == pytest.approx(0.95)
        assert result.loc[12, "screen_in_domain"] == 1
        assert result.loc[13, "screen_in_domain"] == 0
        assert result["screen_model"].iloc[0] == "stub@stub@1"

    def test_action_column(self, stub):
        result, _ = _screen(_frame())
        assert result.loc[10, "screen_action"] == ACTION_KEPT
        assert result.loc[13, "screen_action"] == ACTION_OUT_OF_DOMAIN


class TestFilterMode:
    def test_drops_only_in_domain_rejects(self, stub):
        result, stats = _screen(_frame(), screen_mode=MODE_FILTER)
        assert stats["dropped"] == 1
        assert 12 not in result.index  # in-domain, above threshold
        assert {10, 11, 13} == set(result.index)

    def test_out_of_domain_survives(self, stub):
        """A lanthanide scores 0.99 but was never trained on: dropping it destroys work."""
        result, _ = _screen(_frame(), screen_mode=MODE_FILTER)
        assert 13 in result.index
        assert result.loc[13, "screen_action"] == ACTION_OUT_OF_DOMAIN

    def test_index_is_not_renumbered(self, stub):
        """orig_index is derived from the index, so renumbering would corrupt provenance."""
        result, _ = _screen(_frame(), screen_mode=MODE_FILTER)
        assert result.index.tolist() == [10, 11, 13]

    def test_dropped_rows_are_labelled_before_removal(self, stub):
        annotated, _ = _screen(
            _frame(), screen_mode=MODE_ANNOTATE, screen_threshold=0.9
        )
        assert annotated.loc[12, "screen_action"] == ACTION_KEPT  # annotate never drops
        filtered, _ = _screen(_frame(), screen_mode=MODE_FILTER, screen_threshold=0.9)
        assert ACTION_DROPPED not in set(filtered["screen_action"])  # they are gone

    def test_raw_threshold_overrides_target_precision(self, stub):
        _, stats = _screen(_frame(), screen_mode=MODE_FILTER, screen_threshold=0.4)
        assert stats["threshold"] == pytest.approx(0.4)
        assert stats["dropped"] == 2  # 0.50 and 0.95, both in-domain


class TestBuildScreener:
    def test_requires_a_bundle(self, stub):
        with pytest.raises(ValueError, match="--screen-bundle"):
            build_screener(_args(screen_with="stub"))


class TestPickThreshold:
    @pytest.fixture
    def table(self):
        return pd.DataFrame(
            {
                "threshold": [0.5, 0.7, 0.8, 0.9],
                "precision": [0.79, 0.87, 0.91, 0.95],
                "recall": [0.87, 0.78, 0.65, 0.55],
            }
        ).set_index("threshold")

    def test_picks_the_lowest_threshold_that_qualifies(self, table):
        threshold, row = pick_threshold(table, 0.91)
        assert threshold == pytest.approx(0.8)
        assert row["recall"] == pytest.approx(0.65)

    def test_rounds_up_to_the_next_qualifying_row(self, table):
        assert pick_threshold(table, 0.88)[0] == pytest.approx(0.8)

    def test_unreachable_target_raises_rather_than_silently_capping(self, table):
        """Silently returning the best available would understate the false-positive rate."""
        with pytest.raises(ValueError, match="best available is 0.950"):
            pick_threshold(table, 0.99)


class TestArgparseWiring:
    def test_defaults_are_off(self):
        parser = argparse.ArgumentParser()
        add_screening_args(parser)
        args = parser.parse_args([])
        assert args.screen_with is None
        assert args.screen_mode == MODE_ANNOTATE
        assert not screening_enabled(args)

    def test_filter_risk_is_registered(self):
        parser = argparse.ArgumentParser()
        add_screening_args(parser)
        args = parser.parse_args(
            ["--screen-with", "filter_risk", "--screen-bundle", "/tmp/b"]
        )
        assert args.screen_with == "filter_risk"
        assert args.screen_bundle == Path("/tmp/b")


class TestEndToEnd:
    def test_columns_reach_the_database(self, stub, tmp_path):
        """The annotations must survive create_workflow_db, not just live in the frame."""
        df = _frame().rename(columns={"spin": "spinmult"})
        screened, _ = apply_screening(
            df,
            _args(screen_with="stub", screen_bundle=tmp_path, screen_mode=MODE_FILTER),
            geometry_column="structure",
            charge_column="charge",
            spin_column="spinmult",
            metal_column="metal",
        )
        db_path = tmp_path / "screened.db"
        create_workflow_db(
            csv_path=screened,
            db_path=db_path,
            geometry_column="structure",
            charge_column="charge",
            spin_column="spinmult",
            extra_columns={
                "metal": "TEXT",
                **screening_extra_columns(_args(screen_with="stub")),
            },
        )

        conn = sqlite3.connect(db_path)
        try:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(structures)")}
            assert {
                "screen_model",
                "screen_in_domain",
                "screen_action",
                "screen_p_filtered",
            } <= cols
            rows = conn.execute(
                "SELECT metal, screen_in_domain, screen_action, screen_p_filtered "
                "FROM structures ORDER BY orig_index"
            ).fetchall()
        finally:
            conn.close()

        assert len(rows) == 3  # the in-domain reject was dropped
        assert [r[0] for r in rows] == ["Np", "Np", "Gd"]
        assert rows[-1][1] == 0
        assert rows[-1][2] == ACTION_OUT_OF_DOMAIN
        assert rows[-1][3] == pytest.approx(0.99)

    def test_orig_index_survives_a_drop(self, stub, tmp_path):
        """A filtered run must not renumber surviving rows."""
        df = _frame().rename(columns={"spin": "spinmult"})
        screened, _ = apply_screening(
            df,
            _args(screen_with="stub", screen_bundle=tmp_path, screen_mode=MODE_FILTER),
            geometry_column="structure",
            charge_column="charge",
            spin_column="spinmult",
            metal_column="metal",
        )
        db_path = tmp_path / "gap.db"
        create_workflow_db(
            csv_path=screened,
            db_path=db_path,
            geometry_column="structure",
            charge_column="charge",
            spin_column="spinmult",
            extra_columns=screening_extra_columns(_args(screen_with="stub")),
        )
        conn = sqlite3.connect(db_path)
        try:
            indices = [
                r[0]
                for r in conn.execute(
                    "SELECT orig_index FROM structures ORDER BY orig_index"
                )
            ]
        finally:
            conn.close()
        assert indices == [10, 11, 13]  # 12 is missing, the rest keep their positions


class TestUnscorableRows:
    def test_nan_probability_is_never_dropped(self, stub, monkeypatch):
        """A row the model could not score is not evidence that it is bad."""

        def score(self, df, **kwargs):
            probability = df["want"].astype(float).copy()
            probability.iloc[0] = np.nan
            return pd.DataFrame(
                {
                    "probability": probability,
                    "in_domain": df["metal"] != "Gd",
                    "screen_p_filtered": probability,
                },
                index=df.index,
            )

        monkeypatch.setattr(StubScreener, "score", score)
        result, stats = _screen(_frame(), screen_mode=MODE_FILTER)
        assert 10 in result.index
        assert stats["unscored"] == 1
        assert result.loc[10, "screen_action"] == "unscored"
