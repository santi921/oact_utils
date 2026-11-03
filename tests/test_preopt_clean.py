"""clean.py treats g-xTB scratch as tmp and keeps the pre-opt results."""

from __future__ import annotations

from oact_utilities.workflows.clean import _is_excluded, _match_cleanup_patterns


def test_preopt_scratch_dir_is_tmp() -> None:
    assert _match_cleanup_patterns("preopt_gxtb", True, {"tmp"}) is True
    assert _match_cleanup_patterns("preopt_gxtb", False, {"tmp"}) is False
    assert _match_cleanup_patterns("preopt_gxtb", True, {"bas"}) is False


def test_preopt_results_are_never_cleaned() -> None:
    for kept in (
        "preopt.xyz",
        "preopt_status.json",
        "preopt_config.json",
        "preopt_driver.log",
        "run_preopt.py",
        "input.xyz",
    ):
        assert _is_excluded(kept)
        assert _match_cleanup_patterns(kept, False, {"tmp", "bas"}) is False
