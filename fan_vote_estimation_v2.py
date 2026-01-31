"""
Compatibility wrapper for the project.

Several Task 2 scripts import `fan_vote_estimation_v2`, while the actual
implementation lives in `1_fan_vote_estimation.py` (which is the optimized V2).

This file re-exports the V2 API so analysis scripts remain runnable.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_impl_module():
    """
    Load `1_fan_vote_estimation.py` as a module.

    The file name starts with a digit, so it cannot be imported with a normal
    `import` statement.
    """

    impl_path = Path(__file__).with_name("1_fan_vote_estimation.py")
    spec = importlib.util.spec_from_file_location("_fan_vote_estimation_v2_impl", impl_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from: {impl_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


_impl = _load_impl_module()

load_and_preprocess_data = _impl.load_and_preprocess_data
sample_season_feasible_region_v2 = _impl.sample_season_feasible_region_v2
build_hmm_and_find_map_path_v2 = _impl.build_hmm_and_find_map_path_v2
evaluate_consistency_v2 = _impl.evaluate_consistency_v2

__all__ = [
    "load_and_preprocess_data",
    "sample_season_feasible_region_v2",
    "build_hmm_and_find_map_path_v2",
    "evaluate_consistency_v2",
]

