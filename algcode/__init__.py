import os
import importlib
from typing import Dict, Any, List
from pathlib import Path

from .utils.dataloader import (
    load_data, dataframe_to_trajectory_points, trajectory_points_to_dataframe, TrajectoryPoint,
    scan_categories, scan_datasets, load_ais_dataset
)
from .utils.geo_utils import GeoUtils
from .utils.metrics import calculate_sed_metrics, calculate_navigation_event_recall, evaluate_compression
from .utils.visualization import visualize_trajectories

_ALGORITHMS_DIR = Path(__file__).parent / "algorithms"
_AVAILABLE_ALGORITHMS = {}

def _load_algorithms():
    global _AVAILABLE_ALGORITHMS

    if not _ALGORITHMS_DIR.exists():
        return

    for py_file in _ALGORITHMS_DIR.glob("*.py"):
        module_name = py_file.stem

        if module_name in ['squish', 'semantic_dr']:
            continue

        try:
            module = importlib.import_module(f".algorithms.{module_name}", package=__name__)

            if hasattr(module, 'compress') and hasattr(module, 'DISPLAY_NAME'):
                alg_info = {
                    'module': module,
                    'display_name': getattr(module, 'DISPLAY_NAME', module_name),
                    'default_params': getattr(module, 'DEFAULT_PARAMS', {}),
                    'param_help': getattr(module, 'PARAM_HELP', {}),
                    'compress_func': module.compress
                }
                _AVAILABLE_ALGORITHMS[module_name] = alg_info

        except Exception as e:
            print(f"加载算法模块 {module_name} 失败: {e}")
            continue

_load_algorithms()

def get_available_algorithms() -> Dict[str, Dict[str, Any]]:
    return _AVAILABLE_ALGORITHMS.copy()

def run_algorithm(algorithm_key: str, points, params: Dict):
    if algorithm_key not in _AVAILABLE_ALGORITHMS:
        raise ValueError(f"未知算法: {algorithm_key}")
    alg_info = _AVAILABLE_ALGORITHMS[algorithm_key]
    return alg_info['compress_func'](points, params)

try:
    from .algorithms import (
        dead_reckoning_compress,
        adaptive_dr_compress,
        opening_window_compress,
        dp_compress
    )
    _legacy_algorithms = [
        'dead_reckoning_compress', 'adaptive_dr_compress',
        'opening_window_compress', 'dp_compress'
    ]
except ImportError:
    _legacy_algorithms = []

__version__ = "1.0.0"
__all__ = [
    'load_data', 'dataframe_to_trajectory_points', 'trajectory_points_to_dataframe', 'TrajectoryPoint',
    'scan_categories', 'scan_datasets', 'load_ais_dataset',
    'get_available_algorithms', 'run_algorithm',
    'GeoUtils',
    'calculate_sed_metrics', 'calculate_navigation_event_recall', 'evaluate_compression',
    'visualize_trajectories'
] + _legacy_algorithms
