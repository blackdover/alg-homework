"""
轨迹压缩算法包
包含多种轨迹简化算法实现

作者: Algorithm Engineer
Python版本: 3.10
"""

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

# 动态加载算法
_ALGORITHMS_DIR = Path(__file__).parent / "algorithms"
_AVAILABLE_ALGORITHMS = {}

def _load_algorithms():
    """动态加载算法目录中的所有算法"""
    global _AVAILABLE_ALGORITHMS

    if not _ALGORITHMS_DIR.exists():
        return

    for py_file in _ALGORITHMS_DIR.glob("*.py"):
        module_name = py_file.stem

        # 跳过被禁用的算法
        if module_name in ['squish', 'semantic_dr']:
            continue

        try:
            # 动态导入模块
            module = importlib.import_module(f".algorithms.{module_name}", package=__name__)

            # 检查是否有必需的接口和元数据
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

# 初始化时加载算法
_load_algorithms()

def get_available_algorithms() -> Dict[str, Dict[str, Any]]:
    """
    获取所有可用的算法信息

    返回:
        算法字典，键为算法标识符，值为包含以下字段的字典：
        - module: 算法模块
        - display_name: 显示名称
        - default_params: 默认参数
        - param_help: 参数帮助文本
        - compress_func: 压缩函数
    """
    return _AVAILABLE_ALGORITHMS.copy()

def run_algorithm(algorithm_key: str, points, params: Dict):
    """
    运行指定的算法

    参数:
        algorithm_key: 算法标识符
        points: 输入轨迹点（DataFrame 或其他格式）
        params: 算法参数

    返回:
        压缩后的轨迹
    """
    if algorithm_key not in _AVAILABLE_ALGORITHMS:
        raise ValueError(f"未知算法: {algorithm_key}")

    alg_info = _AVAILABLE_ALGORITHMS[algorithm_key]
    return alg_info['compress_func'](points, params)

# 为了向后兼容，保留旧的导入（如果需要）
try:
    from .algorithms import (
        dead_reckoning_compress,
        adaptive_dr_compress,
        sliding_window_compress,
        opening_window_compress,
        dp_compress
    )
    # 导出旧接口
    _legacy_algorithms = [
        'dead_reckoning_compress', 'adaptive_dr_compress',
        'sliding_window_compress', 'opening_window_compress', 'dp_compress'
    ]
except ImportError:
    _legacy_algorithms = []

__version__ = "1.0.0"
__all__ = [
    # 数据加载
    'load_data', 'dataframe_to_trajectory_points', 'trajectory_points_to_dataframe', 'TrajectoryPoint',
    'scan_categories', 'scan_datasets', 'load_ais_dataset',
    # 新算法接口
    'get_available_algorithms', 'run_algorithm',
    # 工具
    'GeoUtils',
    # 评估
    'calculate_sed_metrics', 'calculate_navigation_event_recall', 'evaluate_compression',
    # 可视化
    'visualize_trajectories'
] + _legacy_algorithms
