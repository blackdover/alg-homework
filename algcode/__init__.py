"""
轨迹压缩算法包 - 这是整个轨迹压缩算法库的主要包文件
包含多种轨迹简化算法实现，用于压缩和优化轨迹数据

作者: Algorithm Engineer  # 代码作者信息
Python版本: 3.10  # 要求的Python版本
"""

import os  # 操作系统接口模块，用于文件和路径操作
import importlib  # 动态导入模块的工具库
from typing import Dict, Any, List  # 类型注解，用于代码类型提示
from pathlib import Path  # 现代路径处理库，提供面向对象的路径操作

from .utils.dataloader import (  # 从utils子包导入数据加载功能
    load_data, dataframe_to_trajectory_points, trajectory_points_to_dataframe, TrajectoryPoint,  # 数据格式转换函数
    scan_categories, scan_datasets, load_ais_dataset  # 数据集扫描和加载函数
)
from .utils.geo_utils import GeoUtils  # 导入地理工具类
from .utils.metrics import calculate_sed_metrics, calculate_navigation_event_recall, evaluate_compression  # 导入评估指标计算函数
from .utils.visualization import visualize_trajectories  # 导入轨迹可视化函数

# 动态加载算法 - 以下变量用于存储算法信息
_ALGORITHMS_DIR = Path(__file__).parent / "algorithms"  # 算法目录路径，相对于当前文件位置
_AVAILABLE_ALGORITHMS = {}  # 全局字典，用于存储所有已加载的算法信息

def _load_algorithms():  # 私有函数，动态加载算法目录中的所有算法
    """动态加载algorithms目录中所有有效的算法模块"""
    global _AVAILABLE_ALGORITHMS  # 声明使用全局变量

    if not _ALGORITHMS_DIR.exists():  # 检查算法目录是否存在
        return  # 如果目录不存在，直接返回

    for py_file in _ALGORITHMS_DIR.glob("*.py"):  # 遍历algorithms目录下的所有Python文件
        module_name = py_file.stem  # 获取文件名（不含扩展名），作为模块名

        # 跳过被禁用的算法（这些算法可能有问题或不完整）
        if module_name in ['squish', 'semantic_dr']:  # 检查是否为禁用的算法
            continue  # 跳过这些算法

        try:  # 尝试动态导入和验证算法模块
            # 动态导入模块，使用相对导入路径
            module = importlib.import_module(f".algorithms.{module_name}", package=__name__)

            # 检查模块是否有必需的接口和元数据（compress函数和DISPLAY_NAME常量）
            if hasattr(module, 'compress') and hasattr(module, 'DISPLAY_NAME'):  # 验证必需属性
                alg_info = {  # 创建算法信息字典
                    'module': module,  # 算法模块对象
                    'display_name': getattr(module, 'DISPLAY_NAME', module_name),  # 显示名称，默认使用模块名
                    'default_params': getattr(module, 'DEFAULT_PARAMS', {}),  # 默认参数字典
                    'param_help': getattr(module, 'PARAM_HELP', {}),  # 参数帮助信息
                    'compress_func': module.compress  # 压缩函数引用
                }
                _AVAILABLE_ALGORITHMS[module_name] = alg_info  # 将算法信息添加到全局字典

        except Exception as e:  # 捕获导入或验证过程中的异常
            print(f"加载算法模块 {module_name} 失败: {e}")  # 输出错误信息
            continue  # 继续处理下一个模块

# 初始化时加载算法 - 在模块导入时自动执行算法加载
_load_algorithms()  # 调用函数加载所有可用算法

def get_available_algorithms() -> Dict[str, Dict[str, Any]]:  # 获取所有可用算法信息的公共接口函数
    """
    获取所有可用的算法信息字典

    返回:
        算法字典，键为算法标识符，值为包含以下字段的字典：
        - module: 算法模块对象
        - display_name: 在界面中显示的算法名称
        - default_params: 算法的默认参数配置
        - param_help: 参数的帮助说明文本
        - compress_func: 实际的压缩函数引用
    """
    return _AVAILABLE_ALGORITHMS.copy()  # 返回字典的副本，避免外部修改

def run_algorithm(algorithm_key: str, points, params: Dict):  # 运行指定算法的统一接口函数
    """
    运行指定的压缩算法

    参数:
        algorithm_key: 算法的唯一标识符字符串（如'dp', 'dr'等）
        points: 输入的轨迹点数据（可以是DataFrame或其他格式）
        params: 算法需要的参数配置字典

    返回:
        压缩后的轨迹数据（通常是DataFrame格式）
    """
    if algorithm_key not in _AVAILABLE_ALGORITHMS:  # 检查算法是否存在
        raise ValueError(f"未知算法: {algorithm_key}")  # 抛出错误，算法不存在

    alg_info = _AVAILABLE_ALGORITHMS[algorithm_key]  # 获取算法信息
    return alg_info['compress_func'](points, params)  # 调用算法的压缩函数并返回结果

# 为了向后兼容，保留旧的导入方式（如果需要）- 允许使用旧版本的函数名
try:  # 尝试导入旧版本的算法函数（为了兼容性）
    from .algorithms import (  # 从algorithms子包导入旧版本的函数
        dead_reckoning_compress,  # 死推算法压缩函数
        adaptive_dr_compress,     # 自适应死推算法压缩函数
        sliding_window_compress,  # 滑动窗口算法压缩函数
        opening_window_compress,  # 开窗算法压缩函数
        dp_compress               # 道格拉斯-普克算法压缩函数
    )
    # 导出旧接口函数名列表
    _legacy_algorithms = [  # 定义旧版本函数名列表
        'dead_reckoning_compress', 'adaptive_dr_compress',
        'sliding_window_compress', 'opening_window_compress', 'dp_compress'
    ]
except ImportError:  # 如果导入失败（旧版本函数不存在）
    _legacy_algorithms = []  # 设置为空列表

__version__ = "1.0.0"  # 包版本号
__all__ = [  # 定义包的公开接口列表（from package import * 时导入的内容）
    # 数据加载相关函数
    'load_data', 'dataframe_to_trajectory_points', 'trajectory_points_to_dataframe', 'TrajectoryPoint',
    'scan_categories', 'scan_datasets', 'load_ais_dataset',
    # 新版本的算法接口
    'get_available_algorithms', 'run_algorithm',
    # 地理工具类
    'GeoUtils',
    # 评估指标计算函数
    'calculate_sed_metrics', 'calculate_navigation_event_recall', 'evaluate_compression',
    # 轨迹可视化函数
    'visualize_trajectories'
] + _legacy_algorithms  # 加上旧版本的兼容函数
