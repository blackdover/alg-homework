"""
Douglas-Peucker (DP) 离线轨迹压缩算法
"""

import pandas as pd
from rdp import rdp
from typing import Dict


def compress(points: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Douglas-Peucker (DP) 离线轨迹压缩算法

    逻辑说明：
        经典的离线轨迹简化算法，用于对比基准。
        epsilon是垂直距离阈值（度），需要转换为米或直接使用。

    参数:
        points: 输入轨迹DataFrame，必须包含列：LAT, LON
        params: 参数字典，必须包含 'epsilon' 键（距离阈值，度）

    返回:
        压缩后的DataFrame
    """
    df = points  # 重命名以保持兼容性

    if len(df) <= 2:
        return df.copy()

    epsilon = params.get('epsilon', 0.0009)

    # 准备坐标点数组（用于rdp库）
    points_array = df[['LAT', 'LON']].values

    # 使用rdp库进行压缩
    # 注意：rdp的epsilon参数是欧氏距离，对于经纬度坐标需要适当调整
    # 这里我们将epsilon转换为近似的度数值
    # 1度纬度 ≈ 111km，所以epsilon度 ≈ epsilon * 111000 米
    # 但rdp使用的是欧氏距离，对于经纬度坐标，我们需要调整
    simplified_indices = rdp(points_array, epsilon=epsilon, return_mask=True)

    # 返回压缩后的DataFrame
    return df[simplified_indices].reset_index(drop=True)


# 算法元数据
DISPLAY_NAME = "Douglas-Peucker"
DEFAULT_PARAMS = {'epsilon': 0.0009}
PARAM_HELP = {'epsilon': '距离阈值（度）'}
