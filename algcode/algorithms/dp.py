import pandas as pd
from typing import Dict, List


def _point_line_distance_m(lat, lon, lat1, lon1, lat2, lon2):
    """
    使用项目中的 GeoUtils 计算点到线段的距离（米）。
    这里在函数内部导入以避免循环依赖。
    """
    from ..utils.geo_utils import GeoUtils
    return GeoUtils.point_to_line_distance(lat, lon, lat1, lon1, lat2, lon2)


def _rdp_indices(df: pd.DataFrame, epsilon_m: float) -> List[int]:
    """
    返回保留点的索引列表（基于 Douglas-Peucker），使用距离单位为米。
    递归实现：在段 [start,end] 中找到最大垂直距离点，若大于阈值则分治。
    """
    indices: List[int] = []

    def recurse(start: int, end: int):
        if end <= start + 1:
            return

        # 选择段的端点
        lat1 = df.iloc[start]['LAT']
        lon1 = df.iloc[start]['LON']
        lat2 = df.iloc[end]['LAT']
        lon2 = df.iloc[end]['LON']

        # 查找最大距离
        max_dist = -1.0
        max_idx = -1
        for i in range(start + 1, end):
            lat = df.iloc[i]['LAT']
            lon = df.iloc[i]['LON']
            dist = _point_line_distance_m(lat, lon, lat1, lon1, lat2, lon2)
            if dist > max_dist:
                max_dist = dist
                max_idx = i

        if max_dist > epsilon_m and max_idx != -1:
            # 递归左段与右段
            recurse(start, max_idx)
            indices.append(max_idx)
            recurse(max_idx, end)

    # 总是保留首尾
    start_idx = 0
    end_idx = len(df) - 1
    indices = [start_idx]  # 初始化为包含起始点的列表
    recurse(start_idx, end_idx)
    # 确保包含结束点
    if end_idx not in indices:
        indices.append(end_idx)
    # 去重并排序
    indices = sorted(list(set(indices)))
    return indices


def compress(points: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Douglas-Peucker 手写实现（离线）。

    参数:
        points: 输入轨迹 DataFrame，必须包含列 LAT, LON, BaseDateTime 可选
        params: 参数字典：
            - epsilon: 当以度为单位时（默认），会按大致比例转换为米（*111000）
            - epsilon_m: 可直接传入以米为单位的阈值（优先）
    返回:
        压缩后的 DataFrame
    """
    df = points
    if len(df) <= 2:
        return df.copy()

    # 优先使用以米为单位的阈值，否则把度近似转换为米
    if 'epsilon_m' in params:
        eps_m = float(params['epsilon_m'])
    else:
        eps_deg = float(params.get('epsilon', 0.0009))
        eps_m = eps_deg * 111000.0

    indices = _rdp_indices(df, eps_m)
    return df.iloc[indices].reset_index(drop=False).rename(columns={"index": "orig_idx"})


# 算法元数据
DISPLAY_NAME = "DP算法"
DEFAULT_PARAMS = {'epsilon': 0.0009}
PARAM_HELP = {'epsilon': '距离阈值（度）'}
