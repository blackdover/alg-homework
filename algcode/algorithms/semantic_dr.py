"""
语义增强 Dead Reckoning 轨迹压缩算法
"""

import pandas as pd
from typing import Dict


def compress(points: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    语义增强 Dead Reckoning 轨迹压缩算法

    在 DR 基础上叠加航行事件约束：急转弯、急加速、长时间无数据点必须保留。

    参数:
        points: 输入轨迹DataFrame
        params: 参数字典，包含：
            - min_threshold: 最低距离阈值（米）
            - max_threshold: 最高距离阈值（米）
            - v_lower: 低速截止点（节）
            - v_upper: 高速截止点（节）
            - cog_threshold: 转向角度阈值（度）
            - sog_threshold: 速度变化阈值（节）
            - time_threshold: 时间间隔阈值（秒）
    """
    from ..geo_utils import GeoUtils

    df = points  # 重命名以保持兼容性

    if len(df) == 0:
        return df.copy()
    if len(df) == 1:
        return df.copy()

    cog_threshold = params.get('cog_threshold', 10.0)
    sog_threshold = params.get('sog_threshold', 1.0)
    time_threshold = params.get('time_threshold', 300.0)  # 5分钟

    compressed_indices = [0]
    last_index = 0

    for i in range(1, len(df)):
        last_point = df.iloc[last_index]
        current_point = df.iloc[i]

        delta_t = (current_point['BaseDateTime'] - last_point['BaseDateTime']).total_seconds()

        # 硬约束检查：任一触发就保留当前点
        should_keep = False

        # 1. 转向约束
        if abs(current_point['COG'] - last_point['COG']) > cog_threshold:
            should_keep = True

        # 2. 速度变化约束
        if abs(current_point['SOG'] - last_point['SOG']) > sog_threshold:
            should_keep = True

        # 3. 时间间隔约束
        if delta_t > time_threshold:
            should_keep = True

        if not should_keep:
            # DR 逻辑：计算预测误差
            speed_knots = last_point['SOG']
            course_deg = last_point['COG']
            speed_mps = GeoUtils.knots_to_mps(speed_knots)

            pred_lat, pred_lon = GeoUtils.predict_position(
                lat_old=last_point['LAT'],
                lon_old=last_point['LON'],
                speed_mps=speed_mps,
                course_deg=course_deg,
                delta_t=delta_t
            )

            error = GeoUtils.haversine_distance(
                lat1=current_point['LAT'],
                lon1=current_point['LON'],
                lat2=pred_lat,
                lon2=pred_lon
            )

            threshold = GeoUtils.get_linear_threshold(speed_knots, params)
            if error >= threshold:
                should_keep = True

        if should_keep:
            compressed_indices.append(i)
            last_index = i

    if compressed_indices[-1] != len(df) - 1:
        compressed_indices.append(len(df) - 1)

    return df.iloc[compressed_indices].reset_index(drop=True)


# 算法元数据
DISPLAY_NAME = "语义增强 DR"
DEFAULT_PARAMS = {
    'min_threshold': 20.0,
    'max_threshold': 500.0,
    'v_lower': 3.0,
    'v_upper': 20.0,
    'cog_threshold': 10.0,
    'sog_threshold': 1.0,
    'time_threshold': 300.0
}
PARAM_HELP = {
    'min_threshold': '最低距离阈值（米）',
    'max_threshold': '最高距离阈值（米）',
    'v_lower': '低速截止点（节）',
    'v_upper': '高速截止点（节）',
    'cog_threshold': '转向角度阈值（度）',
    'sog_threshold': '速度变化阈值（节）',
    'time_threshold': '时间间隔阈值（秒）'
}
