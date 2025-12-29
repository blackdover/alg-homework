import pandas as pd
from typing import Dict


def compress(points: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    自适应阈值 Dead Reckoning 轨迹压缩算法

    算法逻辑：
    通过预测当前位置并与实际位置对比，如果预测误差小于阈值，则认为当前点冗余，可以丢弃。

    参数:
        points: 输入轨迹DataFrame，必须包含列：BaseDateTime, LAT, LON, SOG, COG
        params: 参数字典，包含：
            - min_threshold: 最低距离阈值（米）
            - max_threshold: 最高距离阈值（米）
            - v_lower: 低速截止点（节）
            - v_upper: 高速截止点（节）

    返回:
        压缩后的DataFrame
    """
    from ..utils.geo_utils import GeoUtils

    df = points  # 重命名以保持兼容性

    if len(df) == 0:
        return df.copy()
    if len(df) == 1:
        return df.copy()

    compressed_indices = [0]
    last_index = 0

    for i in range(1, len(df)):
        last_point = df.iloc[last_index]
        current_point = df.iloc[i]

        delta_t = (current_point['BaseDateTime'] - last_point['BaseDateTime']).total_seconds()
        if delta_t <= 0:
            continue

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
            compressed_indices.append(i)
            last_index = i

    if compressed_indices[-1] != len(df) - 1:
        compressed_indices.append(len(df) - 1)

    # 返回并保留原始索引 orig_idx
    return df.iloc[compressed_indices].reset_index(drop=False).rename(columns={"index": "orig_idx"})


# 算法元数据
DISPLAY_NAME = "自适应阈值 DR"
DEFAULT_PARAMS = {
    'min_threshold': 20.0,
    'max_threshold': 500.0,
    'v_lower': 3.0,
    'v_upper': 20.0
}
PARAM_HELP = {
    'min_threshold': '最低距离阈值（米）',
    'max_threshold': '最高距离阈值（米）',
    'v_lower': '低速截止点（节）',
    'v_upper': '高速截止点（节）'
}
