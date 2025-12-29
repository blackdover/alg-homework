import pandas as pd
from typing import Dict


def compress(points: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    基于航位推算(Dead Reckoning)的在线轨迹压缩算法

    逻辑说明：
        这是Online处理模式的核心算法。通过预测当前位置并与实际位置对比，
        如果预测误差小于阈值，则认为当前点冗余，可以丢弃。

    参数:
        points: 输入轨迹DataFrame，必须包含列：BaseDateTime, LAT, LON, SOG, COG
        params: 参数字典，必须包含 'epsilon' 键（距离误差阈值，米）

    返回:
        压缩后的DataFrame
    """
    from ..utils.geo_utils import GeoUtils

    df = points  # 重命名以保持兼容性

    if len(df) == 0:
        return df.copy()

    if len(df) == 1:
        return df.copy()

    threshold_meters = params.get('epsilon', 100.0)

    # 初始化压缩点列表，存入第一个点作为初始锚点
    compressed_indices = [0]
    last_index = 0

    # 从第二个点开始遍历轨迹流
    for i in range(1, len(df)):
        # 获取锚点（上一个保留的点）
        last_point = df.iloc[last_index]
        current_point = df.iloc[i]

        # 计算时间差（秒）
        delta_t = (current_point['BaseDateTime'] - last_point['BaseDateTime']).total_seconds()

        # 如果时间差为0或负值，跳过
        if delta_t <= 0:
            continue

        # 获取锚点的速度和航向
        speed_knots = last_point['SOG']
        course_deg = last_point['COG']

        # 转换为米/秒
        speed_mps = GeoUtils.knots_to_mps(speed_knots)

        # 使用航位推算预测当前时刻的位置
        pred_lat, pred_lon = GeoUtils.predict_position(
            lat_old=last_point['LAT'],
            lon_old=last_point['LON'],
            speed_mps=speed_mps,
            course_deg=course_deg,
            delta_t=delta_t
        )

        # 计算实际点与预测点之间的距离误差
        error = GeoUtils.haversine_distance(
            lat1=current_point['LAT'],
            lon1=current_point['LON'],
            lat2=pred_lat,
            lon2=pred_lon
        )

        # 判断：若误差 >= 阈值，说明预测失效，保留当前点
        if error >= threshold_meters:
            compressed_indices.append(i)
            last_index = i

    # 确保轨迹的最后一个点总是被保留
    if compressed_indices[-1] != len(df) - 1:
        compressed_indices.append(len(df) - 1)

    # 返回压缩后的DataFrame
    return df.iloc[compressed_indices].reset_index(drop=True)


# 算法元数据
DISPLAY_NAME = "固定阈值 DR"
DEFAULT_PARAMS = {'epsilon': 100.0}
PARAM_HELP = {'epsilon': '距离阈值（米）'}
