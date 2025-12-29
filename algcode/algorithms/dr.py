import pandas as pd  # 导入pandas数据处理库，用于处理轨迹数据
from typing import Dict  # 导入类型注解，用于函数参数类型提示


def compress(points: pd.DataFrame, params: Dict) -> pd.DataFrame:  # 主压缩函数，实现航位推算压缩算法
    """
    基于航位推算(Dead Reckoning)的在线轨迹压缩算法

    算法逻辑说明：
        这是在线处理模式的核心算法。通过根据上一保留点的位置、速度和航向预测当前位置，
        然后与实际GPS位置对比，如果预测误差小于阈值，则认为当前点是冗余的，可以丢弃。

    参数:
        points: 输入轨迹数据框，必须包含以下列：BaseDateTime（时间）、LAT（纬度）、LON（经度）、SOG（速度）、COG（航向）
        params: 参数配置字典，必须包含'epsilon'键，表示距离误差阈值（单位：米）

    返回:
        压缩后的数据框，包含原始索引信息
    """
    from ..utils.geo_utils import GeoUtils  # 导入地理工具类，用于距离计算和位置预测

    df = points  # 将输入重命名为df以保持代码兼容性
    # 确保BaseDateTime列是datetime类型，避免后续时间计算出错
    try:
        df = df.copy()  # 创建数据框副本，避免修改原数据
        df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], errors='coerce')  # 转换为datetime类型
    except Exception:  # 如果转换失败（比如已经是datetime类型）
        pass  # 跳过转换，继续执行
    if len(df) == 0:  # 如果数据框为空
        return df.copy()  # 直接返回副本

    if len(df) == 1:  # 如果只有一个数据点
        return df.copy()  # 无法压缩，直接返回

    threshold_meters = params.get('epsilon', 100.0)  # 获取距离阈值参数，默认100米

    # 初始化压缩结果：将第一个点作为初始锚点（基准点）
    compressed_indices = [0]  # 压缩后保留的点索引列表，初始包含第一个点
    last_index = 0  # 记录最后一个保留点的索引，用于后续预测

    # 从第二个点开始遍历整个轨迹流（在线处理模式）
    for i in range(1, len(df)):  # 遍历索引从1到最后一个点
        # 获取锚点（上一个被保留的点）和当前待判断的点
        last_point = df.iloc[last_index]    # 上一个保留点的数据
        current_point = df.iloc[i]          # 当前正在处理的点

        # 计算两个点之间的时间差（转换为秒）
        delta_t = (current_point['BaseDateTime'] - last_point['BaseDateTime']).total_seconds()

        # 如果时间差为0或负值（数据异常），跳过这个点
        if delta_t <= 0:
            continue  # 继续处理下一个点

        # 获取锚点的速度和航向信息，用于位置预测
        speed_knots = last_point['SOG']  # 速度（节，knots）
        course_deg = last_point['COG']   # 航向（度）

        # 将速度从节转换为米/秒（便于计算）
        speed_mps = GeoUtils.knots_to_mps(speed_knots)

        # 使用航位推算根据锚点信息预测当前时刻应该在的位置
        pred_lat, pred_lon = GeoUtils.predict_position(
            lat_old=last_point['LAT'],    # 锚点纬度
            lon_old=last_point['LON'],    # 锚点经度
            speed_mps=speed_mps,          # 速度（米/秒）
            course_deg=course_deg,        # 航向（度）
            delta_t=delta_t               # 时间差（秒）
        )

        # 计算实际GPS位置与预测位置之间的距离误差
        error = GeoUtils.haversine_distance(
            lat1=current_point['LAT'],   # 实际纬度
            lon1=current_point['LON'],   # 实际经度
            lat2=pred_lat,               # 预测纬度
            lon2=pred_lon                # 预测经度
        )

        # 判断是否保留当前点：如果预测误差超过阈值，说明预测不准确，需要保留这个点
        if error >= threshold_meters:  # 如果误差大于等于阈值
            compressed_indices.append(i)  # 将当前点索引加入保留列表
            last_index = i  # 更新锚点为当前点，用于后续预测

    # 确保轨迹的最后一个点总是被保留（无论误差多小）
    if compressed_indices[-1] != len(df) - 1:  # 如果最后一个点没有被保留
        compressed_indices.append(len(df) - 1)  # 添加最后一个点的索引

    # 返回压缩后的数据框，重置索引并保留原始索引信息
    return df.iloc[compressed_indices].reset_index(drop=False).rename(columns={"index": "orig_idx"})


# 算法元数据定义 - 用于GUI界面显示和参数配置
DISPLAY_NAME = "固定阈值 DR"  # 算法在界面中显示的名称
DEFAULT_PARAMS = {'epsilon': 100.0}  # 默认参数配置，距离阈值为100米
PARAM_HELP = {'epsilon': '距离阈值（米）'}  # 参数帮助说明，告诉用户epsilon参数的含义
