import pandas as pd  # 导入pandas数据处理库，用于处理轨迹数据
from typing import Dict  # 导入类型注解，用于函数参数类型提示


def compress(points: pd.DataFrame, params: Dict) -> pd.DataFrame:  # 主压缩函数，实现自适应阈值DR算法
    """
    自适应阈值Dead Reckoning轨迹压缩算法

    算法逻辑：
    通过预测当前位置并与实际位置对比，如果预测误差小于阈值，则认为当前点冗余，可以丢弃。
    与基本DR算法不同，这个算法会根据船速动态调整距离阈值：高速时阈值较大，低速时阈值较小。

    参数:
        points: 输入轨迹数据框，必须包含列：BaseDateTime（时间）、LAT（纬度）、LON（经度）、SOG（速度）、COG（航向）
        params: 参数配置字典，包含：
            - min_threshold: 最低距离阈值（米），用于低速情况
            - max_threshold: 最高距离阈值（米），用于高速情况
            - v_lower: 低速截止点（节），低于此速度使用最小阈值
            - v_upper: 高速截止点（节），高于此速度使用最大阈值

    返回:
        压缩后的数据框，包含原始索引信息
    """
    from ..utils.geo_utils import GeoUtils  # 导入地理工具类，用于距离计算和位置预测

    df = points  # 将输入数据框赋值给df，便于后续处理
    # 确保BaseDateTime列是datetime类型，避免后续时间计算出错
    try:
        df = df.copy()  # 创建数据框副本，避免修改原数据
        df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], errors='coerce')  # 转换为datetime格式
    except Exception:  # 如果转换失败
        pass  # 跳过转换，继续执行
    if len(df) == 0:  # 如果数据框为空
        return df.copy()  # 直接返回副本
    if len(df) == 1:  # 如果只有一个数据点
        return df.copy()  # 无法压缩，直接返回

    compressed_indices = [0]  # 初始化压缩结果索引列表，包含第一个点作为起始锚点
    last_index = 0  # 记录最后一个保留点的索引，用于后续预测计算

    for i in range(1, len(df)):  # 从第二个点开始遍历整个轨迹序列
        last_point = df.iloc[last_index]  # 获取上一个保留点（锚点）的数据
        current_point = df.iloc[i]  # 获取当前正在处理的轨迹点

        delta_t = (current_point['BaseDateTime'] - last_point['BaseDateTime']).total_seconds()  # 计算两点间的时间差（秒）
        if delta_t <= 0:  # 如果时间差无效或为负值（数据异常）
            continue  # 跳过这个异常点，继续处理下一个点

        speed_knots = last_point['SOG']  # 获取锚点的速度信息（单位：节）
        course_deg = last_point['COG']   # 获取锚点的航向信息（单位：度）
        speed_mps = GeoUtils.knots_to_mps(speed_knots)  # 将速度从节转换为米/秒，便于计算

        # 使用航位推算根据锚点状态预测当前时刻的位置
        pred_lat, pred_lon = GeoUtils.predict_position(  # 调用地理工具的位置预测函数
            lat_old=last_point['LAT'],    # 锚点的纬度坐标
            lon_old=last_point['LON'],    # 锚点的经度坐标
            speed_mps=speed_mps,          # 转换为米/秒的速度
            course_deg=course_deg,        # 航向角度（度）
            delta_t=delta_t               # 时间间隔（秒）
        )

        # 计算实际GPS测量位置与航位推算预测位置之间的距离误差
        error = GeoUtils.haversine_distance(  # 使用球面距离公式计算误差
            lat1=current_point['LAT'],   # 实际测量纬度
            lon1=current_point['LON'],   # 实际测量经度
            lat2=pred_lat,               # 预测纬度
            lon2=pred_lon                # 预测经度
        )

        threshold = GeoUtils.get_linear_threshold(speed_knots, params)  # 根据当前速度计算自适应距离阈值

        if error >= threshold:  # 如果预测误差超过了自适应阈值
            compressed_indices.append(i)  # 将当前点加入保留点列表
            last_index = i  # 更新锚点为当前点，用于后续预测

    if compressed_indices[-1] != len(df) - 1:  # 确保轨迹的最后一个点总是被保留
        compressed_indices.append(len(df) - 1)  # 如果最后一个点没有被保留，强制添加它

    # 返回压缩后的数据框，重置索引并保留原始索引信息用于评估
    return df.iloc[compressed_indices].reset_index(drop=False).rename(columns={"index": "orig_idx"})


# 算法元数据定义 - 用于GUI界面显示和参数配置
DISPLAY_NAME = "自适应阈值 DR"  # 算法在界面中显示的名称
DEFAULT_PARAMS = {  # 默认参数配置字典
    'min_threshold': 20.0,   # 最低距离阈值（米），用于低速情况
    'max_threshold': 500.0,  # 最高距离阈值（米），用于高速情况
    'v_lower': 3.0,          # 低速截止点（节），低于此速度使用最小阈值
    'v_upper': 20.0          # 高速截止点（节），高于此速度使用最大阈值
}
PARAM_HELP = {  # 参数帮助说明字典，告诉用户每个参数的含义
    'min_threshold': '最低距离阈值（米）',    # 最小阈值的说明
    'max_threshold': '最高距离阈值（米）',    # 最大阈值的说明
    'v_lower': '低速截止点（节）',           # 低速阈值的说明
    'v_upper': '高速截止点（节）'            # 高速阈值的说明
}
