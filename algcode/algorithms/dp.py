import pandas as pd  # 导入pandas数据处理库，用于处理轨迹数据框
from typing import Dict, List  # 导入类型注解，用于函数参数和返回值的类型提示


def _point_line_distance_m(lat, lon, lat1, lon1, lat2, lon2):  # 私有函数，计算点到线段的距离（米）
    """
    使用项目中的GeoUtils工具类计算点到线段的距离（单位为米）。
    在函数内部导入以避免循环依赖问题（因为其他地方可能也会导入这个模块）。
    """
    from ..utils.geo_utils import GeoUtils  # 动态导入地理工具类，避免循环导入
    return GeoUtils.point_to_line_distance(lat, lon, lat1, lon1, lat2, lon2)  # 调用工具函数计算距离并返回


def _rdp_indices(df: pd.DataFrame, epsilon_m: float) -> List[int]:  # 私有函数，实现RDP算法的核心逻辑
    """
    返回需要保留的轨迹点的索引列表（基于Douglas-Peucker算法），距离单位为米。
    递归实现：在轨迹段[start,end]中找到垂直距离最大的点，如果超过阈值则继续分割。
    """
    indices: List[int] = []  # 初始化结果索引列表，用于存储需要保留的点索引

    def recurse(start: int, end: int):  # 内部递归函数，处理轨迹段[start, end]
        if end <= start + 1:  # 如果段长度小于等于1（只有2个点或更少），不需要处理
            return  # 直接返回

        # 获取当前轨迹段的两个端点坐标
        lat1 = df.iloc[start]['LAT']  # 起始点纬度
        lon1 = df.iloc[start]['LON']  # 起始点经度
        lat2 = df.iloc[end]['LAT']    # 结束点纬度
        lon2 = df.iloc[end]['LON']    # 结束点经度

        # 在当前段中查找垂直距离最大的点
        max_dist = -1.0  # 初始化最大距离为-1
        max_idx = -1     # 初始化最大距离点索引为-1
        for i in range(start + 1, end):  # 遍历段内的所有中间点（不包括端点）
            lat = df.iloc[i]['LAT']  # 当前点的纬度
            lon = df.iloc[i]['LON']  # 当前点的经度
            dist = _point_line_distance_m(lat, lon, lat1, lon1, lat2, lon2)  # 计算点到线段的距离
            if dist > max_dist:  # 如果当前距离大于最大距离
                max_dist = dist  # 更新最大距离
                max_idx = i      # 更新最大距离点索引

        if max_dist > epsilon_m and max_idx != -1:  # 如果最大距离超过阈值且找到了有效点
            # 递归处理左段和右段（分治策略）
            recurse(start, max_idx)  # 递归处理起始点到最大距离点的段
            indices.append(max_idx)  # 将最大距离点加入保留列表
            recurse(max_idx, end)    # 递归处理最大距离点到结束点的段

    # 主函数逻辑：总是保留轨迹的首尾两个点
    start_idx = 0  # 轨迹起始点索引
    end_idx = len(df) - 1  # 轨迹结束点索引
    indices = [start_idx]  # 初始化索引列表，包含起始点
    recurse(start_idx, end_idx)  # 调用递归函数处理整个轨迹
    # 确保结束点被包含在结果中
    if end_idx not in indices:  # 如果结束点不在索引列表中
        indices.append(end_idx)  # 添加结束点
    # 对索引列表去重并排序，确保结果正确
    indices = sorted(list(set(indices)))  # 去重、排序并转换回列表
    return indices  # 返回需要保留的点索引列表


def compress(points: pd.DataFrame, params: Dict) -> pd.DataFrame:  # 主压缩函数，实现Douglas-Peucker算法
    """
    Douglas-Peucker轨迹压缩算法的手写实现（离线版本）。

    参数:
        points: 输入轨迹数据框，必须包含LAT、LON列，BaseDateTime列可选
        params: 参数配置字典：
            - epsilon: 当以度为单位时（默认值），会按大致比例转换为米（*111000）
            - epsilon_m: 可直接传入以米为单位的距离阈值（优先级更高）
    返回:
        压缩后的数据框，包含原始索引信息
    """
    df = points  # 使用输入的数据框
    if len(df) <= 2:  # 如果轨迹点数不超过2个，无法压缩
        return df.copy()  # 直接返回原数据框的副本

    # 处理距离阈值参数：优先使用米单位，否则将度转换为米
    if 'epsilon_m' in params:  # 如果直接提供了米单位的阈值
        eps_m = float(params['epsilon_m'])  # 直接使用米单位的值
    else:  # 否则使用度单位，然后转换为米（1度约等于111公里）
        eps_deg = float(params.get('epsilon', 0.0009))  # 获取度单位的值，默认0.0009度
        eps_m = eps_deg * 111000.0  # 将度转换为米（近似转换）

    indices = _rdp_indices(df, eps_m)  # 调用RDP算法获取需要保留的点索引
    # 返回压缩后的数据框，重置索引并保留原始索引信息
    return df.iloc[indices].reset_index(drop=False).rename(columns={"index": "orig_idx"})


# 算法元数据定义 - 用于GUI界面显示和参数配置
DISPLAY_NAME = "DP算法"  # 算法在界面中显示的名称
DEFAULT_PARAMS = {'epsilon': 0.0009}  # 默认参数配置，epsilon为0.0009度
PARAM_HELP = {'epsilon': '距离阈值（度）'}  # 参数帮助说明，告诉用户epsilon参数的含义
