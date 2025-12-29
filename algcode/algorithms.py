"""
轨迹压缩算法模块
包含多种轨迹简化算法实现

作者: Algorithm Engineer
Python版本: 3.10
"""

import pandas as pd
import numpy as np
import heapq
from typing import Optional, Tuple, Dict, List, Any
from .dataloader import TrajectoryPoint
from rdp import rdp  # 需要安装: pip install rdp


def dead_reckoning_compress(df: pd.DataFrame, threshold_meters: float = 100.0) -> pd.DataFrame:
    """
    基于航位推算(Dead Reckoning)的在线轨迹压缩算法

    逻辑说明：
        这是Online处理模式的核心算法。通过预测当前位置并与实际位置对比，
        如果预测误差小于阈值，则认为当前点冗余，可以丢弃。

    参数:
        df: 输入轨迹DataFrame，必须包含列：BaseDateTime, LAT, LON, SOG, COG
        threshold_meters: 距离误差阈值（米），默认100米

    返回:
        压缩后的DataFrame
    """
    from .geo_utils import GeoUtils

    if len(df) == 0:
        return df.copy()

    if len(df) == 1:
        return df.copy()

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


def adaptive_dr_compress(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    自适应阈值 Dead Reckoning 轨迹压缩算法

    算法逻辑：
    通过预测当前位置并与实际位置对比，如果预测误差小于阈值，则认为当前点冗余，可以丢弃。

    参数:
        df: 输入轨迹DataFrame，必须包含列：BaseDateTime, LAT, LON, SOG, COG
        params: 参数字典，包含：
            - min_threshold: 最低距离阈值（米）
            - max_threshold: 最高距离阈值（米）
            - v_lower: 低速截止点（节）
            - v_upper: 高速截止点（节）

    返回:
        压缩后的DataFrame
    """
    from .geo_utils import GeoUtils

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

    return df.iloc[compressed_indices].reset_index(drop=True)


def semantic_dr_compress(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    语义增强 Dead Reckoning 轨迹压缩算法

    在 DR 基础上叠加航行事件约束：急转弯、急加速、长时间无数据点必须保留。

    参数:
        df: 输入轨迹DataFrame
        params: 参数字典，包含：
            - min_threshold: 最低距离阈值（米）
            - max_threshold: 最高距离阈值（米）
            - v_lower: 低速截止点（节）
            - v_upper: 高速截止点（节）
            - cog_threshold: 转向角度阈值（度）
            - sog_threshold: 速度变化阈值（节）
            - time_threshold: 时间间隔阈值（秒）
    """
    from .geo_utils import GeoUtils

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


def sliding_window_compress(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Sliding Window 轨迹压缩算法

    以最后一个保留点为起点，不断往前看新点；如果新点相对起点→新点这条线的偏差超过阈值，
    就把上一个点保留下来当新起点。

    参数:
        df: 输入轨迹DataFrame
        params: 参数字典，包含：
            - epsilon: 距离阈值（米）
    """
    from .geo_utils import GeoUtils

    if len(df) <= 2:
        return df.copy()

    epsilon = params.get('epsilon', 100.0)
    compressed_indices = [0]  # 起始点

    i = 1
    while i < len(df):
        # 从当前锚点开始，寻找最远的点使得所有中间点都在容差内
        anchor_idx = compressed_indices[-1]

        # 使用二分查找优化：找到第一个超出误差的点
        left, right = i, len(df) - 1
        farthest_valid = i - 1  # 初始化为无效值

        while left <= right:
            mid = (left + right) // 2

            # 检查从i到mid的所有点是否都在anchor->mid线的容差内
            valid = True
            for k in range(i, mid + 1):
                error = GeoUtils.point_to_line_distance(
                    df.iloc[k]['LAT'], df.iloc[k]['LON'],
                    df.iloc[anchor_idx]['LAT'], df.iloc[anchor_idx]['LON'],
                    df.iloc[mid]['LAT'], df.iloc[mid]['LON']
                )
                if error > epsilon:
                    valid = False
                    break

            if valid:
                # mid点有效，可以尝试更远的点
                farthest_valid = mid
                left = mid + 1
            else:
                # mid点无效，尝试更近的点
                right = mid - 1

        # 确定下一个保留点
        if farthest_valid >= i:
            # 找到了有效的远点
            compressed_indices.append(farthest_valid)
            i = farthest_valid + 1
        else:
            # 连当前点都无效，直接保留当前点
            compressed_indices.append(i)
            i += 1

    # 确保包含最后一个点
    if compressed_indices[-1] != len(df) - 1:
        compressed_indices.append(len(df) - 1)

    return df.iloc[compressed_indices].reset_index(drop=True)


def opening_window_compress(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Opening Window 轨迹压缩算法

    从 anchor 点开始开窗，尽可能延长窗口右端；只要窗口内所有点对 anchor→窗口末端的误差都 ≤ ε，
    就继续扩张；一旦出现超阈值点，就输出上一个窗口末端并重开窗口。

    参数:
        df: 输入轨迹DataFrame
        params: 参数字典，包含：
            - epsilon: 距离阈值（米）
    """
    from .geo_utils import GeoUtils

    if len(df) <= 2:
        return df.copy()

    epsilon = params.get('epsilon', 100.0)
    compressed_indices = []

    i = 0
    while i < len(df) - 1:
        # 从当前点开始开窗
        anchor_idx = i
        window_end = i + 1

        # 尽可能延长窗口 - 使用二分查找优化
        left, right = i + 1, len(df) - 1
        best_end = i + 1  # 最小的有效窗口

        while left <= right:
            mid = (left + right) // 2

            # 检查从anchor到mid的所有点是否都在误差范围内
            valid = True
            for k in range(anchor_idx + 1, mid):
                error = GeoUtils.point_to_line_distance(
                    df.iloc[k]['LAT'], df.iloc[k]['LON'],
                    df.iloc[anchor_idx]['LAT'], df.iloc[anchor_idx]['LON'],
                    df.iloc[mid]['LAT'], df.iloc[mid]['LON']
                )
                if error > epsilon:
                    valid = False
                    break

            if valid:
                # 可以扩展更远
                best_end = mid
                left = mid + 1
            else:
                # 不能扩展到mid
                right = mid - 1

        window_end = best_end

        # 保留 anchor 点
        compressed_indices.append(anchor_idx)

        # 移动到窗口末端
        i = window_end

        # 安全检查：防止无限循环
        if i <= anchor_idx:
            # 如果i没有前进，强制前进到下一个点
            i = anchor_idx + 1

    # 确保包含最后一个点
    if not compressed_indices or compressed_indices[-1] != len(df) - 1:
        compressed_indices.append(len(df) - 1)

    return df.iloc[compressed_indices].reset_index(drop=True)


def squish_compress(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    SQUISH 流式轨迹简化算法

    维护一个大小为 B 的点集，每来一个新点就加入；如果超出 B，就删掉最不重要的点
    （重要性用该点在相邻两点形成的线段上的误差衡量），并更新邻居的分数。

    参数:
        df: 输入轨迹DataFrame
        params: 参数字典，包含：
            - buffer_size: 缓冲区大小 B
    """
    from .geo_utils import GeoUtils

    if len(df) <= 2:
        return df.copy()

    buffer_size = params.get('buffer_size', 100)

    class PointInfo:
        def __init__(self, idx: int, importance: float = 0.0):
            self.idx = idx
            self.importance = importance

        def __lt__(self, other):
            return self.importance < other.importance

    # 初始化缓冲区：前两个点
    buffer = [PointInfo(0, float('inf')), PointInfo(1, float('inf'))]
    heapq.heapify(buffer)

    # 计算重要性的辅助函数
    def update_importance(buffer_list: List[PointInfo]) -> None:
        if len(buffer_list) < 3:
            return

        # 按索引排序
        sorted_buffer = sorted(buffer_list, key=lambda x: x.idx)

        for i in range(1, len(sorted_buffer) - 1):
            curr = sorted_buffer[i]
            prev = sorted_buffer[i-1]
            next_p = sorted_buffer[i+1]

            # 计算当前点对 prev->next 线段的误差
            error = GeoUtils.point_to_line_distance(
                df.iloc[curr.idx]['LAT'], df.iloc[curr.idx]['LON'],
                df.iloc[prev.idx]['LAT'], df.iloc[prev.idx]['LON'],
                df.iloc[next_p.idx]['LAT'], df.iloc[next_p.idx]['LON']
            )
            curr.importance = error

    # 处理后续点
    for i in range(2, len(df)):
        # 添加新点
        new_point = PointInfo(i, 0.0)
        buffer.append(new_point)

        # 更新所有点的权重
        update_importance(buffer)

        # 如果超出缓冲区大小，移除最不重要的点
        while len(buffer) > buffer_size:
            # 重新建堆（因为重要性已更新）
            heapq.heapify(buffer)
            removed = heapq.heappop(buffer)

            # 从缓冲区移除该点
            buffer = [p for p in buffer if p.idx != removed.idx]

    # 返回缓冲区中的所有点，按原始顺序排序
    result_indices = sorted([p.idx for p in buffer])
    return df.iloc[result_indices].reset_index(drop=True)


def dp_compress(df: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    """
    Douglas-Peucker (DP) 离线轨迹压缩算法

    逻辑说明：
        经典的离线轨迹简化算法，用于对比基准。
        epsilon是垂直距离阈值（度），需要转换为米或直接使用。

    参数:
        df: 输入轨迹DataFrame，必须包含列：LAT, LON
        epsilon: 距离阈值（度），通常需要根据实际情况调整

    返回:
        压缩后的DataFrame
    """
    if len(df) <= 2:
        return df.copy()

    # 准备坐标点数组（用于rdp库）
    points = df[['LAT', 'LON']].values

    # 使用rdp库进行压缩
    # 注意：rdp的epsilon参数是欧氏距离，对于经纬度坐标需要适当调整
    # 这里我们将epsilon转换为近似的度数值
    # 1度纬度 ≈ 111km，所以epsilon度 ≈ epsilon * 111000 米
    # 但rdp使用的是欧氏距离，对于经纬度坐标，我们需要调整
    simplified_indices = rdp(points, epsilon=epsilon, return_mask=True)

    # 返回压缩后的DataFrame
    return df[simplified_indices].reset_index(drop=True)
