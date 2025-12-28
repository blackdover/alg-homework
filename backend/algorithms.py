"""
轨迹压缩算法集合
手工实现多个轨迹简化算法，用于对比分析

作者: Algorithm Engineer
Python版本: 3.10
"""

import pandas as pd
import numpy as np
import time
import math
import heapq
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass


# ============================================================================
# 数据结构定义 (Data Structures)
# ============================================================================

@dataclass
class TrajectoryPoint:
    """轨迹点数据结构"""
    lat: float
    lon: float
    timestamp: pd.Timestamp
    sog: float  # Speed Over Ground (knots)
    cog: float  # Course Over Ground (degrees)
    mmsi: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'LAT': self.lat,
            'LON': self.lon,
            'BaseDateTime': self.timestamp,
            'SOG': self.sog,
            'COG': self.cog,
            'MMSI': self.mmsi
        }


@dataclass
class CompressionResult:
    """压缩算法结果"""
    algorithm: str
    compressed_points: List[TrajectoryPoint]
    compression_ratio: float
    elapsed_time: float
    metrics: Dict[str, Any]


# ============================================================================
# 地理计算工具类 (Geo Utils)
# ============================================================================

class GeoUtils:
    """地理计算工具类，处理核心数学计算"""

    # 地球半径（米）
    EARTH_RADIUS_M = 6371000.0

    # 节到米/秒的转换系数
    KNOTS_TO_MPS = 0.514444

    @staticmethod
    def knots_to_mps(knots: float) -> float:
        """将航速从节(knots)转换为米/秒(m/s)"""
        return knots * GeoUtils.KNOTS_TO_MPS

    @staticmethod
    def deg_to_rad(degrees: float) -> float:
        """角度转弧度"""
        return math.radians(degrees)

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """计算两点间的地球表面距离（Haversine公式）"""
        # 转换为弧度
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # 计算差值
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        # Haversine公式
        a = math.sin(dlat / 2) ** 2 + \
            math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))

        # 返回距离（米）
        return GeoUtils.EARTH_RADIUS_M * c

    @staticmethod
    def predict_position(lat_old: float, lon_old: float,
                        speed_mps: float, course_deg: float,
                        delta_t: float) -> Tuple[float, float]:
        """航位预测：根据起始点状态和时间差预测新位置"""
        # 计算行驶距离
        distance = speed_mps * delta_t

        # 将航向转换为弧度
        course_rad = math.radians(course_deg)

        # 转换为弧度
        lat_old_rad = math.radians(lat_old)
        lon_old_rad = math.radians(lon_old)

        # 计算经纬度偏移量（使用近似公式）
        dlat_rad = (distance * math.cos(course_rad)) / GeoUtils.EARTH_RADIUS_M
        dlon_rad = (distance * math.sin(course_rad)) / (GeoUtils.EARTH_RADIUS_M * math.cos(lat_old_rad))

        # 计算新位置
        lat_new_rad = lat_old_rad + dlat_rad
        lon_new_rad = lon_old_rad + dlon_rad

        # 转换回度
        lat_new = math.degrees(lat_new_rad)
        lon_new = math.degrees(lon_new_rad)

        return lat_new, lon_new

    @staticmethod
    def get_linear_threshold(speed_knots: float, params: Dict) -> float:
        """根据航速计算动态距离阈值 (连续线性映射方案)"""
        epsilon_min = params.get('min_threshold', 20.0)
        epsilon_max = params.get('max_threshold', 500.0)
        v_lower = params.get('v_lower', 3.0)
        v_upper = params.get('v_upper', 20.0)

        # 1. 低速区间：保持高精度
        if speed_knots <= v_lower:
            return epsilon_min

        # 2. 高速区间：保持最大压缩
        if speed_knots >= v_upper:
            return epsilon_max

        # 3. 过渡区间：线性插值
        k = (epsilon_max - epsilon_min) / (v_upper - v_lower)
        current_epsilon = k * (speed_knots - v_lower) + epsilon_min

        return current_epsilon

    @staticmethod
    def point_to_line_distance(lat: float, lon: float,
                              lat1: float, lon1: float,
                              lat2: float, lon2: float) -> float:
        """计算点到线段的垂直距离（PED - Perpendicular Error Distance）"""
        # 转换为米坐标（近似）
        # 1度纬度 ≈ 111km，1度经度 ≈ 111km * cos(lat)
        avg_lat = (lat1 + lat2) / 2
        cos_lat = math.cos(math.radians(avg_lat))

        # 坐标转换
        x = lon * 111000 * cos_lat
        y = lat * 111000
        x1 = lon1 * 111000 * cos_lat
        y1 = lat1 * 111000
        x2 = lon2 * 111000 * cos_lat
        y2 = lat2 * 111000

        # 计算点到线段的距离
        if x1 == x2 and y1 == y2:
            return math.sqrt((x - x1)**2 + (y - y1)**2)

        # 线段向量
        dx = x2 - x1
        dy = y2 - y1

        # 点到起点向量
        px = x - x1
        py = y - y1

        # 投影长度
        t = max(0, min(1, (px * dx + py * dy) / (dx * dx + dy * dy)))

        # 最近点坐标
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        # 计算距离
        return math.sqrt((x - closest_x)**2 + (y - closest_y)**2)


# ============================================================================
# 数据转换工具 (Data Conversion)
# ============================================================================

def dataframe_to_trajectory_points(df: pd.DataFrame) -> List[TrajectoryPoint]:
    """将 pandas DataFrame 转换为 TrajectoryPoint 列表"""
    points = []
    for _, row in df.iterrows():
        point = TrajectoryPoint(
            lat=row['LAT'],
            lon=row['LON'],
            timestamp=row['BaseDateTime'],
            sog=row['SOG'],
            cog=row['COG'],
            mmsi=row.get('MMSI')
        )
        points.append(point)
    return points


def trajectory_points_to_dataframe(points: List[TrajectoryPoint]) -> pd.DataFrame:
    """将 TrajectoryPoint 列表转换为 pandas DataFrame"""
    data = [point.to_dict() for point in points]
    return pd.DataFrame(data)


# ============================================================================
# 评估指标计算 (Evaluation Metrics)
# ============================================================================

def calculate_sed_metrics(original_df: pd.DataFrame,
                         compressed_df: pd.DataFrame) -> Dict[str, float]:
    """计算 SED (Spatial Error Distance) 相关指标"""
    if len(compressed_df) <= 1:
        return {'mean': 0.0, 'max': 0.0, 'p95': 0.0}

    sed_values = []

    # 为每个原始点找到最近的压缩段，计算 SED
    compressed_coords = compressed_df[['LAT', 'LON']].values

    for i in range(len(compressed_df) - 1):
        segment_start = compressed_df.iloc[i]
        segment_end = compressed_df.iloc[i + 1]

        # 找到该段对应的原始点
        segment_points = []
        start_time = segment_start['BaseDateTime']
        end_time = segment_end['BaseDateTime']

        for _, point in original_df.iterrows():
            if start_time <= point['BaseDateTime'] <= end_time:
                segment_points.append(point)

        # 计算该段内各点的 SED
        for point in segment_points:
            sed = GeoUtils.point_to_line_distance(
                point['LAT'], point['LON'],
                segment_start['LAT'], segment_start['LON'],
                segment_end['LAT'], segment_end['LON']
            )
            sed_values.append(sed)

    if not sed_values:
        return {'mean': 0.0, 'max': 0.0, 'p95': 0.0}

    sed_array = np.array(sed_values)
    return {
        'mean': float(np.mean(sed_array)),
        'max': float(np.max(sed_array)),
        'p95': float(np.percentile(sed_array, 95))
    }


def calculate_navigation_event_recall(original_df: pd.DataFrame,
                                    compressed_df: pd.DataFrame,
                                    cog_threshold: float = 20.0) -> float:
    """计算航行事件保留率（转向事件 recall）"""
    if len(original_df) < 2:
        return 1.0

    # 计算原始轨迹中的转向事件
    original_turns = 0
    for i in range(1, len(original_df)):
        prev_cog = original_df.iloc[i-1]['COG']
        curr_cog = original_df.iloc[i]['COG']
        if abs(curr_cog - prev_cog) > cog_threshold:
            original_turns += 1

    if original_turns == 0:
        return 1.0

    # 计算压缩轨迹中保留的转向事件
    compressed_turns = 0
    for i in range(1, len(compressed_df)):
        prev_cog = compressed_df.iloc[i-1]['COG']
        curr_cog = compressed_df.iloc[i]['COG']
        if abs(curr_cog - prev_cog) > cog_threshold:
            compressed_turns += 1

    return compressed_turns / original_turns


def evaluate_compression(original_df: pd.DataFrame,
                        compressed_df: pd.DataFrame,
                        algorithm_name: str,
                        elapsed_time: float = None) -> Dict[str, Any]:
    """计算并返回压缩算法的评估指标"""
    N = len(original_df)  # 原始点数
    M = len(compressed_df)  # 压缩后点数

    # 压缩率
    compression_ratio = (1 - M / N) * 100 if N > 0 else 0.0

    # SED 指标
    sed_metrics = calculate_sed_metrics(original_df, compressed_df)

    # 航行事件保留率
    event_recall = calculate_navigation_event_recall(original_df, compressed_df)

    return {
        'algorithm': algorithm_name,
        'original_points': N,
        'compressed_points': M,
        'compression_ratio': compression_ratio,
        'sed_mean': sed_metrics['mean'],
        'sed_max': sed_metrics['max'],
        'sed_p95': sed_metrics['p95'],
        'event_recall': event_recall,
        'elapsed_time': elapsed_time
    }


# ============================================================================
# 轨迹压缩算法实现 (Compression Algorithms)
# ============================================================================

def dead_reckoning_compress(points: List[TrajectoryPoint], params: Dict) -> List[TrajectoryPoint]:
    """
    自适应阈值 Dead Reckoning 轨迹压缩算法

    算法逻辑：
    通过预测当前位置并与实际位置对比，如果预测误差小于阈值，则认为当前点冗余，可以丢弃。

    参数:
        points: 轨迹点列表
        params: 参数字典，包含：
            - min_threshold: 最低距离阈值（米）
            - max_threshold: 最高距离阈值（米）
            - v_lower: 低速截止点（节）
            - v_upper: 高速截止点（节）

    返回:
        压缩后的轨迹点列表
    """
    if len(points) == 0:
        return []
    if len(points) == 1:
        return points.copy()

    compressed_indices = [0]  # 存入第一个点作为初始锚点
    last_index = 0

    # 从第二个点开始遍历轨迹流
    for i in range(1, len(points)):
        # 获取锚点（上一个保留的点）
        last_point = points[last_index]
        current_point = points[i]

        # 计算时间差（秒）
        delta_t = (current_point.timestamp - last_point.timestamp).total_seconds()

        # 如果时间差为0或负值，跳过
        if delta_t <= 0:
            continue

        # 获取锚点的速度和航向
        speed_knots = last_point.sog
        course_deg = last_point.cog

        # 转换为米/秒
        speed_mps = GeoUtils.knots_to_mps(speed_knots)

        # 使用航位推算预测当前时刻的位置
        pred_lat, pred_lon = GeoUtils.predict_position(
            lat_old=last_point.lat,
            lon_old=last_point.lon,
            speed_mps=speed_mps,
            course_deg=course_deg,
            delta_t=delta_t
        )

        # 计算实际点与预测点之间的距离误差
        error = GeoUtils.haversine_distance(
            lat1=current_point.lat,
            lon1=current_point.lon,
            lat2=pred_lat,
            lon2=pred_lon
        )

        # 根据当前速度计算动态阈值
        threshold = GeoUtils.get_linear_threshold(speed_knots, params)

        # 判断：若误差 >= 阈值，说明预测失效，保留当前点
        if error >= threshold:
            compressed_indices.append(i)
            last_index = i

    # 确保轨迹的最后一个点总是被保留
    if compressed_indices[-1] != len(points) - 1:
        compressed_indices.append(len(points) - 1)

    # 返回压缩后的点列表
    return [points[i] for i in compressed_indices]


def fixed_epsilon_dr_compress(points: List[TrajectoryPoint], params: Dict) -> List[TrajectoryPoint]:
    """
    固定阈值 Dead Reckoning 轨迹压缩算法

    与自适应 DR 相同，但使用固定阈值而非动态阈值。

    参数:
        points: 轨迹点列表
        params: 参数字典，包含：
            - epsilon: 固定距离阈值（米）
    """
    if len(points) == 0:
        return []
    if len(points) == 1:
        return points.copy()

    epsilon = params.get('epsilon', 100.0)
    compressed_indices = [0]
    last_index = 0

    for i in range(1, len(points)):
        last_point = points[last_index]
        current_point = points[i]

        delta_t = (current_point.timestamp - last_point.timestamp).total_seconds()
        if delta_t <= 0:
            continue

        speed_knots = last_point.sog
        course_deg = last_point.cog
        speed_mps = GeoUtils.knots_to_mps(speed_knots)

        pred_lat, pred_lon = GeoUtils.predict_position(
            lat_old=last_point.lat,
            lon_old=last_point.lon,
            speed_mps=speed_mps,
            course_deg=course_deg,
            delta_t=delta_t
        )

        error = GeoUtils.haversine_distance(
            lat1=current_point.lat,
            lon1=current_point.lon,
            lat2=pred_lat,
            lon2=pred_lon
        )

        if error >= epsilon:
            compressed_indices.append(i)
            last_index = i

    if compressed_indices[-1] != len(points) - 1:
        compressed_indices.append(len(points) - 1)

    return [points[i] for i in compressed_indices]


def semantic_dr_compress(points: List[TrajectoryPoint], params: Dict) -> List[TrajectoryPoint]:
    """
    语义增强 Dead Reckoning 轨迹压缩算法

    在 DR 基础上叠加航行事件约束：急转弯、急加速、长时间无数据点必须保留。

    参数:
        points: 轨迹点列表
        params: 参数字典，包含：
            - min_threshold: 最低距离阈值（米）
            - max_threshold: 最高距离阈值（米）
            - v_lower: 低速截止点（节）
            - v_upper: 高速截止点（节）
            - cog_threshold: 转向角度阈值（度）
            - sog_threshold: 速度变化阈值（节）
            - time_threshold: 时间间隔阈值（秒）
    """
    if len(points) == 0:
        return []
    if len(points) == 1:
        return points.copy()

    cog_threshold = params.get('cog_threshold', 10.0)
    sog_threshold = params.get('sog_threshold', 1.0)
    time_threshold = params.get('time_threshold', 300.0)  # 5分钟

    compressed_indices = [0]
    last_index = 0

    for i in range(1, len(points)):
        last_point = points[last_index]
        current_point = points[i]

        delta_t = (current_point.timestamp - last_point.timestamp).total_seconds()

        # 硬约束检查：任一触发就保留当前点
        should_keep = False

        # 1. 转向约束
        if abs(current_point.cog - last_point.cog) > cog_threshold:
            should_keep = True

        # 2. 速度变化约束
        if abs(current_point.sog - last_point.sog) > sog_threshold:
            should_keep = True

        # 3. 时间间隔约束
        if delta_t > time_threshold:
            should_keep = True

        if not should_keep:
            # DR 逻辑：计算预测误差
            speed_knots = last_point.sog
            course_deg = last_point.cog
            speed_mps = GeoUtils.knots_to_mps(speed_knots)

            pred_lat, pred_lon = GeoUtils.predict_position(
                lat_old=last_point.lat,
                lon_old=last_point.lon,
                speed_mps=speed_mps,
                course_deg=course_deg,
                delta_t=delta_t
            )

            error = GeoUtils.haversine_distance(
                lat1=current_point.lat,
                lon1=current_point.lon,
                lat2=pred_lat,
                lon2=pred_lon
            )

            threshold = GeoUtils.get_linear_threshold(speed_knots, params)
            if error >= threshold:
                should_keep = True

        if should_keep:
            compressed_indices.append(i)
            last_index = i

    if compressed_indices[-1] != len(points) - 1:
        compressed_indices.append(len(points) - 1)

    return [points[i] for i in compressed_indices]


def sliding_window_compress(points: List[TrajectoryPoint], params: Dict) -> List[TrajectoryPoint]:
    """
    Sliding Window 轨迹压缩算法

    以最后一个保留点为起点，不断往前看新点；如果新点相对起点→新点这条线的偏差超过阈值，
    就把上一个点保留下来当新起点。

    参数:
        points: 轨迹点列表
        params: 参数字典，包含：
            - epsilon: 距离阈值（米）
    """
    if len(points) <= 2:
        return points.copy()

    epsilon = params.get('epsilon', 100.0)
    compressed_indices = [0]  # 起始点

    i = 1
    while i < len(points):
        # 从当前锚点开始，寻找最远的点使得所有中间点都在容差内
        anchor_idx = compressed_indices[-1]

        # 找到第一个超出误差的点
        farthest_valid = i
        for j in range(i + 1, len(points)):
            # 检查点 j 对 anchor->j 这条线的误差
            error = GeoUtils.point_to_line_distance(
                points[j].lat, points[j].lon,
                points[anchor_idx].lat, points[anchor_idx].lon,
                points[j].lat, points[j].lon
            )

            if error > epsilon:
                break
            farthest_valid = j

        # 如果找到了超出误差的点，则保留上一个有效点
        if farthest_valid > i:
            compressed_indices.append(farthest_valid)
            i = farthest_valid + 1
        else:
            # 如果连下一个点都超出误差，直接保留它
            compressed_indices.append(i)
            i += 1

    # 确保包含最后一个点
    if compressed_indices[-1] != len(points) - 1:
        compressed_indices.append(len(points) - 1)

    return [points[i] for i in compressed_indices]


def opening_window_compress(points: List[TrajectoryPoint], params: Dict) -> List[TrajectoryPoint]:
    """
    Opening Window 轨迹压缩算法

    从 anchor 点开始开窗，尽可能延长窗口右端；只要窗口内所有点对 anchor→窗口末端的误差都 ≤ ε，
    就继续扩张；一旦出现超阈值点，就输出上一个窗口末端并重开窗口。

    参数:
        points: 轨迹点列表
        params: 参数字典，包含：
            - epsilon: 距离阈值（米）
    """
    if len(points) <= 2:
        return points.copy()

    epsilon = params.get('epsilon', 100.0)
    compressed_indices = []

    i = 0
    while i < len(points) - 1:
        # 从当前点开始开窗
        anchor_idx = i
        window_end = i + 1

        # 尽可能延长窗口
        while window_end < len(points):
            # 检查窗口内所有点对 anchor->window_end 这条线的误差
            valid = True
            for k in range(anchor_idx + 1, window_end):
                error = GeoUtils.point_to_line_distance(
                    points[k].lat, points[k].lon,
                    points[anchor_idx].lat, points[anchor_idx].lon,
                    points[window_end].lat, points[window_end].lon
                )
                if error > epsilon:
                    valid = False
                    break

            if not valid:
                break
            window_end += 1

        # 保留 anchor 点
        compressed_indices.append(anchor_idx)

        # 移动到窗口末端
        i = window_end

    # 确保包含最后一个点
    if not compressed_indices or compressed_indices[-1] != len(points) - 1:
        compressed_indices.append(len(points) - 1)

    return [points[i] for i in compressed_indices]


def squish_compress(points: List[TrajectoryPoint], params: Dict) -> List[TrajectoryPoint]:
    """
    SQUISH 流式轨迹简化算法

    维护一个大小为 B 的点集，每来一个新点就加入；如果超出 B，就删掉最不重要的点
    （重要性用该点在相邻两点形成的线段上的误差衡量），并更新邻居的分数。

    参数:
        points: 轨迹点列表
        params: 参数字典，包含：
            - buffer_size: 缓冲区大小 B
    """
    if len(points) <= 2:
        return points.copy()

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
                points[curr.idx].lat, points[curr.idx].lon,
                points[prev.idx].lat, points[prev.idx].lon,
                points[next_p.idx].lat, points[next_p.idx].lon
            )
            curr.importance = error

    # 处理后续点
    for i in range(2, len(points)):
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
    return [points[i] for i in result_indices]


# ============================================================================
# 统一接口 (Unified Interface)
# ============================================================================

def compress(trajectory: pd.DataFrame,
             algorithm: str,
             params: Dict) -> CompressionResult:
    """
    统一的轨迹压缩接口

    参数:
        trajectory: 输入轨迹 DataFrame
        algorithm: 算法名称 ('dr', 'fixed_dr', 'semantic_dr', 'sliding', 'opening', 'squish')
        params: 算法参数

    返回:
        CompressionResult 对象
    """
    # 转换为 TrajectoryPoint 列表
    points = dataframe_to_trajectory_points(trajectory)

    # 选择算法
    algorithm_funcs = {
        'dr': dead_reckoning_compress,
        'fixed_dr': fixed_epsilon_dr_compress,
        'semantic_dr': semantic_dr_compress,
        'sliding': sliding_window_compress,
        'opening': opening_window_compress,
        'squish': squish_compress
    }

    if algorithm not in algorithm_funcs:
        raise ValueError(f"未知算法: {algorithm}")

    # 执行压缩
    start_time = time.time()
    compressed_points = algorithm_funcs[algorithm](points, params)
    elapsed_time = time.time() - start_time

    # 计算指标
    compressed_df = trajectory_points_to_dataframe(compressed_points)
    metrics = evaluate_compression(trajectory, compressed_df, algorithm, elapsed_time)

    return CompressionResult(
        algorithm=algorithm,
        compressed_points=compressed_points,
        compression_ratio=metrics['compression_ratio'],
        elapsed_time=elapsed_time,
        metrics=metrics
    )

