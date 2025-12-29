"""
轨迹压缩算法集合
包含多种轨迹简化算法：Dead Reckoning、Douglas-Peucker、Sliding Window、Opening Window、SQUISH等

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
import folium
from folium import plugins
from rdp import rdp  # 需要安装: pip install rdp
import argparse
import sys
import os


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
# 模块一：数据加载与预处理 (Data Loading)
# ============================================================================

def load_data(filepath: str, mmsi: Optional[int] = None) -> pd.DataFrame:
    """
    读取CSV格式的AIS船舶数据并进行预处理
    
    参数:
        filepath: CSV文件路径
        mmsi: 可选，筛选特定船舶的MMSI号
    
    返回:
        清洗后的DataFrame，包含列：MMSI, BaseDateTime, LAT, LON, SOG, COG
    """
    # 读取CSV文件
    df = pd.read_csv(filepath)
    
    # 检查必需的列是否存在
    required_columns = ['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺少必需的列: {missing_columns}")
    
    # 筛选特定船舶（如果指定）
    if mmsi is not None:
        df = df[df['MMSI'] == mmsi].copy()
    
    # 转换时间戳为datetime对象
    df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
    
    # 清洗逻辑：过滤无效的经纬度
    # LAT: -90~90, LON: -180~180
    df = df[
        (df['LAT'] >= -90) & (df['LAT'] <= 90) &
        (df['LON'] >= -180) & (df['LON'] <= 180)
    ].copy()
    
    # 过滤掉SOG或COG为NaN或无效值的行
    df = df.dropna(subset=['SOG', 'COG', 'LAT', 'LON'])
    
    # 按时间戳升序排序
    df = df.sort_values('BaseDateTime').reset_index(drop=True)
    
    return df[required_columns].copy()


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
# 模块二：地理计算工具类 (Geo Utils)
# ============================================================================

class GeoUtils:
    """地理计算工具类，处理核心数学计算"""
    
    # 地球半径（米）
    EARTH_RADIUS_M = 6371000.0
    
    # 节到米/秒的转换系数
    KNOTS_TO_MPS = 0.514444
    
    @staticmethod
    def knots_to_mps(knots: float) -> float:
        """
        将航速从节(knots)转换为米/秒(m/s)
        
        参数:
            knots: 航速（节）
        
        返回:
            航速（米/秒）
        """
        return knots * GeoUtils.KNOTS_TO_MPS
    
    @staticmethod
    def deg_to_rad(degrees: float) -> float:
        """
        角度转弧度
        
        参数:
            degrees: 角度（度）
        
        返回:
            弧度
        """
        return math.radians(degrees)
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        计算两点间的地球表面距离（Haversine公式）
        
        参数:
            lat1, lon1: 点A的纬度和经度（度）
            lat2, lon2: 点B的纬度和经度（度）
        
        返回:
            两点间距离（米）
        """
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
        """
        航位预测：根据起始点状态和时间差预测新位置
        
        参数:
            lat_old: 起始点纬度（度）
            lon_old: 起始点经度（度）
            speed_mps: 航速（米/秒）
            course_deg: 航向（度，正北为0度，顺时针）
            delta_t: 时间差（秒）
        
        返回:
            (新纬度, 新经度) 元组（度）
        """
        # 计算行驶距离
        distance = speed_mps * delta_t
        
        # 将航向转换为弧度
        # COG定义：0度=正北，90度=正东，180度=正南，270度=正西（顺时针）
        course_rad = math.radians(course_deg)
        
        # 转换为弧度
        lat_old_rad = math.radians(lat_old)
        lon_old_rad = math.radians(lon_old)
        
        # 计算经纬度偏移量（使用近似公式）
        # 纬度偏移：主要受南北方向影响
        # COG=0度（正北）时，cos(0°)=1，纬度增加
        # COG=180度（正南）时，cos(180°)=-1，纬度减少
        dlat_rad = (distance * math.cos(course_rad)) / GeoUtils.EARTH_RADIUS_M
        
        # 经度偏移：需要考虑纬度的影响
        # COG=90度（正东）时，sin(90°)=1，经度增加
        # COG=270度（正西）时，sin(270°)=-1，经度减少
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
        """
        根据航速计算动态距离阈值 (连续线性映射方案)
        
        数学模型：
        ε(v_i) = min(ε_max, max(ε_min, k·(v_i - v_lower) + ε_min))
        
        参数:
            speed_knots: 当前航速（节）
            params: 阈值配置字典，包含：
                - min_threshold: 最低航速下的阈值（米），如20米
                - max_threshold: 最高航速下的阈值（米），如500米
                - v_lower: 低速截止点（节），如3节
                - v_upper: 高速截止点（节），如20节
        
        返回:
            动态阈值（米）
        """
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
        
        # 3. 过渡区间：线性插值 (Linear Interpolation)
        # 计算斜率 k = (ε_max - ε_min) / (v_upper - v_lower)
        k = (epsilon_max - epsilon_min) / (v_upper - v_lower)
        
        # 线性映射：y = k * (x - x0) + y0
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
# 模块三：核心算法实现 (Core Algorithms)
# ============================================================================

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
    if len(df) <= 2:
        return df.copy()

    epsilon = params.get('epsilon', 100.0)
    compressed_indices = [0]  # 起始点

    i = 1
    while i < len(df):
        # 从当前锚点开始，寻找最远的点使得所有中间点都在容差内
        anchor_idx = compressed_indices[-1]

        # 找到第一个超出误差的点
        farthest_valid = i
        for j in range(i + 1, len(df)):
            # 检查点 j 对 anchor->j 这条线的误差
            error = GeoUtils.point_to_line_distance(
                df.iloc[j]['LAT'], df.iloc[j]['LON'],
                df.iloc[anchor_idx]['LAT'], df.iloc[anchor_idx]['LON'],
                df.iloc[j]['LAT'], df.iloc[j]['LON']
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
    if len(df) <= 2:
        return df.copy()

    epsilon = params.get('epsilon', 100.0)
    compressed_indices = []

    i = 0
    while i < len(df) - 1:
        # 从当前点开始开窗
        anchor_idx = i
        window_end = i + 1

        # 尽可能延长窗口
        while window_end < len(df):
            # 检查窗口内所有点对 anchor->window_end 这条线的误差
            valid = True
            for k in range(anchor_idx + 1, window_end):
                error = GeoUtils.point_to_line_distance(
                    df.iloc[k]['LAT'], df.iloc[k]['LON'],
                    df.iloc[anchor_idx]['LAT'], df.iloc[anchor_idx]['LON'],
                    df.iloc[window_end]['LAT'], df.iloc[window_end]['LON']
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


# ============================================================================
# 模块四：评估指标 (Evaluation Metrics)
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
                        elapsed_time: float = None) -> dict:
    """
    计算并打印压缩算法的评估指标
    
    参数:
        original_df: 原始轨迹DataFrame
        compressed_df: 压缩后的轨迹DataFrame
        algorithm_name: 算法名称
        elapsed_time: 算法运行时间（秒），可选
    
    返回:
        包含评估指标的字典
    """
    N = len(original_df)  # 原始点数
    M = len(compressed_df)  # 压缩后点数
    
    # 压缩率
    compression_ratio = (1 - M / N) * 100 if N > 0 else 0.0

    # SED 指标
    sed_metrics = calculate_sed_metrics(original_df, compressed_df)

    # 航行事件保留率
    event_recall = calculate_navigation_event_recall(original_df, compressed_df)

    # 打印结果
    print(f"\n{'='*60}")
    print(f"算法: {algorithm_name}")
    print(f"{'='*60}")
    print(f"原始点数: {N}")
    print(f"压缩后点数: {M}")
    print(f"压缩率: {compression_ratio:.2f}%")
    print(f"SED均值: {sed_metrics['mean']:.2f} 米")
    print(f"SED最大值: {sed_metrics['max']:.2f} 米")
    print(f"SED 95分位数: {sed_metrics['p95']:.2f} 米")
    print(f"航行事件保留率: {event_recall:.3f}")
    if elapsed_time is not None:
        print(f"运行时间: {elapsed_time:.4f} 秒")
    print(f"{'='*60}\n")

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
# 模块五：可视化展示 (Visualization)
# ============================================================================

def visualize_trajectories(original_df: pd.DataFrame,
                          dr_df: pd.DataFrame,
                          dp_df: Optional[pd.DataFrame] = None,
                          output_file: str = "map.html") -> None:
    """
    使用folium库可视化轨迹压缩结果
    
    参数:
        original_df: 原始轨迹DataFrame
        dr_df: DR算法压缩后的轨迹DataFrame
        dp_df: DP算法压缩后的轨迹DataFrame（可选）
        output_file: 输出HTML文件路径
    """
    # 初始化地图，中心点设为轨迹起点
    center_lat = original_df.iloc[0]['LAT']
    center_lon = original_df.iloc[0]['LON']
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # 图层1：原始轨迹（红色细线）
    original_coords = [[row['LAT'], row['LON']] for _, row in original_df.iterrows()]
    folium.PolyLine(
        original_coords,
        color='red',
        weight=2,
        opacity=0.5,
        popup='Original Trajectory'
    ).add_to(m)
    
    # 图层2：DR压缩轨迹（蓝色线，带标记点）
    dr_coords = [[row['LAT'], row['LON']] for _, row in dr_df.iterrows()]
    folium.PolyLine(
        dr_coords,
        color='blue',
        weight=3,
        opacity=0.8,
        popup='DR Compressed Trajectory'
    ).add_to(m)
    
    # 在DR保留点处添加圆点标记
    for idx, row in dr_df.iterrows():
        folium.CircleMarker(
            location=[row['LAT'], row['LON']],
            radius=5,
            popup=f"DR Point {idx}",
            color='blue',
            fill=True,
            fillColor='blue',
            fillOpacity=0.7
        ).add_to(m)
    
    # 图层3：DP压缩轨迹（绿色线，可选）
    if dp_df is not None:
        dp_coords = [[row['LAT'], row['LON']] for _, row in dp_df.iterrows()]
        folium.PolyLine(
            dp_coords,
            color='green',
            weight=3,
            opacity=0.8,
            popup='DP Compressed Trajectory'
        ).add_to(m)
        
        # 在DP保留点处添加圆点标记
        for idx, row in dp_df.iterrows():
            folium.CircleMarker(
                location=[row['LAT'], row['LON']],
                radius=5,
                popup=f"DP Point {idx}",
                color='green',
                fill=True,
                fillColor='green',
                fillOpacity=0.7
            ).add_to(m)
    
    # 添加起点和终点标记
    folium.Marker(
        location=[original_df.iloc[0]['LAT'], original_df.iloc[0]['LON']],
        popup='Start',
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)
    
    folium.Marker(
        location=[original_df.iloc[-1]['LAT'], original_df.iloc[-1]['LON']],
        popup='End',
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(m)
    
    # 添加图层控制
    folium.LayerControl().add_to(m)
    
    # 保存地图
    m.save(output_file)
    print(f"可视化地图已保存至: {output_file}")


# ============================================================================
# 模块六：统一压缩接口 (Unified Interface)
# ============================================================================

def compress_trajectory(df: pd.DataFrame,
                       algorithm: str,
                       params: Dict) -> CompressionResult:
    """
    统一的轨迹压缩接口

    参数:
        df: 输入轨迹 DataFrame
        algorithm: 算法名称 ('dr', 'adaptive_dr', 'semantic_dr', 'dp', 'sliding', 'opening', 'squish')
        params: 算法参数

    返回:
        CompressionResult 对象
    """
    # 算法函数映射
    algorithm_funcs = {
        'dr': lambda df, p: dead_reckoning_compress(df, p.get('epsilon', 100.0)),
        'adaptive_dr': adaptive_dr_compress,
        'semantic_dr': semantic_dr_compress,
        'dp': lambda df, p: dp_compress(df, p.get('epsilon', 0.0009)),  # DP用度为单位
        'sliding': sliding_window_compress,
        'opening': opening_window_compress,
        'squish': squish_compress
    }

    if algorithm not in algorithm_funcs:
        raise ValueError(f"未知算法: {algorithm}. 支持的算法: {list(algorithm_funcs.keys())}")

    # 执行压缩
    start_time = time.time()
    compressed_df = algorithm_funcs[algorithm](df, params)
    elapsed_time = time.time() - start_time

    # 计算指标
    metrics = evaluate_compression(df, compressed_df, algorithm, elapsed_time)

    # 转换为TrajectoryPoint列表
    compressed_points = dataframe_to_trajectory_points(compressed_df)

    return CompressionResult(
        algorithm=algorithm,
        compressed_points=compressed_points,
        compression_ratio=metrics['compression_ratio'],
        elapsed_time=elapsed_time,
        metrics=metrics
    )


# ============================================================================
# 模块七：命令行界面 (CLI Interface)
# ============================================================================

def create_cli_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='轨迹压缩算法工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python trajectory_compression.py data.csv -a dr -e 100 -o result.csv -m map.html
  python trajectory_compression.py data.csv -a adaptive_dr --min-threshold 20 --max-threshold 500 -v 3 20
  python trajectory_compression.py data.csv -a semantic_dr --cog-threshold 15 --time-threshold 600
  python trajectory_compression.py data.csv -a dp -e 0.001
  python trajectory_compression.py data.csv -a squish -b 200

支持的算法:
  dr: 固定阈值 Dead Reckoning
  adaptive_dr: 自适应阈值 Dead Reckoning
  semantic_dr: 语义增强 Dead Reckoning
  dp: Douglas-Peucker
  sliding: Sliding Window
  opening: Opening Window
  squish: SQUISH 流式算法
        """
    )

    parser.add_argument('input_file', help='输入的轨迹数据文件路径')

    parser.add_argument('-a', '--algorithm', required=True,
                       choices=['dr', 'adaptive_dr', 'semantic_dr', 'dp', 'sliding', 'opening', 'squish'],
                       help='使用的压缩算法')

    parser.add_argument('-o', '--output', help='压缩结果输出文件路径（CSV格式）')

    parser.add_argument('-m', '--map', help='生成的轨迹可视化地图文件路径（HTML格式）')

    parser.add_argument('-e', '--epsilon', type=float, default=100.0,
                       help='距离阈值（米），用于dr/sliding/opening算法；度，用于dp算法')

    # 自适应DR参数
    parser.add_argument('--min-threshold', type=float, default=20.0,
                       help='自适应DR的最低距离阈值（米）')
    parser.add_argument('--max-threshold', type=float, default=500.0,
                       help='自适应DR的最高距离阈值（米）')
    parser.add_argument('-v', '--velocity-range', nargs=2, type=float,
                       metavar=('V_LOWER', 'V_UPPER'), default=[3.0, 20.0],
                       help='自适应DR的速度范围（节）：低速截止点和高速截止点')

    # 语义DR参数
    parser.add_argument('--cog-threshold', type=float, default=10.0,
                       help='语义DR的转向角度阈值（度）')
    parser.add_argument('--sog-threshold', type=float, default=1.0,
                       help='语义DR的速度变化阈值（节）')
    parser.add_argument('--time-threshold', type=float, default=300.0,
                       help='语义DR的时间间隔阈值（秒）')

    # SQUISH参数
    parser.add_argument('-b', '--buffer-size', type=int, default=100,
                       help='SQUISH算法的缓冲区大小')

    parser.add_argument('--mmsi', type=int, help='筛选特定船舶的MMSI号')

    return parser


def main():
    """主函数"""
    parser = create_cli_parser()
    args = parser.parse_args()

    try:
        # 加载数据
        print(f"正在加载数据文件: {args.input_file}")
        df = load_data(args.input_file, args.mmsi)
        print(f"成功加载 {len(df)} 个数据点")

        # 准备算法参数
        params = {}

        if args.algorithm == 'dr':
            params['epsilon'] = args.epsilon
        elif args.algorithm == 'adaptive_dr':
            params.update({
                'min_threshold': args.min_threshold,
                'max_threshold': args.max_threshold,
                'v_lower': args.velocity_range[0],
                'v_upper': args.velocity_range[1]
            })
        elif args.algorithm == 'semantic_dr':
            params.update({
                'min_threshold': args.min_threshold,
                'max_threshold': args.max_threshold,
                'v_lower': args.velocity_range[0],
                'v_upper': args.velocity_range[1],
                'cog_threshold': args.cog_threshold,
                'sog_threshold': args.sog_threshold,
                'time_threshold': args.time_threshold
            })
        elif args.algorithm == 'dp':
            params['epsilon'] = args.epsilon
        elif args.algorithm in ['sliding', 'opening']:
            params['epsilon'] = args.epsilon
        elif args.algorithm == 'squish':
            params['buffer_size'] = args.buffer_size

        # 执行压缩
        print(f"\n执行 {args.algorithm} 算法...")
        result = compress_trajectory(df, args.algorithm, params)

        # 输出结果
        print("压缩完成！")
        print(".1f"
              ".3f")

        # 保存压缩结果
        if args.output:
            compressed_df = trajectory_points_to_dataframe(result.compressed_points)
            compressed_df.to_csv(args.output, index=False)
            print(f"压缩结果已保存至: {args.output}")

        # 生成可视化地图
        if args.map:
            compressed_df = trajectory_points_to_dataframe(result.compressed_points)
            visualize_trajectories(df, compressed_df, output_file=args.map)

    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


# ============================================================================
# 模块八：主执行脚本 (Main Execution)
# ============================================================================

if __name__ == "__main__":
    # 如果有命令行参数，使用CLI模式
    if len(sys.argv) > 1:
        main()
    else:
        # 交互式演示模式
        print("="*60)
        print("轨迹压缩算法演示")
        print("="*60)

        # 尝试加载真实AIS数据
        data_loaded = False
        df = None

        # 尝试从AIS数据目录加载一个示例文件
        sample_file = r"E:\code\homework\alg\AIS Dataset\AIS Data\Tugboat\220584000.csv"

        if os.path.exists(sample_file):
            try:
                print(f"\n正在加载数据: {sample_file}")
                df = load_data(sample_file)
                print(f"成功加载 {len(df)} 个数据点")
                data_loaded = True
            except Exception as e:
                print(f"加载数据失败: {e}")
                data_loaded = False

        # 如果加载失败，生成模拟数据
        if not data_loaded:
            print("\n生成模拟AIS数据...")
            np.random.seed(42)

            # 生成80个点的模拟轨迹
            n_points = 80
            start_lat, start_lon = 34.6, -77.0

            # 模拟直线航行和一次转弯
            times = pd.date_range('2021-01-01 00:00:00', periods=n_points, freq='1min')

            lats = [start_lat]
            lons = [start_lon]
            speeds = []
            courses = []

            current_lat = start_lat
            current_lon = start_lon

            # 前40个点：直线航行（航向约75度）
            for i in range(39):
                speed = 5.0 + np.random.normal(0, 0.3)
                course = 75.0 + np.random.normal(0, 2)

                dt = 60
                distance = GeoUtils.knots_to_mps(speed) * dt
                course_rad = math.radians(course)

                current_lat += (distance * math.cos(course_rad)) / GeoUtils.EARTH_RADIUS_M * 180 / math.pi
                current_lon += (distance * math.sin(course_rad)) / (GeoUtils.EARTH_RADIUS_M * math.cos(math.radians(current_lat))) * 180 / math.pi

                lats.append(current_lat)
                lons.append(current_lon)
                speeds.append(max(0, speed))
                courses.append(course % 360)

            # 后40个点：转弯后继续航行（航向约120度）
            for i in range(40):
                speed = 5.0 + np.random.normal(0, 0.3)
                course = 120.0 + np.random.normal(0, 2)

                dt = 60
                distance = GeoUtils.knots_to_mps(speed) * dt
                course_rad = math.radians(course)

                current_lat += (distance * math.cos(course_rad)) / GeoUtils.EARTH_RADIUS_M * 180 / math.pi
                current_lon += (distance * math.sin(course_rad)) / (GeoUtils.EARTH_RADIUS_M * math.cos(math.radians(current_lat))) * 180 / math.pi

                lats.append(current_lat)
                lons.append(current_lon)
                speeds.append(max(0, speed))
                courses.append(course % 360)

            speeds.insert(0, 5.0)
            courses.insert(0, 75.0)

            df = pd.DataFrame({
                'MMSI': [123456789] * n_points,
                'BaseDateTime': times,
                'LAT': lats,
                'LON': lons,
                'SOG': speeds,
                'COG': courses
            })

            print(f"生成了 {len(df)} 个模拟数据点")

        # 演示多种算法
        algorithms_to_test = [
            ('dr', {'epsilon': 100.0}, '固定阈值 DR'),
            ('adaptive_dr', {
                'min_threshold': 20.0,
                'max_threshold': 500.0,
                'v_lower': 3.0,
                'v_upper': 20.0
            }, '自适应阈值 DR'),
            ('semantic_dr', {
                'min_threshold': 20.0,
                'max_threshold': 500.0,
                'v_lower': 3.0,
                'v_upper': 20.0,
                'cog_threshold': 10.0,
                'sog_threshold': 1.0,
                'time_threshold': 300.0
            }, '语义增强 DR'),
            ('sliding', {'epsilon': 100.0}, 'Sliding Window'),
            ('opening', {'epsilon': 100.0}, 'Opening Window'),
            ('squish', {'buffer_size': 50}, 'SQUISH'),
            ('dp', {'epsilon': 100.0 / 111000.0}, 'Douglas-Peucker')
        ]

        results = []
        for alg, params, name in algorithms_to_test:
            print(f"\n{'='*60}")
            print(f"运行 {name} 算法...")
            print('='*60)

            result = compress_trajectory(df, alg, params)
            results.append((name, result))

        # 生成对比可视化
        print(f"\n{'='*60}")
        print("生成对比可视化地图...")
        print('='*60)

        # 选择几个算法进行可视化对比
        visualize_trajectories(
            original_df=df,
            dr_df=trajectory_points_to_dataframe(results[0][1].compressed_points),  # DR
            dp_df=trajectory_points_to_dataframe(results[-1][1].compressed_points),  # DP
            output_file="trajectory_compression_comparison.html"
        )

        print(f"\n{'='*60}")
        print("算法对比完成！")
        print('='*60)
        print("\n算法性能对比:")
        print("-" * 60)
        print(f"{'算法':<12} {'压缩率':<6} {'SED均值':<8} {'SED最大':<8} {'SED_95%':<8} {'事件保留':<8} {'时间':<6}")
        print("-" * 60)

        for name, result in results:
            print(f"{name:<12} "
                  f"{result.metrics['compression_ratio']:<6.1f} "
                  f"{result.metrics['sed_mean']:<8.2f} "
                  f"{result.metrics['sed_max']:<8.2f} "
                  f"{result.metrics['sed_p95']:<8.2f} "
                  f"{result.metrics['event_recall']:<8.3f} "
                  f"{result.elapsed_time:<6.4f}")

        print("-" * 60)
        print("可视化地图已保存至: trajectory_compression_comparison.html")

