"""
地理计算工具类
处理核心数学计算：距离计算、位置预测、阈值映射等

作者: Algorithm Engineer
Python版本: 3.10
"""

import math
from typing import Dict, Tuple


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
