import math  # 导入数学库，提供三角函数、常量等数学运算
from typing import Dict, Tuple  # 导入类型注解，用于函数参数和返回值的类型提示


class GeoUtils:  # 地理计算工具类，提供各种地理位置相关的数学计算
    """地理计算工具类，处理核心的地理数学计算，包括距离计算、位置预测等"""

    # 地球半径常量（米），用于球面距离计算
    EARTH_RADIUS_M = 6371000.0

    # 航速单位转换常量：1节 = 0.514444米/秒
    KNOTS_TO_MPS = 0.514444

    @staticmethod  # 静态方法装饰器，不需要实例化就可以调用
    def knots_to_mps(knots: float) -> float:  # 将节转换为米/秒的函数
        """
        将航速从节(knots)转换为米/秒(m/s)

        参数:
            knots: 航速数值，单位为节（海里/小时）

        返回:
            转换后的航速，单位为米/秒
        """
        return knots * GeoUtils.KNOTS_TO_MPS  # 使用转换系数进行计算

    @staticmethod  # 静态方法
    def deg_to_rad(degrees: float) -> float:  # 角度转弧度的函数
        """
        将角度从度数转换为弧度制
        """
        return math.radians(degrees)  # 使用math库的radians函数进行转换

    @staticmethod  # 静态方法
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:  # 计算两点间距离的函数
        """
        计算地球表面两点间的球面距离，使用Haversine公式（半正矢公式）
        Haversine公式比简化公式更精确，尤其在远距离计算时
        返回距离（单位：米）
        """
        lat1_rad = math.radians(lat1)  # 将第一个点的纬度转换为弧度
        lon1_rad = math.radians(lon1)  # 将第一个点的经度转换为弧度
        lat2_rad = math.radians(lat2)  # 将第二个点的纬度转换为弧度
        lon2_rad = math.radians(lon2)  # 将第二个点的经度转换为弧度
        dlat = lat2_rad - lat1_rad  # 计算纬度差
        dlon = lon2_rad - lon1_rad  # 计算经度差
        # Haversine公式的第一部分：纬度差的半正矢
        a = math.sin(dlat / 2) ** 2 + \
            math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))  # 计算中央角
        return GeoUtils.EARTH_RADIUS_M * c  # 距离 = 地球半径 × 中央角

    @staticmethod  # 静态方法
    def predict_position(lat_old: float, lon_old: float,  # 位置预测函数
                        speed_mps: float, course_deg: float,
                        delta_t: float) -> Tuple[float, float]:
        """
        航位推算预测：根据船舶当前状态（位置、速度、航向）和时间差预测未来位置
        这是Dead Reckoning算法的核心，用于在线轨迹压缩
        返回 (新纬度, 新经度) 的元组
        """
        distance = speed_mps * delta_t  # 计算在这段时间内行驶的距离（米）
        course_rad = math.radians(course_deg)  # 将航向角度转换为弧度
        lat_old_rad = math.radians(lat_old)  # 将起始纬度转换为弧度
        lon_old_rad = math.radians(lon_old)  # 将起始经度转换为弧度
        dlat_rad = (distance * math.cos(course_rad)) / GeoUtils.EARTH_RADIUS_M  # 计算纬度变化（弧度）
        dlon_rad = (distance * math.sin(course_rad)) / (GeoUtils.EARTH_RADIUS_M * math.cos(lat_old_rad))  # 计算经度变化（弧度）
        lat_new_rad = lat_old_rad + dlat_rad  # 计算新的纬度（弧度）
        lon_new_rad = lon_old_rad + dlon_rad  # 计算新的经度（弧度）
        lat_new = math.degrees(lat_new_rad)  # 将纬度转换回度数
        lon_new = math.degrees(lon_new_rad)  # 将经度转换回度数
        return lat_new, lon_new  # 返回预测的新位置坐标

    @staticmethod  # 静态方法
    def get_linear_threshold(speed_knots: float, params: Dict) -> float:  # 动态阈值计算函数
        """
        根据船舶航速计算动态距离阈值（连续线性映射方案）
        航速越快，阈值越大，因为高速运动下位置预测误差更大
        """
        epsilon_min = params.get('min_threshold', 20.0)  # 获取最小阈值，默认20米
        epsilon_max = params.get('max_threshold', 500.0)  # 获取最大阈值，默认500米
        v_lower = params.get('v_lower', 3.0)  # 获取最低航速阈值，默认3节
        v_upper = params.get('v_upper', 20.0)  # 获取最高航速阈值，默认20节
        if speed_knots <= v_lower:  # 如果航速低于最低阈值
            return epsilon_min  # 返回最小距离阈值
        if speed_knots >= v_upper:  # 如果航速高于最高阈值
            return epsilon_max  # 返回最大距离阈值
        k = (epsilon_max - epsilon_min) / (v_upper - v_lower)  # 计算线性映射斜率
        current_epsilon = k * (speed_knots - v_lower) + epsilon_min  # 计算当前航速对应的阈值
        return current_epsilon  # 返回动态计算的距离阈值

    @staticmethod  # 静态方法
    def point_to_line_distance(lat: float, lon: float,  # 点到线段距离计算函数
                              lat1: float, lon1: float,
                              lat2: float, lon2: float) -> float:
        """计算点到线段的最短垂直距离（单位：米，使用近似平面投影）"""
        avg_lat = (lat1 + lat2) / 2  # 计算线段中点纬度，用于近似计算
        cos_lat = math.cos(math.radians(avg_lat))  # 计算纬度余弦，用于经度距离校正
        x = lon * 111000 * cos_lat  # 将目标点经度转换为米（近似：1度≈111km）
        y = lat * 111000  # 将目标点纬度转换为米
        x1 = lon1 * 111000 * cos_lat  # 将线段起点经度转换为米
        y1 = lat1 * 111000  # 将线段起点纬度转换为米
        x2 = lon2 * 111000 * cos_lat  # 将线段终点经度转换为米
        y2 = lat2 * 111000  # 将线段终点纬度转换为米
        if x1 == x2 and y1 == y2:  # 如果线段起点和终点重合（退化为点）
            return math.sqrt((x - x1)**2 + (y - y1)**2)  # 返回点到点的距离
        dx = x2 - x1  # 线段在x方向的长度
        dy = y2 - y1  # 线段在y方向的长度
        px = x - x1  # 点相对于线段起点的x偏移
        py = y - y1  # 点相对于线段起点的y偏移
        t = max(0, min(1, (px * dx + py * dy) / (dx * dx + dy * dy)))  # 计算投影参数t（限制在[0,1]区间）
        closest_x = x1 + t * dx  # 计算线段上距离目标点最近的点的x坐标
        closest_y = y1 + t * dy  # 计算线段上距离目标点最近的点的y坐标
        return math.sqrt((x - closest_x)**2 + (y - closest_y)**2)  # 返回点到最近点的距离


