"""
基于航位推算(Dead Reckoning)的在线轨迹简化算法
与Douglas-Peucker算法对比实现

作者: Algorithm Engineer
Python版本: 3.10
"""

import pandas as pd
import numpy as np
import time
import math
from typing import Optional, Tuple, Dict
import folium
from folium import plugins
from rdp import rdp  # 需要安装: pip install rdp


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
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"算法: {algorithm_name}")
    print(f"{'='*60}")
    print(f"原始点数: {N}")
    print(f"压缩后点数: {M}")
    print(f"压缩率: {compression_ratio:.2f}%")
    if elapsed_time is not None:
        print(f"运行时间: {elapsed_time:.4f} 秒")
    print(f"{'='*60}\n")
    
    return {
        'algorithm': algorithm_name,
        'original_points': N,
        'compressed_points': M,
        'compression_ratio': compression_ratio,
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
# 模块六：主执行脚本 (Main Execution)
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("基于航位推算(Dead Reckoning)的轨迹压缩算法")
    print("="*60)
    
    # 尝试加载真实AIS数据
    data_loaded = False
    df = None
    
    # 尝试从AIS数据目录加载一个示例文件
    import os
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
        
        # 生成50-100个点的模拟轨迹
        n_points = 80
        start_lat, start_lon = 34.6, -77.0
        
        # 模拟直线航行和一次转弯
        times = pd.date_range('2021-01-01 00:00:00', periods=n_points, freq='1min')
        
        lats = [start_lat]  # 添加起点
        lons = [start_lon]
        speeds = []
        courses = []
        
        current_lat = start_lat
        current_lon = start_lon
        
        # 前40个点：直线航行（航向约75度）
        for i in range(39):  # 减少1个点，因为起点已添加
            speed = 5.0 + np.random.normal(0, 0.3)  # 约5节
            course = 75.0 + np.random.normal(0, 2)  # 约75度
            
            # 计算下一个点位置（简化计算）
            dt = 60  # 1分钟
            distance = GeoUtils.knots_to_mps(speed) * dt
            course_rad = math.radians(course)
            
            current_lat += (distance * math.cos(course_rad)) / GeoUtils.EARTH_RADIUS_M * 180 / math.pi
            current_lon += (distance * math.sin(course_rad)) / (GeoUtils.EARTH_RADIUS_M * math.cos(math.radians(current_lat))) * 180 / math.pi
            
            lats.append(current_lat)
            lons.append(current_lon)
            speeds.append(max(0, speed))
            courses.append(course % 360)
        
        # 后40个点：转弯后继续航行（航向约120度）
        for i in range(40):  # 保持40个点
            speed = 5.0 + np.random.normal(0, 0.3)
            course = 120.0 + np.random.normal(0, 2)  # 转弯后约120度
            
            dt = 60
            distance = GeoUtils.knots_to_mps(speed) * dt
            course_rad = math.radians(course)
            
            current_lat += (distance * math.cos(course_rad)) / GeoUtils.EARTH_RADIUS_M * 180 / math.pi
            current_lon += (distance * math.sin(course_rad)) / (GeoUtils.EARTH_RADIUS_M * math.cos(math.radians(current_lat))) * 180 / math.pi
            
            lats.append(current_lat)
            lons.append(current_lon)
            speeds.append(max(0, speed))
            courses.append(course % 360)
        
        # 为起点添加初始速度和航向
        speeds.insert(0, 5.0)
        courses.insert(0, 75.0)
        
        # 确保时间序列长度匹配
        actual_n_points = len(lats)
        times_actual = pd.date_range('2021-01-01 00:00:00', periods=actual_n_points, freq='1min')
        
        # 创建DataFrame
        df = pd.DataFrame({
            'MMSI': [123456789] * actual_n_points,
            'BaseDateTime': times_actual,
            'LAT': lats,
            'LON': lons,
            'SOG': speeds,
            'COG': courses
        })
        
        print(f"生成了 {len(df)} 个模拟数据点")
    
    # 运行Dead Reckoning压缩算法
    print("\n" + "="*60)
    print("运行 Dead Reckoning 压缩算法...")
    print("="*60)
    threshold = 100.0  # 100米阈值
    
    start_time = time.time()
    dr_compressed = dead_reckoning_compress(df, threshold_meters=threshold)
    dr_time = time.time() - start_time
    
    dr_metrics = evaluate_compression(df, dr_compressed, "Dead Reckoning", dr_time)
    
    # 运行Douglas-Peucker压缩算法（用于对比）
    print("\n" + "="*60)
    print("运行 Douglas-Peucker 压缩算法...")
    print("="*60)
    
    # DP的epsilon需要转换为度（近似）
    # 100米 ≈ 0.0009度（在赤道附近）
    dp_epsilon = threshold / 111000.0  # 粗略转换
    
    start_time = time.time()
    dp_compressed = dp_compress(df, epsilon=dp_epsilon)
    dp_time = time.time() - start_time
    
    dp_metrics = evaluate_compression(df, dp_compressed, "Douglas-Peucker", dp_time)
    
    # 生成可视化地图
    print("\n" + "="*60)
    print("生成可视化地图...")
    print("="*60)
    visualize_trajectories(
        original_df=df,
        dr_df=dr_compressed,
        dp_df=dp_compressed,
        output_file="trajectory_compression_map.html"
    )
    
    print("\n" + "="*60)
    print("算法执行完成！")
    print("="*60)
    print(f"\n对比总结:")
    print(f"  DR压缩率: {dr_metrics['compression_ratio']:.2f}%")
    print(f"  DP压缩率: {dp_metrics['compression_ratio']:.2f}%")
    print(f"  DR运行时间: {dr_time:.4f} 秒")
    print(f"  DP运行时间: {dp_time:.4f} 秒")

