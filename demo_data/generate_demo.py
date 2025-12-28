#!/usr/bin/env python3
"""
生成演示轨迹数据
用于测试和展示轨迹压缩算法效果
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_demo_trajectory(n_points=200, ship_type='cargo', seed=42):
    """
    生成演示轨迹数据

    参数:
        n_points: 轨迹点数量
        ship_type: 船舶类型 ('cargo', 'tugboat', 'passenger')
        seed: 随机种子
    """
    np.random.seed(seed)

    # 设置船舶类型参数
    if ship_type == 'cargo':
        base_speed = 12.0
        speed_variation = 3.0
        course_change_prob = 0.02
        mmsi = 123456789
    elif ship_type == 'tugboat':
        base_speed = 5.0
        speed_variation = 2.0
        course_change_prob = 0.05
        mmsi = 220584000
    elif ship_type == 'passenger':
        base_speed = 15.0
        speed_variation = 4.0
        course_change_prob = 0.03
        mmsi = 345678901
    else:
        base_speed = 10.0
        speed_variation = 3.0
        course_change_prob = 0.03
        mmsi = 999999999

    # 初始化
    start_time = datetime(2021, 1, 1, 8, 0, 0)  # 上午8点开始
    current_lat, current_lon = 34.6, -77.0  # 起始位置（北卡罗来纳州附近）
    current_course = 75.0  # 初始航向（东北偏东）
    current_speed = base_speed

    # 数据容器
    times = []
    lats = []
    lons = []
    speeds = []
    courses = []

    for i in range(n_points):
        # 记录当前点
        times.append(start_time + timedelta(minutes=i))
        lats.append(current_lat)
        lons.append(current_lon)
        speeds.append(max(0, current_speed))
        courses.append(current_course % 360)

        # 随机改变航向和速度
        if np.random.random() < course_change_prob:
            # 较大的航向改变（模拟转弯）
            course_change = np.random.normal(0, 25)
            current_course += course_change
        else:
            # 小幅随机扰动
            current_course += np.random.normal(0, 3)

        if np.random.random() < 0.1:  # 10% 概率改变速度
            speed_change = np.random.normal(0, speed_variation * 0.5)
            current_speed = max(0, base_speed + speed_change)

        # 计算下一位置
        speed_mps = current_speed * 0.514444  # knots to m/s
        delta_t = 60  # 1分钟
        course_rad = np.radians(current_course)

        # 简化的位置计算（适用于小距离）
        dlat = (speed_mps * np.cos(course_rad) * delta_t) / 6371000 * 180 / np.pi
        dlon = (speed_mps * np.sin(course_rad) * delta_t) / (6371000 * np.cos(np.radians(current_lat))) * 180 / np.pi

        current_lat += dlat
        current_lon += dlon

        # 限制在合理范围内
        current_lat = np.clip(current_lat, -90, 90)
        current_lon = np.clip(current_lon, -180, 180)

    # 创建 DataFrame
    df = pd.DataFrame({
        'MMSI': [mmsi] * n_points,
        'BaseDateTime': times,
        'LAT': lats,
        'LON': lons,
        'SOG': speeds,
        'COG': courses
    })

    return df


def generate_complex_trajectory(n_points=300, seed=123):
    """
    生成复杂轨迹（包含多种航行模式）
    """
    np.random.seed(seed)

    start_time = datetime(2021, 1, 1, 6, 0, 0)
    current_lat, current_lon = 1.2833, 103.8333  # 新加坡附近
    mmsi = 999888777

    times, lats, lons, speeds, courses = [], [], [], [], []

    # 第一段：港口出发，直线加速
    for i in range(50):
        times.append(start_time + timedelta(minutes=i))
        lats.append(current_lat)
        lons.append(current_lon)
        speeds.append(5.0 + i * 0.1)  # 逐渐加速
        courses.append(45.0)  # 东北方向

        # 直线运动
        speed_mps = (5.0 + i * 0.1) * 0.514444
        dlat = (speed_mps * np.cos(np.radians(45.0)) * 60) / 6371000 * 180 / np.pi
        dlon = (speed_mps * np.sin(np.radians(45.0)) * 60) / (6371000 * np.cos(np.radians(current_lat))) * 180 / np.pi

        current_lat += dlat
        current_lon += dlon

    # 第二段：巡航，少量扰动
    for i in range(50, 150):
        times.append(start_time + timedelta(minutes=i))
        lats.append(current_lat)
        lons.append(current_lon)
        speeds.append(12.0 + np.random.normal(0, 1))
        course = 45.0 + np.random.normal(0, 2)
        courses.append(course)

        speed_mps = (12.0 + np.random.normal(0, 1)) * 0.514444
        dlat = (speed_mps * np.cos(np.radians(course)) * 60) / 6371000 * 180 / np.pi
        dlon = (speed_mps * np.sin(np.radians(course)) * 60) / (6371000 * np.cos(np.radians(current_lat))) * 180 / np.pi

        current_lat += dlat
        current_lon += dlon

    # 第三段：转弯进入港口
    for i in range(150, 250):
        times.append(start_time + timedelta(minutes=i))
        lats.append(current_lat)
        lons.append(current_lon)

        # 逐渐减速
        speed = max(3.0, 12.0 - (i - 150) * 0.05)
        speeds.append(speed)

        # 逐渐转向
        progress = (i - 150) / 100
        course = 45.0 - progress * 90.0  # 从东北转向正北
        courses.append(course)

        speed_mps = speed * 0.514444
        dlat = (speed_mps * np.cos(np.radians(course)) * 60) / 6371000 * 180 / np.pi
        dlon = (speed_mps * np.sin(np.radians(course)) * 60) / (6371000 * np.cos(np.radians(current_lat))) * 180 / np.pi

        current_lat += dlat
        current_lon += dlon

    # 第四段：低速停泊
    for i in range(250, n_points):
        times.append(start_time + timedelta(minutes=i))
        lats.append(current_lat + np.random.normal(0, 0.0001))
        lons.append(current_lon + np.random.normal(0, 0.0001))
        speeds.append(np.random.uniform(0, 1))  # 低速或静止
        courses.append(np.random.uniform(0, 360))  # 随机方向

    df = pd.DataFrame({
        'MMSI': [mmsi] * n_points,
        'BaseDateTime': times,
        'LAT': lats,
        'LON': lons,
        'SOG': speeds,
        'COG': courses
    })

    return df


def main():
    """生成演示数据集"""
    print("生成演示轨迹数据...")

    # 创建目录
    os.makedirs('demo_data', exist_ok=True)

    # 生成不同类型的演示数据
    datasets = [
        ('cargo_demo', 'cargo', 200),
        ('tugboat_demo', 'tugboat', 150),
        ('passenger_demo', 'passenger', 180),
        ('complex_demo', 'mixed', 300)
    ]

    for name, ship_type, n_points in datasets:
        print(f"生成 {name} 数据集...")

        if name == 'complex_demo':
            df = generate_complex_trajectory(n_points, seed=999)
        else:
            df = generate_demo_trajectory(n_points, ship_type, seed=hash(name) % 10000)

        # 保存为 CSV
        filename = f'demo_data/{name}.csv'
        df.to_csv(filename, index=False)
        print(f"  保存至: {filename} ({len(df)} 点)")

        # 显示基本统计信息
        print(f"  时间范围: {df['BaseDateTime'].min()} - {df['BaseDateTime'].max()}")
        print(f"  平均速度: {df['SOG'].mean():.2f} 节")
        print(f"  速度范围: {df['SOG'].min():.2f} - {df['SOG'].max():.2f} 节")
        print(f"  航程: {len(df)} 点")
        print()

    print("演示数据生成完成！")
    print("\n使用方法:")
    print("1. 将 demo_data 目录复制到 AIS Dataset/ 相应船型目录下")
    print("2. 重启后端服务")
    print("3. 在前端选择相应数据集进行测试")


if __name__ == '__main__':
    main()
