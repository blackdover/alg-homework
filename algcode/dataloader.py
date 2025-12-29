"""
轨迹数据加载与预处理模块
包含数据加载、清洗、转换等功能

作者: Algorithm Engineer
Python版本: 3.10
"""

import pandas as pd
import os
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass


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


def scan_categories(data_root: str) -> List[str]:
    """
    扫描数据类别（子目录名）

    参数:
        data_root: 数据根目录路径

    返回:
        类别名称列表
    """
    if not os.path.exists(data_root):
        return []
    return [d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))]


def scan_datasets(category_path: str) -> List[Tuple[str, int]]:
    """
    扫描类别下的数据集，返回 (文件名, 点数) 列表

    参数:
        category_path: 类别目录路径

    返回:
        数据集信息列表，每个元素为 (文件名, 点数)
    """
    datasets = []
    if not os.path.exists(category_path):
        return datasets

    for file in os.listdir(category_path):
        if file.endswith('.csv'):
            filepath = os.path.join(category_path, file)
            try:
                # 快速读取并计数（只读取必要的列以提高性能）
                df = pd.read_csv(filepath, usecols=['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG'])
                point_count = len(df)
                datasets.append((file, point_count))
            except Exception:
                # 如果读取失败，跳过该文件
                continue
    return datasets


def load_ais_dataset(data_root: str, category: str, filename: str, mmsi: Optional[int] = None) -> pd.DataFrame:
    """
    从AIS数据集目录加载指定文件

    参数:
        data_root: 数据根目录路径
        category: 类别名称
        filename: 文件名
        mmsi: 可选，筛选特定船舶的MMSI号

    返回:
        清洗后的DataFrame
    """
    filepath = os.path.join(data_root, category, filename)
    return load_data(filepath, mmsi)