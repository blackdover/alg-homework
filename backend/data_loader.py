"""
数据集加载和处理工具
用于发现、采样和按船型/速度段组织 AIS 数据集

作者: Algorithm Engineer
Python版本: 3.10
"""

import pandas as pd
import numpy as np
import os
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import glob


class DatasetInfo:
    """数据集信息类"""

    def __init__(self, name: str, path: str, ship_type: Optional[str] = None,
                 sample_size: Optional[int] = None, time_range: Optional[Tuple[str, str]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.path = path
        self.ship_type = ship_type
        self.sample_size = sample_size
        self.time_range = time_range
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'path': self.path,
            'ship_type': self.ship_type,
            'sample_size': self.sample_size,
            'time_range': self.time_range,
            'metadata': self.metadata
        }


class DataLoader:
    """AIS 数据集加载器"""

    # 船型映射（基于 AIS 船型编码）
    SHIP_TYPE_MAPPING = {
        'Tugboat': ['Tug'],
        'Cargo': ['Cargo', 'Bulk Carrier', 'Container Ship'],
        'Passenger': ['Passenger', 'Cruise'],
        'Fishing': ['Fishing'],
        'Tanker': ['Tanker', 'Oil Tanker'],
        'Other': []
    }

    # 速度段定义
    SPEED_BINS = {
        'low': (0, 3),      # 低速 (0-3 knots)
        'medium': (3, 12),  # 中速 (3-12 knots)
        'high': (12, 50)    # 高速 (12+ knots)
    }

    def __init__(self, data_root: str = "AIS Dataset/AIS Data/", max_datasets: int = None, ship_types: List[str] = None):
        """
        初始化数据加载器

        参数:
            data_root: 数据集根目录路径
            max_datasets: 最大数据集数量限制（用于测试）
            ship_types: 要包含的船舶类型列表，None表示全部
        """
        self.data_root = Path(data_root)
        self.max_datasets = max_datasets
        self.ship_types = ship_types
        self.datasets = self._discover_datasets()

    def _discover_datasets(self) -> List[DatasetInfo]:
        """自动发现数据集"""
        datasets = []

        if not self.data_root.exists():
            print(f"警告: 数据目录不存在 {self.data_root}")
            # 如果数据目录不存在，返回模拟数据集列表
            return self._create_mock_datasets()

        print(f"正在扫描数据目录: {self.data_root}")

        # 遍历数据目录
        ship_type_dirs = list(self.data_root.iterdir())
        if self.ship_types:
            # 只处理指定类型的船舶
            ship_type_dirs = [d for d in ship_type_dirs if d.is_dir() and d.name in self.ship_types]

        for ship_type_dir in ship_type_dirs:
            if not ship_type_dir.is_dir():
                continue

            ship_type = ship_type_dir.name
            print(f"  扫描船型: {ship_type}")

            # 查找该船型目录下的 CSV 文件
            csv_files = list(ship_type_dir.glob("*.csv"))

            # 如果设置了最大数据集数量，进行采样
            if self.max_datasets and len(csv_files) > self.max_datasets // len(ship_type_dirs):
                import random
                csv_files = random.sample(csv_files, min(len(csv_files), self.max_datasets // len(ship_type_dirs)))

            print(f"    发现 {len(csv_files)} 个文件")

            for csv_file in csv_files:
                try:
                    # 轻量级验证：只检查文件是否存在和基本文件名格式
                    # 真正的格式验证推迟到实际加载数据时
                    dataset = DatasetInfo(
                        name=f"{ship_type}_{csv_file.stem}",
                        path=str(csv_file),
                        ship_type=ship_type
                    )

                    # 延迟加载统计信息，只在真正需要时计算
                    dataset.sample_size = None  # 标记为未计算
                    dataset.time_range = None
                    dataset.metadata = {}

                    datasets.append(dataset)

                    # 如果达到最大数量限制，停止扫描
                    if self.max_datasets and len(datasets) >= self.max_datasets:
                        break

                except Exception as e:
                    print(f"警告: 无法处理文件 {csv_file}: {e}")
                    continue

            if self.max_datasets and len(datasets) >= self.max_datasets:
                break

        # 如果没有找到真实数据，返回模拟数据集
        if not datasets:
            print("未找到真实数据集，使用模拟数据")
            datasets = self._create_mock_datasets()
        else:
            print(f"发现 {len(datasets)} 个数据集")

        return datasets

    def _validate_csv_structure(self, df: pd.DataFrame) -> bool:
        """验证 CSV 文件结构"""
        required_columns = ['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG']
        return all(col in df.columns for col in required_columns)

    def _analyze_speed_distribution(self, df: pd.DataFrame) -> Dict[str, float]:
        """分析速度分布"""
        speeds = df['SOG'].dropna()
        if len(speeds) == 0:
            return {'low': 0, 'medium': 0, 'high': 0}

        total = len(speeds)
        return {
            'low': len(speeds[(speeds >= self.SPEED_BINS['low'][0]) &
                             (speeds < self.SPEED_BINS['low'][1])]) / total * 100,
            'medium': len(speeds[(speeds >= self.SPEED_BINS['medium'][0]) &
                                (speeds < self.SPEED_BINS['medium'][1])]) / total * 100,
            'high': len(speeds[(speeds >= self.SPEED_BINS['high'][0]) &
                              (speeds < self.SPEED_BINS['high'][1])]) / total * 100
        }

    def _create_mock_datasets(self) -> List[DatasetInfo]:
        """创建模拟数据集（用于测试）"""
        return [
            DatasetInfo(
                name="tugboat_sample",
                path="mock_data/tugboat.csv",
                ship_type="Tugboat",
                sample_size=1000,
                time_range=("2021-01-01", "2021-01-02"),
                metadata={
                    'speed_distribution': {'low': 40.0, 'medium': 45.0, 'high': 15.0}
                }
            ),
            DatasetInfo(
                name="cargo_sample",
                path="mock_data/cargo.csv",
                ship_type="Cargo",
                sample_size=1500,
                time_range=("2021-01-01", "2021-01-03"),
                metadata={
                    'speed_distribution': {'low': 20.0, 'medium': 60.0, 'high': 20.0}
                }
            ),
            DatasetInfo(
                name="passenger_sample",
                path="mock_data/passenger.csv",
                ship_type="Passenger",
                sample_size=800,
                time_range=("2021-01-01", "2021-01-02"),
                metadata={
                    'speed_distribution': {'low': 30.0, 'medium': 50.0, 'high': 20.0}
                }
            )
        ]

    def get_datasets(self, ship_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取数据集列表"""
        datasets = self.datasets
        if ship_type:
            datasets = [d for d in datasets if d.ship_type == ship_type]

        # 对于少量数据集，尝试计算统计信息
        result = []
        for d in datasets[:20]:  # 只处理前20个数据集以避免超时
            if d.sample_size is None and d.path.startswith(str(self.data_root)):
                try:
                    # 尝试快速计算基本统计信息
                    df_sample = pd.read_csv(d.path, nrows=1000)  # 只读取前1000行作为样本
                    d.sample_size = len(df_sample)

                    if 'BaseDateTime' in df_sample.columns:
                        try:
                            df_sample['BaseDateTime'] = pd.to_datetime(df_sample['BaseDateTime'], errors='coerce')
                            valid_times = df_sample['BaseDateTime'].dropna()
                            if len(valid_times) > 0:
                                d.time_range = (
                                    valid_times.min().strftime('%Y-%m-%d'),
                                    valid_times.max().strftime('%Y-%m-%d')
                                )
                        except:
                            pass

                    if 'SOG' in df_sample.columns:
                        try:
                            speeds = pd.to_numeric(df_sample['SOG'], errors='coerce').dropna()
                            if len(speeds) > 0:
                                speed_stats = self._analyze_speed_distribution(df_sample)
                                d.metadata['speed_distribution'] = speed_stats
                        except:
                            pass

                except Exception as e:
                    print(f"警告: 无法计算数据集统计信息 {d.name}: {e}")

            result.append(d.to_dict())

        # 对于剩余的数据集，直接返回基本信息
        for d in datasets[20:]:
            result.append(d.to_dict())

        return result

    def load_dataset(self, dataset_name: str, mmsi: Optional[int] = None,
                    speed_segment: Optional[str] = None,
                    max_samples: Optional[int] = None) -> pd.DataFrame:
        """
        加载指定数据集

        参数:
            dataset_name: 数据集名称
            mmsi: 可选，筛选特定船舶
            speed_segment: 可选，速度段 ('low', 'medium', 'high')
            max_samples: 可选，最大采样数量

        返回:
            清洗后的 DataFrame
        """
        # 查找数据集
        dataset = None
        for d in self.datasets:
            if d.name == dataset_name:
                dataset = d
                break

        if not dataset:
            raise ValueError(f"数据集不存在: {dataset_name}")

        # 加载数据
        if dataset.path.startswith("mock_data"):
            # 生成模拟数据
            return self._generate_mock_data(dataset, speed_segment, max_samples)
        else:
            # 加载真实数据
            return self._load_real_data(dataset.path, mmsi, speed_segment, max_samples)

    def _load_real_data(self, filepath: str, mmsi: Optional[int] = None,
                       speed_segment: Optional[str] = None,
                       max_samples: Optional[int] = None) -> pd.DataFrame:
        """加载真实 AIS 数据"""
        print(f"正在加载数据文件: {filepath}")

        # 对于大文件，先读取少量行来检查格式
        try:
            df_sample = pd.read_csv(filepath, nrows=5)
            if not self._validate_csv_structure(df_sample):
                raise ValueError(f"文件格式不符合要求: {filepath}")
        except Exception as e:
            raise ValueError(f"无法读取文件 {filepath}: {e}")

        # 读取完整文件
        try:
            df = pd.read_csv(filepath)
            print(f"成功读取 {len(df)} 行数据")
        except Exception as e:
            raise ValueError(f"读取文件失败 {filepath}: {e}")

        # 验证必需的列
        required_columns = ['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"缺少必需的列: {missing_columns}")

        print(f"数据列: {list(df.columns)}")

        # 筛选特定船舶
        if mmsi is not None:
            original_count = len(df)
            df = df[df['MMSI'] == mmsi].copy()
            print(f"MMSI 筛选: {original_count} -> {len(df)}")

        # 转换时间戳
        try:
            df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
        except Exception as e:
            print(f"警告: 时间格式转换失败: {e}，尝试其他格式")
            # 尝试其他常见格式
            for fmt in ['%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S', '%d-%m-%Y %H:%M:%S']:
                try:
                    df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], format=fmt)
                    break
                except:
                    continue
            else:
                raise ValueError(f"无法解析时间格式")

        # 清洗数据：过滤无效的经纬度
        original_count = len(df)
        df = df[
            (df['LAT'] >= -90) & (df['LAT'] <= 90) &
            (df['LON'] >= -180) & (df['LON'] <= 180)
        ].copy()
        print(f"经纬度过滤: {original_count} -> {len(df)}")

        # 过滤掉无效值
        df = df.dropna(subset=['SOG', 'COG', 'LAT', 'LON'])

        # 确保数值类型正确
        df['SOG'] = pd.to_numeric(df['SOG'], errors='coerce')
        df['COG'] = pd.to_numeric(df['COG'], errors='coerce')
        df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
        df['LON'] = pd.to_numeric(df['LON'], errors='coerce')

        # 再次过滤 NaN 值
        df = df.dropna(subset=['SOG', 'COG', 'LAT', 'LON'])

        print(f"数据清洗后: {len(df)} 行")

        # 按时间排序
        df = df.sort_values('BaseDateTime').reset_index(drop=True)

        # 按速度段筛选
        if speed_segment and speed_segment in self.SPEED_BINS:
            min_speed, max_speed = self.SPEED_BINS[speed_segment]
            original_count = len(df)
            df = df[(df['SOG'] >= min_speed) & (df['SOG'] < max_speed)].copy()
            print(f"速度段筛选 ({speed_segment}): {original_count} -> {len(df)}")

        # 采样（对于大数据集）
        if max_samples and len(df) > max_samples:
            print(f"数据采样: {len(df)} -> {max_samples}")
            df = df.sample(n=max_samples, random_state=42).sort_values('BaseDateTime').reset_index(drop=True)

        final_df = df[required_columns].copy()
        print(f"最终数据集: {len(final_df)} 行")

        return final_df

    def _generate_mock_data(self, dataset: DatasetInfo,
                           speed_segment: Optional[str] = None,
                           max_samples: Optional[int] = None) -> pd.DataFrame:
        """生成模拟 AIS 数据"""
        np.random.seed(42)

        # 根据数据集类型确定参数
        if "tugboat" in dataset.name.lower():
            base_speed = 5.0
            speed_variation = 2.0
            course_change_prob = 0.1
        elif "cargo" in dataset.name.lower():
            base_speed = 12.0
            speed_variation = 3.0
            course_change_prob = 0.05
        elif "passenger" in dataset.name.lower():
            base_speed = 15.0
            speed_variation = 4.0
            course_change_prob = 0.08
        else:
            base_speed = 10.0
            speed_variation = 3.0
            course_change_prob = 0.05

        # 确定样本数量
        n_samples = max_samples or dataset.sample_size or 1000

        # 生成时间序列
        start_time = pd.Timestamp('2021-01-01 00:00:00')
        times = [start_time + pd.Timedelta(minutes=i) for i in range(n_samples)]

        # 初始化位置
        lat, lon = 34.6, -77.0
        current_course = 75.0
        current_speed = base_speed

        lats, lons, speeds, courses, mmsis = [], [], [], [], []

        for i in range(n_samples):
            # 随机改变航向和速度
            if np.random.random() < course_change_prob:
                current_course += np.random.normal(0, 15)
                current_course = current_course % 360

            if np.random.random() < 0.1:  # 10% 概率改变速度
                current_speed = max(0, base_speed + np.random.normal(0, speed_variation))

            # 计算新位置
            speed_mps = current_speed * 0.514444  # knots to m/s
            delta_t = 60  # 1分钟

            # 简化的位置计算
            lat_rad = np.radians(lat)
            lon_rad = np.radians(lon)
            course_rad = np.radians(current_course)

            dlat = (speed_mps * np.cos(course_rad) * delta_t) / 6371000 * 180 / np.pi
            dlon = (speed_mps * np.sin(course_rad) * delta_t) / (6371000 * np.cos(lat_rad)) * 180 / np.pi

            lat += dlat
            lon += dlon

            # 限制在合理范围内
            lat = np.clip(lat, -90, 90)
            lon = np.clip(lon, -180, 180)

            # 记录数据
            lats.append(lat)
            lons.append(lon)
            speeds.append(current_speed)
            courses.append(current_course)
            mmsis.append(123456789)

        # 创建 DataFrame
        df = pd.DataFrame({
            'MMSI': mmsis,
            'BaseDateTime': times,
            'LAT': lats,
            'LON': lons,
            'SOG': speeds,
            'COG': courses
        })

        # 按速度段筛选
        if speed_segment and speed_segment in self.SPEED_BINS:
            min_speed, max_speed = self.SPEED_BINS[speed_segment]
            df = df[(df['SOG'] >= min_speed) & (df['SOG'] < max_speed)].copy()

        return df.reset_index(drop=True)
