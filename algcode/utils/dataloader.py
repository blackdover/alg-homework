import pandas as pd
import os
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass


@dataclass
class TrajectoryPoint:
    lat: float
    lon: float
    timestamp: pd.Timestamp
    sog: float
    cog: float
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
    df = pd.read_csv(filepath)

    required_columns = ['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺少必需的列: {missing_columns}")

    if mmsi is not None:
        df = df[df['MMSI'] == mmsi].copy()

    df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])

    df = df[
        (df['LAT'] >= -90) & (df['LAT'] <= 90) &
        (df['LON'] >= -180) & (df['LON'] <= 180)
    ].copy()

    df = df.dropna(subset=['SOG', 'COG', 'LAT', 'LON'])

    df = df.sort_values('BaseDateTime').reset_index(drop=True)

    return df[required_columns].copy()


def dataframe_to_trajectory_points(df: pd.DataFrame) -> List[TrajectoryPoint]:
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
    data = [point.to_dict() for point in points]
    return pd.DataFrame(data)


def scan_categories(data_root: str) -> List[str]:
    if not os.path.exists(data_root):
        return []
    return [d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))]


def scan_datasets(category_path: str) -> List[Tuple[str, int]]:
    datasets = []
    if not os.path.exists(category_path):
        return datasets

    for file in os.listdir(category_path):
        if file.endswith('.csv'):
            filepath = os.path.join(category_path, file)
            try:
                df = pd.read_csv(filepath, usecols=['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG'])
                point_count = len(df)
                datasets.append((file, point_count))
            except Exception:
                continue
    return datasets


def load_ais_dataset(data_root: str, category: str, filename: str, mmsi: Optional[int] = None) -> pd.DataFrame:
    filepath = os.path.join(data_root, category, filename)
    return load_data(filepath, mmsi)


