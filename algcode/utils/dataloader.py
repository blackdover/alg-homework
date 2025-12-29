import pandas as pd  # 导入pandas数据处理库，用于数据框操作
import os  # 导入操作系统接口模块，用于文件路径操作
from typing import Optional, Tuple, Dict, List, Any  # 导入类型注解，用于函数参数和返回值的类型提示
from dataclasses import dataclass  # 导入数据类装饰器，用于创建简单的数据容器类


@dataclass  # 使用dataclass装饰器自动生成__init__、__repr__等方法
class TrajectoryPoint:  # 定义轨迹点的数据结构类
    """轨迹点数据结构，用于表示船舶在特定时刻的位置和运动信息"""
    lat: float  # 纬度坐标
    lon: float  # 经度坐标
    timestamp: pd.Timestamp  # 时间戳，使用pandas的时间戳类型
    sog: float  # 对地速度（Speed Over Ground），单位为节（knots）
    cog: float  # 对地航向（Course Over Ground），单位为度
    mmsi: Optional[int] = None  # 船舶MMSI识别号，可选参数

    def to_dict(self) -> Dict[str, Any]:  # 实例方法，将轨迹点转换为字典格式
        """将轨迹点对象转换为字典格式，便于数据处理"""
        return {  # 返回包含所有字段的字典
            'LAT': self.lat,           # 纬度
            'LON': self.lon,           # 经度
            'BaseDateTime': self.timestamp,  # 时间戳
            'SOG': self.sog,           # 对地速度
            'COG': self.cog,           # 对地航向
            'MMSI': self.mmsi          # 船舶识别号
        }


def load_data(filepath: str, mmsi: Optional[int] = None) -> pd.DataFrame:  # 加载AIS船舶数据的函数
    """
    读取CSV格式的AIS船舶数据并进行预处理和清洗

    参数:
        filepath: CSV文件的完整路径字符串
        mmsi: 可选参数，指定要筛选的特定船舶MMSI识别号，如果为None则加载所有船舶

    返回:
        清洗后的pandas DataFrame，包含标准化的AIS数据列：MMSI, BaseDateTime, LAT, LON, SOG, COG
    """
    # 使用pandas读取CSV文件，自动推断列的数据类型
    df = pd.read_csv(filepath)

    # 数据质量检查：验证必需的数据列是否存在
    required_columns = ['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG']  # 定义必需的列名列表
    missing_columns = [col for col in required_columns if col not in df.columns]  # 找出缺失的列
    if missing_columns:  # 如果有缺失的列
        raise ValueError(f"缺少必需的列: {missing_columns}")  # 抛出错误，中止执行

    # 根据MMSI筛选特定船舶的数据（如果指定了mmsi参数）
    if mmsi is not None:  # 如果提供了MMSI筛选条件
        df = df[df['MMSI'] == mmsi].copy()  # 筛选指定MMSI的行，并创建副本

    # 将时间戳字符串转换为pandas的datetime对象，便于时间计算
    df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])

    # 数据清洗：过滤掉无效的地理坐标
    # 纬度范围：-90度到90度，经度范围：-180度到180度
    df = df[  # 使用布尔索引过滤数据
        (df['LAT'] >= -90) & (df['LAT'] <= 90) &      # 纬度在有效范围内
        (df['LON'] >= -180) & (df['LON'] <= 180)      # 经度在有效范围内
    ].copy()  # 创建过滤后的数据副本

    # 过滤掉包含NaN（缺失值）的关键数据行
    df = df.dropna(subset=['SOG', 'COG', 'LAT', 'LON'])  # 删除在这些列中有NaN的行

    # 按时间戳升序排序，确保轨迹点按时间顺序排列
    df = df.sort_values('BaseDateTime').reset_index(drop=True)  # 排序并重置索引

    return df[required_columns].copy()  # 返回只包含必需列的DataFrame副本


def dataframe_to_trajectory_points(df: pd.DataFrame) -> List[TrajectoryPoint]:  # DataFrame转轨迹点列表的函数
    """将pandas DataFrame格式的轨迹数据转换为TrajectoryPoint对象的列表"""
    points = []  # 初始化空列表，用于存储转换后的轨迹点
    for _, row in df.iterrows():  # 遍历DataFrame的每一行（_表示行索引，不使用）
        point = TrajectoryPoint(  # 创建TrajectoryPoint对象
            lat=row['LAT'],              # 纬度
            lon=row['LON'],              # 经度
            timestamp=row['BaseDateTime'], # 时间戳
            sog=row['SOG'],              # 对地速度
            cog=row['COG'],              # 对地航向
            mmsi=row.get('MMSI')         # MMSI识别号（使用get避免KeyError）
        )
        points.append(point)  # 将创建的轨迹点添加到列表
    return points  # 返回轨迹点对象列表


def trajectory_points_to_dataframe(points: List[TrajectoryPoint]) -> pd.DataFrame:  # 轨迹点列表转DataFrame的函数
    """将TrajectoryPoint对象列表转换为pandas DataFrame格式"""
    data = [point.to_dict() for point in points]  # 为每个轨迹点调用to_dict()方法，生成字典列表
    return pd.DataFrame(data)  # 使用字典列表创建DataFrame并返回


def scan_categories(data_root: str) -> List[str]:  # 扫描数据类别的函数
    """
    扫描数据根目录下的所有子目录，作为不同的数据类别

    参数:
        data_root: 数据根目录的路径字符串

    返回:
        类别名称的字符串列表，每个类别对应一个子目录
    """
    if not os.path.exists(data_root):  # 检查目录是否存在
        return []  # 如果不存在，返回空列表
    return [d for d in os.listdir(data_root)  # 遍历目录内容
            if os.path.isdir(os.path.join(data_root, d))]  # 只保留目录（子类别）


def scan_datasets(category_path: str) -> List[Tuple[str, int]]:  # 扫描数据集的函数
    """
    扫描指定类别目录下的所有CSV数据集文件，返回文件名和数据点数

    参数:
        category_path: 类别目录的完整路径

    返回:
        数据集信息列表，每个元素是(文件名, 点数)的元组
    """
    datasets = []  # 初始化数据集列表
    if not os.path.exists(category_path):  # 检查类别目录是否存在
        return datasets  # 如果不存在，返回空列表

    for file in os.listdir(category_path):  # 遍历类别目录下的所有文件
        if file.endswith('.csv'):  # 只处理CSV文件
            filepath = os.path.join(category_path, file)  # 构建完整的文件路径
            try:  # 尝试读取文件获取点数信息
                # 快速读取CSV文件，只读取必要的列以提高性能
                df = pd.read_csv(filepath, usecols=['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG'])
                point_count = len(df)  # 获取数据点数量
                datasets.append((file, point_count))  # 添加到数据集列表
            except Exception:  # 如果读取失败（文件损坏、格式错误等）
                # 如果读取失败，跳过该文件
                continue  # 继续处理下一个文件
    return datasets  # 返回数据集信息列表


def load_ais_dataset(data_root: str, category: str, filename: str, mmsi: Optional[int] = None) -> pd.DataFrame:  # 加载AIS数据集的高级函数
    """
    从AIS数据集目录结构中加载指定的数据集文件

    参数:
        data_root: 数据根目录路径
        category: 数据类别（子目录名）
        filename: 数据文件名
        mmsi: 可选参数，筛选特定船舶的MMSI识别号

    返回:
        经过清洗和预处理的pandas DataFrame
    """
    filepath = os.path.join(data_root, category, filename)  # 构建完整的文件路径
    return load_data(filepath, mmsi)  # 调用load_data函数进行实际的数据加载和清洗


