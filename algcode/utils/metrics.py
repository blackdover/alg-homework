import numpy as np  # 导入numpy数值计算库，用于数组操作和数学函数
import pandas as pd  # 导入pandas数据处理库，用于数据框操作
from typing import Dict, Optional  # 导入类型注解，用于函数参数和返回值的类型提示


def _ensure_datetime(s: pd.Series) -> pd.Series:  # 私有辅助函数，确保时间序列格式正确
    """确保pandas Series是datetime64格式，如果不是则转换"""
    if np.issubdtype(s.dtype, np.datetime64):  # 如果已经是datetime64类型
        return s  # 直接返回
    return pd.to_datetime(s, errors="coerce", utc=True)  # 转换为datetime，容错处理，转为UTC时间


def _angle_diff_deg(a: float, b: float) -> float:  # 私有辅助函数，计算角度差
    """计算两个角度之间的最小差值（考虑360度圆周特性）"""
    d = abs(a - b) % 360.0  # 计算绝对差并取360度的模
    return min(d, 360.0 - d)  # 返回两个方向中的最小角度差


def calculate_sed_metrics(original_df: pd.DataFrame,  # 计算SED误差指标的主函数
                         compressed_df: pd.DataFrame,  # 原始轨迹数据框
                         time_col: str = "BaseDateTime",  # 时间列名，默认BaseDateTime
                         lat_col: str = "LAT",  # 纬度列名，默认LAT
                         lon_col: str = "LON",  # 经度列名，默认LON
                         idx_col: str = "orig_idx") -> Dict[str, float]:  # 原始索引列名，默认orig_idx
    """
    计算真正的SED（Synchronized Euclidean Distance同步欧几里得距离）。
    compressed_df必须包含原始索引列idx_col（指向original_df的行号）。
    对每个压缩段，在时间轴上对段内原始点做线性插值并计算地理距离（米）。
    这是轨迹压缩质量评估的核心指标。
    """
    from .geo_utils import GeoUtils  # 导入地理工具类，用于距离计算

    if len(compressed_df) <= 1 or len(original_df) <= 1:  # 如果任一数据框点数太少
        return {"mean": 0.0, "max": 0.0, "p95": 0.0}  # 返回零误差

    if idx_col not in compressed_df.columns:  # 检查压缩数据是否有原始索引列
        raise ValueError(f"compressed_df 缺少 {idx_col}，请让算法输出保留原始索引（orig_idx）")

    o = original_df.reset_index(drop=True).copy()  # 复制原始数据，重置索引
    o[time_col] = _ensure_datetime(o[time_col])  # 确保时间列格式正确

    c = compressed_df.sort_values(idx_col).reset_index(drop=True).copy()  # 复制压缩数据，按原始索引排序
    c[time_col] = _ensure_datetime(c[time_col])  # 确保时间列格式正确

    o = o.dropna(subset=[time_col, lat_col, lon_col]).reset_index(drop=True)  # 删除原始数据中的缺失值
    c = c.dropna(subset=[time_col, lat_col, lon_col, idx_col]).reset_index(drop=True)  # 删除压缩数据中的缺失值

    if len(c) <= 1 or len(o) <= 1:  # 再次检查数据有效性
        return {"mean": 0.0, "max": 0.0, "p95": 0.0}  # 返回零误差

    sed_values = []  # 初始化SED误差值列表

    for i in range(len(c) - 1):  # 遍历压缩轨迹的每两个相邻点（构成压缩段）
        s = c.iloc[i]  # 当前压缩段的起始点
        e = c.iloc[i + 1]  # 当前压缩段的结束点

        si = int(s[idx_col])  # 获取起始点对应的原始轨迹索引
        ei = int(e[idx_col])  # 获取结束点对应的原始轨迹索引
        if ei <= si:  # 如果索引无效（结束索引不大于起始索引）
            continue  # 跳过这个压缩段

        t0 = s[time_col]  # 压缩段起始时间
        t1 = e[time_col]  # 压缩段结束时间
        if pd.isna(t0) or pd.isna(t1):  # 如果时间数据缺失
            continue  # 跳过这个压缩段

        denom = (t1 - t0).total_seconds()  # 计算时间间隔（秒）
        if denom <= 0:  # 如果时间间隔无效
            continue  # 跳过这个压缩段

        lat0, lon0 = float(s[lat_col]), float(s[lon_col])  # 压缩段起始点的坐标
        lat1, lon1 = float(e[lat_col]), float(e[lon_col])  # 压缩段结束点的坐标

        # 提取当前压缩段对应的原始轨迹区间（不含压缩段的端点）
        seg = o.iloc[si + 1: ei].copy()  # 获取原始轨迹中si+1到ei-1的点
        if seg.empty:  # 如果区间内没有原始点
            continue  # 跳过这个压缩段

        dt = (seg[time_col] - t0).dt.total_seconds().to_numpy(dtype=float)  # 计算每个原始点相对起始时间的时间差
        alpha = dt / denom  # 计算时间插值系数（0-1之间）
        mask = (alpha >= 0.0) & (alpha <= 1.0)  # 创建掩码，只保留时间系数在有效范围内的点
        if not np.any(mask):  # 如果没有有效点
            continue  # 跳过这个压缩段

        alpha = alpha[mask]  # 过滤时间系数
        lat_actual = seg.loc[seg.index[mask], lat_col].to_numpy(dtype=float)  # 获取有效点的实际纬度
        lon_actual = seg.loc[seg.index[mask], lon_col].to_numpy(dtype=float)  # 获取有效点的实际经度

        lat_hat = lat0 + alpha * (lat1 - lat0)  # 根据时间系数线性插值预测纬度
        lon_hat = lon0 + alpha * (lon1 - lon0)  # 根据时间系数线性插值预测经度

        for la, lo, lh, lnh in zip(lat_actual, lon_actual, lat_hat, lon_hat):  # 遍历每个原始点
            sed = GeoUtils.haversine_distance(la, lo, lh, lnh)  # 计算实际位置与预测位置的距离误差
            sed_values.append(sed)  # 将误差值添加到列表

    if not sed_values:  # 如果没有计算出任何SED误差值
        return {"mean": 0.0, "max": 0.0, "p95": 0.0}  # 返回零误差指标

    arr = np.asarray(sed_values, dtype=float)  # 将SED值列表转换为numpy数组
    return {"mean": float(arr.mean()), "max": float(arr.max()), "p95": float(np.percentile(arr, 95))}  # 返回统计指标


def calculate_navigation_event_recall(original_df: pd.DataFrame,  # 计算航行事件保留率函数
                                    compressed_df: pd.DataFrame,  # 原始轨迹数据框
                                    cog_threshold: float = 20.0,  # COG变化阈值，默认20度
                                    match_window_s: int = 30) -> float:  # 时间匹配窗口，默认30秒
    """
    计算航行事件保留率（使用时间窗口匹配）。
    航行事件指航向（COG）发生显著变化的事件。
    """
    if len(original_df) < 2:  # 如果原始轨迹点数太少
        return 1.0  # 返回完美保留率

    time_col = "BaseDateTime"  # 时间列名
    o = original_df.copy()  # 复制原始数据
    c = compressed_df.copy()  # 复制压缩数据
    o[time_col] = _ensure_datetime(o[time_col])  # 确保原始数据时间格式正确
    c[time_col] = _ensure_datetime(c[time_col])  # 确保压缩数据时间格式正确

    def find_turn_times(df: pd.DataFrame):  # 内部函数，查找轨迹中的转弯时间点
        times = []  # 初始化转弯时间列表
        for i in range(1, len(df)):  # 遍历轨迹点（从第二个点开始）
            a = df.iloc[i - 1].get('COG')  # 获取前一个点的航向
            b = df.iloc[i].get('COG')  # 获取当前点的航向
            if pd.isna(a) or pd.isna(b):  # 如果航向数据缺失
                continue  # 跳过这个点
            try:
                da = float(a)  # 转换为浮点数
                db = float(b)
            except Exception:  # 如果转换失败
                continue  # 跳过这个点
            if _angle_diff_deg(da, db) > cog_threshold:  # 如果航向变化超过阈值
                t = df.iloc[i][time_col]  # 获取转弯发生的时间
                if pd.isna(t):  # 如果时间数据缺失
                    continue  # 跳过
                times.append(t)  # 将转弯时间添加到列表
        return times  # 返回转弯时间列表

    orig_times = find_turn_times(o)  # 在原始轨迹中查找转弯时间
    if len(orig_times) == 0:  # 如果原始轨迹没有转弯事件
        return 1.0  # 返回完美保留率（没有事件需要保留）
    comp_times = find_turn_times(c)  # 在压缩轨迹中查找转弯时间

    matched = 0
    for ot in orig_times:
        found = False
        for ct in comp_times:
            if abs((ct - ot).total_seconds()) <= match_window_s:
                found = True
                break
        if found:
            matched += 1
    return matched / len(orig_times)


def evaluate_compression(original_df: pd.DataFrame,  # 评估压缩算法性能的主函数
                        compressed_df: pd.DataFrame,  # 原始轨迹数据框
                        algorithm_name: str,  # 压缩算法名称
                        elapsed_time: float = None,  # 算法运行耗时（可选）
                        calculate_sed: bool = True) -> dict:  # 是否计算SED指标，默认True
    N = len(original_df)  # 原始轨迹点数
    M = len(compressed_df)  # 压缩后轨迹点数
    compression_ratio = (1 - M / N) * 100 if N > 0 else 0.0  # 计算压缩率（百分比）
    if calculate_sed:  # 如果需要计算SED指标
        sed_metrics = calculate_sed_metrics(original_df, compressed_df)  # 计算SED误差指标
    else:  # 如果不需要计算SED
        sed_metrics = {'mean': 0.0, 'max': 0.0, 'p95': 0.0}  # 使用默认零值
    event_recall = calculate_navigation_event_recall(original_df, compressed_df)  # 计算航行事件保留率

    # 计算轨迹相似度指标（综合SED和事件保留率的评分）
    similarity_score = calculate_trajectory_similarity(original_df, compressed_df, sed_metrics)
    print(f"\n{'='*60}")
    print(f"算法: {algorithm_name}")
    print(f"{'='*60}")
    print(f"原始点数: {N}")
    print(f"压缩后点数: {M}")
    print(f"压缩率: {compression_ratio:.2f}%")
    if calculate_sed:
        print(f"SED均值: {sed_metrics['mean']:.2f} 米")
        print(f"SED最大值: {sed_metrics['max']:.2f} 米")
        print(f"SED 95分位数: {sed_metrics['p95']:.2f} 米")
    print(f"航行事件保留率: {event_recall:.3f}")
    print(f"轨迹相似度: {similarity_score:.3f}")
    if elapsed_time is not None:  # 如果提供了运行时间
        print(f"运行时间: {elapsed_time:.4f} 秒")  # 打印运行时间
    print(f"{'='*60}\n")  # 打印分隔线
    result = {  # 构建结果字典
        'algorithm': algorithm_name,  # 算法名称
        'original_points': N,  # 原始点数
        'compressed_points': M,  # 压缩后点数
        'compression_ratio': compression_ratio,  # 压缩率
        'event_recall': event_recall,  # 事件保留率
        'trajectory_similarity': similarity_score,  # 轨迹相似度
        'elapsed_time': elapsed_time  # 运行耗时
    }
    if calculate_sed:  # 如果计算了SED指标
        result.update({  # 将SED指标添加到结果中
            'sed_mean': sed_metrics['mean'],  # SED均值
            'sed_max': sed_metrics['max'],   # SED最大值
            'sed_p95': sed_metrics['p95']    # SED 95%分位数
        })
    return result  # 返回完整的评估结果字典


# 定义计算轨迹相似度指标的函数
def calculate_trajectory_similarity(original_df: pd.DataFrame,  # 计算轨迹相似度函数
                                  compressed_df: pd.DataFrame,  # 原始轨迹
                                  sed_metrics: Dict[str, float]) -> float:  # 压缩轨迹和SED指标
    """
    轨迹相似度（0~1）：用 SED 均值归一化后转成分数（使用总轨迹长度作为基准）。
    """
    from .geo_utils import GeoUtils

    if len(original_df) <= 1 or len(compressed_df) <= 1:
        return 0.0

    total_distance = 0.0
    for i in range(1, len(original_df)):
        total_distance += GeoUtils.haversine_distance(
            float(original_df.iloc[i - 1]['LAT']), float(original_df.iloc[i - 1]['LON']),
            float(original_df.iloc[i]['LAT']), float(original_df.iloc[i]['LON'])
        )
    total_distance = max(total_distance, 1e-6)

    normalized_sed = sed_metrics.get('mean', 0.0) / total_distance

    keep_ratio = len(compressed_df) / max(len(original_df), 1)
    compression_penalty = 0.0
    if keep_ratio < 0.05:
        compression_penalty = (0.05 - keep_ratio) * 2.0

    score = 1.0 / (1.0 + normalized_sed + compression_penalty)
    return float(np.clip(score, 0.0, 1.0))


