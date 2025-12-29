"""
轨迹压缩评估指标模块
包含SED、事件保留率等评估指标计算

作者: Algorithm Engineer
Python版本: 3.10
"""

import numpy as np
import pandas as pd
from typing import Dict


def calculate_sed_metrics(original_df: pd.DataFrame,
                         compressed_df: pd.DataFrame) -> Dict[str, float]:
    """计算 SED (Spatial Error Distance) 相关指标 - 优化版本"""
    from .geo_utils import GeoUtils

    if len(compressed_df) <= 1:
        return {'mean': 0.0, 'max': 0.0, 'p95': 0.0}

    sed_values = []

    # 确保数据按时间排序
    original_sorted = original_df.sort_values('BaseDateTime').reset_index(drop=True)
    compressed_sorted = compressed_df.sort_values('BaseDateTime').reset_index(drop=True)

    # 为每个压缩段计算SED
    for i in range(len(compressed_sorted) - 1):
        segment_start = compressed_sorted.iloc[i]
        segment_end = compressed_sorted.iloc[i + 1]

        start_time = segment_start['BaseDateTime']
        end_time = segment_end['BaseDateTime']

        # 使用向量化操作找到时间范围内的点 - 这比循环快得多
        mask = (original_sorted['BaseDateTime'] >= start_time) & (original_sorted['BaseDateTime'] <= end_time)
        segment_points = original_sorted[mask]

        if len(segment_points) == 0:
            continue

        # 计算该段内各点的 SED - 仍然需要循环但优化了数据访问
        for _, point in segment_points.iterrows():
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
                        elapsed_time: float = None,
                        calculate_sed: bool = True) -> dict:
    """
    计算并打印压缩算法的评估指标

    参数:
        original_df: 原始轨迹DataFrame
        compressed_df: 压缩后的轨迹DataFrame
        algorithm_name: 算法名称
        elapsed_time: 算法运行时间（秒），可选
        calculate_sed: 是否计算SED指标（耗时操作），默认True

    返回:
        包含评估指标的字典
    """
    N = len(original_df)  # 原始点数
    M = len(compressed_df)  # 压缩后点数

    # 压缩率
    compression_ratio = (1 - M / N) * 100 if N > 0 else 0.0

    # SED 指标（可选，耗时）
    if calculate_sed:
        sed_metrics = calculate_sed_metrics(original_df, compressed_df)
    else:
        sed_metrics = {'mean': 0.0, 'max': 0.0, 'p95': 0.0}

    # 航行事件保留率
    event_recall = calculate_navigation_event_recall(original_df, compressed_df)

    # 打印结果
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
    if elapsed_time is not None:
        print(f"运行时间: {elapsed_time:.4f} 秒")
    print(f"{'='*60}\n")

    result = {
        'algorithm': algorithm_name,
        'original_points': N,
        'compressed_points': M,
        'compression_ratio': compression_ratio,
        'event_recall': event_recall,
        'elapsed_time': elapsed_time
    }

    # 只有在启用SED计算时才包含SED指标
    if calculate_sed:
        result.update({
            'sed_mean': sed_metrics['mean'],
            'sed_max': sed_metrics['max'],
            'sed_p95': sed_metrics['p95']
        })

    return result
