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
    original_sorted = original_df.sort_values('BaseDateTime').reset_index(drop=True)
    compressed_sorted = compressed_df.sort_values('BaseDateTime').reset_index(drop=True)

    for i in range(len(compressed_sorted) - 1):
        segment_start = compressed_sorted.iloc[i]
        segment_end = compressed_sorted.iloc[i + 1]
        start_time = segment_start['BaseDateTime']
        end_time = segment_end['BaseDateTime']
        mask = (original_sorted['BaseDateTime'] >= start_time) & (original_sorted['BaseDateTime'] <= end_time)
        segment_points = original_sorted[mask]
        if len(segment_points) == 0:
            continue
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
    original_turns = 0
    for i in range(1, len(original_df)):
        prev_cog = original_df.iloc[i-1]['COG']
        curr_cog = original_df.iloc[i]['COG']
        if abs(curr_cog - prev_cog) > cog_threshold:
            original_turns += 1
    if original_turns == 0:
        return 1.0
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
    N = len(original_df)
    M = len(compressed_df)
    compression_ratio = (1 - M / N) * 100 if N > 0 else 0.0
    if calculate_sed:
        sed_metrics = calculate_sed_metrics(original_df, compressed_df)
    else:
        sed_metrics = {'mean': 0.0, 'max': 0.0, 'p95': 0.0}
    event_recall = calculate_navigation_event_recall(original_df, compressed_df)
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
    if calculate_sed:
        result.update({
            'sed_mean': sed_metrics['mean'],
            'sed_max': sed_metrics['max'],
            'sed_p95': sed_metrics['p95']
        })
    return result


