#!/usr/bin/env python3

import pandas as pd
from algcode import load_data, dead_reckoning_compress
import time

def test_sed_performance():
    # 加载数据并压缩
    df = load_data(r'E:\code\homework\alg\AIS Dataset\AIS Data\Tugboat\220584000.csv')
    result = dead_reckoning_compress(df, 100.0)

    print(f'Original: {len(df)} points')
    print(f'Compressed: {len(result)} points')

    # 估算SED计算的复杂度
    n_compressed = len(result)
    n_original = len(df)
    estimated_operations = n_compressed * n_original

    print(f'Estimated operations for SED calculation: {estimated_operations:,} ({n_compressed} * {n_original})')
    print('This is why it takes so long!')

    # 测试优化后的SED计算
    print('\nTesting optimized SED calculation...')

    # 使用更高效的方法：预排序并使用二分查找
    original_sorted = df.sort_values('BaseDateTime').reset_index(drop=True)
    compressed_sorted = result.sort_values('BaseDateTime').reset_index(drop=True)

    start_time = time.time()

    sed_values = []
    for i in range(len(compressed_sorted) - 1):
        segment_start = compressed_sorted.iloc[i]
        segment_end = compressed_sorted.iloc[i + 1]

        start_time_val = segment_start['BaseDateTime']
        end_time_val = segment_end['BaseDateTime']

        # 使用向量化操作找到时间范围内的点
        mask = (original_sorted['BaseDateTime'] >= start_time_val) & (original_sorted['BaseDateTime'] <= end_time_val)
        segment_points = original_sorted[mask]

        # 计算SED
        for _, point in segment_points.iterrows():
            from algcode.geo_utils import GeoUtils
            sed = GeoUtils.point_to_line_distance(
                point['LAT'], point['LON'],
                segment_start['LAT'], segment_start['LON'],
                segment_end['LAT'], segment_end['LON']
            )
            sed_values.append(sed)

    elapsed = time.time() - start_time
    print(f'Optimized SED calculation: {elapsed:.2f} seconds')

    if sed_values:
        import numpy as np
        sed_array = np.array(sed_values)
        print(f'SED stats - Mean: {np.mean(sed_array):.2f}, Max: {np.max(sed_array):.2f}, P95: {np.percentile(sed_array, 95):.2f}')

if __name__ == "__main__":
    test_sed_performance()
