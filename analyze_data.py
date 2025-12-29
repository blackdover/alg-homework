#!/usr/bin/env python3

import pandas as pd
from algcode import load_data

def analyze_data():
    df = load_data(r'E:\code\homework\alg\AIS Dataset\AIS Data\Tugboat\220584000.csv')
    print(f'Loaded {len(df)} data points')

    # 检查几个时间差样本
    print('Sample time differences:')
    for i in range(1, min(10, len(df))):
        delta_t = (df.iloc[i]['BaseDateTime'] - df.iloc[i-1]['BaseDateTime']).total_seconds()
        print(f'  Point {i}: {delta_t} seconds')

    # 查找大的时间跳跃
    print('\nLarge time gaps:')
    count_large_gaps = 0
    max_gap = 0
    max_gap_idx = 0

    for i in range(1, len(df)):
        delta_t = (df.iloc[i]['BaseDateTime'] - df.iloc[i-1]['BaseDateTime']).total_seconds()
        if delta_t > 3600:  # 大于1小时
            count_large_gaps += 1
            if delta_t > max_gap:
                max_gap = delta_t
                max_gap_idx = i
            if count_large_gaps <= 5:  # 只显示前5个
                print(f'  Between points {i-1} and {i}: {delta_t} seconds ({delta_t/86400:.1f} days)')

    print(f'\nTotal large gaps (>1 hour): {count_large_gaps}')
    print(f'Maximum gap: {max_gap} seconds ({max_gap/86400:.1f} days) at index {max_gap_idx}')

    # 测试DR算法在不同大小的数据集上的表现
    print('\nTesting DR algorithm performance:')

    # 测试小数据集
    df_small = df.head(100).copy()
    from algcode import dead_reckoning_compress
    import time

    start_time = time.time()
    result_small = dead_reckoning_compress(df_small, 100.0)
    elapsed_small = time.time() - start_time
    print(f'Small dataset (100 points): {elapsed_small:.3f}s, compressed to {len(result_small)} points')

    # 测试中等数据集
    df_medium = df.head(1000).copy()
    start_time = time.time()
    result_medium = dead_reckoning_compress(df_medium, 100.0)
    elapsed_medium = time.time() - start_time
    print(f'Medium dataset (1000 points): {elapsed_medium:.3f}s, compressed to {len(result_medium)} points')

if __name__ == "__main__":
    analyze_data()
