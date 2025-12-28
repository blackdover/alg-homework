#!/usr/bin/env python3
"""测试优化后的数据加载器"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from data_loader import DataLoader
import time

def test_loading():
    print("测试优化后的数据加载器...")
    print("=" * 50)

    # 测试1: 加载10个数据集
    print("测试1: 加载10个数据集")
    start_time = time.time()
    dl1 = DataLoader(max_datasets=10)
    elapsed1 = time.time() - start_time
    datasets1 = dl1.get_datasets()
    print(f"  耗时: {elapsed1:.2f}秒")
    print(f"  数据集数量: {len(datasets1)}")
    print()

    # 测试2: 只加载Tugboat类型
    print("测试2: 只加载Tugboat类型")
    start_time = time.time()
    dl2 = DataLoader(max_datasets=50, ship_types=['Tugboat'])
    elapsed2 = time.time() - start_time
    datasets2 = dl2.get_datasets()
    print(f"  耗时: {elapsed2:.2f}秒")
    print(f"  数据集数量: {len(datasets2)}")
    print()

    # 显示一些示例数据集
    if datasets1:
        print("示例数据集:")
        for i, d in enumerate(datasets1[:3]):
            print(f"  {d['name']}: {d['ship_type']}")

    print("\n测试完成!")

if __name__ == "__main__":
    test_loading()

