#!/usr/bin/env python3
"""测试数据加载器"""

from backend.data_loader import DataLoader

def main():
    print("测试数据加载器...")
    dl = DataLoader()
    print("发现的数据集:")

    datasets = dl.get_datasets()
    for i, d in enumerate(datasets[:10]):  # 只显示前10个
        print(f"  {d['name']}: {d['ship_type']}, {d.get('sample_size', 'N/A')} 点")

    print(f"\n总共发现 {len(datasets)} 个数据集")

    # 测试加载一个数据集
    if datasets:
        print(f"\n测试加载数据集: {datasets[0]['name']}")
        try:
            df = dl.load_dataset(datasets[0]['name'], max_samples=100)
            print(f"成功加载 {len(df)} 行数据")
            print(f"列: {list(df.columns)}")
            print(f"时间范围: {df['BaseDateTime'].min()} - {df['BaseDateTime'].max()}")
        except Exception as e:
            print(f"加载失败: {e}")

if __name__ == "__main__":
    main()
