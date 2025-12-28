import os
from pathlib import Path
import pandas as pd

base = r"E:\code\homework\alg\AIS Dataset\AIS Data"
cats = ['Cargo', 'Fishing', 'Passenger', 'Pleasure Craft and Sailing', 'Tanker', 'Tugboat']

print("="*70)
print("AIS数据集统计")
print("="*70)

print("\n【文件数量统计】")
total = 0
for cat in cats:
    path = os.path.join(base, cat)
    if os.path.exists(path):
        count = len(list(Path(path).rglob("*.csv")))
        total += count
        print(f"  {cat:<30}: {count:>6} files")

print(f"\n  总计: {total} files")

print("\n【样本文件分析 (Tugboat/220584000.csv)】")
sample_file = os.path.join(base, "Tugboat", "220584000.csv")
if os.path.exists(sample_file):
    df = pd.read_csv(sample_file)
    print(f"  总行数: {len(df):,}")
    print(f"  列数: {len(df.columns)}")
    print(f"  列名: {', '.join(df.columns)}")
    if 'BaseDateTime' in df.columns:
        df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
        print(f"  时间范围: {df['BaseDateTime'].min()} 至 {df['BaseDateTime'].max()}")
    if 'MMSI' in df.columns:
        print(f"  MMSI: {df['MMSI'].iloc[0]}")
    file_size = os.path.getsize(sample_file) / (1024*1024)
    print(f"  文件大小: {file_size:.2f} MB")

print("\n" + "="*70)

