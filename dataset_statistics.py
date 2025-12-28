"""
AIS数据集统计脚本
统计E:\code\homework\alg\AIS Dataset目录下的数据信息
"""

import os
import pandas as pd
from pathlib import Path
from collections import defaultdict

def get_directory_size(path):
    """计算目录总大小（MB）"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.csv'):
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)  # 转换为MB

def count_lines_in_file(filepath):
    """快速统计CSV文件行数（不包括表头）"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for line in f) - 1  # 减去表头
    except:
        return 0

def analyze_sample_file(filepath):
    """分析样本文件的基本信息"""
    try:
        df = pd.read_csv(filepath, nrows=1000)  # 只读前1000行用于分析
        total_lines = count_lines_in_file(filepath)
        
        info = {
            'total_rows': total_lines,
            'columns': list(df.columns),
            'sample_rows': len(df)
        }
        
        if 'BaseDateTime' in df.columns:
            df_full = pd.read_csv(filepath)
            df_full['BaseDateTime'] = pd.to_datetime(df_full['BaseDateTime'])
            info['date_range'] = (df_full['BaseDateTime'].min(), df_full['BaseDateTime'].max())
            info['unique_mmsi'] = df_full['MMSI'].nunique() if 'MMSI' in df_full.columns else 1
        
        return info
    except Exception as e:
        return {'error': str(e)}

def main():
    base_path = r"E:\code\homework\alg\AIS Dataset\AIS Data"
    
    print("="*80)
    print("AIS数据集统计报告")
    print("="*80)
    
    categories = ['Cargo', 'Fishing', 'Passenger', 'Pleasure Craft and Sailing', 'Tanker', 'Tugboat']
    
    stats = {}
    total_files = 0
    total_size_mb = 0
    
    for category in categories:
        category_path = os.path.join(base_path, category)
        if not os.path.exists(category_path):
            continue
        
        # 统计文件数量
        csv_files = list(Path(category_path).rglob('*.csv'))
        file_count = len(csv_files)
        total_files += file_count
        
        # 计算总大小
        size_mb = get_directory_size(category_path)
        total_size_mb += size_mb
        
        # 分析一个样本文件
        sample_file = None
        if csv_files:
            sample_file = str(csv_files[0])
            sample_info = analyze_sample_file(sample_file)
        else:
            sample_info = {}
        
        stats[category] = {
            'file_count': file_count,
            'size_mb': round(size_mb, 2),
            'sample_file': os.path.basename(sample_file) if sample_file else None,
            'sample_info': sample_info
        }
    
    # 打印统计结果
    print(f"\n{'类别':<30} {'文件数':<15} {'总大小(MB)':<15} {'样本文件':<20}")
    print("-"*80)
    
    for category, data in stats.items():
        print(f"{category:<30} {data['file_count']:<15} {data['size_mb']:<15} {data['sample_file'] or 'N/A':<20}")
    
    print("-"*80)
    print(f"{'总计':<30} {total_files:<15} {round(total_size_mb, 2):<15}")
    
    # 打印样本文件详细信息
    print("\n" + "="*80)
    print("样本文件详细信息")
    print("="*80)
    
    for category, data in stats.items():
        if data['sample_file']:
            print(f"\n【{category}】")
            print(f"  样本文件: {data['sample_file']}")
            sample_info = data['sample_info']
            if 'error' not in sample_info:
                print(f"  总行数: {sample_info.get('total_rows', 'N/A')}")
                print(f"  列数: {len(sample_info.get('columns', []))}")
                print(f"  列名: {', '.join(sample_info.get('columns', []))}")
                if 'date_range' in sample_info:
                    print(f"  时间范围: {sample_info['date_range'][0]} 至 {sample_info['date_range'][1]}")
                if 'unique_mmsi' in sample_info:
                    print(f"  唯一MMSI数: {sample_info['unique_mmsi']}")
            else:
                print(f"  错误: {sample_info['error']}")
    
    print("\n" + "="*80)
    print("统计完成！")
    print("="*80)

if __name__ == "__main__":
    main()

