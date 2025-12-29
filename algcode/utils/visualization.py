import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, Dict, List
import numpy as np


def visualize_trajectories(original_df: pd.DataFrame,
                          dr_df: pd.DataFrame,
                          dp_df: Optional[pd.DataFrame] = None,
                          output_file: str = "visualization.png") -> None:
    trajectories = {
        '原始轨迹': original_df,
        'DR压缩': dr_df
    }
    if dp_df is not None:
        trajectories['DP压缩'] = dp_df
    visualize_multiple_trajectories(trajectories, output_file)


def visualize_multiple_trajectories(trajectories: Dict[str, pd.DataFrame],
                                  output_file: str = "visualization.png") -> None:
    """使用matplotlib创建多子图轨迹可视化，白底布局，底部显示原始轨迹"""
    if not trajectories:
        raise ValueError("至少需要一个轨迹")

    # 分离原始轨迹和其他压缩轨迹
    original_trajectory = None
    compressed_trajectories = {}

    for name, df in trajectories.items():
        if len(df) == 0:
            continue
        if name == '原始轨迹':
            original_trajectory = df
        else:
            compressed_trajectories[name] = df

    if original_trajectory is None:
        raise ValueError("必须包含原始轨迹")

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建子图布局：上部显示压缩轨迹，下部显示原始轨迹
    n_compressed = len(compressed_trajectories)
    if n_compressed == 0:
        # 只有原始轨迹
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        axes = [ax]
        original_ax = ax
    else:
        # 创建GridSpec布局
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(n_compressed + 1, 1, height_ratios=[1] * n_compressed + [0.8])

        axes = []
        for i in range(n_compressed):
            ax = fig.add_subplot(gs[i, 0])
            axes.append(ax)

        # 底部原始轨迹
        original_ax = fig.add_subplot(gs[-1, 0])

    # 颜色设置
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    # 绘制压缩轨迹
    for i, (name, df) in enumerate(compressed_trajectories.items()):
        ax = axes[i]
        ax.set_facecolor('white')

        # 绘制压缩轨迹
        color = colors[i % len(colors)]
        ax.plot(df['LON'], df['LAT'], 'o-', color=color, linewidth=2, markersize=4,
                label=f'{name} ({len(df)} 点)', alpha=0.8)

        # 设置标题和标签
        ax.set_title(f'{name} 轨迹压缩结果', fontsize=12, fontweight='bold')
        ax.set_xlabel('经度 (°)', fontsize=10)
        ax.set_ylabel('纬度 (°)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        # 设置坐标轴范围
        all_lons = []
        all_lats = []
        for traj_df in [original_trajectory] + list(compressed_trajectories.values()):
            all_lons.extend(traj_df['LON'].tolist())
            all_lats.extend(traj_df['LAT'].tolist())

        if all_lons and all_lats:
            lon_margin = (max(all_lons) - min(all_lons)) * 0.05
            lat_margin = (max(all_lats) - min(all_lats)) * 0.05
            ax.set_xlim(min(all_lons) - lon_margin, max(all_lons) + lon_margin)
            ax.set_ylim(min(all_lats) - lat_margin, max(all_lats) + lat_margin)

    # 绘制底部原始轨迹
    original_ax.set_facecolor('white')
    original_ax.plot(original_trajectory['LON'], original_trajectory['LAT'], 'o-',
                    color='red', linewidth=2, markersize=3, alpha=0.7,
                    label=f'原始轨迹 ({len(original_trajectory)} 点)')

    # 标记起点和终点
    original_ax.plot(original_trajectory.iloc[0]['LON'], original_trajectory.iloc[0]['LAT'],
                    'go', markersize=8, label='起点')
    original_ax.plot(original_trajectory.iloc[-1]['LON'], original_trajectory.iloc[-1]['LAT'],
                    'ro', markersize=8, label='终点')

    original_ax.set_title('原始轨迹', fontsize=12, fontweight='bold')
    original_ax.set_xlabel('经度 (°)', fontsize=10)
    original_ax.set_ylabel('纬度 (°)', fontsize=10)
    original_ax.grid(True, alpha=0.3)
    original_ax.legend(loc='upper right')

    # 设置原始轨迹的坐标轴范围
    if all_lons and all_lats:
        lon_margin = (max(all_lons) - min(all_lons)) * 0.05
        lat_margin = (max(all_lats) - min(all_lats)) * 0.05
        original_ax.set_xlim(min(all_lons) - lon_margin, max(all_lons) + lon_margin)
        original_ax.set_ylim(min(all_lats) - lat_margin, max(all_lats) + lat_margin)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"轨迹可视化已保存至: {output_file}")