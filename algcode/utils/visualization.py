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

    # 创建子图布局：按 2 列排列压缩轨迹，上方为压缩结果，底部为原始轨迹（跨两列）
    n_compressed = len(compressed_trajectories)
    if n_compressed == 0:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        axes = [ax]
        original_ax = ax
    else:
        cols = 2
        rows = int(np.ceil(n_compressed / cols))
        rows = min(rows, 3)  # 限制为最多 3 行（2x3 布局）
        # 总高度：每行约 3.0，高度再加一个原始轨迹区域（3.0）
        fig_height = rows * 3.0 + 3.0
        fig_width = 12
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = gridspec.GridSpec(rows + 1, cols, height_ratios=[1] * rows + [0.9])

        axes = []
        for i in range(n_compressed):
            r = i // cols
            ccol = i % cols
            ax = fig.add_subplot(gs[r, ccol])
            axes.append(ax)

        # 底部原始轨迹跨两列
        original_ax = fig.add_subplot(gs[-1, :])

    # 颜色设置
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    # 绘制压缩轨迹
    # 先计算用于投影的平均纬度，把经纬转换为米（近似投影）
    all_lats = []
    all_lons = []
    for traj_df in [original_trajectory] + list(compressed_trajectories.values()):
        all_lats.extend(traj_df['LAT'].tolist())
        all_lons.extend(traj_df['LON'].tolist())

    if len(all_lats) == 0 or len(all_lons) == 0:
        mean_lat = 0.0
    else:
        mean_lat = float(np.mean(all_lats))
    cos_lat = np.cos(np.radians(mean_lat))
    meter_per_deg = 111000.0
    lon_scale = meter_per_deg * cos_lat
    lat_scale = meter_per_deg

    def project_lonlat(df):
        xs = (df['LON'].to_numpy(dtype=float)) * lon_scale
        ys = (df['LAT'].to_numpy(dtype=float)) * lat_scale
        return xs, ys

    # 全局范围（投影后）
    all_x = []
    all_y = []
    for traj_df in [original_trajectory] + list(compressed_trajectories.values()):
        xs, ys = project_lonlat(traj_df)
        all_x.extend(xs.tolist())
        all_y.extend(ys.tolist())

    x_min, x_max = (min(all_x), max(all_x)) if all_x else (0, 1)
    y_min, y_max = (min(all_y), max(all_y)) if all_y else (0, 1)
    x_margin = (x_max - x_min) * 0.05 if x_max != x_min else 1.0
    y_margin = (y_max - y_min) * 0.05 if y_max != y_min else 1.0

    for i, (name, df) in enumerate(compressed_trajectories.items()):
        ax = axes[i]
        ax.set_facecolor('white')

        # 投影并绘制压缩轨迹（米）
        xs, ys = project_lonlat(df)
        color = colors[i % len(colors)]
        ax.plot(xs, ys, 'o-', color=color, linewidth=2, markersize=4,
                label=f'{name} ({len(df)} 点)', alpha=0.8)

        # 设置标题和标签（以米为单位）
        ax.set_title(f'{name} 轨迹压缩结果', fontsize=12, fontweight='bold')
        ax.set_xlabel('Easting (m)', fontsize=10)
        ax.set_ylabel('Northing (m)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        # 设置坐标轴范围（投影后，带边距）
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_aspect('equal', adjustable='box')

    # 绘制底部原始轨迹
    # 绘制底部原始轨迹（投影到米坐标）
    original_ax.set_facecolor('white')
    orig_xs = (original_trajectory['LON'].to_numpy(dtype=float)) * lon_scale
    orig_ys = (original_trajectory['LAT'].to_numpy(dtype=float)) * lat_scale
    original_ax.plot(orig_xs, orig_ys, 'o-',
                    color='red', linewidth=2, markersize=3, alpha=0.7,
                    label=f'原始轨迹 ({len(original_trajectory)} 点)')

    # 标记起点和终点（使用投影坐标）
    if len(orig_xs) > 0 and len(orig_ys) > 0:
        original_ax.plot(orig_xs[0], orig_ys[0], 'go', markersize=8, label='起点')
        original_ax.plot(orig_xs[-1], orig_ys[-1], 'ro', markersize=8, label='终点')

    original_ax.set_title('原始轨迹', fontsize=12, fontweight='bold')
    original_ax.set_xlabel('Easting (m)', fontsize=10)
    original_ax.set_ylabel('Northing (m)', fontsize=10)
    original_ax.grid(True, alpha=0.3)
    original_ax.legend(loc='upper right')

    # 设置原始轨迹的坐标轴范围（投影后）
    original_ax.set_xlim(x_min - x_margin, x_max + x_margin)
    original_ax.set_ylim(y_min - y_margin, y_max + y_margin)
    original_ax.set_aspect('equal', adjustable='box')

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"轨迹可视化已保存至: {output_file}")