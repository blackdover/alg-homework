import pandas as pd  # 导入pandas数据处理库，用于处理轨迹数据框
import matplotlib.pyplot as plt  # 导入matplotlib绘图库，用于创建可视化图表
import matplotlib.gridspec as gridspec  # 导入子图网格布局模块
from typing import Optional, Dict, List  # 导入类型注解，用于函数参数类型提示
import numpy as np  # 导入numpy数值计算库，用于数组操作


def visualize_trajectories(original_df: pd.DataFrame,  # 可视化轨迹对比的函数（简化版本）
                          dr_df: pd.DataFrame,  # DR压缩后的轨迹数据框
                          dp_df: Optional[pd.DataFrame] = None,  # DP压缩后的轨迹数据框（可选）
                          output_file: str = "visualization.png") -> None:  # 输出文件路径，默认visualization.png
    # 构建轨迹字典，至少包含原始轨迹和DR压缩轨迹
    trajectories = {  # 创建轨迹数据字典
        '原始轨迹': original_df,  # 原始轨迹数据
        'DR压缩': dr_df  # DR算法压缩后的轨迹
    }
    if dp_df is not None:  # 如果提供了DP压缩结果
        trajectories['DP压缩'] = dp_df  # 添加DP压缩轨迹到字典
    visualize_multiple_trajectories(trajectories, output_file)  # 调用通用可视化函数


def visualize_multiple_trajectories(trajectories: Dict[str, pd.DataFrame],  # 多轨迹可视化函数
                                  output_file: str = "visualization.png") -> None:  # 输出文件路径
    """使用matplotlib创建多子图轨迹可视化，白底布局，底部显示原始轨迹作为对比基准"""
    if not trajectories:  # 如果轨迹字典为空
        raise ValueError("至少需要一个轨迹")  # 抛出错误，至少需要一个轨迹

    # 分离原始轨迹和其他压缩轨迹，以便分别处理
    original_trajectory = None  # 初始化原始轨迹变量
    compressed_trajectories = {}  # 初始化压缩轨迹字典

    for name, df in trajectories.items():  # 遍历所有提供的轨迹
        if len(df) == 0:  # 如果轨迹数据为空
            continue  # 跳过空轨迹
        if name == '原始轨迹':  # 如果是原始轨迹
            original_trajectory = df  # 保存原始轨迹
        else:  # 如果是压缩轨迹
            compressed_trajectories[name] = df  # 添加到压缩轨迹字典

    if original_trajectory is None:  # 如果没有找到原始轨迹
        raise ValueError("必须包含原始轨迹")  # 抛出错误，必须要有原始轨迹作为对比基准

    # 设置matplotlib的中文字体支持，确保中文标签正常显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 设置中文字体列表
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 创建子图布局：按 2 列排列压缩轨迹，上方为压缩结果，底部为原始轨迹（跨两列）
    n_compressed = len(compressed_trajectories)  # 获取压缩轨迹的数量
    if n_compressed == 0:  # 如果没有压缩轨迹，只有原始轨迹
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))  # 创建单个子图
        axes = [ax]  # 轴列表只包含一个轴
        original_ax = ax  # 原始轨迹也使用同一个轴
    else:  # 如果有压缩轨迹
        cols = 2  # 固定2列布局
        rows = int(np.ceil(n_compressed / cols))  # 根据轨迹数量计算需要的行数
        rows = min(rows, 3)  # 限制为最多 3 行（2x3 布局）
        # 总高度：每行约 3.0，高度再加一个原始轨迹区域（3.0）
        fig_height = rows * 3.0 + 3.0  # 计算图表总高度
        fig_width = 12  # 固定图表宽度
        fig = plt.figure(figsize=(fig_width, fig_height))  # 创建指定大小的图表
        gs = gridspec.GridSpec(rows + 1, cols, height_ratios=[1] * rows + [0.9])  # 创建网格布局

        axes = []  # 初始化压缩轨迹的轴列表
        for i in range(n_compressed):  # 为每个压缩轨迹创建子图
            r = i // cols  # 计算行索引
            ccol = i % cols  # 计算列索引
            ax = fig.add_subplot(gs[r, ccol])  # 在网格中添加子图
            axes.append(ax)  # 添加到轴列表

        # 底部原始轨迹跨两列
        original_ax = fig.add_subplot(gs[-1, :])  # 在最后一行跨所有列创建原始轨迹子图

    # 颜色设置 - 为不同的轨迹分配不同颜色
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']  # 预定义颜色列表

    # 绘制压缩轨迹
    # 先计算用于投影的平均纬度，把经纬度坐标转换为米坐标（近似UTM投影）
    all_lats = []  # 收集所有轨迹的纬度值
    all_lons = []  # 收集所有轨迹的经度值
    for traj_df in [original_trajectory] + list(compressed_trajectories.values()):  # 遍历所有轨迹
        all_lats.extend(traj_df['LAT'].tolist())  # 添加当前轨迹的所有纬度
        all_lons.extend(traj_df['LON'].tolist())  # 添加当前轨迹的所有经度

    if len(all_lats) == 0 or len(all_lons) == 0:  # 如果没有坐标数据
        mean_lat = 0.0  # 默认平均纬度为0
    else:  # 如果有坐标数据
        mean_lat = float(np.mean(all_lats))  # 计算所有轨迹的平均纬度
    cos_lat = np.cos(np.radians(mean_lat))  # 计算平均纬度的余弦值
    meter_per_deg = 111000.0  # 地球上1度对应的距离（米）
    lon_scale = meter_per_deg * cos_lat  # 经度每度的距离（米），考虑纬度影响
    lat_scale = meter_per_deg  # 纬度每度的距离（米）

    def project_lonlat(df):  # 定义坐标投影函数，将经纬度转换为米坐标
        xs = (df['LON'].to_numpy(dtype=float)) * lon_scale  # 经度转换为米坐标
        ys = (df['LAT'].to_numpy(dtype=float)) * lat_scale  # 纬度转换为米坐标
        return xs, ys  # 返回x、y坐标数组

    # 计算全局范围（投影后的坐标），用于设置统一的坐标轴范围
    all_x = []  # 收集所有x坐标
    all_y = []  # 收集所有y坐标
    for traj_df in [original_trajectory] + list(compressed_trajectories.values()):  # 遍历所有轨迹
        xs, ys = project_lonlat(traj_df)  # 投影当前轨迹
        all_x.extend(xs.tolist())  # 添加x坐标
        all_y.extend(ys.tolist())  # 添加y坐标

    x_min, x_max = (min(all_x), max(all_x)) if all_x else (0, 1)  # 计算x坐标范围
    y_min, y_max = (min(all_y), max(all_y)) if all_y else (0, 1)  # 计算y坐标范围
    x_margin = (x_max - x_min) * 0.05 if x_max != x_min else 1.0  # 计算x方向边距
    y_margin = (y_max - y_min) * 0.05 if y_max != y_min else 1.0  # 计算y方向边距

    for i, (name, df) in enumerate(compressed_trajectories.items()):  # 遍历每个压缩轨迹
        ax = axes[i]  # 获取对应的子图轴
        ax.set_facecolor('white')  # 设置子图背景为白色

        # 投影并绘制压缩轨迹（坐标单位为米）
        xs, ys = project_lonlat(df)  # 将轨迹坐标投影到米坐标系
        color = colors[i % len(colors)]  # 为当前轨迹选择颜色
        ax.plot(xs, ys, 'o-', color=color, linewidth=2, markersize=4,  # 绘制轨迹线和点
                label=f'{name} ({len(df)} 点)', alpha=0.8)  # 设置标签显示轨迹名和点数

        # 设置标题和标签（以米为单位）
        ax.set_title(f'{name} 轨迹压缩结果', fontsize=12, fontweight='bold')  # 设置子图标题
        ax.set_xlabel('Easting (m)', fontsize=10)  # 设置x轴标签（东向距离，米）
        ax.set_ylabel('Northing (m)', fontsize=10)  # 设置y轴标签（北向距离，米）
        ax.grid(True, alpha=0.3)  # 显示网格线，透明度0.3
        ax.legend(loc='upper right')  # 在右上角显示图例

        # 设置坐标轴范围（投影后，带边距）
        ax.set_xlim(x_min - x_margin, x_max + x_margin)  # 设置x轴范围，包含边距
        ax.set_ylim(y_min - y_margin, y_max + y_margin)  # 设置y轴范围，包含边距
        ax.set_aspect('equal', adjustable='box')  # 设置等比例缩放

    # 绘制底部原始轨迹作为对比基准
    # 绘制底部原始轨迹（投影到米坐标系）
    original_ax.set_facecolor('white')  # 设置原始轨迹子图背景为白色
    orig_xs = (original_trajectory['LON'].to_numpy(dtype=float)) * lon_scale  # 投影原始轨迹经度
    orig_ys = (original_trajectory['LAT'].to_numpy(dtype=float)) * lat_scale  # 投影原始轨迹纬度
    original_ax.plot(orig_xs, orig_ys, 'o-',  # 绘制原始轨迹线和点
                    color='red', linewidth=2, markersize=3, alpha=0.7,  # 设置红色，半透明
                    label=f'原始轨迹 ({len(original_trajectory)} 点)')  # 设置标签显示点数

    # 标记起点和终点（使用投影坐标）
    if len(orig_xs) > 0 and len(orig_ys) > 0:  # 如果原始轨迹有数据点
        original_ax.plot(orig_xs[0], orig_ys[0], 'go', markersize=8, label='起点')  # 绿色圆点标记起点
        original_ax.plot(orig_xs[-1], orig_ys[-1], 'ro', markersize=8, label='终点')  # 红色圆点标记终点

    original_ax.set_title('原始轨迹', fontsize=12, fontweight='bold')  # 设置原始轨迹子图标题
    original_ax.set_xlabel('Easting (m)', fontsize=10)  # 设置x轴标签
    original_ax.set_ylabel('Northing (m)', fontsize=10)  # 设置y轴标签
    original_ax.grid(True, alpha=0.3)  # 显示网格
    original_ax.legend(loc='upper right')  # 显示图例

    # 设置原始轨迹的坐标轴范围（投影后，带边距）
    original_ax.set_xlim(x_min - x_margin, x_max + x_margin)  # 设置x轴范围
    original_ax.set_ylim(y_min - y_margin, y_max + y_margin)  # 设置y轴范围
    original_ax.set_aspect('equal', adjustable='box')  # 设置等比例缩放

    # 调整布局，确保子图之间的间距合理
    plt.tight_layout()

    # 保存图像到指定文件
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')  # 高分辨率保存
    plt.close()  # 关闭图表，释放内存

    print(f"轨迹可视化已保存至: {output_file}")