"""
轨迹压缩可视化模块
包含轨迹可视化功能

作者: Algorithm Engineer
Python版本: 3.10
"""

import pandas as pd
from typing import Optional, Dict
import folium
from folium import plugins


def visualize_trajectories(original_df: pd.DataFrame,
                          dr_df: pd.DataFrame,
                          dp_df: Optional[pd.DataFrame] = None,
                          output_file: str = "map.html") -> None:
    """
    使用folium库可视化轨迹压缩结果（向后兼容接口）

    参数:
        original_df: 原始轨迹DataFrame
        dr_df: DR算法压缩后的轨迹DataFrame
        dp_df: DP算法压缩后的轨迹DataFrame（可选）
        output_file: 输出HTML文件路径
    """
    # 转换为新的多轨迹接口
    trajectories = {
        '原始轨迹': original_df,
        'DR压缩': dr_df
    }
    if dp_df is not None:
        trajectories['DP压缩'] = dp_df

    visualize_multiple_trajectories(trajectories, output_file)


def visualize_multiple_trajectories(trajectories: Dict[str, pd.DataFrame],
                                  output_file: str = "map.html") -> None:
    """
    使用folium库可视化多个轨迹，支持图层切换

    参数:
        trajectories: 轨迹字典，键为轨迹名称，值为DataFrame
        output_file: 输出HTML文件路径
    """
    if not trajectories:
        raise ValueError("至少需要一个轨迹")

    # 使用第一个轨迹确定地图中心
    first_trajectory = next(iter(trajectories.values()))
    if len(first_trajectory) == 0:
        raise ValueError("轨迹不能为空")

    center_lat = first_trajectory.iloc[0]['LAT']
    center_lon = first_trajectory.iloc[0]['LON']
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    # 定义颜色方案
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'black']

    # 为每条轨迹创建独立的 FeatureGroup，这样 folium 的 LayerControl 可以正确管理图层显示
    for i, (name, df) in enumerate(trajectories.items()):
        if len(df) == 0:
            continue

        color = colors[i % len(colors)]

        # 创建轨迹图层组，名称包含点数，方便 LayerControl 显示
        fg = folium.FeatureGroup(name=f"{name} ({len(df)} 点)")

        # 创建轨迹线并加入该图层组
        coords = [[row['LAT'], row['LON']] for _, row in df.iterrows()]
        folium.PolyLine(
            coords,
            color=color,
            weight=3 if name != '原始轨迹' else 2,
            opacity=0.8 if name != '原始轨迹' else 0.5,
            popup=f'{name} ({len(df)} 点)'
        ).add_to(fg)

        # 对于压缩轨迹，添加点标记到同一图层组
        if name != '原始轨迹':
            for idx, row in df.iterrows():
                folium.CircleMarker(
                    location=[row['LAT'], row['LON']],
                    radius=4,
                    popup=f"{name} 点 {idx}",
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(fg)

        # 对于原始轨迹，添加起点/终点标记到原始图层组
        if name == '原始轨迹':
            original_df = df
            folium.Marker(
                location=[original_df.iloc[0]['LAT'], original_df.iloc[0]['LON']],
                popup='起点',
                icon=folium.Icon(color='green', icon='play')
            ).add_to(fg)

            folium.Marker(
                location=[original_df.iloc[-1]['LAT'], original_df.iloc[-1]['LON']],
                popup='终点',
                icon=folium.Icon(color='red', icon='stop')
            ).add_to(fg)

        # 将该轨迹图层组添加到地图
        fg.add_to(m)

    # 使用 folium 的 LayerControl 管理各轨迹图层的显示/隐藏
    folium.LayerControl(collapsed=False).add_to(m)

    # 保存地图
    m.save(output_file)
    print(f"可视化地图已保存至: {output_file}")


def _add_layer_controls(m: folium.Map, trajectories: Dict[str, pd.DataFrame]) -> None:
    # 旧的自定义图层控制已被替换为 folium.LayerControl，保留空函数以兼容其它调用（无操作）
    return
