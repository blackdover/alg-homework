import pandas as pd
from typing import Optional, Dict
import folium
from folium import plugins


def visualize_trajectories(original_df: pd.DataFrame,
                          dr_df: pd.DataFrame,
                          dp_df: Optional[pd.DataFrame] = None,
                          output_file: str = "map.html") -> None:
    trajectories = {
        '原始轨迹': original_df,
        'DR压缩': dr_df
    }
    if dp_df is not None:
        trajectories['DP压缩'] = dp_df
    visualize_multiple_trajectories(trajectories, output_file)


def visualize_multiple_trajectories(trajectories: Dict[str, pd.DataFrame],
                                  output_file: str = "map.html") -> None:
    if not trajectories:
        raise ValueError("至少需要一个轨迹")
    first_trajectory = next(iter(trajectories.values()))
    if len(first_trajectory) == 0:
        raise ValueError("轨迹不能为空")
    center_lat = first_trajectory.iloc[0]['LAT']
    center_lon = first_trajectory.iloc[0]['LON']
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'black']
    for i, (name, df) in enumerate(trajectories.items()):
        if len(df) == 0:
            continue
        color = colors[i % len(colors)]
        fg = folium.FeatureGroup(name=f"{name} ({len(df)} 点)")
        coords = [[row['LAT'], row['LON']] for _, row in df.iterrows()]
        folium.PolyLine(
            coords,
            color=color,
            weight=3 if name != '原始轨迹' else 2,
            opacity=0.8 if name != '原始轨迹' else 0.5,
            popup=f'{name} ({len(df)} 点)'
        ).add_to(fg)
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
        fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # 添加自定义CSS修改图层控制字体
    custom_css = """
    <style>
    .leaflet-control-layers {
        font-family: 'Microsoft YaHei', '微软雅黑', 'SimHei', '黑体', Arial, sans-serif !important;
        font-size: 25px !important;
        font-weight: normal !important;
    }
    .leaflet-control-layers label {
        font-family: 'Microsoft YaHei', '微软雅黑', 'SimHei', '黑体', Arial, sans-serif !important;
        font-size: 25px !important;
        line-height: 1.4 !important;
    }
    .leaflet-control-layers input[type="checkbox"],
    .leaflet-control-layers input[type="radio"] {
        margin-right: 5px !important;
    }
    </style>
    """

    # 将CSS添加到地图的HTML中
    m.get_root().html.add_child(folium.Element(custom_css))

    m.save(output_file)
    print(f"可视化地图已保存至: {output_file}")


def _add_layer_controls(m: folium.Map, trajectories: Dict[str, pd.DataFrame]) -> None:
    return


