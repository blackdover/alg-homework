import pandas as pd
from typing import Dict

def compress(pts: pd.DataFrame, p: Dict) -> pd.DataFrame:
    from ..utils.geo_utils import GeoUtils

    df = pts
    now = [0]
    last = 0

    for i in range(1, len(df)):
        lastpoint = df.iloc[last]
        currentpoint = df.iloc[i]
        chang_t = (currentpoint['BaseDateTime'] - lastpoint['BaseDateTime']).total_seconds()
        if chang_t <= 0:
            continue
        speed = lastpoint['SOG']
        fangxiang = lastpoint['COG']
        speed2 = GeoUtils.knots_to_mps(speed)

        next_lat, next_lon = GeoUtils.predict_position(
            lat_old=lastpoint['LAT'],
            lon_old=lastpoint['LON'],
            speed_mps=speed2,
            course_deg=fangxiang,
            delta_t=chang_t
        )

        error = GeoUtils.distance_betweeen(
            lat1=currentpoint['LAT'],
            lon1=currentpoint['LON'],
            lat2=next_lat,
            lon2=next_lon
        )

        epsilon_min = p.get('min_threshold', 20.0)
        epsilon_max = p.get('max_threshold', 500.0)
        v_lower = p.get('v_lower', 3.0)
        v_upper = p.get('v_upper', 20.0)
        if speed <= v_lower:
            epsilon = epsilon_min
        elif speed >= v_upper:
            epsilon = epsilon_max
        else:
            k = (epsilon_max - epsilon_min) / (v_upper - v_lower)
            epsilon = k * (speed - v_lower) + epsilon_min

        if error >= epsilon:
            now.append(i)
            last = i

    if now[-1] != len(df) - 1:
        now.append(len(df) - 1)
    return df.iloc[now].reset_index(drop=False).rename(columns={"index": "orig_idx"})

DISPLAY_NAME = "自适应阈值 DR"
DEFAULT_PARAMS = {
    'min_threshold': 20.0,
    'max_threshold': 500.0,
    'v_lower': 3.0,
    'v_upper': 20.0
}
PARAM_HELP = {
    'min_threshold': '最低距离阈值（米）',
    'max_threshold': '最高距离阈值（米）',
    'v_lower': '低速截止点（节）',
    'v_upper': '高速截止点（节）'
}
