import pandas as pd
from typing import Dict

def compress(pts: pd.DataFrame, p: Dict) -> pd.DataFrame:
    from ..utils.geo_utils import GeoUtils

    df = pts
    compressedindices = [0]
    lastindex = 0

    for i in range(1, len(df)):
        lastpoint = df.iloc[lastindex]
        currentpoint = df.iloc[i]

        deltat = (currentpoint['BaseDateTime'] - lastpoint['BaseDateTime']).total_seconds()
        if deltat <= 0:
            continue

        speedknots = lastpoint['SOG']
        coursedegr = lastpoint['COG']
        speedmps = GeoUtils.knots_to_mps(speedknots)

        predlat, predlon = GeoUtils.predict_position(
            lat_old=lastpoint['LAT'],
            lon_old=lastpoint['LON'],
            speed_mps=speedmps,
            course_deg=coursedegr,
            delta_t=deltat
        )

        error = GeoUtils.haversine_distance(
            lat1=currentpoint['LAT'],
            lon1=currentpoint['LON'],
            lat2=predlat,
            lon2=predlon
        )

        epsilon_min = p.get('min_threshold', 20.0)
        epsilon_max = p.get('max_threshold', 500.0)
        v_lower = p.get('v_lower', 3.0)
        v_upper = p.get('v_upper', 20.0)
        if speedknots <= v_lower:
            threshold = epsilon_min
        elif speedknots >= v_upper:
            threshold = epsilon_max
        else:
            k = (epsilon_max - epsilon_min) / (v_upper - v_lower)
            threshold = k * (speedknots - v_lower) + epsilon_min

        if error >= threshold:
            compressedindices.append(i)
            lastindex = i

    if compressedindices[-1] != len(df) - 1:
        compressedindices.append(len(df) - 1)

    return df.iloc[compressedindices].reset_index(drop=False).rename(columns={"index": "orig_idx"})


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
