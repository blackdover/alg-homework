import pandas as pd
from typing import Dict


def compress(pts: pd.DataFrame, p: Dict) -> pd.DataFrame:
    from ..utils.geo_utils import GeoUtils

    df = pts
    epsilon = p.get('epsilon', 100.0)
    now = [0]
    last = 0
    for i in range(1, len(df)):
        lastpoint = df.iloc[last]
        currentpoint = df.iloc[i]
        chang_t = (currentpoint['BaseDateTime'] - lastpoint['BaseDateTime']).total_seconds()
        if chang_t <= 0:
            continue
        speedknots = lastpoint['SOG']
        coursedegr = lastpoint['COG']
        speedmps = GeoUtils.knots_to_mps(speedknots)
        next_lat, next_lon = GeoUtils.predict_position(
            lat_old=lastpoint['LAT'],
            lon_old=lastpoint['LON'],
            speed_mps=speedmps,
            course_deg=coursedegr,
            delta_t=chang_t
        )
        error = GeoUtils.haversine_distance(
            lat1=currentpoint['LAT'],
            lon1=currentpoint['LON'],
            lat2=next_lat,
            lon2=next_lon
        )
        if error >= epsilon:
            now.append(i)
            last = i
    if now[-1] != len(df) - 1:
        now.append(len(df) - 1)

    return df.iloc[now].reset_index(drop=False).rename(columns={"index": "orig_idx"})

DISPLAY_NAME = "固定阈值 DR"
DEFAULT_PARAMS = {'epsilon': 100.0}
PARAM_HELP = {'epsilon': '距离阈值（米）'}
