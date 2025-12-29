import pandas as pd
from typing import Dict


def compress(pts: pd.DataFrame, p: Dict) -> pd.DataFrame:
    from ..utils.geo_utils import GeoUtils

    df = pts
    threshold = p.get('epsilon', 100.0)

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

        if error >= threshold:
            compressedindices.append(i)
            lastindex = i

    if compressedindices[-1] != len(df) - 1:
        compressedindices.append(len(df) - 1)

    return df.iloc[compressedindices].reset_index(drop=False).rename(columns={"index": "orig_idx"})

DISPLAY_NAME = "固定阈值 DR"
DEFAULT_PARAMS = {'epsilon': 100.0}
PARAM_HELP = {'epsilon': '距离阈值（米）'}
