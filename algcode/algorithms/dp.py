import pandas as pd
from typing import Dict, List

def compress(pts: pd.DataFrame, p: Dict) -> pd.DataFrame:
    from ..utils.geo_utils import GeoUtils
    df = pts
    eps_deg = float(p.get('epsilon', 0.0009))
    epsilon = eps_deg * 111000.0

    now: List[int] = []
    def recurse(start: int, end: int):
        if end <= start + 1:
            return

        lat1 = df.iloc[start]['LAT']
        lon1 = df.iloc[start]['LON']
        lat2 = df.iloc[end]['LAT']
        lon2 = df.iloc[end]['LON']
        
        maxdist = -1.0
        maxi = -1
        for i in range(start + 1, end):
            lat = df.iloc[i]['LAT']
            lon = df.iloc[i]['LON']
            dist = GeoUtils.point_to_line_distance(lat, lon, lat1, lon1, lat2, lon2)
            if dist > maxdist:
                maxdist = dist
                maxi = i

        if maxdist > epsilon and maxi != -1:
            recurse(start, maxi)
            now.append(maxi)
            recurse(maxi, end)

    start = 0
    end = len(df) - 1
    now = [start]
    recurse(start, end)
    if end not in now:
        now.append(end)
    now = sorted(list(set(now)))
    return df.iloc[now].reset_index(drop=False).rename(columns={"index": "orig_idx"})



DISPLAY_NAME = "DP算法"
DEFAULT_PARAMS = {'epsilon': 0.0009}
PARAM_HELP = {'epsilon': '距离阈值（度）'}
