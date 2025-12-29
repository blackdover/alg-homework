import pandas as pd
from typing import Dict, List


def _point_line_distance_m(lat, lon, lat1, lon1, lat2, lon2):
    from ..utils.geo_utils import GeoUtils
    return GeoUtils.point_to_line_distance(lat, lon, lat1, lon1, lat2, lon2)


def _rdp_indices(df: pd.DataFrame, epsilon: float) -> List[int]:
    indices: List[int] = []

    def recurse(start: int, end: int):
        if end <= start + 1:
            return

        lat1 = df.iloc[start]['LAT']
        lon1 = df.iloc[start]['LON']
        lat2 = df.iloc[end]['LAT']
        lon2 = df.iloc[end]['LON']

        maxdist = -1.0
        maxidx = -1
        for i in range(start + 1, end):
            lat = df.iloc[i]['LAT']
            lon = df.iloc[i]['LON']
            dist = _point_line_distance_m(lat, lon, lat1, lon1, lat2, lon2)
            if dist > maxdist:
                maxdist = dist
                maxidx = i

        if maxdist > epsilon and maxidx != -1:
            recurse(start, maxidx)
            indices.append(maxidx)
            recurse(maxidx, end)

    startidx = 0
    endidx = len(df) - 1
    indices = [startidx]
    recurse(startidx, endidx)
    if endidx not in indices:
        indices.append(endidx)
    indices = sorted(list(set(indices)))
    return indices


def compress(pts: pd.DataFrame, p: Dict) -> pd.DataFrame:
    df = pts
    if len(df) <= 2:
        return df.copy()

    if 'epsilon_m' in p:
        epsilon = float(p['epsilon_m'])
    else:
        eps_deg = float(p.get('epsilon', 0.0009))
        epsilon = eps_deg * 111000.0

    indices = _rdp_indices(df, epsilon)
    return df.iloc[indices].reset_index(drop=False).rename(columns={"index": "orig_idx"})



DISPLAY_NAME = "DP算法"
DEFAULT_PARAMS = {'epsilon': 0.0009}
PARAM_HELP = {'epsilon': '距离阈值（度）'}
