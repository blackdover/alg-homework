import pandas as pd
from typing import Dict


def compress(pts: pd.DataFrame, p: Dict) -> pd.DataFrame:
    from ..utils.geo_utils import GeoUtils

    df = pts
    epsilon = p.get('epsilon', 100.0)
    compressedindices = []

    i = 0
    while i < len(df) - 1:
        anchoridx = i
        
        bestend = i + 1

        left, right = i + 1, len(df) - 1
        

        while left <= right:
            mid = (left + right) // 2

            valid = True
            for k in range(anchoridx + 1, mid):
                error = GeoUtils.point_to_line_distance(
                    df.iloc[k]['LAT'], df.iloc[k]['LON'],
                    df.iloc[anchoridx]['LAT'], df.iloc[anchoridx]['LON'],
                    df.iloc[mid]['LAT'], df.iloc[mid]['LON']
                )
                if error > epsilon:
                    valid = False
                    break

            if valid:
                bestend = mid
                left = mid + 1
            else:
                right = mid - 1


        compressedindices.append(anchoridx)

        i = bestend

        if i <= anchoridx:
            i = anchoridx + 1

    if not compressedindices or compressedindices[-1] != len(df) - 1:
        compressedindices.append(len(df) - 1)

    return df.iloc[compressedindices].reset_index(drop=False).rename(columns={"index": "orig_idx"})


DISPLAY_NAME = "开窗算法"
DEFAULT_PARAMS = {'epsilon': 100.0}
PARAM_HELP = {'epsilon': '距离阈值（米）'}
