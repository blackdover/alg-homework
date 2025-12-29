import pandas as pd
from typing import Dict


def compress(pts: pd.DataFrame, p: Dict) -> pd.DataFrame:
    from ..utils.geo_utils import GeoUtils

    df = pts
    epsilon = p.get('epsilon', 100.0)
    compressedindices = [0]

    i = 1
    while i < len(df):
        anchoridx = compressedindices[-1]

        left, right = i, len(df) - 1
        farthestvalid = i - 1

        while left <= right:
            mid = (left + right) // 2

            valid = True
            for k in range(i, mid + 1):
                error = GeoUtils.point_to_line_distance(
                    df.iloc[k]['LAT'], df.iloc[k]['LON'],
                    df.iloc[anchoridx]['LAT'], df.iloc[anchoridx]['LON'],
                    df.iloc[mid]['LAT'], df.iloc[mid]['LON']
                )
                if error > epsilon:
                    valid = False
                    break

            if valid:
                farthestvalid = mid
                left = mid + 1
            else:
                right = mid - 1

        if farthestvalid >= i:
            compressedindices.append(farthestvalid)
            i = farthestvalid + 1
        else:
            compressedindices.append(i)
            i += 1

    if compressedindices[-1] != len(df) - 1:
        compressedindices.append(len(df) - 1)

    return df.iloc[compressedindices].reset_index(drop=False).rename(columns={"index": "orig_idx"})


DISPLAY_NAME = "滑动窗口算法"
DEFAULT_PARAMS = {'epsilon': 100.0}
PARAM_HELP = {'epsilon': '距离阈值（米）'}
