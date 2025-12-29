import pandas as pd
from typing import Dict


def compress(pts: pd.DataFrame, p: Dict) -> pd.DataFrame:
    from ..utils.geo_utils import GeoUtils

    df = pts
    epsilon = p.get('epsilon', 100.0)
    now = []
    i = 0
    while i < len(df) - 1:
        aim = i
        best_end = i + 1
        left, right = i + 1, len(df) - 1

        while left <= right:
            mid = (left + right) // 2
            valid = True
            for k in range(aim + 1, mid):
                error = GeoUtils.point_to_line_distance(
                    df.iloc[k]['LAT'], df.iloc[k]['LON'],
                    df.iloc[aim]['LAT'], df.iloc[aim]['LON'],
                    df.iloc[mid]['LAT'], df.iloc[mid]['LON']
                )
                if error > epsilon:
                    valid = False
                    break
            if valid:
                best_end = mid
                left = mid + 1
            else:
                right = mid - 1
        now.append(aim)
        i = best_end
        if i <= aim:
            i = aim + 1
    if not now or now[-1] != len(df) - 1:
        now.append(len(df) - 1)
    return df.iloc[now].reset_index(drop=False).rename(columns={"index": "orig_idx"})

DISPLAY_NAME = "开窗算法"
DEFAULT_PARAMS = {'epsilon': 100.0}
PARAM_HELP = {'epsilon': '距离阈值（米）'}
