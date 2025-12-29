import pandas as pd
from typing import Dict


def compress(points: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Sliding Window 轨迹压缩算法

    以最后一个保留点为起点，不断往前看新点；如果新点相对起点→新点这条线的偏差超过阈值，
    就把上一个点保留下来当新起点。

    参数:
        points: 输入轨迹DataFrame
        params: 参数字典，包含：
            - epsilon: 距离阈值（米）
    """
    from ..utils.geo_utils import GeoUtils

    df = points  # 重命名以保持兼容性

    if len(df) <= 2:
        return df.copy()

    epsilon = params.get('epsilon', 100.0)
    compressed_indices = [0]  # 起始点

    i = 1
    while i < len(df):
        # 从当前锚点开始，寻找最远的点使得所有中间点都在容差内
        anchor_idx = compressed_indices[-1]

        # 使用二分查找优化：找到第一个超出误差的点
        left, right = i, len(df) - 1
        farthest_valid = i - 1  # 初始化为无效值

        while left <= right:
            mid = (left + right) // 2

            # 检查从i到mid的所有点是否都在anchor->mid线的容差内
            valid = True
            for k in range(i, mid + 1):
                error = GeoUtils.point_to_line_distance(
                    df.iloc[k]['LAT'], df.iloc[k]['LON'],
                    df.iloc[anchor_idx]['LAT'], df.iloc[anchor_idx]['LON'],
                    df.iloc[mid]['LAT'], df.iloc[mid]['LON']
                )
                if error > epsilon:
                    valid = False
                    break

            if valid:
                # mid点有效，可以尝试更远的点
                farthest_valid = mid
                left = mid + 1
            else:
                # mid点无效，尝试更近的点
                right = mid - 1

        # 确定下一个保留点
        if farthest_valid >= i:
            # 找到了有效的远点
            compressed_indices.append(farthest_valid)
            i = farthest_valid + 1
        else:
            # 连当前点都无效，直接保留当前点
            compressed_indices.append(i)
            i += 1

    # 确保包含最后一个点
    if compressed_indices[-1] != len(df) - 1:
        compressed_indices.append(len(df) - 1)

    return df.iloc[compressed_indices].reset_index(drop=False).rename(columns={"index": "orig_idx"})


# 算法元数据
DISPLAY_NAME = "Sliding Window"
DEFAULT_PARAMS = {'epsilon': 100.0}
PARAM_HELP = {'epsilon': '距离阈值（米）'}
