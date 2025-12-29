import pandas as pd
from typing import Dict


def compress(points: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Opening Window 轨迹压缩算法

    从 anchor 点开始开窗，尽可能延长窗口右端；只要窗口内所有点对 anchor→窗口末端的误差都 ≤ ε，
    就继续扩张；一旦出现超阈值点，就输出上一个窗口末端并重开窗口。

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
    compressed_indices = []

    i = 0
    while i < len(df) - 1:
        # 从当前点开始开窗
        anchor_idx = i
        window_end = i + 1

        # 尽可能延长窗口 - 使用二分查找优化
        left, right = i + 1, len(df) - 1
        best_end = i + 1  # 最小的有效窗口

        while left <= right:
            mid = (left + right) // 2

            # 检查从anchor到mid的所有点是否都在误差范围内
            valid = True
            for k in range(anchor_idx + 1, mid):
                error = GeoUtils.point_to_line_distance(
                    df.iloc[k]['LAT'], df.iloc[k]['LON'],
                    df.iloc[anchor_idx]['LAT'], df.iloc[anchor_idx]['LON'],
                    df.iloc[mid]['LAT'], df.iloc[mid]['LON']
                )
                if error > epsilon:
                    valid = False
                    break

            if valid:
                # 可以扩展更远
                best_end = mid
                left = mid + 1
            else:
                # 不能扩展到mid
                right = mid - 1

        window_end = best_end

        # 保留 anchor 点
        compressed_indices.append(anchor_idx)

        # 移动到窗口末端
        i = window_end

        # 安全检查：防止无限循环
        if i <= anchor_idx:
            # 如果i没有前进，强制前进到下一个点
            i = anchor_idx + 1

    # 确保包含最后一个点
    if not compressed_indices or compressed_indices[-1] != len(df) - 1:
        compressed_indices.append(len(df) - 1)

    return df.iloc[compressed_indices].reset_index(drop=False).rename(columns={"index": "orig_idx"})


# 算法元数据
DISPLAY_NAME = "Opening Window"
DEFAULT_PARAMS = {'epsilon': 100.0}
PARAM_HELP = {'epsilon': '距离阈值（米）'}
