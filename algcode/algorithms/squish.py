"""
SQUISH 流式轨迹简化算法
"""

import pandas as pd
import heapq
from typing import Dict, List


def compress(points: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    SQUISH 流式轨迹简化算法

    维护一个大小为 B 的点集，每来一个新点就加入；如果超出 B，就删掉最不重要的点
    （重要性用该点在相邻两点形成的线段上的误差衡量），并更新邻居的分数。

    参数:
        points: 输入轨迹DataFrame
        params: 参数字典，包含：
            - buffer_size: 缓冲区大小 B
    """
    from ..geo_utils import GeoUtils

    df = points  # 重命名以保持兼容性

    if len(df) <= 2:
        return df.copy()

    buffer_size = params.get('buffer_size', 100)

    class PointInfo:
        def __init__(self, idx: int, importance: float = 0.0):
            self.idx = idx
            self.importance = importance

        def __lt__(self, other):
            return self.importance < other.importance

    # 初始化缓冲区：前两个点
    buffer = [PointInfo(0, float('inf')), PointInfo(1, float('inf'))]
    heapq.heapify(buffer)

    # 计算重要性的辅助函数
    def update_importance(buffer_list: List[PointInfo]) -> None:
        if len(buffer_list) < 3:
            return

        # 按索引排序
        sorted_buffer = sorted(buffer_list, key=lambda x: x.idx)

        for i in range(1, len(sorted_buffer) - 1):
            curr = sorted_buffer[i]
            prev = sorted_buffer[i-1]
            next_p = sorted_buffer[i+1]

            # 计算当前点对 prev->next 线段的误差
            error = GeoUtils.point_to_line_distance(
                df.iloc[curr.idx]['LAT'], df.iloc[curr.idx]['LON'],
                df.iloc[prev.idx]['LAT'], df.iloc[prev.idx]['LON'],
                df.iloc[next_p.idx]['LAT'], df.iloc[next_p.idx]['LON']
            )
            curr.importance = error

    # 处理后续点
    for i in range(2, len(df)):
        # 添加新点
        new_point = PointInfo(i, 0.0)
        buffer.append(new_point)

        # 更新所有点的权重
        update_importance(buffer)

        # 如果超出缓冲区大小，移除最不重要的点
        while len(buffer) > buffer_size:
            # 重新建堆（因为重要性已更新）
            heapq.heapify(buffer)
            removed = heapq.heappop(buffer)

            # 从缓冲区移除该点
            buffer = [p for p in buffer if p.idx != removed.idx]

    # 返回缓冲区中的所有点，按原始顺序排序
    result_indices = sorted([p.idx for p in buffer])
    return df.iloc[result_indices].reset_index(drop=True)


# 算法元数据
DISPLAY_NAME = "SQUISH"
DEFAULT_PARAMS = {'buffer_size': 100}
PARAM_HELP = {'buffer_size': '缓冲区大小'}
