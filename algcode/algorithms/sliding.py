import pandas as pd  # 导入pandas数据处理库，用于处理轨迹数据框
from typing import Dict  # 导入类型注解，用于函数参数类型提示


def compress(points: pd.DataFrame, params: Dict) -> pd.DataFrame:  # 主压缩函数，实现Sliding Window算法
    """
    Sliding Window轨迹压缩算法

    算法原理：
    以最后一个保留点为锚点，不断向前看新点；如果新点相对"锚点→新点"这条线的偏差超过阈值，
    就把上一个点保留下来作为新的锚点。

    这是一种增量式的在线算法，每次只考虑当前窗口内的点，逐步构建压缩结果。

    参数:
        points: 输入轨迹数据框，必须包含LAT、LON等地理坐标列
        params: 参数配置字典，包含：
            - epsilon: 距离阈值（米），点到线段的最大允许误差

    返回:
        压缩后的数据框，包含原始索引信息
    """
    from ..utils.geo_utils import GeoUtils  # 导入地理工具类，用于点到线段距离计算

    df = points  # 将输入数据框赋值给df，便于处理

    if len(df) <= 2:  # 如果轨迹点数不超过2个，无法有效压缩
        return df.copy()  # 直接返回原数据框的副本

    epsilon = params.get('epsilon', 100.0)  # 获取距离阈值参数，默认100米
    compressed_indices = [0]  # 初始化压缩结果索引列表，包含起始点作为第一个锚点

    i = 1  # 初始化当前位置索引，从第二个点开始处理
    while i < len(df):  # 当还有未处理的点时继续循环
        # 从当前锚点开始，寻找最远的点使得从i到该点的所有中间点都在误差容差内
        anchor_idx = compressed_indices[-1]  # 获取当前锚点（最后一个保留点）的索引

        # 使用二分查找优化算法：快速找到第一个超出误差容差的点位置
        left, right = i, len(df) - 1  # 二分查找的搜索范围：从当前位置到轨迹末尾
        farthest_valid = i - 1  # 初始化最远有效点为无效值（i-1表示没有找到有效点）

        while left <= right:  # 二分查找主循环
            mid = (left + right) // 2  # 计算中间位置作为候选点

            # 检查从当前位置i到中间点mid的所有点是否都在"锚点->中间点"这条线的误差容差内
            valid = True  # 假设当前候选点有效
            for k in range(i, mid + 1):  # 遍历从i到mid的所有点（包括端点）
                error = GeoUtils.point_to_line_distance(  # 计算点到线段的距离
                    df.iloc[k]['LAT'], df.iloc[k]['LON'],          # 当前检查的点坐标
                    df.iloc[anchor_idx]['LAT'], df.iloc[anchor_idx]['LON'],  # 锚点坐标
                    df.iloc[mid]['LAT'], df.iloc[mid]['LON']       # 中间点坐标
                )
                if error > epsilon:  # 如果任意点的误差超过了阈值
                    valid = False  # 标记候选点无效
                    break  # 跳出检查循环

            if valid:  # 如果中间点有效（所有点都在误差范围内）
                # 可以尝试扩展到更远的点
                farthest_valid = mid  # 更新最远有效点
                left = mid + 1  # 在右侧继续搜索更远的点
            else:  # 如果中间点无效
                # 需要在左侧搜索更近的点
                right = mid - 1  # 在左侧继续搜索

        # 根据二分查找结果确定下一个要保留的点
        if farthest_valid >= i:  # 如果找到了有效的远点（最远有效点有效）
            # 找到了有效的远点，将其作为下一个保留点
            compressed_indices.append(farthest_valid)  # 添加最远有效点到保留列表
            i = farthest_valid + 1  # 将当前位置跳到最远有效点的下一个点
        else:  # 如果连当前点都无效（farthest_valid < i）
            # 连当前点都无效，直接保留当前点作为新的锚点
            compressed_indices.append(i)  # 添加当前点到保留列表
            i += 1  # 移动到下一个点

    # 确保轨迹的最后一个点总是被保留（无论误差多大）
    if compressed_indices[-1] != len(df) - 1:  # 如果最后一个保留点不是轨迹末尾
        compressed_indices.append(len(df) - 1)  # 强制添加最后一个点

    # 返回压缩后的数据框，重置索引并保留原始索引信息用于评估
    return df.iloc[compressed_indices].reset_index(drop=False).rename(columns={"index": "orig_idx"})


# 算法元数据定义 - 用于GUI界面显示和参数配置
DISPLAY_NAME = "滑动窗口算法"  # 算法在界面中显示的名称
DEFAULT_PARAMS = {'epsilon': 100.0}  # 默认参数配置，距离阈值为100米
PARAM_HELP = {'epsilon': '距离阈值（米）'}  # 参数帮助说明，告诉用户epsilon参数的含义
