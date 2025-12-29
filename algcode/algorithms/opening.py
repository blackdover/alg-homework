import pandas as pd  # 导入pandas数据处理库，用于处理轨迹数据框
from typing import Dict  # 导入类型注解，用于函数参数类型提示


def compress(points: pd.DataFrame, params: Dict) -> pd.DataFrame:  # 主压缩函数，实现Opening Window算法
    """
    Opening Window轨迹压缩算法

    算法原理：
    从锚点开始开窗，尽可能延长窗口右端；只要窗口内所有点对"锚点→窗口末端"这条线的误差都≤ε，
    就继续扩张；一旦出现超阈值点，就输出上一个窗口末端作为保留点并重开窗口。

    这是一种贪婪的在线算法，每次都试图找到尽可能长的有效窗口。

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
    compressed_indices = []  # 初始化压缩后保留的点索引列表

    i = 0  # 初始化当前位置索引，从轨迹起点开始
    while i < len(df) - 1:  # 当还有未处理的点时继续循环
        # 从当前位置开始创建一个新的压缩窗口
        anchor_idx = i  # 设置窗口的起始锚点为当前位置
        window_end = i + 1  # 窗口初始结束点为下一个点

        # 使用二分查找优化窗口扩张过程，找到窗口可以扩展到的最远位置
        left, right = i + 1, len(df) - 1  # 二分查找的搜索范围：从i+1到轨迹末尾
        best_end = i + 1  # 初始化最佳结束点为最小有效窗口（只有两个点）

        while left <= right:  # 二分查找主循环
            mid = (left + right) // 2  # 计算中间位置

            # 检查从锚点到中间点的所有轨迹点是否都在误差允许范围内
            valid = True  # 假设当前窗口有效
            for k in range(anchor_idx + 1, mid):  # 遍历锚点之后到中间点之前的所有点
                error = GeoUtils.point_to_line_distance(  # 计算点到线段的距离
                    df.iloc[k]['LAT'], df.iloc[k]['LON'],          # 当前检查的点坐标
                    df.iloc[anchor_idx]['LAT'], df.iloc[anchor_idx]['LON'],  # 锚点坐标
                    df.iloc[mid]['LAT'], df.iloc[mid]['LON']       # 中间点坐标
                )
                if error > epsilon:  # 如果任意点的误差超过了阈值
                    valid = False  # 标记窗口无效
                    break  # 跳出检查循环

            if valid:  # 如果当前窗口有效（所有点都在误差范围内）
                # 可以尝试扩展到更远的点
                best_end = mid  # 更新最佳结束点
                left = mid + 1  # 在右侧继续搜索
            else:  # 如果当前窗口无效
                # 不能扩展到中间点，需要在左侧搜索更近的点
                right = mid - 1  # 在左侧继续搜索

        window_end = best_end  # 设置最终的窗口结束点

        # 保留窗口的起始锚点作为压缩后的关键点
        compressed_indices.append(anchor_idx)

        # 将当前位置移动到窗口结束点，为下一个窗口做准备
        i = window_end

        # 安全检查：防止算法陷入无限循环
        if i <= anchor_idx:  # 如果位置没有前进（异常情况）
            # 如果i没有前进，强制前进到下一个点，避免死循环
            i = anchor_idx + 1

    # 确保轨迹的最后一个点总是被保留（无论误差多大）
    if not compressed_indices or compressed_indices[-1] != len(df) - 1:  # 如果索引列表为空或最后一个点不是轨迹末尾
        compressed_indices.append(len(df) - 1)  # 添加最后一个点的索引

    # 返回压缩后的数据框，重置索引并保留原始索引信息用于评估
    return df.iloc[compressed_indices].reset_index(drop=False).rename(columns={"index": "orig_idx"})


# 算法元数据定义 - 用于GUI界面显示和参数配置
DISPLAY_NAME = "开窗算法"  # 算法在界面中显示的名称
DEFAULT_PARAMS = {'epsilon': 100.0}  # 默认参数配置，距离阈值为100米
PARAM_HELP = {'epsilon': '距离阈值（米）'}  # 参数帮助说明，告诉用户epsilon参数的含义
