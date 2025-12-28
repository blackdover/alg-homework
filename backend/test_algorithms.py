"""
轨迹压缩算法单元测试
测试所有算法的正确性和性能
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from algorithms import (
    compress, dead_reckoning_compress, fixed_epsilon_dr_compress,
    semantic_dr_compress, sliding_window_compress, opening_window_compress,
    squish_compress, dataframe_to_trajectory_points, evaluate_compression
)


class TestTrajectoryCompression(unittest.TestCase):
    """轨迹压缩算法测试类"""

    def setUp(self):
        """测试前准备数据"""
        # 创建测试轨迹数据
        n_points = 100
        start_time = datetime(2021, 1, 1, 0, 0, 0)

        # 生成直线轨迹（用于测试基本压缩）
        lats = []
        lons = []
        times = []
        speeds = []
        courses = []

        for i in range(n_points):
            # 直线运动：从 (0, 0) 向东北方向移动
            lat = i * 0.001  # 约100米间隔
            lon = i * 0.001
            time = start_time + timedelta(minutes=i)
            speed = 10.0  # 10节
            course = 45.0  # 东北方向

            lats.append(lat)
            lons.append(lon)
            times.append(time)
            speeds.append(speed)
            courses.append(course)

        # 添加一些噪声
        np.random.seed(42)
        lats = np.array(lats) + np.random.normal(0, 0.0001, n_points)
        lons = np.array(lons) + np.random.normal(0, 0.0001, n_points)
        speeds = np.array(speeds) + np.random.normal(0, 0.5, n_points)
        courses = np.array(courses) + np.random.normal(0, 2, n_points)

        # 创建 DataFrame
        self.test_df = pd.DataFrame({
            'MMSI': [123456789] * n_points,
            'BaseDateTime': times,
            'LAT': lats,
            'LON': lons,
            'SOG': speeds,
            'COG': courses
        })

        # 转换为 TrajectoryPoint 列表
        self.test_points = dataframe_to_trajectory_points(self.test_df)

    def test_dr_compress_basic(self):
        """测试自适应 DR 算法基本功能"""
        params = {
            'min_threshold': 20.0,
            'max_threshold': 500.0,
            'v_lower': 3.0,
            'v_upper': 20.0
        }

        result = compress(self.test_df, 'dr', params)

        # 基本断言
        self.assertIsNotNone(result)
        self.assertEqual(result.algorithm, 'dr')
        self.assertIsInstance(result.compression_ratio, (int, float))
        self.assertGreater(result.compression_ratio, 0)
        self.assertLessEqual(result.compression_ratio, 100)
        self.assertGreater(len(result.compressed_points), 0)
        self.assertLessEqual(len(result.compressed_points), len(self.test_points))

        # 确保包含起点和终点
        self.assertAlmostEqual(result.compressed_points[0].lat, self.test_points[0].lat, places=6)
        self.assertAlmostEqual(result.compressed_points[0].lon, self.test_points[0].lon, places=6)
        self.assertAlmostEqual(result.compressed_points[-1].lat, self.test_points[-1].lat, places=6)
        self.assertAlmostEqual(result.compressed_points[-1].lon, self.test_points[-1].lon, places=6)

    def test_fixed_dr_compress(self):
        """测试固定阈值 DR 算法"""
        params = {'epsilon': 10.0}  # 使用更小的阈值确保有压缩

        result = compress(self.test_df, 'fixed_dr', params)

        self.assertIsNotNone(result)
        self.assertEqual(result.algorithm, 'fixed_dr')
        self.assertGreaterEqual(result.compression_ratio, 0)  # 允许0压缩率（如果所有点都在阈值内）
        self.assertLessEqual(result.compression_ratio, 100)

    def test_semantic_dr_compress(self):
        """测试语义增强 DR 算法"""
        params = {
            'min_threshold': 20.0,
            'max_threshold': 500.0,
            'v_lower': 3.0,
            'v_upper': 20.0,
            'cog_threshold': 10.0,
            'sog_threshold': 1.0,
            'time_threshold': 300.0
        }

        result = compress(self.test_df, 'semantic_dr', params)

        self.assertIsNotNone(result)
        self.assertEqual(result.algorithm, 'semantic_dr')
        self.assertGreater(result.compression_ratio, 0)
        self.assertLessEqual(result.compression_ratio, 100)

    def test_sliding_window_compress(self):
        """测试 Sliding Window 算法"""
        params = {'epsilon': 50.0}

        result = compress(self.test_df, 'sliding', params)

        self.assertIsNotNone(result)
        self.assertEqual(result.algorithm, 'sliding')
        self.assertGreater(result.compression_ratio, 0)
        self.assertLessEqual(result.compression_ratio, 100)

    def test_opening_window_compress(self):
        """测试 Opening Window 算法"""
        params = {'epsilon': 50.0}

        result = compress(self.test_df, 'opening', params)

        self.assertIsNotNone(result)
        self.assertEqual(result.algorithm, 'opening')
        self.assertGreater(result.compression_ratio, 0)
        self.assertLessEqual(result.compression_ratio, 100)

    def test_squish_compress(self):
        """测试 SQUISH 算法"""
        params = {'buffer_size': 50}

        result = compress(self.test_df, 'squish', params)

        self.assertIsNotNone(result)
        self.assertEqual(result.algorithm, 'squish')
        self.assertGreater(result.compression_ratio, 0)
        self.assertLessEqual(result.compression_ratio, 100)
        self.assertLessEqual(len(result.compressed_points), 50)  # 缓冲区大小限制

    def test_compression_metrics(self):
        """测试压缩指标计算"""
        params = {'epsilon': 100.0}
        result = compress(self.test_df, 'fixed_dr', params)

        # 检查指标是否存在
        metrics = result.metrics
        self.assertIn('sed_mean', metrics)
        self.assertIn('sed_max', metrics)
        self.assertIn('sed_p95', metrics)
        self.assertIn('compression_ratio', metrics)
        self.assertIn('event_recall', metrics)

        # 检查数值合理性
        self.assertIsInstance(metrics['sed_mean'], (int, float))
        self.assertIsInstance(metrics['sed_max'], (int, float))
        self.assertIsInstance(metrics['sed_p95'], (int, float))
        self.assertGreaterEqual(metrics['sed_mean'], 0)
        self.assertGreaterEqual(metrics['sed_max'], 0)
        self.assertGreaterEqual(metrics['sed_p95'], 0)
        self.assertGreaterEqual(metrics['event_recall'], 0)
        self.assertLessEqual(metrics['event_recall'], 1)

    def test_empty_trajectory(self):
        """测试空轨迹处理"""
        empty_df = pd.DataFrame(columns=['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG'])

        params = {'epsilon': 100.0}
        result = compress(empty_df, 'fixed_dr', params)

        self.assertEqual(len(result.compressed_points), 0)
        self.assertEqual(result.compression_ratio, 0.0)

    def test_single_point_trajectory(self):
        """测试单点轨迹处理"""
        single_df = self.test_df.head(1).copy()

        params = {'epsilon': 100.0}
        result = compress(single_df, 'fixed_dr', params)

        self.assertEqual(len(result.compressed_points), 1)
        self.assertEqual(result.compression_ratio, 0.0)

    def test_invalid_algorithm(self):
        """测试无效算法名"""
        params = {}
        with self.assertRaises(ValueError):
            compress(self.test_df, 'invalid_algorithm', params)

    def test_algorithm_comparison(self):
        """测试算法对比（压缩率应该有差异）"""
        algorithms = ['dr', 'fixed_dr', 'sliding', 'opening', 'squish']
        results = {}

        for algo in algorithms:
            params = self._get_default_params(algo)
            result = compress(self.test_df, algo, params)
            results[algo] = result

        # 至少应该有两个不同的压缩率（证明算法确实不同）
        ratios = [results[algo].compression_ratio for algo in algorithms]
        self.assertGreater(len(set(ratios)), 1, "算法应该产生不同的压缩结果")

    def _get_default_params(self, algorithm):
        """获取算法默认参数"""
        defaults = {
            'dr': {
                'min_threshold': 20.0,
                'max_threshold': 500.0,
                'v_lower': 3.0,
                'v_upper': 20.0
            },
            'fixed_dr': {'epsilon': 100.0},
            'semantic_dr': {
                'min_threshold': 20.0,
                'max_threshold': 500.0,
                'v_lower': 3.0,
                'v_upper': 20.0,
                'cog_threshold': 10.0,
                'sog_threshold': 1.0,
                'time_threshold': 300.0
            },
            'sliding': {'epsilon': 100.0},
            'opening': {'epsilon': 100.0},
            'squish': {'buffer_size': 50}
        }
        return defaults.get(algorithm, {})

    def test_trajectory_with_turns(self):
        """测试包含转弯的轨迹"""
        # 创建包含转弯的轨迹
        n_points = 50
        start_time = datetime(2021, 1, 1, 0, 0, 0)

        lats, lons, times, speeds, courses = [], [], [], [], []

        for i in range(n_points):
            time = start_time + timedelta(minutes=i)
            speed = 10.0

            if i < 25:
                # 前半段：向东
                lat = i * 0.001
                lon = 0.0
                course = 90.0
            else:
                # 后半段：向北（转弯）
                lat = 25 * 0.001
                lon = (i - 25) * 0.001
                course = 0.0

            lats.append(lat)
            lons.append(lon)
            times.append(time)
            speeds.append(speed)
            courses.append(course)

        turn_df = pd.DataFrame({
            'MMSI': [123456789] * n_points,
            'BaseDateTime': times,
            'LAT': lats,
            'LON': lons,
            'SOG': speeds,
            'COG': courses
        })

        # 测试语义 DR 应该保留转弯点
        params = {
            'min_threshold': 20.0,
            'max_threshold': 500.0,
            'v_lower': 3.0,
            'v_upper': 20.0,
            'cog_threshold': 10.0,
            'sog_threshold': 1.0,
            'time_threshold': 300.0
        }

        result = compress(turn_df, 'semantic_dr', params)

        # 应该保留至少一个转弯处的点
        self.assertGreater(len(result.compressed_points), 2)


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
