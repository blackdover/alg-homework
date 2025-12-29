"""
轨迹压缩算法 GUI 界面
提供 PyQt5 可视化界面用于轨迹压缩算法比较

作者: Algorithm Engineer
Python版本: 3.10
"""

import pandas as pd
import time
import math
import os
import sys
import argparse
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass
import concurrent.futures
from pathlib import Path

# PyQt5 导入
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QComboBox, QPushButton, QCheckBox, QGroupBox,
    QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit, QSplitter, QFrame,
    QProgressBar, QMessageBox, QFileDialog, QScrollArea, QTableWidget,
    QTableWidgetItem, QHeaderView, QTabWidget, QSizePolicy
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView

# 导入算法包
from algcode import (
    load_data, dataframe_to_trajectory_points, trajectory_points_to_dataframe,
    scan_categories, scan_datasets, load_ais_dataset,
    get_available_algorithms, run_algorithm,
    evaluate_compression, visualize_trajectories, GeoUtils, TrajectoryPoint
)


@dataclass
class CompressionResult:
    """压缩算法结果"""
    algorithm: str
    compressed_points: List['TrajectoryPoint']
    compression_ratio: float
    elapsed_time: float
    metrics: Dict[str, Any]


@dataclass
class AlgorithmInfo:
    """算法信息"""
    key: str
    display_name: str
    default_params: Dict[str, Any]
    param_help: Dict[str, str]


class CompressionWorker(QThread):
    """后台压缩工作线程"""
    progress = pyqtSignal(str)  # 进度信息
    finished = pyqtSignal(dict)  # 完成信号，包含结果
    error = pyqtSignal(str)     # 错误信号

    def __init__(self, original_df: pd.DataFrame, selected_algorithms: List[str],
                 algorithm_params: Dict[str, Dict], include_original: bool):
        super().__init__()
        self.original_df = original_df
        self.selected_algorithms = selected_algorithms
        self.algorithm_params = algorithm_params
        self.include_original = include_original

    def run(self):
        try:
            results = {}

            if self.include_original:
                results['original'] = self.original_df
                self.progress.emit("加载原始轨迹...")

            # 并行运行算法
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_alg = {}
                for alg in self.selected_algorithms:
                    params = self.algorithm_params.get(alg, {})
                    future = executor.submit(self._run_single_algorithm, alg, params)
                    future_to_alg[future] = alg

                for future in concurrent.futures.as_completed(future_to_alg):
                    alg = future_to_alg[future]
                    try:
                        result_df = future.result()
                        results[alg] = result_df
                        self.progress.emit(f"完成算法: {alg}")
                    except Exception as e:
                        self.progress.emit(f"算法 {alg} 执行失败: {str(e)}")

            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))

    def _run_single_algorithm(self, algorithm: str, params: Dict) -> pd.DataFrame:
        """运行单个算法"""
        return run_algorithm(algorithm, self.original_df, params)


class TrajectoryCompressionGUI(QMainWindow):
    """轨迹压缩 GUI 主窗口"""

    def __init__(self):
        super().__init__()
        self.data_root = r"E:\code\homework\alg\AIS Dataset\AIS Data"
        self.current_df = None
        self.compression_results = {}
        self.worker = None

        # 动态加载算法配置
        self.algorithms = {}
        available_algs = get_available_algorithms()
        for alg_key, alg_info in available_algs.items():
            self.algorithms[alg_key] = AlgorithmInfo(
                key=alg_key,
                display_name=alg_info['display_name'],
                default_params=alg_info['default_params'],
                param_help=alg_info['param_help']
            )

        self.init_ui()
        self.load_categories()

    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("轨迹压缩算法比较工具")
        self.setGeometry(100, 100, 1400, 900)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)

        # 左侧控制面板
        self.create_control_panel()
        main_layout.addWidget(self.control_panel, 1)

        # 右侧内容区域（缩小比例以让左侧控件更多空间，避免下拉显示被截断）
        self.create_content_area()
        main_layout.addWidget(self.content_area, 2)

    def create_control_panel(self):
        """创建控制面板"""
        self.control_panel = QWidget()
        self.control_panel.setMaximumWidth(350)
        self.control_panel.setMinimumWidth(320)
        layout = QVBoxLayout(self.control_panel)

        # 数据集选择
        dataset_group = QGroupBox("数据集选择")
        dataset_layout = QVBoxLayout(dataset_group)

        # 类别选择
        category_layout = QHBoxLayout()
        category_layout.addWidget(QLabel("类别:"))
        self.category_combo = QComboBox()
        self.category_combo.currentTextChanged.connect(self.on_category_changed)
        category_layout.addWidget(self.category_combo)
        dataset_layout.addLayout(category_layout)

        # 数据集选择
        dataset_select_layout = QHBoxLayout()
        dataset_select_layout.addWidget(QLabel("数据集:"))
        self.dataset_combo = QComboBox()
        # 提升下拉可见项，避免被截断
        try:
            self.dataset_combo.setMaxVisibleItems(15)
        except Exception:
            pass
        self.dataset_combo.setMinimumWidth(220)
        dataset_select_layout.addWidget(self.dataset_combo)
        dataset_layout.addLayout(dataset_select_layout)

        layout.addWidget(dataset_group)

        # 算法选择
        algorithm_group = QGroupBox("算法选择")
        algorithm_layout = QVBoxLayout(algorithm_group)

        # 原始轨迹复选框
        self.original_checkbox = QCheckBox("原始轨迹")
        self.original_checkbox.setChecked(True)
        algorithm_layout.addWidget(self.original_checkbox)

        # 算法复选框
        self.algorithm_checkboxes = {}
        for key, info in self.algorithms.items():
            checkbox = QCheckBox(info.display_name)
            checkbox.stateChanged.connect(self.update_parameter_panel)  # 连接信号
            self.algorithm_checkboxes[key] = checkbox
            algorithm_layout.addWidget(checkbox)

        # 默认选择几个算法
        for key in ['dr', 'adaptive_dr', 'dp']:
            if key in self.algorithm_checkboxes:
                self.algorithm_checkboxes[key].setChecked(True)

        layout.addWidget(algorithm_group)

        # 参数设置
        self.create_parameter_panel(layout)

        # 操作按钮
        self.create_action_buttons(layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # 添加伸缩空间
        layout.addStretch()

        # 初始化参数面板
        self.update_parameter_panel()

    def create_parameter_panel(self, parent_layout):
        """创建参数设置面板"""
        param_group = QGroupBox("参数设置")
        param_layout = QVBoxLayout(param_group)

        # 滚动区域用于参数
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        self.param_layout = QVBoxLayout(scroll_widget)

        # 动态参数控件存储
        self.param_widgets = {}

        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        param_layout.addWidget(scroll_area)

        parent_layout.addWidget(param_group)

    def create_action_buttons(self, parent_layout):
        """创建操作按钮"""
        button_layout = QVBoxLayout()

        self.run_button = QPushButton("生成并显示")
        self.run_button.clicked.connect(self.on_run_compression)
        self.run_button.setMinimumHeight(35)
        button_layout.addWidget(self.run_button)

        self.export_button = QPushButton("导出结果")
        self.export_button.clicked.connect(self.on_export_results)
        self.export_button.setEnabled(False)
        button_layout.addWidget(self.export_button)

        parent_layout.addLayout(button_layout)

    def create_content_area(self):
        """创建内容显示区域"""
        self.content_area = QWidget()
        layout = QVBoxLayout(self.content_area)

        # 创建标签页
        self.tab_widget = QTabWidget()

        # 可视化标签页
        self.create_visualization_tab()
        self.tab_widget.addTab(self.visualization_tab, "轨迹可视化")

        # 评估指标标签页
        self.create_metrics_tab()
        self.tab_widget.addTab(self.metrics_tab, "评估指标")

        layout.addWidget(self.tab_widget)

    def create_visualization_tab(self):
        """创建可视化标签页"""
        self.visualization_tab = QWidget()
        layout = QVBoxLayout(self.visualization_tab)

        # WebView 用于显示地图
        self.web_view = QWebEngineView()
        # 优先让 WebView 占满可用空间
        self.web_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.web_view.setMinimumHeight(500)
        layout.addWidget(self.web_view)

        # 图层控制说明
        control_label = QLabel("使用地图右上角的图层控制来切换显示不同轨迹")
        control_label.setStyleSheet("color: #666; font-style: italic;")
        # layout.addWidget(control_label)

        # 使 web_view 占据大部分垂直空间，说明文本不占伸缩空间
        layout.setStretch(0, 8)  # web_view
        layout.setStretch(1, 0)  # control_label

    def create_metrics_tab(self):
        """创建评估指标标签页"""
        self.metrics_tab = QWidget()
        layout = QVBoxLayout(self.metrics_tab)

        # 指标表格
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(6)
        self.metrics_table.setHorizontalHeaderLabels([
            '算法', '压缩率', 'SED均值', 'SED最大', 'SED_95%', '事件保留'
        ])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.metrics_table)

    def load_categories(self):
        """加载数据类别"""
        categories = scan_categories(self.data_root)
        self.category_combo.clear()
        self.category_combo.addItem("请选择类别", "")
        for category in categories:
            self.category_combo.addItem(category, category)

    def on_category_changed(self, category):
        """类别选择变化"""
        if not category:
            self.dataset_combo.clear()
            return
        # 仅加载每个类别的前 N 个文件以避免切换时卡顿
        datasets = scan_datasets(os.path.join(self.data_root, category))[:10]

        self.dataset_combo.clear()
        self.dataset_combo.addItem("请选择数据集", "")
        for filename, point_count in datasets:
            display_text = f"{Path(filename).stem} ({point_count} pts)"
            self.dataset_combo.addItem(display_text, filename)

        # 如果存在更多文件，可在下拉提示中显示
        if len(scan_datasets(os.path.join(self.data_root, category))) > len(datasets):
            self.dataset_combo.setToolTip("仅显示前10个文件；若需更多，请到数据目录查看或重启以加载全部。")

    def update_parameter_panel(self):
        """更新参数面板"""
        # 检查参数布局是否已初始化
        if not hasattr(self, 'param_layout'):
            return
        # 清空现有参数控件（递归删除布局与控件，避免残留空布局导致显示错乱）
        def _clear_layout(layout):
            while layout.count():
                item = layout.takeAt(0)
                if item is None:
                    continue
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    child_layout = item.layout()
                    if child_layout is not None:
                        _clear_layout(child_layout)

        _clear_layout(self.param_layout)
        self.param_widgets.clear()

        # 添加选中算法的参数
        for alg_key, checkbox in self.algorithm_checkboxes.items():
            if checkbox.isChecked():
                alg_info = self.algorithms[alg_key]
                self.add_algorithm_parameters(alg_key, alg_info)

    def add_algorithm_parameters(self, alg_key: str, alg_info: AlgorithmInfo):
        """添加算法参数控件"""
        # 算法标题
        title_label = QLabel(f"{alg_info.display_name} 参数")
        title_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        self.param_layout.addWidget(title_label)

        # 参数控件
        for param_key, default_value in alg_info.default_params.items():
            param_layout = QHBoxLayout()

            # 参数标签
            label_text = alg_info.param_help.get(param_key, param_key)
            label = QLabel(f"  {label_text}:")
            param_layout.addWidget(label)

            # 参数输入控件
            if isinstance(default_value, int):
                widget = QSpinBox()
                widget.setValue(int(default_value))
                widget.setMinimum(1)
                widget.setMaximum(10000)
            elif isinstance(default_value, float):
                widget = QDoubleSpinBox()
                widget.setValue(float(default_value))
                widget.setDecimals(4)
                widget.setMinimum(0.0001)
                widget.setMaximum(10000.0)
            else:
                widget = QLineEdit(str(default_value))

            widget.setObjectName(f"{alg_key}_{param_key}")
            param_layout.addWidget(widget)

            self.param_widgets[f"{alg_key}_{param_key}"] = widget
            self.param_layout.addLayout(param_layout)

    def on_run_compression(self):
        """运行压缩"""
        # 检查数据集选择
        if self.dataset_combo.currentData() is None:
            QMessageBox.warning(self, "警告", "请选择一个数据集")
            return

        # 检查算法选择
        selected_algorithms = []
        for alg_key, checkbox in self.algorithm_checkboxes.items():
            if checkbox.isChecked():
                selected_algorithms.append(alg_key)

        if not selected_algorithms and not self.original_checkbox.isChecked():
            QMessageBox.warning(self, "警告", "请至少选择原始轨迹或一个算法")
            return

        # 加载数据集
        category = self.category_combo.currentData()
        filename = self.dataset_combo.currentData()

        try:
            self.current_df = load_ais_dataset(self.data_root, category, filename)
            print(f"加载数据集: {category}/{filename}, 点数: {len(self.current_df)}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载数据集失败: {str(e)}")
            return

        # 获取算法参数
        algorithm_params = {}
        for alg_key in selected_algorithms:
            params = {}
            for param_key in self.algorithms[alg_key].default_params.keys():
                widget_key = f"{alg_key}_{param_key}"
                if widget_key in self.param_widgets:
                    widget = self.param_widgets[widget_key]
                    if isinstance(widget, QSpinBox):
                        params[param_key] = widget.value()
                    elif isinstance(widget, QDoubleSpinBox):
                        params[param_key] = widget.value()
                    elif isinstance(widget, QLineEdit):
                        params[param_key] = widget.text()
            algorithm_params[alg_key] = params

        # 启动后台压缩
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不确定进度
        self.run_button.setEnabled(False)

        self.worker = CompressionWorker(
            self.current_df,
            selected_algorithms,
            algorithm_params,
            self.original_checkbox.isChecked()
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_compression_finished)
        self.worker.error.connect(self.on_compression_error)
        self.worker.start()

    def on_progress(self, message):
        """处理进度更新"""
        print(f"进度: {message}")

    def on_compression_finished(self, results):
        """压缩完成"""
        self.compression_results = results
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        self.export_button.setEnabled(True)

        # 生成可视化
        self.generate_visualization()

        # 计算并显示指标
        self.calculate_and_display_metrics()

        QMessageBox.information(self, "完成", "轨迹压缩完成！")

    def on_compression_error(self, error_msg):
        """压缩错误"""
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        QMessageBox.critical(self, "错误", f"压缩过程中出错: {error_msg}")

    def generate_visualization(self):
        """生成轨迹可视化"""
        if not self.compression_results:
            return

        # 创建临时HTML文件
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(suffix='.html', delete=False)
        temp_file.close()

        try:
            # 调用可视化函数（需要修改以支持多轨迹）
            self.create_interactive_visualization(temp_file.name)

            # 在WebView中加载
            self.web_view.load(QUrl.fromLocalFile(temp_file.name))

            # 切换到可视化标签页
            self.tab_widget.setCurrentIndex(0)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"生成可视化失败: {str(e)}")

    def create_interactive_visualization(self, output_file: str):
        """创建交互式可视化"""
        from algcode.visualization import visualize_multiple_trajectories

        # 准备轨迹字典
        trajectories = {}
        for name, df in self.compression_results.items():
            if name == 'original':
                trajectories['原始轨迹'] = df
            else:
                # 获取算法显示名称
                alg_info = self.algorithms.get(name)
                display_name = alg_info.display_name if alg_info else name
                trajectories[display_name] = df

        # 生成多轨迹可视化
        visualize_multiple_trajectories(trajectories, output_file)

    def calculate_and_display_metrics(self):
        """计算并显示评估指标"""
        # 避免直接对 DataFrame 使用布尔检查（会抛出 ValueError）
        if self.current_df is None or (hasattr(self.current_df, 'empty') and self.current_df.empty) or not self.compression_results:
            return

        self.metrics_table.setRowCount(0)

        for alg_name, compressed_df in self.compression_results.items():
            if alg_name == 'original':
                continue

            try:
                # 计算指标
                metrics = evaluate_compression(
                    self.current_df, compressed_df, alg_name, 0.0, True
                )

                # 添加到表格
                row = self.metrics_table.rowCount()
                self.metrics_table.insertRow(row)

                self.metrics_table.setItem(row, 0, QTableWidgetItem(alg_name))
                self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{metrics['compression_ratio']:.1f}"))
                self.metrics_table.setItem(row, 2, QTableWidgetItem(f"{metrics['sed_mean']:.2f}"))
                self.metrics_table.setItem(row, 3, QTableWidgetItem(f"{metrics['sed_max']:.2f}"))
                self.metrics_table.setItem(row, 4, QTableWidgetItem(f"{metrics['sed_p95']:.2f}"))
                self.metrics_table.setItem(row, 5, QTableWidgetItem(f"{metrics['event_recall']:.3f}"))

            except Exception as e:
                print(f"计算 {alg_name} 指标失败: {e}")

    def on_export_results(self):
        """导出结果"""
        if not self.compression_results:
            QMessageBox.warning(self, "警告", "没有可导出的结果")
            return

        # 选择导出目录
        export_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not export_dir:
            return

        try:
            # 导出每个算法的结果
            for alg_name, df in self.compression_results.items():
                if alg_name == 'original':
                    filename = "original_trajectory.csv"
                else:
                    filename = f"{alg_name}_compressed.csv"

                filepath = os.path.join(export_dir, filename)
                df.to_csv(filepath, index=False)

            QMessageBox.information(self, "成功", f"结果已导出到: {export_dir}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")


# ============================================================================
# 统一压缩接口 (Unified Interface)
# ============================================================================

def compress_trajectory(df: pd.DataFrame,
                       algorithm: str,
                       params: Dict,
                       calculate_sed: bool = True) -> CompressionResult:
    """
    统一的轨迹压缩接口

    参数:
        df: 输入轨迹 DataFrame
        algorithm: 算法名称 ('dr', 'adaptive_dr', 'semantic_dr', 'dp', 'sliding', 'opening', 'squish')
        params: 算法参数
        calculate_sed: 是否计算SED指标（耗时操作），默认True

    返回:
        CompressionResult 对象
    """
    # 运行指定算法（使用动态加载的 run_algorithm）
    start_time = time.time()
    compressed_df = run_algorithm(algorithm, df, params)
    elapsed_time = time.time() - start_time

    # 计算指标
    metrics = evaluate_compression(df, compressed_df, algorithm, elapsed_time, calculate_sed)

    # 转换为TrajectoryPoint列表
    compressed_points = dataframe_to_trajectory_points(compressed_df)

    return CompressionResult(
        algorithm=algorithm,
        compressed_points=compressed_points,
        compression_ratio=metrics['compression_ratio'],
        elapsed_time=elapsed_time,
        metrics=metrics
    )


# ============================================================================
# 命令行界面 (CLI Interface)
# ============================================================================

def create_cli_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='轨迹压缩算法工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python trajectory_compression.py data.csv -a dr -e 100 -o result.csv -m map.html
  python trajectory_compression.py data.csv -a adaptive_dr --min-threshold 20 --max-threshold 500 -v 3 20
  python trajectory_compression.py data.csv -a semantic_dr --cog-threshold 15 --time-threshold 600
  python trajectory_compression.py data.csv -a dp -e 0.001
  python trajectory_compression.py data.csv -a squish -b 200

支持的算法:
  dr: 固定阈值 Dead Reckoning
  adaptive_dr: 自适应阈值 Dead Reckoning
  semantic_dr: 语义增强 Dead Reckoning
  dp: Douglas-Peucker
  sliding: Sliding Window
  opening: Opening Window
  squish: SQUISH 流式算法
        """
    )

    parser.add_argument('input_file', help='输入的轨迹数据文件路径')

    parser.add_argument('-a', '--algorithm', required=True,
                       choices=['dr', 'adaptive_dr', 'semantic_dr', 'dp', 'sliding', 'opening', 'squish'],
                       help='使用的压缩算法')

    parser.add_argument('-o', '--output', help='压缩结果输出文件路径（CSV格式）')

    parser.add_argument('-m', '--map', help='生成的轨迹可视化地图文件路径（HTML格式）')

    parser.add_argument('-e', '--epsilon', type=float, default=100.0,
                       help='距离阈值（米），用于dr/sliding/opening算法；度，用于dp算法')

    # 自适应DR参数
    parser.add_argument('--min-threshold', type=float, default=20.0,
                       help='自适应DR的最低距离阈值（米）')
    parser.add_argument('--max-threshold', type=float, default=500.0,
                       help='自适应DR的最高距离阈值（米）')
    parser.add_argument('-v', '--velocity-range', nargs=2, type=float,
                       metavar=('V_LOWER', 'V_UPPER'), default=[3.0, 20.0],
                       help='自适应DR的速度范围（节）：低速截止点和高速截止点')

    # 语义DR参数
    parser.add_argument('--cog-threshold', type=float, default=10.0,
                       help='语义DR的转向角度阈值（度）')
    parser.add_argument('--sog-threshold', type=float, default=1.0,
                       help='语义DR的速度变化阈值（节）')
    parser.add_argument('--time-threshold', type=float, default=300.0,
                       help='语义DR的时间间隔阈值（秒）')

    # SQUISH参数
    parser.add_argument('-b', '--buffer-size', type=int, default=100,
                       help='SQUISH算法的缓冲区大小')

    parser.add_argument('--mmsi', type=int, help='筛选特定船舶的MMSI号')

    return parser


def main():
    """主函数"""
    parser = create_cli_parser()
    args = parser.parse_args()

    try:
        # 加载数据
        print(f"正在加载数据文件: {args.input_file}")
        df = load_data(args.input_file, args.mmsi)
        print(f"成功加载 {len(df)} 个数据点")

        # 准备算法参数
        params = {}

        if args.algorithm == 'dr':
            params['epsilon'] = args.epsilon
        elif args.algorithm == 'adaptive_dr':
            params.update({
                'min_threshold': args.min_threshold,
                'max_threshold': args.max_threshold,
                'v_lower': args.velocity_range[0],
                'v_upper': args.velocity_range[1]
            })
        elif args.algorithm == 'semantic_dr':
            params.update({
                'min_threshold': args.min_threshold,
                'max_threshold': args.max_threshold,
                'v_lower': args.velocity_range[0],
                'v_upper': args.velocity_range[1],
                'cog_threshold': args.cog_threshold,
                'sog_threshold': args.sog_threshold,
                'time_threshold': args.time_threshold
            })
        elif args.algorithm == 'dp':
            params['epsilon'] = args.epsilon
        elif args.algorithm in ['sliding', 'opening']:
            params['epsilon'] = args.epsilon
        elif args.algorithm == 'squish':
            params['buffer_size'] = args.buffer_size

        # 执行压缩
        print(f"\n执行 {args.algorithm} 算法...")
        result = compress_trajectory(df, args.algorithm, params)

        # 输出结果
        print("压缩完成！")
        print(".1f"
              ".3f")

        # 保存压缩结果
        if args.output:
            compressed_df = trajectory_points_to_dataframe(result.compressed_points)
            compressed_df.to_csv(args.output, index=False)
            print(f"压缩结果已保存至: {args.output}")

        # 生成可视化地图
        if args.map:
            compressed_df = trajectory_points_to_dataframe(result.compressed_points)
            visualize_trajectories(df, compressed_df, output_file=args.map)

    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


# ============================================================================
# 主执行脚本 (Main Execution)
# ============================================================================

def main_cli():
    """命令行模式"""
    parser = create_cli_parser()
    args = parser.parse_args()

    try:
        # 加载数据
        print(f"正在加载数据文件: {args.input_file}")
        df = load_data(args.input_file, args.mmsi)
        print(f"成功加载 {len(df)} 个数据点")

        # 准备算法参数
        params = {}

        if args.algorithm == 'dr':
            params['epsilon'] = args.epsilon
        elif args.algorithm == 'adaptive_dr':
            params.update({
                'min_threshold': args.min_threshold,
                'max_threshold': args.max_threshold,
                'v_lower': args.velocity_range[0],
                'v_upper': args.velocity_range[1]
            })
        elif args.algorithm == 'semantic_dr':
            params.update({
                'min_threshold': args.min_threshold,
                'max_threshold': args.max_threshold,
                'v_lower': args.velocity_range[0],
                'v_upper': args.velocity_range[1],
                'cog_threshold': args.cog_threshold,
                'sog_threshold': args.sog_threshold,
                'time_threshold': args.time_threshold
            })
        elif args.algorithm == 'dp':
            params['epsilon'] = args.epsilon
        elif args.algorithm in ['sliding', 'opening']:
            params['epsilon'] = args.epsilon
        elif args.algorithm == 'squish':
            params['buffer_size'] = args.buffer_size

        # 执行压缩
        print(f"\n执行 {args.algorithm} 算法...")
        result = compress_trajectory(df, args.algorithm, params)

        # 输出结果
        print("压缩完成！")
        print(".1f"
              ".3f")

        # 保存压缩结果
        if args.output:
            compressed_df = trajectory_points_to_dataframe(result.compressed_points)
            compressed_df.to_csv(args.output, index=False)
            print(f"压缩结果已保存至: {args.output}")

        # 生成可视化地图
        if args.map:
            compressed_df = trajectory_points_to_dataframe(result.compressed_points)
            visualize_trajectories(df, compressed_df, output_file=args.map)

    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


def main_gui():
    """GUI模式"""
    app = QApplication(sys.argv)

    # 设置应用程序属性
    app.setApplicationName("轨迹压缩算法工具")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Algorithm Engineer")

    # 创建主窗口
    window = TrajectoryCompressionGUI()
    window.show()

    # 运行应用程序
    sys.exit(app.exec_())


if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        # 显示帮助信息
        main_cli()
    elif len(sys.argv) > 1:
        # 命令行模式
        main_cli()
    else:
        # GUI模式
        main_gui()

