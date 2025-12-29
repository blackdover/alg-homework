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
# 如果此模块作为脚本执行（python algcode/ui.py）
# Python 会将 __package__ 设置为 None，这会导致相对导入失败。通过将
# 项目根目录插入 sys.path 并设置 __package__ 来修复，以便 "from .utils..." 正常工作。
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    __package__ = "algcode"

# 导入包内工具与算法接口
from .utils.dataloader import (
    load_data, dataframe_to_trajectory_points, trajectory_points_to_dataframe,
    scan_categories, scan_datasets, load_ais_dataset, TrajectoryPoint
)
from .utils.geo_utils import GeoUtils
from .utils.visualization import visualize_multiple_trajectories
from . import get_available_algorithms, run_algorithm, evaluate_compression


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
        self.setWindowTitle("轨迹压缩算法课设")
        self.setGeometry(100, 100, 1400, 900)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建可调整大小的分隔器布局
        main_splitter = QSplitter(Qt.Horizontal, central_widget)

        # 左侧控制面板
        self.create_control_panel()
        main_splitter.addWidget(self.control_panel)

        # 右侧内容区域
        self.create_content_area()
        main_splitter.addWidget(self.content_area)

        # 设置初始比例 (左侧:右侧 = 1:2)
        main_splitter.setSizes([350, 700])
        # 设置最小宽度，避免拖拽时控件被压缩过小
        self.control_panel.setMinimumWidth(320)
        self.content_area.setMinimumWidth(500)

        # 设置为主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.addWidget(main_splitter)

    def create_control_panel(self):
        """创建控制面板"""
        self.control_panel = QWidget()
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

        self.run_button = QPushButton("计算结果")
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
        self.tab_widget.addTab(self.visualization_tab, "轨迹")

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
        self.metrics_table.setColumnCount(7)
        self.metrics_table.setHorizontalHeaderLabels([
            '算法', '压缩率', 'SED均值', 'SED最大', 'SED_95%', '事件保留', '综合得分'
        ])
        # 设置等宽分布列
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
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
        # 算法标题（直接使用算法显示名，去掉“参数”字样）
        title_label = QLabel(f"{alg_info.display_name}")
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

        # 创建临时PNG文件
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_file.close()

        try:
            # 调用可视化函数
            self.create_interactive_visualization(temp_file.name)

            # 在标签页中显示图像
            self.display_image_in_tab(temp_file.name)

            # 切换到可视化标签页
            self.tab_widget.setCurrentIndex(0)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"生成可视化失败: {str(e)}")

    def display_image_in_tab(self, image_path: str):
        """在可视化标签页中显示图像，支持缩放"""
        from PyQt5.QtGui import QPixmap, QPainter
        from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QVBoxLayout, QWidget, QHBoxLayout, QPushButton, QLabel
        from PyQt5.QtCore import Qt

        # 定义支持滚轮缩放的GraphicsView类
        class ZoomableGraphicsView(QGraphicsView):
            def __init__(self):
                super().__init__()
                self._zoom_factor = 1.0

            def wheelEvent(self, event):
                # 滚轮缩放
                factor = 1.2 if event.angleDelta().y() > 0 else 1/1.2
                self.scale(factor, factor)
                self._zoom_factor *= factor

        # 创建主布局
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        # 创建控制按钮布局
        control_layout = QHBoxLayout()
        zoom_in_btn = QPushButton("放大")
        zoom_out_btn = QPushButton("缩小")
        fit_btn = QPushButton("适应")
        # reset_btn = QPushButton("重置缩放")

        zoom_in_btn.setMaximumWidth(80)
        zoom_out_btn.setMaximumWidth(80)
        fit_btn.setMaximumWidth(80)
        # reset_btn.setMaximumWidth(80)

        control_layout.addWidget(zoom_in_btn)
        control_layout.addWidget(zoom_out_btn)
        control_layout.addWidget(fit_btn)
        # control_layout.addWidget(reset_btn)
        control_layout.addStretch()

        main_layout.addLayout(control_layout)

        # 创建图形视图和场景
        graphics_view = ZoomableGraphicsView()
        scene = QGraphicsScene()
        graphics_view.setScene(scene)

        # 设置视图属性
        graphics_view.setRenderHint(QPainter.Antialiasing)
        graphics_view.setRenderHint(QPainter.SmoothPixmapTransform)
        graphics_view.setDragMode(QGraphicsView.ScrollHandDrag)
        graphics_view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        graphics_view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        # 加载图片
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            error_label = QLabel("图片加载失败")
            main_layout.addWidget(error_label)
        else:
            pixmap_item = QGraphicsPixmapItem(pixmap)
            scene.addItem(pixmap_item)

            # 适应窗口大小
            graphics_view.fitInView(pixmap_item, Qt.KeepAspectRatio)

            main_layout.addWidget(graphics_view)

            # 连接按钮信号
            def zoom_in():
                factor = 1.2
                graphics_view.scale(factor, factor)

            def zoom_out():
                factor = 1/1.2
                graphics_view.scale(factor, factor)

            def fit_to_window():
                graphics_view.fitInView(pixmap_item, Qt.KeepAspectRatio)

            def reset_zoom():
                graphics_view.resetTransform()
                graphics_view.fitInView(pixmap_item, Qt.KeepAspectRatio)

            zoom_in_btn.clicked.connect(zoom_in)
            zoom_out_btn.clicked.connect(zoom_out)
            fit_btn.clicked.connect(fit_to_window)
            # reset_btn.clicked.connect(reset_zoom)

        # 替换可视化标签页的内容
        if self.tab_widget.count() > 0:
            self.tab_widget.removeTab(0)
        self.tab_widget.insertTab(0, main_widget, "轨迹可视化")
        self.tab_widget.setCurrentIndex(0)

    def create_interactive_visualization(self, output_file: str):
        """创建交互式可视化"""
        from algcode.utils.visualization import visualize_multiple_trajectories

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

        # 收集所有算法的指标并计算综合得分后排序
        metrics_list = []
        for alg_name, compressed_df in self.compression_results.items():
            if alg_name == 'original':
                continue
            try:
                metrics = evaluate_compression(self.current_df, compressed_df, alg_name, 0.0, True)

                # 计算综合得分：权重可调整
                # composite = 0.6 * similarity + 0.3 * event_recall - 0.1 * (compression_ratio / 100)
                sim = float(metrics.get('trajectory_similarity', 0.0))
                event_rec = float(metrics.get('event_recall', 0.0))
                compression_ratio = float(metrics.get('compression_ratio', 0.0))  # 0-100
                composite = 0.6 * sim + 0.3 * event_rec - 0.1 * (compression_ratio / 100.0)
                metrics['composite_score'] = composite

                metrics_list.append((alg_name, metrics))

            except Exception as e:
                print(f"计算 {alg_name} 指标失败: {e}")

        # 按综合得分降序排序并填充表格
        metrics_list.sort(key=lambda x: x[1].get('composite_score', 0.0), reverse=True)
        for alg_name, metrics in metrics_list:
            row = self.metrics_table.rowCount()
            self.metrics_table.insertRow(row)
            try:
                self.metrics_table.setItem(row, 0, QTableWidgetItem(alg_name))
                self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{metrics.get('compression_ratio', 0.0):.1f}"))
                self.metrics_table.setItem(row, 2, QTableWidgetItem(f"{metrics.get('sed_mean', 0.0):.2f}"))
                self.metrics_table.setItem(row, 3, QTableWidgetItem(f"{metrics.get('sed_max', 0.0):.2f}"))
                self.metrics_table.setItem(row, 4, QTableWidgetItem(f"{metrics.get('sed_p95', 0.0):.2f}"))
                self.metrics_table.setItem(row, 5, QTableWidgetItem(f"{metrics.get('event_recall', 0.0):.3f}"))
                self.metrics_table.setItem(row, 6, QTableWidgetItem(f"{metrics.get('composite_score', 0.0):.5f}"))
            except Exception as e:
                print(f"填充表格时出错 {alg_name}: {e}")

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


def main_gui():
    """GUI 模式入口：创建并运行主窗口"""
    app = QApplication(sys.argv)
    app.setApplicationName("轨迹压缩算法课设")

    window = TrajectoryCompressionGUI()
    window.show()
    sys.exit(app.exec_())


