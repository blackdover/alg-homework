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

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QComboBox, QPushButton, QCheckBox, QGroupBox,
    QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit, QSplitter, QFrame,
    QProgressBar, QMessageBox, QFileDialog, QScrollArea, QTableWidget,
    QTableWidgetItem, QHeaderView, QTabWidget, QSizePolicy
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView

if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    __package__ = "algcode"

from .utils.dataloader import (
    load_data, dataframe_to_trajectory_points, trajectory_points_to_dataframe,
    scan_categories, scan_datasets, load_ais_dataset, TrajectoryPoint
)
from .utils.geo_utils import GeoUtils
from .utils.visualization import visualize_multiple_trajectories
from . import get_available_algorithms, run_algorithm, evaluate_compression


@dataclass
class CompressionResult:
    algorithm: str
    compressed_points: List['TrajectoryPoint']
    compression_ratio: float
    elapsed_time: float
    metrics: Dict[str, Any]


@dataclass
class AlgorithmInfo:
    key: str
    display_name: str
    default_params: Dict[str, Any]
    param_help: Dict[str, str]


class CompressionWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

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
        return run_algorithm(algorithm, self.original_df, params)


class TrajectoryCompressionGUI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.data_root = r"E:\code\homework\alg\AIS Dataset\AIS Data"
        self.current_df = None
        self.compression_results = {}
        self.worker = None

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
        self.setWindowTitle("轨迹压缩算法课设")
        self.setGeometry(100, 100, 1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_splitter = QSplitter(Qt.Horizontal, central_widget)

        self.create_control_panel()
        main_splitter.addWidget(self.control_panel)

        self.create_content_area()
        main_splitter.addWidget(self.content_area)

        main_splitter.setSizes([350, 700])
        self.control_panel.setMinimumWidth(320)
        self.content_area.setMinimumWidth(500)

        main_layout = QVBoxLayout(central_widget)
        main_layout.addWidget(main_splitter)

    def create_control_panel(self):
        self.control_panel = QWidget()
        self.control_panel.setMinimumWidth(320)
        layout = QVBoxLayout(self.control_panel)

        dataset_group = QGroupBox("数据集选择")
        dataset_layout = QVBoxLayout(dataset_group)

        category_layout = QHBoxLayout()
        category_layout.addWidget(QLabel("类别:"))
        self.category_combo = QComboBox()
        self.category_combo.currentTextChanged.connect(self.on_category_changed)
        category_layout.addWidget(self.category_combo)
        dataset_layout.addLayout(category_layout)

        dataset_select_layout = QHBoxLayout()
        dataset_select_layout.addWidget(QLabel("数据集:"))
        self.dataset_combo = QComboBox()
        try:
            self.dataset_combo.setMaxVisibleItems(15)
        except Exception:
            pass
        self.dataset_combo.setMinimumWidth(220)
        dataset_select_layout.addWidget(self.dataset_combo)
        dataset_layout.addLayout(dataset_select_layout)

        layout.addWidget(dataset_group)

        algorithm_group = QGroupBox("算法选择")
        algorithm_layout = QVBoxLayout(algorithm_group)

        self.original_checkbox = QCheckBox("原始轨迹")
        self.original_checkbox.setChecked(True)
        algorithm_layout.addWidget(self.original_checkbox)

        self.algorithm_checkboxes = {}
        for key, info in self.algorithms.items():
            checkbox = QCheckBox(info.display_name)
            checkbox.stateChanged.connect(self.update_parameter_panel)
            self.algorithm_checkboxes[key] = checkbox
            algorithm_layout.addWidget(checkbox)

        for key in ['dr', 'adaptive_dr', 'dp']:
            if key in self.algorithm_checkboxes:
                self.algorithm_checkboxes[key].setChecked(True)

        layout.addWidget(algorithm_group)

        self.create_parameter_panel(layout)

        self.create_action_buttons(layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addStretch()

        self.update_parameter_panel()

    def create_parameter_panel(self, parent_layout):
        param_group = QGroupBox("参数设置")
        param_layout = QVBoxLayout(param_group)

        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        self.param_layout = QVBoxLayout(scroll_widget)

        self.param_widgets = {}

        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        param_layout.addWidget(scroll_area)

        parent_layout.addWidget(param_group)

    def create_action_buttons(self, parent_layout):
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
        self.content_area = QWidget()
        layout = QVBoxLayout(self.content_area)

        self.tab_widget = QTabWidget()

        self.create_visualization_tab()
        self.tab_widget.addTab(self.visualization_tab, "轨迹")

        self.create_metrics_tab()
        self.tab_widget.addTab(self.metrics_tab, "评估指标")

        layout.addWidget(self.tab_widget)

    def create_visualization_tab(self):
        self.visualization_tab = QWidget()
        layout = QVBoxLayout(self.visualization_tab)

        self.web_view = QWebEngineView()
        self.web_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.web_view.setMinimumHeight(500)
        layout.addWidget(self.web_view)

        layout.setStretch(0, 8)

    def create_metrics_tab(self):
        self.metrics_tab = QWidget()
        layout = QVBoxLayout(self.metrics_tab)

        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(7)
        self.metrics_table.setHorizontalHeaderLabels([
            '算法', '压缩率', 'SED均值', 'SED最大', 'SED_95%', '事件保留', '综合得分'
        ])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.metrics_table)

    def load_categories(self):
        categories = scan_categories(self.data_root)
        self.category_combo.clear()
        self.category_combo.addItem("请选择类别", "")
        for category in categories:
            self.category_combo.addItem(category, category)

    def on_category_changed(self, category):
        if not category:
            self.dataset_combo.clear()
            return
        datasets = scan_datasets(os.path.join(self.data_root, category))[:10]

        self.dataset_combo.clear()
        self.dataset_combo.addItem("请选择数据集", "")
        for filename, point_count in datasets:
            display_text = f"{Path(filename).stem} ({point_count} pts)"
            self.dataset_combo.addItem(display_text, filename)

        if len(scan_datasets(os.path.join(self.data_root, category))) > len(datasets):
            self.dataset_combo.setToolTip("仅显示前10个文件；若需更多，请到数据目录查看或重启以加载全部。")

    def update_parameter_panel(self):
        if not hasattr(self, 'param_layout'):
            return
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

        for alg_key, checkbox in self.algorithm_checkboxes.items():
            if checkbox.isChecked():
                alg_info = self.algorithms[alg_key]
                self.add_algorithm_parameters(alg_key, alg_info)

    def add_algorithm_parameters(self, alg_key: str, alg_info: AlgorithmInfo):
        title_label = QLabel(f"{alg_info.display_name}")
        title_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        self.param_layout.addWidget(title_label)

        for param_key, default_value in alg_info.default_params.items():
            param_layout = QHBoxLayout()

            label_text = alg_info.param_help.get(param_key, param_key)
            label = QLabel(f"  {label_text}:")
            param_layout.addWidget(label)

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
        if self.dataset_combo.currentData() is None:
            QMessageBox.warning(self, "警告", "请选择一个数据集")
            return

        selected_algorithms = []
        for alg_key, checkbox in self.algorithm_checkboxes.items():
            if checkbox.isChecked():
                selected_algorithms.append(alg_key)

        if not selected_algorithms and not self.original_checkbox.isChecked():
            QMessageBox.warning(self, "警告", "请至少选择原始轨迹或一个算法")
            return

        category = self.category_combo.currentData()
        filename = self.dataset_combo.currentData()

        try:
            self.current_df = load_ais_dataset(self.data_root, category, filename)
            print(f"加载数据集: {category}/{filename}, 点数: {len(self.current_df)}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载数据集失败: {str(e)}")
            return

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

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
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
        print(f"进度: {message}")

    def on_compression_finished(self, results):
        self.compression_results = results
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        self.export_button.setEnabled(True)

        self.generate_visualization()

        self.calculate_and_display_metrics()

        QMessageBox.information(self, "完成", "轨迹压缩完成！")

    def on_compression_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        QMessageBox.critical(self, "错误", f"压缩过程中出错: {error_msg}")

    def generate_visualization(self):
        if not self.compression_results:
            return

        import tempfile
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_file.close()

        try:
            self.create_interactive_visualization(temp_file.name)

            self.display_image_in_tab(temp_file.name)

            self.tab_widget.setCurrentIndex(0)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"生成可视化失败: {str(e)}")

    def display_image_in_tab(self, image_path: str):
        from PyQt5.QtGui import QPixmap, QPainter
        from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QVBoxLayout, QWidget, QHBoxLayout, QPushButton, QLabel
        from PyQt5.QtCore import Qt

        class ZoomableGraphicsView(QGraphicsView):
            def __init__(self):
                super().__init__()
                self._zoom_factor = 1.0

            def wheelEvent(self, event):
                factor = 1.2 if event.angleDelta().y() > 0 else 1/1.2
                self.scale(factor, factor)
                self._zoom_factor *= factor

        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        control_layout = QHBoxLayout()
        zoom_in_btn = QPushButton("放大")
        zoom_out_btn = QPushButton("缩小")
        fit_btn = QPushButton("适应")

        zoom_in_btn.setMaximumWidth(80)
        zoom_out_btn.setMaximumWidth(80)
        fit_btn.setMaximumWidth(80)

        control_layout.addWidget(zoom_in_btn)
        control_layout.addWidget(zoom_out_btn)
        control_layout.addWidget(fit_btn)
        control_layout.addStretch()

        main_layout.addLayout(control_layout)

        graphics_view = ZoomableGraphicsView()
        scene = QGraphicsScene()
        graphics_view.setScene(scene)

        graphics_view.setRenderHint(QPainter.Antialiasing)
        graphics_view.setRenderHint(QPainter.SmoothPixmapTransform)
        graphics_view.setDragMode(QGraphicsView.ScrollHandDrag)
        graphics_view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        graphics_view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            error_label = QLabel("图片加载失败")
            main_layout.addWidget(error_label)
        else:
            pixmap_item = QGraphicsPixmapItem(pixmap)
            scene.addItem(pixmap_item)

            graphics_view.fitInView(pixmap_item, Qt.KeepAspectRatio)

            main_layout.addWidget(graphics_view)

            def zoom_in():
                factor = 1.2
                graphics_view.scale(factor, factor)

            def zoom_out():
                factor = 1/1.2
                graphics_view.scale(factor, factor)

            def fit_to_window():
                graphics_view.fitInView(pixmap_item, Qt.KeepAspectRatio)

            zoom_in_btn.clicked.connect(zoom_in)
            zoom_out_btn.clicked.connect(zoom_out)
            fit_btn.clicked.connect(fit_to_window)

        if self.tab_widget.count() > 0:
            self.tab_widget.removeTab(0)
        self.tab_widget.insertTab(0, main_widget, "轨迹可视化")
        self.tab_widget.setCurrentIndex(0)

    def create_interactive_visualization(self, output_file: str):
        from algcode.utils.visualization import visualize_multiple_trajectories

        trajectories = {}
        for name, df in self.compression_results.items():
            if name == 'original':
                trajectories['原始轨迹'] = df
            else:
                alg_info = self.algorithms.get(name)
                display_name = alg_info.display_name if alg_info else name
                trajectories[display_name] = df

        visualize_multiple_trajectories(trajectories, output_file)

    def calculate_and_display_metrics(self):
        if self.current_df is None or (hasattr(self.current_df, 'empty') and self.current_df.empty) or not self.compression_results:
            return

        self.metrics_table.setRowCount(0)

        metrics_list = []
        for alg_name, compressed_df in self.compression_results.items():
            if alg_name == 'original':
                continue
            try:
                metrics = evaluate_compression(self.current_df, compressed_df, alg_name, 0.0, True)

                sim = float(metrics.get('trajectory_similarity', 0.0))
                event_rec = float(metrics.get('event_recall', 0.0))
                compression_ratio = float(metrics.get('compression_ratio', 0.0))
                composite = 0.6 * sim + 0.3 * event_rec - 0.1 * (compression_ratio / 100.0)
                metrics['composite_score'] = composite

                metrics_list.append((alg_name, metrics))

            except Exception as e:
                print(f"计算 {alg_name} 指标失败: {e}")

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
        if not self.compression_results:
            QMessageBox.warning(self, "警告", "没有可导出的结果")
            return

        export_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not export_dir:
            return

        try:
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
    app = QApplication(sys.argv)
    app.setApplicationName("轨迹压缩算法课设")

    window = TrajectoryCompressionGUI()
    window.show()
    sys.exit(app.exec_())


