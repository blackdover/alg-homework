# 导入基础数据处理库
import pandas as pd  # 用于数据处理和分析的数据框库
import time  # 时间处理模块
import math  # 数学函数库
import os  # 操作系统接口，用于文件和路径操作
import sys  # 系统相关的参数和函数
import argparse  # 命令行参数解析库
from typing import Optional, Tuple, Dict, List, Any  # 类型注解，用于代码类型提示
from dataclasses import dataclass  # 数据类装饰器，用于创建简单的数据容器类
import concurrent.futures  # 并发执行库，用于并行处理任务
from pathlib import Path  # 路径处理库，提供面向对象的路径操作

# PyQt5 导入 - GUI界面库的所有组件
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,  # 基础GUI组件和布局
    QGridLayout, QLabel, QComboBox, QPushButton, QCheckBox, QGroupBox,  # 控件组件
    QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit, QSplitter, QFrame,  # 输入和分割控件
    QProgressBar, QMessageBox, QFileDialog, QScrollArea, QTableWidget,  # 进度条、对话框、文件选择、滚动区域、表格
    QTableWidgetItem, QHeaderView, QTabWidget, QSizePolicy  # 表格项、表头视图、标签页、大小策略
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QUrl  # 核心功能：线程、信号、Qt常量、URL
from PyQt5.QtWebEngineWidgets import QWebEngineView  # Web引擎视图，用于显示地图

# 处理相对导入问题：当此文件作为主脚本直接运行时（python algcode/ui.py），
# Python会将__package__设置为None，导致"from .utils..."这样的相对导入失败。
# 通过将项目根目录添加到sys.path并手动设置__package__来解决这个问题。
if __name__ == "__main__" and __package__ is None:  # 检查是否作为主脚本运行且包名未设置
    import sys  # 重新导入sys（虽然已经导入了，但为了清晰）
    from pathlib import Path  # 重新导入Path
    project_root = str(Path(__file__).resolve().parent.parent)  # 获取项目根目录（alg文件夹）
    if project_root not in sys.path:  # 如果项目根目录不在Python路径中
        sys.path.insert(0, project_root)  # 将项目根目录插入到Python路径的开头
    __package__ = "algcode"  # 手动设置包名为algcode，使相对导入正常工作

# 导入项目内部的工具模块和算法接口
from .utils.dataloader import (  # 数据加载工具模块
    load_data, dataframe_to_trajectory_points, trajectory_points_to_dataframe,  # 数据转换函数
    scan_categories, scan_datasets, load_ais_dataset, TrajectoryPoint  # 数据集扫描和AIS数据集加载
)
from .utils.geo_utils import GeoUtils  # 地理工具类
from .utils.visualization import visualize_multiple_trajectories  # 多轨迹可视化函数
from . import get_available_algorithms, run_algorithm, evaluate_compression  # 算法接口函数


@dataclass  # 使用dataclass装饰器自动生成__init__、__repr__等方法
class CompressionResult:  # 定义压缩结果的数据类，用于存储单个算法的压缩结果
    """压缩算法结果数据类，包含算法运行后的所有关键信息"""
    algorithm: str  # 算法名称，如'dp'、'dr'等
    compressed_points: List['TrajectoryPoint']  # 压缩后的轨迹点列表（使用字符串引用避免循环导入）
    compression_ratio: float  # 压缩率，原始点数/压缩后点数的比例
    elapsed_time: float  # 算法运行耗时（秒）
    metrics: Dict[str, Any]  # 评估指标字典，包含各种误差和相似度指标


@dataclass  # 数据类装饰器
class AlgorithmInfo:  # 定义算法信息的数据类，用于存储算法的元数据
    """算法信息数据类，存储算法的基本信息和参数配置"""
    key: str  # 算法的唯一标识符，如'dp'、'dr'
    display_name: str  # 算法的显示名称，如'道格拉斯-普克算法'
    default_params: Dict[str, Any]  # 默认参数字典，key为参数名，value为默认值
    param_help: Dict[str, str]  # 参数帮助字典，key为参数名，value为参数说明文本


class CompressionWorker(QThread):  # 继承自QThread，用于在后台执行压缩任务
    """后台压缩工作线程类，用于并行运行多个轨迹压缩算法，避免阻塞GUI界面"""
    progress = pyqtSignal(str)  # 进度信息信号，向GUI发送当前处理状态的字符串消息
    finished = pyqtSignal(dict)  # 完成信号，发送包含所有算法结果的字典
    error = pyqtSignal(str)     # 错误信号，当压缩过程中出现异常时发送错误信息

    def __init__(self, original_df: pd.DataFrame, selected_algorithms: List[str],  # 构造函数，接收压缩任务的参数
                 algorithm_params: Dict[str, Dict], include_original: bool):
        super().__init__()  # 调用父类QThread的构造函数
        self.original_df = original_df  # 原始轨迹数据框
        self.selected_algorithms = selected_algorithms  # 用户选择的算法列表
        self.algorithm_params = algorithm_params  # 各算法的参数配置字典
        self.include_original = include_original  # 是否包含原始轨迹在结果中

    def run(self):  # 重写QThread的run方法，线程启动时自动执行
        try:  # 使用try-except捕获可能的异常
            results = {}  # 初始化结果字典，用于存储所有算法的压缩结果

            if self.include_original:  # 如果用户选择了包含原始轨迹
                results['original'] = self.original_df  # 将原始数据添加到结果中
                self.progress.emit("加载原始轨迹...")  # 发送进度信号给GUI显示

            # 使用线程池并行运行多个压缩算法，提高处理效率
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:  # 创建4个工作线程的线程池
                future_to_alg = {}  # 创建future到算法名的映射字典

                for alg in self.selected_algorithms:  # 遍历用户选择的所有算法
                    params = self.algorithm_params.get(alg, {})  # 获取该算法的参数配置，默认空字典
                    future = executor.submit(self._run_single_algorithm, alg, params)  # 提交算法运行任务到线程池
                    future_to_alg[future] = alg  # 记录future对应的算法名

                for future in concurrent.futures.as_completed(future_to_alg):  # 遍历完成的future对象
                    alg = future_to_alg[future]  # 获取对应的算法名
                    try:  # 捕获单个算法执行的异常
                        result_df = future.result()  # 获取算法的执行结果（阻塞等待）
                        results[alg] = result_df  # 将结果存储到结果字典中
                        self.progress.emit(f"完成算法: {alg}")  # 发送完成进度信号
                    except Exception as e:  # 如果算法执行失败
                        self.progress.emit(f"算法 {alg} 执行失败: {str(e)}")  # 发送错误进度信号

            self.finished.emit(results)  # 发送完成信号，包含所有结果

        except Exception as e:  # 捕获整个run方法的异常
            self.error.emit(str(e))  # 发送错误信号给GUI处理

    def _run_single_algorithm(self, algorithm: str, params: Dict) -> pd.DataFrame:  # 私有方法，运行单个压缩算法
        """运行单个算法的辅助方法，由线程池调用"""
        return run_algorithm(algorithm, self.original_df, params)  # 调用全局的run_algorithm函数执行压缩


class TrajectoryCompressionGUI(QMainWindow):  # 主窗口类，继承自QMainWindow
    """轨迹压缩算法的可视化GUI主窗口类，提供完整的用户界面和交互功能"""

    def __init__(self):  # 构造函数，初始化GUI的所有组件和数据
        super().__init__()  # 调用父类QMainWindow的构造函数
        self.data_root = r"E:\code\homework\alg\AIS Dataset\AIS Data"  # AIS数据集的根目录路径
        self.current_df = None  # 当前加载的数据集（pandas DataFrame）
        self.compression_results = {}  # 压缩结果字典，存储所有算法的输出结果
        self.worker = None  # 后台工作线程对象，用于执行压缩任务

        # 动态加载算法配置：从algorithms模块获取所有可用的压缩算法信息
        self.algorithms = {}  # 算法信息字典，key为算法标识，value为AlgorithmInfo对象
        available_algs = get_available_algorithms()  # 获取所有可用算法的配置信息
        for alg_key, alg_info in available_algs.items():  # 遍历每个算法的配置
            self.algorithms[alg_key] = AlgorithmInfo(  # 创建AlgorithmInfo对象存储算法信息
                key=alg_key,  # 算法唯一标识符
                display_name=alg_info['display_name'],  # 算法显示名称（用于界面显示）
                default_params=alg_info['default_params'],  # 默认参数配置
                param_help=alg_info['param_help']  # 参数帮助说明
            )

        self.init_ui()  # 初始化用户界面
        self.load_categories()  # 加载数据类别到界面

    def init_ui(self):  # 初始化用户界面的主方法
        """初始化整个GUI界面的布局和组件"""
        self.setWindowTitle("轨迹压缩算法课设")  # 设置窗口标题
        self.setGeometry(100, 100, 1400, 900)  # 设置窗口初始位置和大小（x=100,y=100,宽1400,高900）

        # 创建中央部件作为主窗口的中心组件
        central_widget = QWidget()  # 创建一个基础的QWidget作为容器
        self.setCentralWidget(central_widget)  # 将其设置为窗口的中央部件

        # 创建水平分隔器布局，将界面分为左右两部分，用户可以拖拽调整大小比例
        main_splitter = QSplitter(Qt.Horizontal, central_widget)  # 水平分割器，父控件为central_widget

        # 创建左侧控制面板（参数设置、算法选择等）
        self.create_control_panel()  # 调用方法创建控制面板
        main_splitter.addWidget(self.control_panel)  # 将控制面板添加到分割器的左侧

        # 创建右侧内容区域（可视化显示、评估指标等）
        self.create_content_area()  # 调用方法创建内容显示区域
        main_splitter.addWidget(self.content_area)  # 将内容区域添加到分割器的右侧

        # 设置初始大小比例 (左侧控制面板:右侧内容区域 = 1:2)
        main_splitter.setSizes([350, 700])  # 左侧350像素，右侧700像素
        # 设置最小宽度限制，避免用户拖拽时将控件压缩得过小而影响使用
        self.control_panel.setMinimumWidth(320)  # 控制面板最小宽度320像素
        self.content_area.setMinimumWidth(500)  # 内容区域最小宽度500像素

        # 将分隔器设置为中央部件的主布局
        main_layout = QVBoxLayout(central_widget)  # 创建垂直布局
        main_layout.addWidget(main_splitter)  # 将分隔器添加到布局中

    def create_control_panel(self):  # 创建左侧控制面板的方法
        """创建控制面板，包含数据集选择、算法选择、参数设置等所有控制组件"""
        self.control_panel = QWidget()  # 创建控制面板的容器控件
        self.control_panel.setMinimumWidth(320)  # 设置最小宽度，确保控件不会被压缩过小
        layout = QVBoxLayout(self.control_panel)  # 创建垂直布局管理器

        # 创建数据集选择分组框
        dataset_group = QGroupBox("数据集选择")  # 分组框提供视觉分组和标题
        dataset_layout = QVBoxLayout(dataset_group)  # 分组框内部使用垂直布局

        # 类别选择行：标签 + 下拉框的水平布局
        category_layout = QHBoxLayout()  # 水平布局放置标签和下拉框
        category_layout.addWidget(QLabel("类别:"))  # 添加"类别"标签
        self.category_combo = QComboBox()  # 创建类别选择下拉框
        self.category_combo.currentTextChanged.connect(self.on_category_changed)  # 连接选择变化信号到处理函数
        category_layout.addWidget(self.category_combo)  # 将下拉框添加到水平布局
        dataset_layout.addLayout(category_layout)  # 将类别选择行添加到数据集分组框

        # 数据集选择行：标签 + 下拉框的水平布局
        dataset_select_layout = QHBoxLayout()  # 水平布局
        dataset_select_layout.addWidget(QLabel("数据集:"))  # 添加"数据集"标签
        self.dataset_combo = QComboBox()  # 创建数据集选择下拉框
        # 尝试设置下拉列表的最大可见项数，避免选项过多时被截断（某些Qt版本可能不支持此方法）
        try:
            self.dataset_combo.setMaxVisibleItems(15)  # 设置最多显示15个选项
        except Exception:  # 如果设置失败（版本兼容性问题），跳过不处理
            pass  # 不做任何处理，继续执行
        self.dataset_combo.setMinimumWidth(220)  # 设置最小宽度，确保有足够空间显示数据集名称
        dataset_select_layout.addWidget(self.dataset_combo)  # 将下拉框添加到布局
        dataset_layout.addLayout(dataset_select_layout)  # 添加到数据集分组框

        layout.addWidget(dataset_group)  # 将完整的数据集选择分组框添加到控制面板主布局

        # 创建算法选择分组框
        algorithm_group = QGroupBox("算法选择")  # 分组框标题
        algorithm_layout = QVBoxLayout(algorithm_group)  # 垂直布局容纳所有算法选项

        # 添加原始轨迹复选框（始终可用，用于对比显示）
        self.original_checkbox = QCheckBox("原始轨迹")  # 创建复选框控件
        self.original_checkbox.setChecked(True)  # 默认选中，方便用户对比
        algorithm_layout.addWidget(self.original_checkbox)  # 添加到算法选择布局

        # 为每个可用算法创建复选框控件
        self.algorithm_checkboxes = {}  # 字典存储算法复选框，key为算法标识，value为QCheckBox对象
        for key, info in self.algorithms.items():  # 遍历所有加载的算法信息
            checkbox = QCheckBox(info.display_name)  # 使用算法的显示名称创建复选框
            checkbox.stateChanged.connect(self.update_parameter_panel)  # 连接状态变化信号到参数面板更新函数
            self.algorithm_checkboxes[key] = checkbox  # 存储复选框引用，便于后续操作
            algorithm_layout.addWidget(checkbox)  # 将复选框添加到布局

        # 设置默认选中的算法，给用户提供常用算法的预选
        for key in ['dr', 'adaptive_dr', 'dp']:  # 预选三种常用算法：距离阈值、适应性距离阈值、道格拉斯-普克
            if key in self.algorithm_checkboxes:  # 检查算法是否存在（防止配置变化导致的错误）
                self.algorithm_checkboxes[key].setChecked(True)  # 设置为选中状态

        layout.addWidget(algorithm_group)  # 将算法选择分组框添加到控制面板主布局

        # 创建参数设置面板（动态显示选中算法的参数）
        self.create_parameter_panel(layout)  # 调用方法创建参数设置区域

        # 创建操作按钮区域（计算结果、导出结果等）
        self.create_action_buttons(layout)  # 调用方法创建按钮控件

        # 创建进度条用于显示压缩处理的进度
        self.progress_bar = QProgressBar()  # 创建进度条控件
        self.progress_bar.setVisible(False)  # 初始隐藏，只在处理时显示
        layout.addWidget(self.progress_bar)  # 添加到主布局

        # 添加伸缩空间，将上面的控件推到顶部，剩余空间留空
        layout.addStretch()  # 添加弹性空间

        # 初始化参数面板，根据当前选中的算法显示对应参数控件
        self.update_parameter_panel()  # 调用方法初始化参数面板内容

    def create_parameter_panel(self, parent_layout):  # 创建参数设置面板的方法
        """创建参数设置面板，包含一个可滚动的区域用于显示算法参数控件"""
        param_group = QGroupBox("参数设置")  # 创建分组框，标题为"参数设置"
        param_layout = QVBoxLayout(param_group)  # 分组框内部使用垂直布局

        # 创建滚动区域，因为算法参数可能很多，需要滚动查看
        scroll_area = QScrollArea()  # 创建滚动区域控件
        scroll_widget = QWidget()  # 创建滚动区域内部的容器控件
        self.param_layout = QVBoxLayout(scroll_widget)  # 为容器创建垂直布局，用于放置参数控件

        # 参数控件存储字典：用于保存所有动态创建的参数输入控件，便于后续获取用户输入的值
        self.param_widgets = {}  # key格式为"alg_key_param_key"，value为对应的输入控件

        scroll_area.setWidget(scroll_widget)  # 将容器控件设置为滚动区域的内容
        scroll_area.setWidgetResizable(True)  # 允许滚动区域根据内容自动调整大小
        param_layout.addWidget(scroll_area)  # 将滚动区域添加到参数分组框布局

        parent_layout.addWidget(param_group)  # 将完整参数分组框添加到父布局

    def create_action_buttons(self, parent_layout):  # 创建操作按钮区域的方法
        """创建操作按钮区域，包含计算结果和导出结果两个主要功能按钮"""
        button_layout = QVBoxLayout()  # 创建垂直布局容纳按钮

        self.run_button = QPushButton("计算结果")  # 创建"计算结果"按钮，启动轨迹压缩处理
        self.run_button.clicked.connect(self.on_run_compression)  # 连接点击信号到压缩处理函数
        self.run_button.setMinimumHeight(35)  # 设置最小高度，使按钮更易点击
        button_layout.addWidget(self.run_button)  # 添加到按钮布局

        self.export_button = QPushButton("导出结果")  # 创建"导出结果"按钮，将结果保存到文件
        self.export_button.clicked.connect(self.on_export_results)  # 连接点击信号到导出处理函数
        self.export_button.setEnabled(False)  # 初始禁用，只有计算完成后才启用
        button_layout.addWidget(self.export_button)  # 添加到按钮布局

        parent_layout.addLayout(button_layout)  # 将按钮布局添加到父布局

    def create_content_area(self):  # 创建右侧内容显示区域的方法
        """创建内容显示区域，使用标签页形式组织可视化和评估结果"""
        self.content_area = QWidget()  # 创建内容区域的容器控件
        layout = QVBoxLayout(self.content_area)  # 设置垂直布局

        # 创建标签页控件，用于在轨迹可视化和评估指标之间切换
        self.tab_widget = QTabWidget()  # 创建多标签页控件

        # 创建轨迹可视化标签页
        self.create_visualization_tab()  # 调用方法创建可视化标签页内容
        self.tab_widget.addTab(self.visualization_tab, "轨迹")  # 添加为第一个标签页

        # 创建评估指标标签页
        self.create_metrics_tab()  # 调用方法创建评估指标标签页内容
        self.tab_widget.addTab(self.metrics_tab, "评估指标")  # 添加为第二个标签页

        layout.addWidget(self.tab_widget)  # 将标签页控件添加到内容区域布局

    def create_visualization_tab(self):  # 创建轨迹可视化标签页的方法
        """创建可视化标签页，包含Web视图用于显示交互式地图和轨迹"""
        self.visualization_tab = QWidget()  # 创建标签页容器
        layout = QVBoxLayout(self.visualization_tab)  # 设置垂直布局

        # 创建Web引擎视图用于显示地图可视化（使用Leaflet.js等Web技术）
        self.web_view = QWebEngineView()  # 创建Web视图控件
        # 设置大小策略为扩展模式，允许控件占用所有可用空间
        self.web_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.web_view.setMinimumHeight(500)  # 设置最小高度，确保有足够空间显示地图
        layout.addWidget(self.web_view)  # 添加Web视图到布局

        # 创建图层控制使用说明标签（目前被注释掉，可能影响界面简洁性）
        control_label = QLabel("使用地图右上角的图层控制来切换显示不同轨迹")  # 说明文本
        control_label.setStyleSheet("color: #666; font-style: italic;")  # 设置灰色斜体样式
        # layout.addWidget(control_label)  # 已注释，不显示说明文本

        # 设置布局的伸缩因子，web_view占据8份空间，说明文本（如果显示）占据0份空间
        layout.setStretch(0, 8)  # web_view占据主要空间
        layout.setStretch(1, 0)  # control_label（如果添加）不占伸缩空间

    def create_metrics_tab(self):  # 创建评估指标标签页的方法
        """创建评估指标标签页，显示各算法的性能对比表格"""
        self.metrics_tab = QWidget()  # 创建标签页容器
        layout = QVBoxLayout(self.metrics_tab)  # 设置垂直布局

        # 创建表格控件用于显示算法评估指标
        self.metrics_table = QTableWidget()  # 创建表格控件
        self.metrics_table.setColumnCount(7)  # 设置7列，对应不同的评估指标
        self.metrics_table.setHorizontalHeaderLabels([  # 设置表头标签
            '算法', '压缩率', 'SED均值', 'SED最大', 'SED_95%', '事件保留', '综合得分'  # 7个指标列
        ])
        # 设置列宽模式为Stretch，使所有列平均分布宽度
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.metrics_table)  # 将表格添加到布局

    def load_categories(self):  # 加载数据类别到界面
        """加载AIS数据集的类别信息，填充类别选择下拉框"""
        categories = scan_categories(self.data_root)  # 调用工具函数扫描数据根目录，获取所有类别
        self.category_combo.clear()  # 清空类别下拉框
        self.category_combo.addItem("请选择类别", "")  # 添加默认的提示选项，值为空字符串
        for category in categories:  # 遍历所有找到的类别
            self.category_combo.addItem(category, category)  # 添加类别项，显示文本和数据都是类别名

    def on_category_changed(self, category):  # 类别选择变化的事件处理函数
        """当用户选择类别时，动态加载该类别下的数据集文件列表"""
        if not category:  # 如果选择的是空类别（默认提示选项）
            self.dataset_combo.clear()  # 清空数据集下拉框
            return  # 结束处理

        # 为避免界面卡顿，只加载每个类别的部分文件（前10个），提高响应速度
        datasets = scan_datasets(os.path.join(self.data_root, category))[:10]  # 获取前10个数据集文件

        self.dataset_combo.clear()  # 清空数据集下拉框
        self.dataset_combo.addItem("请选择数据集", "")  # 添加默认提示选项
        for filename, point_count in datasets:  # 遍历获取的数据集
            display_text = f"{Path(filename).stem} ({point_count} pts)"  # 格式化显示文本：文件名(点数)
            self.dataset_combo.addItem(display_text, filename)  # 添加选项，显示文本和实际文件名

        # 如果该类别下有更多文件，在下拉框的工具提示中显示说明
        if len(scan_datasets(os.path.join(self.data_root, category))) > len(datasets):  # 检查是否有更多文件
            self.dataset_combo.setToolTip("仅显示前10个文件；若需更多，请到数据目录查看或重启以加载全部。")  # 设置提示文本

    def update_parameter_panel(self):  # 更新参数设置面板的方法
        """根据当前选中的算法动态更新参数面板，显示对应算法的参数输入控件"""
        # 检查参数布局是否已经初始化（防止在界面初始化前调用）
        if not hasattr(self, 'param_layout'):  # 如果param_layout属性不存在
            return  # 直接返回，不进行更新

        # 定义嵌套函数：递归清空布局中的所有控件，避免内存泄漏和界面错乱
        def _clear_layout(layout):  # 内部辅助函数，递归清理布局
            while layout.count():  # 当布局中还有项目时循环
                item = layout.takeAt(0)  # 移除第一个项目（takeAt返回QLayoutItem对象）
                if item is None:  # 如果项目为空，跳过
                    continue
                widget = item.widget()  # 获取项目中的控件
                if widget is not None:  # 如果是控件
                    widget.deleteLater()  # 标记控件为稍后删除，释放内存
                else:  # 如果是子布局
                    child_layout = item.layout()  # 获取子布局
                    if child_layout is not None:  # 如果子布局存在
                        _clear_layout(child_layout)  # 递归清理子布局

        _clear_layout(self.param_layout)  # 调用清理函数清空现有参数控件
        self.param_widgets.clear()  # 清空参数控件字典

        # 为所有选中的算法添加对应的参数输入控件
        for alg_key, checkbox in self.algorithm_checkboxes.items():  # 遍历所有算法复选框
            if checkbox.isChecked():  # 如果该算法被选中
                alg_info = self.algorithms[alg_key]  # 获取算法信息对象
                self.add_algorithm_parameters(alg_key, alg_info)  # 调用方法添加该算法的参数控件

    def add_algorithm_parameters(self, alg_key: str, alg_info: AlgorithmInfo):  # 添加单个算法参数控件的方法
        """为指定的算法创建参数输入控件，并添加到参数面板"""
        # 创建算法标题标签，直接使用算法显示名称作为标题
        title_label = QLabel(f"{alg_info.display_name}")  # 创建标签显示算法名
        title_label.setStyleSheet("font-weight: bold; margin-top: 10px;")  # 设置粗体和上边距样式
        self.param_layout.addWidget(title_label)  # 将标题标签添加到参数布局

        # 为算法的每个参数创建输入控件
        for param_key, default_value in alg_info.default_params.items():  # 遍历算法的所有默认参数
            param_layout = QHBoxLayout()  # 为每个参数创建水平布局（标签+输入框）

            # 创建参数标签，优先使用帮助文本，否则使用参数键名
            label_text = alg_info.param_help.get(param_key, param_key)  # 获取参数说明文本
            label = QLabel(f"  {label_text}:")  # 创建标签，前面加两个空格作为缩进
            param_layout.addWidget(label)  # 将标签添加到参数布局

            # 根据参数默认值的类型创建相应的输入控件
            if isinstance(default_value, int):  # 如果是整数参数
                widget = QSpinBox()  # 创建整数微调框
                widget.setValue(int(default_value))  # 设置默认值
                widget.setMinimum(1)  # 设置最小值
                widget.setMaximum(10000)  # 设置最大值
            elif isinstance(default_value, float):  # 如果是浮点数参数
                widget = QDoubleSpinBox()  # 创建浮点数微调框
                widget.setValue(float(default_value))  # 设置默认值
                widget.setDecimals(4)  # 设置小数位数
                widget.setMinimum(0.0001)  # 设置最小值
                widget.setMaximum(10000.0)  # 设置最大值
            else:  # 如果是其他类型（字符串等）
                widget = QLineEdit(str(default_value))  # 创建文本输入框

            widget.setObjectName(f"{alg_key}_{param_key}")  # 设置控件对象名，便于后续查找
            param_layout.addWidget(widget)  # 将输入控件添加到参数布局

            self.param_widgets[f"{alg_key}_{param_key}"] = widget  # 将控件存储到字典中
            self.param_layout.addLayout(param_layout)  # 将整个参数行添加到主参数布局

    def on_run_compression(self):  # "计算结果"按钮的点击事件处理函数
        """启动轨迹压缩计算过程，包括数据验证、加载和后台处理"""
        # 验证用户是否选择了数据集
        if self.dataset_combo.currentData() is None:  # 检查数据集下拉框是否有有效选择
            QMessageBox.warning(self, "警告", "请选择一个数据集")  # 显示警告对话框
            return  # 结束处理

        # 收集用户选择的算法列表
        selected_algorithms = []  # 初始化选中算法列表
        for alg_key, checkbox in self.algorithm_checkboxes.items():  # 遍历所有算法复选框
            if checkbox.isChecked():  # 如果该算法被选中
                selected_algorithms.append(alg_key)  # 添加到选中列表

        # 验证至少选择了一种显示内容（原始轨迹或算法）
        if not selected_algorithms and not self.original_checkbox.isChecked():  # 如果既没有选算法也没有选原始轨迹
            QMessageBox.warning(self, "警告", "请至少选择原始轨迹或一个算法")  # 显示警告
            return  # 结束处理

        # 获取用户选择的数据集信息
        category = self.category_combo.currentData()  # 获取选择的类别
        filename = self.dataset_combo.currentData()  # 获取选择的文件名

        # 加载AIS数据集文件
        try:
            self.current_df = load_ais_dataset(self.data_root, category, filename)  # 调用加载函数
            print(f"加载数据集: {category}/{filename}, 点数: {len(self.current_df)}")  # 控制台输出加载信息
        except Exception as e:  # 捕获加载过程中的异常
            QMessageBox.critical(self, "错误", f"加载数据集失败: {str(e)}")  # 显示错误对话框
            return  # 结束处理

        # 收集所有选中算法的参数设置值
        algorithm_params = {}  # 初始化算法参数字典
        for alg_key in selected_algorithms:  # 遍历每个选中的算法
            params = {}  # 为当前算法创建参数字典
            for param_key in self.algorithms[alg_key].default_params.keys():  # 遍历算法的所有参数
                widget_key = f"{alg_key}_{param_key}"  # 构造控件键名
                if widget_key in self.param_widgets:  # 如果该参数控件存在
                    widget = self.param_widgets[widget_key]  # 获取参数输入控件
                    if isinstance(widget, QSpinBox):  # 如果是整数输入框
                        params[param_key] = widget.value()  # 获取整数值
                    elif isinstance(widget, QDoubleSpinBox):  # 如果是浮点数输入框
                        params[param_key] = widget.value()  # 获取浮点数值
                    elif isinstance(widget, QLineEdit):  # 如果是文本输入框
                        params[param_key] = widget.text()  # 获取文本值
            algorithm_params[alg_key] = params  # 将算法参数添加到总字典

        # 启动后台压缩处理线程
        self.progress_bar.setVisible(True)  # 显示进度条
        self.progress_bar.setRange(0, 0)  # 设置为不确定进度模式（循环动画）
        self.run_button.setEnabled(False)  # 禁用计算按钮，防止重复点击

        # 创建并启动后台工作线程
        self.worker = CompressionWorker(  # 创建压缩工作线程对象
            self.current_df,  # 传递当前加载的数据集
            selected_algorithms,  # 传递选中的算法列表
            algorithm_params,  # 传递算法参数配置
            self.original_checkbox.isChecked()  # 传递是否包含原始轨迹的设置
        )
        self.worker.progress.connect(self.on_progress)  # 连接进度信号到处理函数
        self.worker.finished.connect(self.on_compression_finished)  # 连接完成信号到处理函数
        self.worker.error.connect(self.on_compression_error)  # 连接错误信号到处理函数
        self.worker.start()  # 启动后台工作线程

    def on_progress(self, message):  # 进度更新信号的处理函数
        """接收后台线程的进度信息并输出到控制台"""
        print(f"进度: {message}")  # 在控制台打印进度消息

    def on_compression_finished(self, results):  # 压缩完成信号的处理函数
        """处理压缩任务完成事件，更新界面状态并触发后续处理"""
        self.compression_results = results  # 保存压缩结果
        self.progress_bar.setVisible(False)  # 隐藏进度条
        self.run_button.setEnabled(True)  # 重新启用计算按钮
        self.export_button.setEnabled(True)  # 启用导出按钮（现在有结果可导出了）

        # 生成轨迹可视化结果
        self.generate_visualization()  # 调用可视化生成方法

        # 计算并显示各算法的评估指标
        self.calculate_and_display_metrics()  # 调用指标计算和显示方法

        QMessageBox.information(self, "完成", "轨迹压缩完成！")  # 显示完成提示对话框

    def on_compression_error(self, error_msg):  # 压缩错误信号的处理函数
        """处理压缩过程中的错误，恢复界面状态并显示错误信息"""
        self.progress_bar.setVisible(False)  # 隐藏进度条
        self.run_button.setEnabled(True)  # 重新启用计算按钮
        QMessageBox.critical(self, "错误", f"压缩过程中出错: {error_msg}")  # 显示错误对话框

    def generate_visualization(self):  # 生成轨迹可视化的主方法
        """生成压缩结果的轨迹可视化，包括原始轨迹和各算法的压缩结果"""
        if not self.compression_results:  # 如果没有压缩结果
            return  # 直接返回，不进行可视化

        # 创建临时PNG文件用于存储可视化结果
        import tempfile  # 导入临时文件模块
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)  # 创建临时PNG文件，不自动删除
        temp_file.close()  # 关闭文件句柄，释放资源

        try:
            # 调用可视化创建方法，生成多轨迹对比图
            self.create_interactive_visualization(temp_file.name)  # 传入临时文件路径

            # 在可视化标签页中显示生成的图像
            self.display_image_in_tab(temp_file.name)  # 调用显示方法

            # 自动切换到可视化标签页，让用户看到结果
            self.tab_widget.setCurrentIndex(0)  # 设置当前标签页为第一个（轨迹可视化）

        except Exception as e:  # 捕获可视化过程中的异常
            QMessageBox.critical(self, "错误", f"生成可视化失败: {str(e)}")  # 显示错误对话框

    def display_image_in_tab(self, image_path: str):  # 在可视化标签页中显示图像的方法
        """在可视化标签页中显示轨迹图像，支持鼠标滚轮缩放和控制按钮操作"""
        # 导入所需的PyQt5组件
        from PyQt5.QtGui import QPixmap, QPainter  # 图像处理和绘制
        from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QVBoxLayout, QWidget, QHBoxLayout, QPushButton, QLabel  # 图形视图和控件
        from PyQt5.QtCore import Qt  # Qt常量

        # 定义自定义的可缩放图形视图类，支持鼠标滚轮缩放
        class ZoomableGraphicsView(QGraphicsView):  # 继承自QGraphicsView
            def __init__(self):
                super().__init__()  # 调用父类构造函数
                self._zoom_factor = 1.0  # 初始化缩放因子

            def wheelEvent(self, event):  # 重写滚轮事件处理方法
                # 根据滚轮方向确定缩放因子（向上滚放大，向下滚缩小）
                factor = 1.2 if event.angleDelta().y() > 0 else 1/1.2  # 放大因子或缩小因子
                self.scale(factor, factor)  # 对视图应用缩放变换
                self._zoom_factor *= factor  # 更新缩放因子

        # 创建主容器控件
        main_widget = QWidget()  # 创建主控件
        main_layout = QVBoxLayout(main_widget)  # 设置垂直布局

        # 创建缩放控制按钮的水平布局
        control_layout = QHBoxLayout()  # 水平布局容纳控制按钮
        zoom_in_btn = QPushButton("放大")  # 放大按钮
        zoom_out_btn = QPushButton("缩小")  # 缩小按钮
        fit_btn = QPushButton("适应")  # 适应窗口按钮
        # reset_btn = QPushButton("重置缩放")  # 重置缩放按钮（已注释）

        # 设置按钮的最大宽度，保持界面整洁
        zoom_in_btn.setMaximumWidth(80)
        zoom_out_btn.setMaximumWidth(80)
        fit_btn.setMaximumWidth(80)
        # reset_btn.setMaximumWidth(80)

        # 将按钮添加到控制布局
        control_layout.addWidget(zoom_in_btn)
        control_layout.addWidget(zoom_out_btn)
        control_layout.addWidget(fit_btn)
        # control_layout.addWidget(reset_btn)
        control_layout.addStretch()  # 添加伸缩空间，将按钮推到左侧

        main_layout.addLayout(control_layout)  # 将控制按钮布局添加到主布局

        # 创建图形视图和场景系统用于显示图像
        graphics_view = ZoomableGraphicsView()  # 创建自定义的可缩放图形视图
        scene = QGraphicsScene()  # 创建图形场景
        graphics_view.setScene(scene)  # 将场景设置到视图中

        # 设置视图的渲染属性以获得更好的显示效果
        graphics_view.setRenderHint(QPainter.Antialiasing)  # 启用抗锯齿渲染
        graphics_view.setRenderHint(QPainter.SmoothPixmapTransform)  # 启用平滑像素变换
        graphics_view.setDragMode(QGraphicsView.ScrollHandDrag)  # 设置拖拽模式为手型拖拽
        graphics_view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)  # 缩放中心为鼠标位置
        graphics_view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)  # 调整大小时鼠标位置不变

        # 加载并显示图像
        pixmap = QPixmap(image_path)  # 从文件路径创建像素图
        if pixmap.isNull():  # 检查图像是否加载成功
            error_label = QLabel("图片加载失败")  # 创建错误提示标签
            main_layout.addWidget(error_label)  # 添加错误标签到布局
        else:  # 图像加载成功
            pixmap_item = QGraphicsPixmapItem(pixmap)  # 创建像素图图形项
            scene.addItem(pixmap_item)  # 将图像项添加到场景中

            # 初始时适应窗口大小显示
            graphics_view.fitInView(pixmap_item, Qt.KeepAspectRatio)  # 适应视图，保持宽高比

            main_layout.addWidget(graphics_view)  # 将图形视图添加到主布局

            # 连接控制按钮的点击信号到对应的缩放函数
            def zoom_in():  # 放大函数
                factor = 1.2  # 放大因子
                graphics_view.scale(factor, factor)  # 对视图应用放大缩放

            def zoom_out():  # 缩小函数
                factor = 1/1.2  # 缩小因子
                graphics_view.scale(factor, factor)  # 对视图应用缩小缩放

            def fit_to_window():  # 适应窗口函数
                graphics_view.fitInView(pixmap_item, Qt.KeepAspectRatio)  # 适应视图大小，保持比例

            def reset_zoom():  # 重置缩放函数（已注释）
                graphics_view.resetTransform()  # 重置所有变换
                graphics_view.fitInView(pixmap_item, Qt.KeepAspectRatio)  # 重新适应窗口

            # 连接按钮点击信号到相应的处理函数
            zoom_in_btn.clicked.connect(zoom_in)
            zoom_out_btn.clicked.connect(zoom_out)
            fit_btn.clicked.connect(fit_to_window)
            # reset_btn.clicked.connect(reset_zoom)

        # 替换可视化标签页的内容（更新显示）
        if self.tab_widget.count() > 0:  # 如果标签页已存在
            self.tab_widget.removeTab(0)  # 移除第一个标签页
        self.tab_widget.insertTab(0, main_widget, "轨迹可视化")  # 插入新的可视化标签页
        self.tab_widget.setCurrentIndex(0)  # 设置为当前活动标签页

    def create_interactive_visualization(self, output_file: str):  # 创建交互式轨迹可视化的方法
        """创建多轨迹对比的可视化图像，显示原始轨迹和各压缩算法的结果"""
        from algcode.utils.visualization import visualize_multiple_trajectories  # 导入可视化函数

        # 准备轨迹数据字典，组织成可视化函数所需的格式
        trajectories = {}  # 初始化轨迹字典
        for name, df in self.compression_results.items():  # 遍历所有压缩结果
            if name == 'original':  # 如果是原始轨迹
                trajectories['原始轨迹'] = df  # 使用固定名称
            else:  # 如果是压缩算法结果
                # 获取算法的显示名称，用于在可视化中标识轨迹
                alg_info = self.algorithms.get(name)  # 从算法信息字典获取算法信息
                display_name = alg_info.display_name if alg_info else name  # 使用显示名称或键名作为后备
                trajectories[display_name] = df  # 添加到轨迹字典

        # 调用可视化函数生成多轨迹对比图像
        visualize_multiple_trajectories(trajectories, output_file)  # 生成可视化并保存到指定文件

    def calculate_and_display_metrics(self):  # 计算并显示各算法评估指标的方法
        """计算所有压缩算法的性能指标并在表格中显示，按综合得分排序"""
        # 安全检查数据有效性，避免对空的DataFrame进行布尔检查（pandas会抛出ValueError）
        if self.current_df is None or (hasattr(self.current_df, 'empty') and self.current_df.empty) or not self.compression_results:
            return  # 如果数据无效，直接返回

        self.metrics_table.setRowCount(0)  # 清空表格，准备填充新数据

        # 收集所有算法的评估指标，计算综合得分后排序显示
        metrics_list = []  # 初始化指标列表，用于存储(算法名, 指标字典)元组
        for alg_name, compressed_df in self.compression_results.items():  # 遍历所有压缩结果
            if alg_name == 'original':  # 跳过原始轨迹（不需要评估）
                continue
            try:  # 尝试计算该算法的评估指标
                # 调用评估函数计算各项指标（相似度、压缩率、事件保留等）
                metrics = evaluate_compression(self.current_df, compressed_df, alg_name, 0.0, True)

                # 计算综合得分：结合相似度(60%)、事件保留(30%)，减去压缩率惩罚(10%)
                # composite = 0.6 * similarity + 0.3 * event_recall - 0.1 * (compression_ratio / 100)
                sim = float(metrics.get('trajectory_similarity', 0.0))  # 轨迹相似度
                event_rec = float(metrics.get('event_recall', 0.0))  # 事件保留率
                compression_ratio = float(metrics.get('compression_ratio', 0.0))  # 压缩率(0-100)
                composite = 0.6 * sim + 0.3 * event_rec - 0.1 * (compression_ratio / 100.0)  # 综合得分计算
                metrics['composite_score'] = composite  # 添加综合得分到指标字典

                metrics_list.append((alg_name, metrics))  # 将(算法名, 指标)添加到列表

            except Exception as e:  # 捕获指标计算异常
                print(f"计算 {alg_name} 指标失败: {e}")  # 输出错误信息到控制台

        # 按综合得分降序排序（得分高的算法排在前面），然后填充到表格中
        metrics_list.sort(key=lambda x: x[1].get('composite_score', 0.0), reverse=True)  # 降序排序
        for alg_name, metrics in metrics_list:  # 遍历排序后的指标列表
            row = self.metrics_table.rowCount()  # 获取当前行数
            self.metrics_table.insertRow(row)  # 插入新行
            try:  # 尝试填充表格单元格
                # 依次填充各列：算法名、压缩率、SED均值、SED最大值、SED_95%、事件保留率、综合得分
                self.metrics_table.setItem(row, 0, QTableWidgetItem(alg_name))  # 算法名称
                self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{metrics.get('compression_ratio', 0.0):.1f}"))  # 压缩率（1位小数）
                self.metrics_table.setItem(row, 2, QTableWidgetItem(f"{metrics.get('sed_mean', 0.0):.2f}"))  # SED均值（2位小数）
                self.metrics_table.setItem(row, 3, QTableWidgetItem(f"{metrics.get('sed_max', 0.0):.2f}"))  # SED最大值（2位小数）
                self.metrics_table.setItem(row, 4, QTableWidgetItem(f"{metrics.get('sed_p95', 0.0):.2f}"))  # SED 95%分位数（2位小数）
                self.metrics_table.setItem(row, 5, QTableWidgetItem(f"{metrics.get('event_recall', 0.0):.3f}"))  # 事件保留率（3位小数）
                self.metrics_table.setItem(row, 6, QTableWidgetItem(f"{metrics.get('composite_score', 0.0):.5f}"))  # 综合得分（5位小数）
            except Exception as e:  # 捕获表格填充异常
                print(f"填充表格时出错 {alg_name}: {e}")  # 输出错误信息

    def on_export_results(self):  # 导出结果按钮的点击处理函数
        """将压缩结果导出为CSV文件到用户选择的目录"""
        if not self.compression_results:  # 检查是否有结果可导出
            QMessageBox.warning(self, "警告", "没有可导出的结果")  # 显示警告对话框
            return  # 结束处理

        # 让用户选择导出目录
        export_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")  # 打开目录选择对话框
        if not export_dir:  # 如果用户取消选择
            return  # 结束处理

        try:  # 尝试执行导出操作
            # 遍历所有压缩结果，为每个算法/原始轨迹创建CSV文件
            for alg_name, df in self.compression_results.items():  # 遍历结果字典
                if alg_name == 'original':  # 如果是原始轨迹
                    filename = "original_trajectory.csv"  # 使用固定文件名
                else:  # 如果是压缩算法结果
                    filename = f"{alg_name}_compressed.csv"  # 使用算法名作为文件名

                filepath = os.path.join(export_dir, filename)  # 构建完整的文件路径
                df.to_csv(filepath, index=False)  # 导出DataFrame为CSV文件，不包含索引列

            QMessageBox.information(self, "成功", f"结果已导出到: {export_dir}")  # 显示成功提示

        except Exception as e:  # 捕获导出过程中的异常
            QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")  # 显示错误对话框


def main_gui():  # GUI模式的主入口函数
    """GUI 模式入口：创建并运行轨迹压缩主窗口"""
    app = QApplication(sys.argv)  # 创建PyQt5应用程序对象，传入命令行参数
    app.setApplicationName("轨迹压缩算法课设")  # 设置应用程序名称

    window = TrajectoryCompressionGUI()  # 创建主窗口实例
    window.show()  # 显示主窗口
    sys.exit(app.exec_())  # 进入应用程序事件循环，程序结束时退出


