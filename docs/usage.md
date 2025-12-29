# 轨迹压缩算法对比平台使用指南

## 概述

这是一个基于多种轨迹压缩算法的在线对比平台，支持上传 AIS 船舶轨迹数据，并使用不同的算法进行压缩和性能评估。

## 系统架构

- **前端**: 原生 HTML/CSS/JavaScript，采用蓝白色 tdesign 设计风格
- **后端**: Python FastAPI，提供 REST API 接口
- **算法**: 手工实现的 5 种轨迹压缩算法

## 支持的算法

### 1. 自适应 Dead Reckoning (DR)
- **原理**: 基于航位推算，根据当前航速动态调整距离阈值
- **参数**:
  - `min_threshold`: 最低距离阈值（米）
  - `max_threshold`: 最高距离阈值（米）
  - `v_lower`: 低速截止点（节）
  - `v_upper`: 高速截止点（节）

### 2. 固定阈值 Dead Reckoning
- **原理**: 使用固定距离阈值的 DR 算法
- **参数**:
  - `epsilon`: 距离阈值（米）

### 3. 语义增强 Dead Reckoning
- **原理**: 在 DR 基础上增加航行事件约束（转弯、速度变化、时间间隔）
- **参数**:
  - DR 参数 + `cog_threshold`（转向阈值°）、`sog_threshold`（速度变化阈值节）、`time_threshold`（时间间隔秒）

### 4. Sliding Window
- **原理**: 以最后一个保留点为起点，不断往前看新点，偏差超阈值则保留上一点
- **参数**:
  - `epsilon`: 距离阈值（米）

### 5. Opening Window
- **原理**: 从锚点开始扩展窗口，尽可能包含更多点直到超出误差
- **参数**:
  - `epsilon`: 距离阈值（米）

### 6. SQUISH
- **原理**: 维护固定大小缓冲区，持续删除最不重要点
- **参数**:
  - `buffer_size`: 缓冲区大小

## 安装和运行

### 环境要求
- Python 3.8+
- Node.js（可选，用于前端开发）

### 安装依赖
```bash
cd backend
pip install -r requirements.txt
```

### 运行后端服务
```bash
cd backend
python run_server.py
```

服务将在 `http://localhost:8000` 启动，API 文档位于 `http://localhost:8000/docs`

### 运行前端
直接在浏览器中打开 `frontend/index.html`，或使用本地服务器：
```bash
cd frontend
python -m http.server 8080
```

然后访问 `http://localhost:8080`

## 数据集管理

### 支持的数据格式
- CSV 格式，必需列：`MMSI`, `BaseDateTime`, `LAT`, `LON`, `SOG`, `COG`
- 时间格式：ISO 格式或 pandas 可识别格式
- 坐标系：WGS84

### 数据集目录结构
```
AIS Dataset/
├── Tugboat/
│   ├── 220584000.csv
│   └── ...
├── Cargo/
│   └── ...
└── Passenger/
    └── ...
```

### 添加新数据集
1. 将 CSV 文件放入相应的船型目录
2. 重启后端服务
3. 前端将自动发现新数据集

## 使用流程

### 1. 选择数据集
- 从下拉列表选择数据集
- 可选：输入特定 MMSI 筛选船舶
- 可选：选择速度段（低速/中速/高速）
- 可选：设置最大采样数量

### 2. 选择算法
- 勾选要对比的算法
- 为每个算法设置参数（自动显示相应参数面板）

### 3. 运行压缩
- 点击"运行压缩"按钮
- 等待任务完成（可查看进度）

### 4. 查看结果
- **性能对比表**: 压缩率、运行时间、SED 指标、事件保留率
- **可视化地图**: 支持图层开关，查看不同算法的压缩效果
- **导出功能**: 下载 CSV 格式的结果数据

## 评估指标

### 基本指标
- **压缩率**: (1 - 压缩点数/原始点数) × 100%
- **运行时间**: 算法执行时间（毫秒）

### SED 指标 (Spatial Error Distance)
- **Mean**: 平均空间误差距离
- **Max**: 最大空间误差距离
- **P95**: 95% 分位数误差距离

### 语义指标
- **事件保留率**: 转向事件（|ΔCOG|>20°）保留比例

## API 接口

### GET /datasets
获取可用数据集列表
```json
{
  "datasets": [
    {
      "name": "tugboat_sample",
      "path": "AIS Dataset/Tugboat/220584000.csv",
      "ship_type": "Tugboat",
      "sample_size": 1000,
      "time_range": ["2021-01-01", "2021-01-02"],
      "metadata": {
        "speed_distribution": {"low": 40.0, "medium": 45.0, "high": 15.0}
      }
    }
  ]
}
```

### POST /run
提交压缩任务
```json
{
  "dataset_name": "tugboat_sample",
  "algorithms": ["dr", "fixed_dr", "semantic_dr"],
  "params": {
    "dr": {"min_threshold": 20.0, "max_threshold": 500.0},
    "fixed_dr": {"epsilon": 100.0}
  },
  "mmsi": 220584000,
  "speed_segment": "medium",
  "max_samples": 1000
}
```

### GET /jobs/{job_id}
查询任务状态
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "progress": 100.0,
  "message": "所有算法执行完成"
}
```

### GET /results/{job_id}
获取压缩结果（包含轨迹数据）
```json
{
  "results": [
    {
      "algorithm": "dr",
      "compression_ratio": 65.2,
      "elapsed_time": 0.123,
      "metrics": {...},
      "compressed_trajectory": [
        {"timestamp": "2021-01-01T00:00:00", "lat": 34.6, "lon": -77.0, ...}
      ]
    }
  ]
}
```

### GET /metrics/{job_id}
获取评估指标
```json
{
  "metrics": [
    {
      "algorithm": "dr",
      "compression_ratio": 65.2,
      "elapsed_time": 123,
      "sed_mean": 45.2,
      "sed_max": 156.8,
      "sed_p95": 89.3,
      "event_recall": 0.85
    }
  ]
}
```

## 算法参数推荐

### 通用设置
- **低速场景** (0-3节): min_threshold=10-20m, max_threshold=100-200m
- **中速场景** (3-12节): min_threshold=20-50m, max_threshold=200-500m
- **高速场景** (12+节): min_threshold=50-100m, max_threshold=500-1000m

### 语义增强参数
- **转向阈值**: 10-30°（根据船舶类型调整）
- **速度变化阈值**: 1-3节
- **时间间隔阈值**: 300-600秒（5-10分钟）

### 几何算法参数
- **epsilon**: 50-200m（根据数据密度调整）
- **buffer_size**: 50-200（根据内存限制调整）

## 故障排除

### 常见问题

1. **前端无法连接后端**
   - 检查后端服务是否运行在 8000 端口
   - 检查防火墙设置

2. **数据集加载失败**
   - 检查 CSV 文件格式是否正确
   - 检查文件路径和权限

3. **算法执行失败**
   - 检查参数设置是否合理
   - 查看后端日志以获取详细错误信息

4. **地图显示异常**
   - 检查轨迹数据坐标是否有效
   - 尝试刷新页面

### 性能优化

- **大数据集**: 设置 `max_samples` 限制处理点数
- **内存优化**: SQUISH 算法的 `buffer_size` 不宜过大
- **并发处理**: 后端支持多算法并行执行

## 开发和扩展

### 添加新算法
1. 在 `backend/algorithms.py` 中实现算法函数
2. 添加到 `compress` 函数的算法映射
3. 在前端添加相应的参数面板和图层控制

### 自定义评估指标
在 `evaluate_compression` 函数中添加新的指标计算

### 数据集预处理
扩展 `backend/data_loader.py` 以支持更多数据格式和预处理逻辑

## 许可证

本项目采用 MIT 许可证。



