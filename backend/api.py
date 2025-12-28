"""
FastAPI 后端服务
提供轨迹压缩算法的 REST API 接口

作者: Algorithm Engineer
"""

import uuid
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd

try:
    # 尝试相对导入（当作为包运行时）
    from .algorithms import compress
    from .data_loader import DataLoader
except ImportError:
    # 如果相对导入失败，使用绝对导入（当直接运行脚本时）
    from algorithms import compress
    from data_loader import DataLoader


# ============================================================================
# 数据模型 (Data Models)
# ============================================================================

class AlgorithmParams(BaseModel):
    """算法参数模型"""
    # DR 相关参数
    min_threshold: Optional[float] = Field(20.0, description="最低距离阈值（米）")
    max_threshold: Optional[float] = Field(500.0, description="最高距离阈值（米）")
    v_lower: Optional[float] = Field(3.0, description="低速截止点（节）")
    v_upper: Optional[float] = Field(20.0, description="高速截止点（节）")

    # 固定阈值 DR 参数
    epsilon: Optional[float] = Field(100.0, description="固定距离阈值（米）")

    # 语义增强 DR 参数
    cog_threshold: Optional[float] = Field(10.0, description="转向角度阈值（度）")
    sog_threshold: Optional[float] = Field(1.0, description="速度变化阈值（节）")
    time_threshold: Optional[float] = Field(300.0, description="时间间隔阈值（秒）")

    # 几何算法参数
    buffer_size: Optional[int] = Field(100, description="SQUISH 缓冲区大小")


class CompressionRequest(BaseModel):
    """压缩请求模型"""
    dataset_name: str = Field(..., description="数据集名称")
    algorithms: List[str] = Field(..., description="要运行的算法列表")
    params: Dict[str, AlgorithmParams] = Field(default_factory=dict, description="各算法参数")
    mmsi: Optional[int] = Field(None, description="特定船舶 MMSI")
    speed_segment: Optional[str] = Field(None, description="速度段筛选")
    max_samples: Optional[int] = Field(None, description="最大采样数量")


class JobStatus(BaseModel):
    """任务状态模型"""
    job_id: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    created_at: datetime
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    message: Optional[str] = None


class CompressionResult(BaseModel):
    """压缩结果模型"""
    algorithm: str
    compression_ratio: float
    elapsed_time: float
    metrics: Dict[str, Any]
    compressed_trajectory: List[Dict[str, Any]]


# ============================================================================
# 全局变量和任务管理 (Global Variables & Task Management)
# ============================================================================

app = FastAPI(
    title="轨迹压缩算法服务",
    description="基于多种算法的在线轨迹压缩服务",
    version="1.0.0"
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],  # 前端地址
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有请求头
)

# 数据加载器 - 限制数据集数量以提高启动速度
# 只加载前100个数据集作为示例
data_loader = DataLoader(max_datasets=100)

# 任务存储
jobs: Dict[str, JobStatus] = {}
results: Dict[str, List[CompressionResult]] = {}

# 线程池执行器
executor = ThreadPoolExecutor(max_workers=4)


# ============================================================================
# 辅助函数 (Helper Functions)
# ============================================================================

def get_algorithm_default_params(algorithm: str) -> Dict[str, Any]:
    """获取算法的默认参数"""
    defaults = {
        'dr': {
            'min_threshold': 20.0,
            'max_threshold': 500.0,
            'v_lower': 3.0,
            'v_upper': 20.0
        },
        'fixed_dr': {
            'epsilon': 100.0
        },
        'semantic_dr': {
            'min_threshold': 20.0,
            'max_threshold': 500.0,
            'v_lower': 3.0,
            'v_upper': 20.0,
            'cog_threshold': 10.0,
            'sog_threshold': 1.0,
            'time_threshold': 300.0
        },
        'sliding': {
            'epsilon': 100.0
        },
        'opening': {
            'epsilon': 100.0
        },
        'squish': {
            'buffer_size': 100
        }
    }
    return defaults.get(algorithm, {})


def run_compression_job(job_id: str, request: CompressionRequest):
    """后台执行压缩任务"""
    try:
        # 更新任务状态
        jobs[job_id].status = 'running'
        jobs[job_id].message = '正在加载数据集...'

        # 加载数据集
        df = data_loader.load_dataset(
            dataset_name=request.dataset_name,
            mmsi=request.mmsi,
            speed_segment=request.speed_segment,
            max_samples=request.max_samples
        )

        jobs[job_id].message = f'数据集加载完成，共 {len(df)} 个点'

        # 执行所有算法
        compression_results = []
        total_algorithms = len(request.algorithms)

        for i, algorithm in enumerate(request.algorithms):
            jobs[job_id].progress = (i / total_algorithms) * 100
            jobs[job_id].message = f'正在运行算法: {algorithm}'

            # 获取算法参数
            params = request.params.get(algorithm, AlgorithmParams())
            param_dict = params.dict(exclude_unset=True)

            # 合并默认参数
            default_params = get_algorithm_default_params(algorithm)
            param_dict = {**default_params, **param_dict}

            try:
                # 执行压缩
                result = compress(df, algorithm, param_dict)

                # 转换为响应格式
                compressed_trajectory = [
                    {
                        'timestamp': point.timestamp.isoformat(),
                        'lat': point.lat,
                        'lon': point.lon,
                        'sog': point.sog,
                        'cog': point.cog,
                        'mmsi': point.mmsi
                    }
                    for point in result.compressed_points
                ]

                compression_result = CompressionResult(
                    algorithm=result.algorithm,
                    compression_ratio=result.compression_ratio,
                    elapsed_time=result.elapsed_time,
                    metrics=result.metrics,
                    compressed_trajectory=compressed_trajectory
                )

                compression_results.append(compression_result)

            except Exception as e:
                print(f"算法 {algorithm} 执行失败: {e}")
                # 继续执行其他算法

        # 存储结果
        results[job_id] = compression_results

        # 更新任务状态
        jobs[job_id].status = 'completed'
        jobs[job_id].completed_at = datetime.now()
        jobs[job_id].progress = 100.0
        jobs[job_id].message = '所有算法执行完成'

    except Exception as e:
        # 任务失败
        jobs[job_id].status = 'failed'
        jobs[job_id].completed_at = datetime.now()
        jobs[job_id].message = f'任务执行失败: {str(e)}'
        print(f"任务 {job_id} 失败: {e}")


# ============================================================================
# API 路由 (API Routes)
# ============================================================================

@app.get("/")
async def root():
    """根路径"""
    return {"message": "轨迹压缩算法服务", "version": "1.0.0"}


@app.get("/datasets")
async def get_datasets(ship_type: Optional[str] = None):
    """
    获取可用数据集列表

    参数:
        ship_type: 可选，按船型筛选
    """
    try:
        datasets = data_loader.get_datasets(ship_type)
        return {"datasets": datasets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取数据集失败: {str(e)}")


@app.post("/run", response_model=JobStatus)
async def run_compression(request: CompressionRequest, background_tasks: BackgroundTasks):
    """
    提交轨迹压缩任务

    支持批量运行多个算法，每个算法可以有独立的参数。
    返回任务 ID，可用于查询结果。
    """
    # 验证算法列表
    valid_algorithms = ['dr', 'fixed_dr', 'semantic_dr', 'sliding', 'opening', 'squish']
    invalid_algorithms = [alg for alg in request.algorithms if alg not in valid_algorithms]
    if invalid_algorithms:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的算法: {invalid_algorithms}. 支持的算法: {valid_algorithms}"
        )

    # 创建任务
    job_id = str(uuid.uuid4())
    job = JobStatus(
        job_id=job_id,
        status='pending',
        created_at=datetime.now(),
        message='任务已提交，等待执行'
    )

    jobs[job_id] = job

    # 后台执行任务
    background_tasks.add_task(run_compression_job, job_id, request)

    return job


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """获取任务状态"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="任务不存在")

    return jobs[job_id]


@app.get("/results/{job_id}")
async def get_compression_results(job_id: str):
    """获取压缩结果"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="任务不存在")

    job = jobs[job_id]
    if job.status != 'completed':
        return JSONResponse(
            status_code=202,
            content={"status": job.status, "message": job.message, "progress": job.progress}
        )

    if job_id not in results:
        raise HTTPException(status_code=500, detail="结果数据丢失")

    return {"results": [result.dict() for result in results[job_id]]}


@app.get("/metrics/{job_id}")
async def get_compression_metrics(job_id: str):
    """获取压缩评估指标"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="任务不存在")

    job = jobs[job_id]
    if job.status != 'completed':
        return JSONResponse(
            status_code=202,
            content={"status": job.status, "message": job.message, "progress": job.progress}
        )

    if job_id not in results:
        raise HTTPException(status_code=500, detail="结果数据丢失")

    # 提取指标数据
    metrics_data = []
    for result in results[job_id]:
        metrics_data.append({
            'algorithm': result.algorithm,
            'compression_ratio': result.compression_ratio,
            'elapsed_time': result.elapsed_time,
            'original_points': result.metrics.get('original_points'),
            'compressed_points': result.metrics.get('compressed_points'),
            'sed_mean': result.metrics.get('sed_mean'),
            'sed_max': result.metrics.get('sed_max'),
            'sed_p95': result.metrics.get('sed_p95'),
            'event_recall': result.metrics.get('event_recall')
        })

    return {"metrics": metrics_data}


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """删除任务及其结果"""
    if job_id in jobs:
        del jobs[job_id]
    if job_id in results:
        del results[job_id]

    return {"message": "任务已删除"}


# ============================================================================
# 启动服务器 (Server Startup)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
