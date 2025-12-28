#!/usr/bin/env python3
"""
轨迹压缩算法服务启动脚本

使用方法:
    python run_server.py

服务器将在 http://localhost:8000 启动
API 文档: http://localhost:8000/docs
"""

import sys
import os
from pathlib import Path

# 添加当前目录到 Python 路径，以便正确导入模块
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import uvicorn
from api import app

if __name__ == "__main__":
    print("启动轨迹压缩算法服务...")
    print("API 文档: http://localhost:8000/docs")
    print("服务器地址: http://localhost:8000")

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 开发模式下自动重载
        log_level="info"
    )
