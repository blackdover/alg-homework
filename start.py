#!/usr/bin/env python3
"""
轨迹压缩算法对比平台启动脚本

启动完整的系统（后端API + 前端静态文件服务）
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path


def check_requirements():
    """检查依赖是否已安装"""
    try:
        import fastapi
        import uvicorn
        import pandas
        import numpy
        print("后端依赖检查通过")
    except ImportError as e:
        print(f"缺少后端依赖: {e}")
        print("请运行: cd backend && pip install -r requirements.txt")
        return False
    return True


def start_backend():
    """启动后端服务"""
    print("启动后端服务...")
    backend_dir = Path(__file__).parent / "backend"

    if not backend_dir.exists():
        print("✗ 后端目录不存在:", backend_dir)
        return None

    if not (backend_dir / "run_server.py").exists():
        print("✗ 后端启动脚本不存在:", backend_dir / "run_server.py")
        return None

    os.chdir(backend_dir)

    # 启动 FastAPI 服务
    try:
        process = subprocess.Popen([
            sys.executable, "run_server.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # 等待一小段时间检查进程是否正常启动
        time.sleep(3)

        if process.poll() is None:
            print("后端服务启动成功 (http://localhost:8000)")
            # 检查端口是否真的在监听
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', 8000))
            sock.close()
            if result == 0:
                print("端口 8000 正在监听")
            else:
                print("端口 8000 未在监听，可能启动失败")
                # 读取错误输出
                if process.stderr:
                    error_output = process.stderr.read()
                    if error_output:
                        print("后端启动错误:", error_output)
        else:
            print("后端服务启动失败")
            stdout, stderr = process.communicate()
            if stderr:
                print("错误信息:", stderr)
            if stdout:
                print("标准输出:", stdout)
            return None

        return process

    except Exception as e:
        print(f"启动后端服务时出错: {e}")
        return None


def start_frontend():
    """启动前端静态文件服务"""
    print("启动前端服务...")
    frontend_dir = Path(__file__).parent / "frontend"

    if not frontend_dir.exists():
        print("✗ 前端目录不存在:", frontend_dir)
        return None

    if not (frontend_dir / "index.html").exists():
        print("✗ 前端主页面不存在:", frontend_dir / "index.html")
        return None

    # 使用 Python 的内置 HTTP 服务器
    try:
        process = subprocess.Popen([
            sys.executable, "-m", "http.server", "8080"
        ], cwd=frontend_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # 等待一小段时间检查进程是否正常启动
        time.sleep(2)

        if process.poll() is None:
            print("前端服务启动成功 (http://localhost:8080)")
            # 检查端口是否真的在监听
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', 8080))
            sock.close()
            if result == 0:
                print("端口 8080 正在监听")
            else:
                print("端口 8080 未在监听，可能启动失败")
                # 读取错误输出
                if process.stderr:
                    error_output = process.stderr.read()
                    if error_output:
                        print("前端启动错误:", error_output)
        else:
            print("前端服务启动失败")
            stdout, stderr = process.communicate()
            if stderr:
                print("错误信息:", stderr)
            if stdout:
                print("标准输出:", stdout)
            return None

        return process

    except Exception as e:
        print(f"启动前端服务时出错: {e}")
        return None


def main():
    """主函数"""
    print("=" * 60)
    print("轨迹压缩算法对比平台")
    print("=" * 60)

    # 检查依赖
    if not check_requirements():
        return

    # 启动服务
    backend_process = start_backend()
    if backend_process is None:
        print("后端服务启动失败，无法继续")
        return

    time.sleep(2)  # 等待后端启动

    frontend_process = start_frontend()
    if frontend_process is None:
        print("前端服务启动失败，正在停止后端服务")
        if backend_process:
            backend_process.terminate()
        return

    time.sleep(1)  # 等待前端启动

    print("\n" + "=" * 60)
    print("服务启动完成！")
    print("后端 API: http://localhost:8000")
    print("前端界面: http://localhost:8080")
    print("API 文档: http://localhost:8000/docs")
    print("=" * 60)
    print("按 Ctrl+C 停止服务")

    # 自动打开浏览器
    try:
        webbrowser.open("http://localhost:8080")
    except Exception as e:
        print(f"无法自动打开浏览器: {e}")

    try:
        # 等待用户中断
        while True:
            if backend_process.poll() is not None:
                print("⚠ 后端服务意外退出")
                break
            if frontend_process.poll() is not None:
                print("⚠ 前端服务意外退出")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在停止服务...")
    finally:
        # 确保进程被终止
        if backend_process and backend_process.poll() is None:
            backend_process.terminate()
            try:
                backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_process.kill()

        if frontend_process and frontend_process.poll() is None:
            frontend_process.terminate()
            try:
                frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                frontend_process.kill()

        print("服务已停止")


if __name__ == "__main__":
    main()
