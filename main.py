import sys
from pathlib import Path

# 确保项目根目录在sys.path中，以便直接运行脚本时包导入正常工作
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    # 从包中导入并运行GUI入口
    from algcode.ui import main_gui
    main_gui()

if __name__ == "__main__":
    main()


