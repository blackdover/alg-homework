import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    from algcode.ui import main_gui
    main_gui()

if __name__ == "__main__":
    main()
