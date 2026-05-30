import os
import sys

# 将 vtss 目录加入 sys.path 以支持其内部平级导入
_vtss_dir = os.path.dirname(os.path.abspath(__file__))
if _vtss_dir not in sys.path:
    sys.path.insert(0, _vtss_dir)
