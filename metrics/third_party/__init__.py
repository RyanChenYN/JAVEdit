import os
import sys

# 将 third_party 目录加入 sys.path，
# 使得 utmosv2 内部的 'from utmosv2.xxx import' 绝对导入能正确解析
_third_party_dir = os.path.dirname(os.path.abspath(__file__))
if _third_party_dir not in sys.path:
    sys.path.insert(0, _third_party_dir)
