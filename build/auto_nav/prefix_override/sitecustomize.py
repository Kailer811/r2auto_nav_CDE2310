import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/ramanathan/colcon_ws/src/auto_nav/auto_nav/install/auto_nav'
