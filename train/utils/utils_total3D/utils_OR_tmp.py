import os,sys,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from utils.utils_rui import Arrow3D, vis_axis, vis_cube_plt, vis_axis_xyz
print(vis_axis_xyz)