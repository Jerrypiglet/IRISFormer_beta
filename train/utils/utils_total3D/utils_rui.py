from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import numpy as np

def clip(subjectPolygon, clipPolygon):
   # https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])

   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0]
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return ((n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3)

   outputList = subjectPolygon
   cp1 = clipPolygon[-1]

   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]

      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
   return(outputList)

def vis_cube_plt(Xs, ax, color=None, linestyle='-', label=None, if_face_idx_text=False, if_vertex_idx_text=False, text_shift=[0., 0., 0.], fontsize_scale=1.):
   # https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
   index1 = [0, 1, 2, 3, 0, 4, 5, 6, 7, 4]
   index2 = [[1, 5], [2, 6], [3, 7]]
   index3 = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [0, 3, 7, 4]] # faces
   if color is None:
      color = list(np.random.choice(range(256), size=3) / 255.)
   #   print(color)
   ax.plot3D(Xs[index1, 0], Xs[index1, 1], Xs[index1, 2], color=color, linestyle=linestyle)
   for index in index2:
      ax.plot3D(Xs[index, 0], Xs[index, 1], Xs[index, 2], color=color, linestyle=linestyle)
   if label is not None:
      # ax.text3D(Xs[0, 0]+text_shift[0], Xs[0, 1]+text_shift[1], Xs[0, 2]+text_shift[2], label, color=color, fontsize=10*fontsize_scale)
      ax.text3D(Xs.mean(axis=0)[0], Xs.mean(axis=0)[1], Xs.mean(axis=0)[2], label, color=color, fontsize=10*fontsize_scale)
   if if_vertex_idx_text:
      for vertex_idx, V in enumerate(Xs):
         ax.text3D(V[0]+text_shift[0], V[1]+text_shift[1], V[2]+text_shift[2], str(vertex_idx), color=color, fontsize=10*fontsize_scale)
   if if_face_idx_text:
      for face_idx, index in enumerate(index3):
         X_center = Xs[index, :].mean(0)
         # print(X_center.shape)
         ax.text3D(X_center[0]+text_shift[0], X_center[1]+text_shift[1], X_center[2]+text_shift[2], str(face_idx), color='grey', fontsize=30*fontsize_scale)
         ax.scatter3D(X_center[0], X_center[1], X_center[2], color=[0.8, 0.8, 0.8], s=30)
         for cross_index in [[index[0], index[2]], [index[1], index[3]]]:
            ax.plot3D(Xs[cross_index, 0], Xs[cross_index, 1], Xs[cross_index, 2], color=[0.8, 0.8, 0.8], linestyle='--', linewidth=1)


   

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
        
def vis_axis(ax):
    for vec, tag, tag_loc in zip([([0, 1], [0, 0], [0, 0]), ([0, 0], [0, 1], [0, 0]), ([0, 0], [0, 0], [0, 1])], [r'$X_w$', r'$Y_w$', r'$Z_w$'], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        a = Arrow3D(vec[0], vec[1], vec[2], mutation_scale=20,
                lw=1, arrowstyle="->", color="k")
        ax.text3D(tag_loc[0], tag_loc[1], tag_loc[2], tag)
        ax.add_artist(a)

def vis_axis_xyz(ax, x, y, z, origin=[0., 0., 0.], suffix='_w', color='k'):
    for vec, tag, tag_loc in zip([([origin[0], (origin+x)[0]], [origin[1], (origin+x)[1]], [origin[2], (origin+x)[2]]), \
       ([origin[0], (origin+y)[0]], [origin[1], (origin+y)[1]], [origin[2], (origin+y)[2]]), \
          ([origin[0], (origin+z)[0]], [origin[1], (origin+z)[1]], [origin[2], (origin+z)[2]])], [r'$X%s$'%suffix, r'$Y%s$'%suffix, r'$Z%s$'%suffix], [origin+x, origin+y, origin+z]):
        a = Arrow3D(vec[0], vec[1], vec[2], mutation_scale=20,
                lw=1, arrowstyle="->", color=color)
        ax.text3D(tag_loc[0], tag_loc[1], tag_loc[2], tag, color=color)
        ax.add_artist(a)