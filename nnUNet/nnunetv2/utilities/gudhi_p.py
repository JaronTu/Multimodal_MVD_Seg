import gudhi.rips_complex
import numpy as np
import pandas as pd
import gudhi
import torch
from sklearn import manifold
from pylab import *
from torch_topological.nn import CubicalComplex
from gudhi import persistence_graphical_tools
import cv2
from torch_topological.nn.data import batch_iter

# img_pred = cv2.imread('/home/turenzhe/下载/1-2-imageonline.co-3836423.png')
img_pred = cv2.imread('/home/turenzhe/下载/1-2-1.png')
# gudhi.rips_complex.RipsComplex(points)
# print(img_pred.shape)
# print(np.unique(img_pred))
img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2GRAY)
# print(np.unique(img_pred[:,:,0]))
# print(np.unique(img_pred[:,:,1]))
# print(np.unique(img_pred[:,:,2]))
rips = gudhi.RipsComplex(points=img_pred)
# rips = gudhi.rips_complex.RipsComplex(points=img_pred)
simplex_tree = rips.create_simplex_tree(max_dimension=2)
persistence = simplex_tree.persistence()
persistence_diagram = persistence_graphical_tools.plot_persistence_diagram(persistence)

# gudhi.plot_persistence_barcode(persistence)
# img_pred = torch.from_numpy(img_pred)
# cubical_complex = CubicalComplex(
#             dim=2,
#             superlevel=False
#         )
# img_pred = cubical_complex(img_pred)
# # img_pred = gd.cubical_complex(img_pred)
# pers_pred = [
#     x for x in batch_iter(img_pred)
# ]
# gd.plot_persistence_diagram(pers_pred)


# import gudhi.cubical_complex
# from sklearn.datasets import fetch_openml
# import matplotlib.pyplot as plt
# import gudhi as gd
#
# X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
#
# # X[17] is an '8'
# cc = gd.CubicalComplex(top_dimensional_cells=X[17], dimensions=[28, 28])
# gudhi.cubical_complex.CubicalComplex
# diag = cc.persistence()
# gd.plot_persistence_diagram(diag, legend=True)
# plt.show()
# print(type(X))
# print(y)