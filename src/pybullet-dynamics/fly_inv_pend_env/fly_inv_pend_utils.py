import pickle
import numpy as np
import cv2
import os
import shutil
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def exhaust(dim, pos_val=1, neg_val=0):
    id = 0
    max_id = 2**dim - 1
    u_list = []    
    while id <= max_id:
        tmp_id = id
        u = np.zeros(dim)
        for i in range(dim):
            flag = tmp_id % 2
            tmp_id = tmp_id // 2
            u[i] = pos_val if flag else neg_val
        id = id + 1
        u_list.append(u)
    u_list = np.asanyarray(u_list)
    return u_list


def get_constraints(vertices: np.ndarray):
    hull = ConvexHull(vertices)
    vertices = vertices[hull.vertices]
    center = np.mean(vertices, axis=0)
    A = hull.equations[:, :-1]
    b = -hull.equations[:, -1]
    for i in range(len(b)):
        if A[i, :] @ center > b[i]:
            A[i, :] = -A[i, :]
            b[i] = -b[i]
    return A, b