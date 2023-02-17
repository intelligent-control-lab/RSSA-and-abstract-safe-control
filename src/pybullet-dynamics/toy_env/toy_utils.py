from cProfile import label
import pickle
import pybullet as p
import numpy as np
import cv2
import os
import shutil
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d

def compare(a, b, ratio_flag=False):
    a = np.asanyarray(a)
    b = np.asanyarray(b)
    a = np.squeeze(a)
    b = np.squeeze(b)
    res = np.max(np.abs(a - b))
    print(f'max diff: {res}')
    
    if ratio_flag:
        tmp = np.abs(a - b)
        ratio_a = tmp / np.abs(a)
        ratio_b = tmp / np.abs(b)
        ratio = np.concatenate((ratio_a, ratio_b))
        ratio_max = np.max(ratio)
        print(f'max diff ratio: {ratio_max}')
        return res, ratio_max

    return res


def draw_sample_points(
    points: np.ndarray, 
    fig_name, 
    s=4, alpha=0.1, c='b', 
    save_path='./src/pybullet-dynamics/toy_env/imgs/',
    show_flag=False, close_flag=True,
):
    '''
    points should of the shape (sample_size, 2) or (sample_size, 3)
    '''
    fig_save_path = save_path + fig_name
    if points.shape[1] == 2:
        plt.scatter(points[:, 0], points[:, 1], s=s, alpha=alpha, c=c)
    elif points.shape[1] == 3:
        ax = plt.axes(projection='3d')
        ax.view_init(20, -20)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=s, alpha=alpha, c=c)
    plt.savefig(fig_save_path)
    if show_flag:
        plt.show()
    if close_flag:
        plt.close()


def draw_2d_polyhedron(
    points: np.ndarray,
    fig_name,
    alpha=0.1, c='b',
    save_path='./src/pybullet-dynamics/toy_env/imgs/',
    show_flag=False, close_flag=True,
):
    '''
    points should of the shape (sample_size, 2)
    '''
    fig_save_path = save_path + fig_name
    hull = ConvexHull(points)
    plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], alpha=alpha, c=c)
    plt.savefig(fig_save_path)
    if show_flag:
        plt.show()
    if close_flag:
        plt.close()


def video_sequential_record(
    movie_name,
    movie_path,
    fig_path, 
    num,    # how many pictures to record
    prefix='', suffix='.jpg',
    fps=24,
):  
    img = cv2.imread(fig_path + prefix + '0.jpg')
    height = img.shape[0]
    width = img.shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(movie_path + movie_name, fourcc, fps, (width, height))

    for i in range(num):
        img = cv2.imread(fig_path + prefix + str(i) + suffix)
        video.write(img)
    video.release()




if __name__ == '__main__':
    # points = np.array([
    #     [1, 0], [1, 1], [0, 0], [0, 1], [2, 1]
    # ])
    # draw_2d_polyhedron(points, '2d_Polygedron.jpg')
    video_sequential_record(
        movie_name='IK_p_with_different_p.mp4',
        movie_path='./src/pybullet-dynamics/toy_env/movies/',
        fig_path='./src/pybullet-dynamics/toy_env/imgs/IK_p_with_different_p/',
        num=100,
        prefix='',
    )   