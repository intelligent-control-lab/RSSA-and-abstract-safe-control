import pickle
import pybullet as p
import numpy as np
import cv2
import os
import shutil
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


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

def draw_GP_confidence_interval(
    mean,
    std,
    label,
    y_name,
    img_name,
    x_name='step',
    img_path='./src/pybullet-dynamics/SCARA_env/imgs/parameter_learning/', 
    alpha=0.3,
    if_close=False,
    if_save=False,
    ylim=(0.0, 2.0),
):
    mean = np.squeeze(np.asanyarray(mean))
    std = np.squeeze(np.asanyarray(std))
    x = np.arange(len(mean))
    plt.plot(x, mean, label=label)
    plt.fill_between(x, mean + std, mean - std, alpha=alpha)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.ylim(ylim)
    if if_save:
        plt.savefig(img_path + img_name)
    if if_close:
        plt.close()
        
def exhaust(dim):
    id = 0
    max_id = 2**dim - 1
    u_list = []    
    while id <= max_id:
        tmp_id = id
        u = np.zeros(dim)
        for i in range(dim):
            flag = tmp_id % 2
            tmp_id = tmp_id // 2
            u[i] = 1 if flag else 0
        id = id + 1
        u_list.append(u)
    u_list = np.asanyarray(u_list)
    return u_list

def quick_plot(sequential_data, img_name, path='./src/pybullet-dynamics/SCARA_env/imgs/latent_test/', xlabel=None, ylabel=None):
    data = np.asanyarray(sequential_data)
    data= np.squeeze(data)
    assert len(data.shape) == 1
    plt.plot(data)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.savefig(path + img_name)
    plt.close()
    
def quick_scatter(x_data, y_data, img_name, path='./src/pybullet-dynamics/SCARA_env/imgs/latent_test/', xlabel=None, ylabel=None):
    plt.scatter(x_data, y_data)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.savefig(path + img_name)
    plt.close()
    
def quick_hist(sequential_data, img_name, path='./src/pybullet-dynamics/SCARA_env/imgs/latent_test/', xlabel=None, ylabel=None, bins=40):
    data = np.asanyarray(sequential_data)
    data= np.squeeze(data)
    assert len(data.shape) == 1
    plt.hist(data, bins=bins)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.savefig(path + img_name)
    plt.close()


if __name__ == '__main__':
    video_sequential_record(
        movie_name='convex_rssa_fake_m_2.mp4',
        movie_path='./src/pybullet-dynamics/SCARA_env/movies/',
        fig_path='./src/pybullet-dynamics/SCARA_env/imgs/convex_rssa/',
        num=400,
        prefix='',
    )   