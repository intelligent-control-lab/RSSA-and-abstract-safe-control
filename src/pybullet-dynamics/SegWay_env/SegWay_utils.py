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
    img_path='./src/pybullet-dynamics/SegWay_env/imgs/parameter_learning/', 
    alpha=0.3,
    if_close=False,
    if_save=False,
):
    mean = np.squeeze(np.asanyarray(mean))
    std = np.squeeze(np.asanyarray(std))
    x = np.arange(len(mean))
    plt.plot(x, mean, label=label)
    plt.legend()
    plt.fill_between(x, mean + std, mean - std, alpha=alpha)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.ylim((-1.0, 5.0))
    if if_save:
        plt.savefig(img_path + img_name)
    if if_close:
        plt.close()


if __name__ == '__main__':
    video_sequential_record(
        movie_name='SegWay_env_test.mp4',
        movie_path='./src/pybullet-dynamics/SegWay_env/movies/',
        fig_path='./src/pybullet-dynamics/SegWay_env/imgs/env_test/',
        num=960,
        prefix='',
    )   