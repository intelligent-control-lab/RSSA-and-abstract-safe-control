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


'''
useful classes 
'''
class ObjectModel:
    def __init__(
        self,
        physics_client_id,
        cartesian_pos,
        shape_type='sphere',
        half_extend=[0.05, 0.05, 0.05],
        radius=0.03,
        rgba=[0, 1, 0, 0.5], 
        create_collision_flag=False,
    ):
        self.physics_client_id = physics_client_id
        self.cartesian_pos = cartesian_pos
        self.shape_type = shape_type

        if self.shape_type == 'box':
            self.half_extend = half_extend
            self.visual_id = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=half_extend,
                visualFramePosition=[0, 0, 0],
                physicsClientId=self.physics_client_id
            )
            
            if create_collision_flag:
                self.collision_id = p.createCollisionShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=half_extend,
                    collisionFramePosition=[0, 0, 0],
                    physicsClientId=self.physics_client_id
                )
            else: 
                self.collision_id = -1

        elif self.shape_type == 'sphere':
            self.radius = radius
            self.visual_id = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=radius,
                visualFramePosition=[0, 0, 0],
                physicsClientId=self.physics_client_id
            )
            
            if create_collision_flag:
                self.collision_id = p.createCollisionShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=radius,
                    collisionFramePosition=[0, 0, 0],
                    physicsClientId=self.physics_client_id
                )
            else: 
                self.collision_id = -1

        self.id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=self.collision_id,
            baseVisualShapeIndex=self.visual_id,
            basePosition=self.cartesian_pos,
            baseInertialFramePosition=[0, 0, 0],
            useMaximalCoordinates=False,
            physicsClientId=self.physics_client_id
        )
        
        p.changeVisualShape(self.id, -1, rgbaColor=rgba, physicsClientId=self.physics_client_id)


class Monitor:
    def __init__(self):
        self.data = dict()
        self.name_set = set()
        self.if_monitor = False

    def start(self):
        self.if_monitor = True

    def update(self, **kwargs):
        if not self.if_monitor:
            return

        for name, value in kwargs.items():
            if name not in self.name_set:
                self.data[name] = []
                self.name_set.add(name)
            self.data[name].append(value)

    def close(self, store_path='./src/pybullet-dynamics/panda_rod_env/data/monitor.pkl'):
        if not self.if_monitor:
            return 

        with open(store_path, 'wb') as file:
            pickle.dump(self.data, file)
        print('successfully collect data!')



'''
useful functions
'''
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


def bgr_to_rgb(image):
    b = image[:, :, 0].copy()
    g = image[:, :, 1].copy()
    r = image[:, :, 2].copy()

    image[:, :, 0] = r
    image[:, :, 1] = g
    image[:, :, 2] = b

    return image


def video_record(
    movie_name,
    images, 
):
    height = images[0].shape[0]
    width = images[0].shape[1]
    fps = 24
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(movie_name, fourcc, fps, (width, height))

    for img in images:
        video.write(img)
    video.release()


def cross_product(vec1, vec2):
    S = np.array(([0, -vec1[2], vec1[1]],
                  [vec1[2], 0, -vec1[0]],
                  [-vec1[1], vec1[0], 0]))

    return np.dot(S, vec2)


def calculate_orientation_error(desired: np.ndarray, current: np.ndarray):
    assert desired.shape[0] == current.shape[0]
    """
    Optimized function to determine orientation error
    """
    rc1 = current[0:3, 0]
    rc2 = current[0:3, 1]
    rc3 = current[0:3, 2]
    rd1 = desired[0:3, 0]
    rd2 = desired[0:3, 1]
    rd3 = desired[0:3, 2]

    orientation_error = 0.5 * (cross_product(rc1, rd1) + cross_product(rc2, rd2) + cross_product(rc3, rd3))
    return orientation_error


def get_file_nums(path):
    cnt = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            cnt = cnt + 1
    print(f'file numbers in {path} is {cnt}')


def remove_files_in_dir(path):
    shutil.rmtree(path)
    os.mkdir(path)


def to_np(data):
    return np.squeeze(np.asanyarray(data))


def get_L2_norm(data_1, data_2):
    assert data_1.shape == data_2.shape

    if len(data_1.shape) == 1:
        return np.abs(data_1 - data_2)

    dis = []
    for i in range(data_1.shape[0]):
        tmp = np.linalg.norm(data_1[i] - data_2[i])
        dis.append(tmp)
    dis = np.asanyarray(dis)
    return dis


def get_largest_singular_value(sigma):
    if len(sigma.shape) == 1:
        return np.sqrt(sigma)

    assert sigma[0].shape[0] == sigma[0].shape[1]

    largest_singular = []
    for i in range(sigma.shape[0]):
        tmp = np.max(np.linalg.eig(sigma[i])[0])
        largest_singular.append(np.sqrt(tmp))
    largest_singular = np.asanyarray(largest_singular)
    return largest_singular


def get_images_of_array(data):
    for key in data.keys():
        data[key] = to_np(data[key])
        data[key] = data[key].reshape(data[key].shape[0], -1)
        interval = data[key].shape[0]

    images = []
    for i in range(interval):
        fig = plt.figure()        
        canvas = FigureCanvasAgg(fig)
        ax = fig.gca()

        for key in data.keys():
            ax.plot(np.arange(len(data[key][i])), data[key][i], label=key)
        ax.legend()

        canvas.draw()
        buf = canvas.buffer_rgba()
        img = np.asanyarray(buf)[:, :, :3]
        img = bgr_to_rgb(img)
        images.append(img)
        plt.close(fig)
    
    return images


def plot_images_of_components(data, ids, plt_array, figsize=(10, 10)):
    for key in data.keys():
        data[key] = to_np(data[key])
        data[key] = data[key].reshape(data[key].shape[0], -1)

    assert plt_array[0] * plt_array[1] == len(ids)

    fig = plt.figure(figsize=figsize)
    plt_ids = np.arange(1, len(ids) + 1)
    for id, plt_id in zip(ids, plt_ids):
        ax = fig.add_subplot(plt_array[0], plt_array[1], plt_id)
        for key in data.keys():
            tmp = data[key][:, id]
            ax.plot(np.arange(len(tmp)), tmp, label=key)

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels)
    plt.legend()

    plt.tight_layout()
    

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


if __name__ == '__main__':
    # remove_files_in_dir('./src/pybullet-dynamics/panda_rod_env/movies')
    get_file_nums('./src/pybullet-dynamics/panda_rod_env/data/nn_raw_data_diff_1')