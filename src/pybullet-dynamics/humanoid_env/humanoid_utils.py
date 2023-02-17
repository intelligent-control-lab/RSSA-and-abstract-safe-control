from cProfile import label
from dataclasses import replace
import pickle
from typing import Dict
import numpy as np
import torch as th
import cv2
import os
from datetime import datetime
from loguru import logger
import shutil
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

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
    
    
def genetate_v_data(
    in_path='./data/test/',
    out_path='./data/v_data/',
    package_num=480,
):
    for i in range(package_num):
        print(i)
        in_data = th.load(in_path + str(i) + '.pth')
        out_data = {}
        out_data['x'] = in_data['last_x']
        out_data['u'] = in_data['u']
        out_data['v'] = (in_data['z'][:, 1] - in_data['last_z'][:, 1]) * 60 # dt = 1 / 60
        th.save(out_data, out_path + str(i) + '.pth')
        

def genetate_z_data(
    in_path='./data/test/',
    out_path='./data/z_data/',
    package_num=480,
):
    for i in range(package_num):
        print(i)
        in_data = th.load(in_path + str(i) + '.pth')
        out_data = {}
        out_data['x'] = in_data['last_x']
        out_data['z'] = in_data['last_z'] 
        th.save(out_data, out_path + str(i) + '.pth')
        

def shuffle_data(
    path='./data/v_data/',
    shuffle_times=10000,
    package_num=480,
):
    data_indices = np.arange(package_num)
    data: Dict = th.load(path + str(0) + '.pth')
    for val in data.values():
        batch_size = val.shape[0]
        batch_indices = np.arange(batch_size)
        break
    keys = list(data.keys())
    for i in range(shuffle_times):
        print(i)
        a, b = np.random.choice(data_indices, size=(2, ), replace=False)
        data_a = th.load(path + str(a) + '.pth')
        data_b = th.load(path + str(b) + '.pth')
        exchange_indices = np.random.choice(batch_indices, size=(batch_size // 2, ), replace=False)
        for key in keys:
            val_a = data_a[key]
            val_b = data_b[key]
            tmp = val_a[exchange_indices].clone()
            val_a[exchange_indices] = val_b[exchange_indices].clone() 
            val_b[exchange_indices] = tmp
        th.save(data_a, path + str(a) + '.pth')
        th.save(data_b, path + str(b) + '.pth')
    
    
def turn_on_log(log_root_path, yaml_path: str):
    now = datetime.now()
    date_time = now.strftime('%Y-%m-%d__%H-%M-%S')
    log_path =log_root_path + date_time
    os.mkdir(log_path)
    logger.add(log_path + '/log.log')
    yaml_name = yaml_path.split('/')[-1]
    shutil.copy(src=yaml_path, dst=log_path + '/' + yaml_name)
    return log_path, date_time


def collect_data(
    env,
    collect_turns=480,
    save_path='./src/pybullet-dynamics/humanoid_env/data/test/',
):
    obses = env.reset()
    u_ref = env.calc_u_ref(obses)
    for _ in range(2):
        obses = env.step(u_ref)
        z = env.get_z()
    for i in range(collect_turns):
        print(i)
        u_ref = env.calc_u_ref(obses)
        last_obses = obses.detach().clone()
        last_z = z.detach().clone()
        obses = env.step(u_ref)
        z = env.get_z()
        env.render(mode='human')
        data = {
            'last_x': last_obses,
            'last_z': last_z,
            'x': obses,
            'z': z,
            'u': u_ref,
        }
        th.save(data, save_path + str(i) + '.pth')
        
        
def collect_M(
    env,
    ssa,
    collect_turns=240,
    state_num_each_turn=400,
    save_path='./src/pybullet-dynamics/humanoid_env/data/latent/',
    clip_actions=1.0,
    root_state_random_degree=0.1,
    position_random_degree=0.2,
    velocity_random_degree=0.1,
):  
    # import ipdb; ipdb.set_trace()
    obses = env.reset()
    for _ in range(1):
        root_states, dof_states = env.get_states()
        env.reset_manual(root_states=root_states[0].repeat(env.num_envs, 1), dof_states=dof_states[0].repeat(env.num_envs, 1))
        u_ref = env.calc_u_ref(obses)
        obses = env.step(u_ref)
    M_ref_list = []
    for i in range(collect_turns):
        print(i)
        root_states, dof_states = env.get_states()
        u_ref = env.calc_u_ref(obses)
        delta_root_states = th.rand(size=(state_num_each_turn, 13), device=env.device) * root_state_random_degree * 2 - root_state_random_degree
        delta_positions = th.rand(size=(state_num_each_turn, env.dof), device=env.device) * position_random_degree * 2 - position_random_degree
        delta_velocities = th.rand(size=(state_num_each_turn, env.dof), device=env.device) * velocity_random_degree * 2 - velocity_random_degree
        
        # to get M_ref_list
        delta_root_states[0] = delta_root_states[0] * 0.0
        delta_positions[0] = delta_positions[0] * 0.0
        delta_velocities[0] = delta_velocities[0] * 0.0
        
        z_list = []
        M_list = []
        for j in range(state_num_each_turn):
            delta_root_state = delta_root_states[j]
            delta_position = delta_positions[j]
            delta_velocity = delta_velocities[j]
            random_root_states = root_states + delta_root_state[None, :]
            random_dof_states = dof_states + th.cat((delta_position, delta_velocity))[None, :]
            env.reset_manual(root_states=random_root_states[0].repeat(env.num_envs, 1), dof_states=random_dof_states[0].repeat(env.num_envs, 1))
            z = env.get_z()
            z_list.append(z[0])
            u = th.rand(size=u_ref.shape, device=env.device) * clip_actions * 2 - clip_actions
            u[0] = u[0] * 0.0
            u[1] = u[0] + clip_actions
            u[2] = u[0] - clip_actions
            env.step(u)
            z_next = env.get_z()
            v = (z_next[:, 1] - z[:, 1]) * 60
            v, _ = th.sort(v)
            v_min = v[0]
            v_max = v[-1]
            if (-v_min * v_max) > 0:
                M = th.min(-v_min, v_max)
            else:
                M = th.zeros_like(v_min)
            M_list.append(M)
            
            # to get M_ref_list
            if j == 0:
                M_ref_list.append(M)
        
        env.reset_manual(root_states=root_states[0].repeat(env.num_envs, 1), dof_states=dof_states[0].repeat(env.num_envs, 1))
        z = th.vstack(z_list)
        M = th.vstack(M_list)
        f_z, g_z = ssa.get_f_z_and_g_z(z)
        v_ref = th.zeros_like(M)
        safe_mask = (z[:, 0] >= ssa.z_min)
        unsafe_mask = ~safe_mask
        data = {
            'z': z,
            'M': M,
            'f_z': f_z, 'g_z': g_z,
            'v_ref': v_ref,
            'safe_mask': safe_mask, 'unsafe_mask': unsafe_mask,
        }
        th.save(data, save_path + f'{i}.pth')
        
        obses = env.step(u_ref)
    
    # to get M_ref_list
    M_ref_list = th.vstack(M_ref_list).squeeze_()
    dot_M_list = th.diff(M_ref_list) * 60
    dot_M_max = th.max(dot_M_list)
    dot_M_data = {
        'M_ref_list': M_ref_list,
        'dot_M_list': dot_M_list,
        'dot_M_max': dot_M_max,
    }
    print(f'dot_M_max: {dot_M_max}')
    th.save(dot_M_data, save_path + 'dot_M.pth')
            

if __name__ == '__main__':
    # images = []
    # img_path = './imgs/ssa/'
    # for i in range(480):
    #     print(i)
    #     rgb = cv2.imread(img_path + str(i) + '.png')
    #     images.append(bgr_to_rgb(rgb))
    # video_record(movie_name='./movies/unstable_ssa.mp4', images=images)
    
    # genetate_v_data()
    # genetate_z_data()
    
    shuffle_data(path='./src/pybullet-dynamics/humanoid_env/data/latent/', package_num=240)