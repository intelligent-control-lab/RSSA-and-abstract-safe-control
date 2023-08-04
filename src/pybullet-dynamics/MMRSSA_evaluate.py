from matplotlib import pyplot as plt
import numpy as np
import yaml
from loguru import logger
import time
from datetime import datetime
import os
import shutil
import pickle
from tqdm import tqdm

from RSSA_safety_index import SafetyIndex
from MMRSSA_additive import MMAddRSSA
from MMRSSA_multiplicative import MMMulRSSA
from MMRSSA_gaussian_multiplicative import GaussianMulRSSA
from MMRSSA_gaussian_additive import GaussianAddRSSA
from RSSA_utils import Monitor

from SegWay_env.SegWay_multimodal_env import SegWayAdditiveNoiseEnv, SegWayMultiplicativeNoiseEnv
from SCARA_env.SCARA_utils import draw_GP_confidence_interval
from SegWay_env.SegWay_utils import generate_gif


additive_rssa_types = ['gaussian_additive_mmrssa', 'additive_mmrssa', 'additive_none']
multiplicative_rssa_types=['gaussian_multiplicative_mmrssa', 'multiplicative_mmrssa', 'multiplicative_none']
plt.rcParams['figure.dpi'] = 500


def evaluate_in_MM_SegWay(
    rssa_types,
    yaml_path = './src/pybullet-dynamics/SegWay_env/SegWay_multimodal_params.yaml',
    log_root_path='./src/pybullet-dynamics/SegWay_env/log/',
    num_steps=100
):
    now = datetime.now()
    date_time = now.strftime('%Y-%m-%d__%H-%M-%S')
    log_path =log_root_path + date_time
    os.makedirs(log_path)
    shutil.copy(src=yaml_path, dst=log_path + '/SegWay_multimodal_params.yaml')
    
    with open(yaml_path, 'r', encoding='utf-8') as file:
        file_data = file.read()
    SegWay_kwargs = yaml.load(file_data, Loader=yaml.FullLoader)
    robot_kwargs = SegWay_kwargs['robot_kwargs']
    safe_control_kwargs = SegWay_kwargs['safe_control_kwargs']
    speed_kwargs = SegWay_kwargs['speed_kwargs']
    
    Env = SegWayAdditiveNoiseEnv if 'additive' in rssa_types[0] else SegWayMultiplicativeNoiseEnv
    print(f'Env: {Env.__name__}')
    env = Env(
            dt=robot_kwargs['dt'],
            K_m=robot_kwargs['K_m'],
            K_b=robot_kwargs['K_b'],
            m_0=robot_kwargs['m_0'],
            m=robot_kwargs['m'],
            J_0=robot_kwargs['J_0'],
            L=robot_kwargs['L'],
            l=robot_kwargs['l'],
            R=robot_kwargs['R'],
            g=robot_kwargs['g'],
            q_limit=robot_kwargs['q_limit'],
            dq_limit=robot_kwargs['dq_limit'],
            u_limit=robot_kwargs['u_limit'],
            a_safe_limit=robot_kwargs['a_safe_limit'],
        )   
    
    q_d = np.asanyarray(speed_kwargs['q_d'])
    dq_d = np.asanyarray(speed_kwargs['dq_d'])    
    store_data = {}
    # rssa_type_list = safe_control_kwargs['rssa_types']
    for rssa_type in rssa_types:
        print(f'Now evaluating: {rssa_type}')
        env.reset()
        env.robot.q = np.asanyarray(speed_kwargs['q_init'])
        
        rssa = get_rssa(rssa_type, env, safe_control_kwargs)
        monitor = Monitor()
        for i in tqdm(range(num_steps)):
            u = env.robot.PD_control(q_d=q_d, dq_d=dq_d)
            if rssa is not None:
                u = rssa.safe_control(u)
            env.step(u)
            monitor.update(
                q=env.robot.q,
                dq=env.robot.dq,
                u=u,
                dis_a_limit=env.a_safe_limit['high'] - env.robot.q[1],
            )
            env.render(img_name=str(i) + '.jpg', save_path=f'./src/pybullet-dynamics/SegWay_env/imgs/mm_evaluate/{rssa_type}/')
        store_data[rssa_type] = monitor.data
        generate_gif(rssa_type + '.gif', f'./src/pybullet-dynamics/SegWay_env/imgs/mm_evaluate/{rssa_type}/',
                     f'./src/pybullet-dynamics/SegWay_env/movies/mm_evaluate/', num_fig=num_steps)
    
    with open(log_path + '/SegWay_safe_control.pkl', 'wb') as file:
        pickle.dump(store_data, file)
    pkl_path = log_path + '/SegWay_safe_control.pkl'    
    return pkl_path, log_path
    
def get_rssa(rssa_type, env, safe_control_kwargs):
    param_dict = safe_control_kwargs['param_dict']
    if rssa_type == 'multiplicative_none' or 'additive_none':
        rssa = None
    elif rssa_type == 'gaussian_additive_mmrssa':
        rssa = GaussianAddRSSA(
            env=env,
            safety_index_params=param_dict,
            p_gaussian=safe_control_kwargs['p_gaussian'],
            sample_points_num=safe_control_kwargs['sample_points_num'],
            gamma=safe_control_kwargs['gamma'],
            fast_SegWay=safe_control_kwargs['fast_SegWay'],
        )
    elif rssa_type == 'gaussian_multiplicative_mmrssa':
        rssa = GaussianMulRSSA(
            env=env,
            safety_index_params=param_dict,
            p_gaussian=safe_control_kwargs['p_gaussian'],
            sample_points_num=safe_control_kwargs['sample_points_num'],
            gamma=safe_control_kwargs['gamma'],
            fast_SegWay=safe_control_kwargs['fast_SegWay'],
        )
    elif rssa_type == 'additive_mmrssa':
        rssa = MMAddRSSA(
            env=env,
            safety_index_params=param_dict,
            sample_points_num=safe_control_kwargs['sample_points_num'],
            gamma=safe_control_kwargs['gamma'],
            fast_SegWay=safe_control_kwargs['fast_SegWay'],
        )
    elif rssa_type == 'multipicative_mmrssa':
        rssa = MMMulRSSA(
            env=env,
            safety_index_params=param_dict,
            p_init=safe_control_kwargs['p_init'],
            sample_points_num=safe_control_kwargs['sample_points_num'],
            gamma=safe_control_kwargs['gamma'],
            fast_SegWay=safe_control_kwargs['fast_SegWay'],
        )
    else:
        raise Exception('Known safe control')
    return rssa

def draw_dis(data, offset=0.0, truncate=500):
    for rssa_type in rssa_types:
        try:
            plt.plot(np.array(data[rssa_type]['dis_wall'])[:truncate] + offset, label=rssa_type)
        except:
            pass
    plt.xlabel('step')
    plt.ylabel('dis_wall')
    plt.legend()
    plt.plot(np.zeros_like(data[rssa_type]['dis_wall'][:truncate]) + offset, linestyle='--', c='k', linewidth=0.75)
    plt.savefig(log_path + '/dis.png')
    plt.close()
    
def draw_CI(data, offset=0.0, truncate=500):
    
    for rssa_type in rssa_types:
        draw_GP_confidence_interval(
            data[rssa_type]['m_2_mean'][:truncate], 
            data[rssa_type]['m_2_std'][:truncate], 
            y_name='m_2_pred', 
            img_name='GP.png', 
            label=rssa_type, 
            if_close=False,
            img_path=log_path+'/',
        )
    plt.legend()
    plt.savefig(log_path + '/GP.png')
    plt.close()

if __name__ == '__main__':
    pkl_path, log_path = evaluate_in_MM_SegWay(rssa_types=additive_rssa_types, num_steps=20)
    pkl_path, log_path = evaluate_in_MM_SegWay(rssa_types=multiplicative_rssa_types, num_steps=20)
    # with open(pkl_path, 'rb') as file:
    #     data = pickle.load(file)
    # draw_dis(data)
    # draw_CI(data)
    