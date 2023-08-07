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


def evaluate_in_MM_SegWay(
    rssa_types,
    yaml_path = './src/pybullet-dynamics/SegWay_env/SegWay_multimodal_params.yaml',
    log_root_path='./src/pybullet-dynamics/SegWay_env/log/',
    num_steps=100,
    render=False
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
            u_ref = env.robot.PD_control(q_d=q_d, dq_d=dq_d)
            if rssa is not None:
                u = rssa.safe_control(u_ref)
            else:
                u = u_ref
            env.step(u)
            monitor.update(
                q=env.robot.q,
                dq=env.robot.dq,
                u_ref=u_ref,
                u=u,
                dis_a_limit=env.a_safe_limit['high'] - env.robot.q[1],
            )
            if render:
                env.render(img_name=str(i) + '.jpg', save_path=f'./src/pybullet-dynamics/SegWay_env/imgs/mm_evaluate/{rssa_type}/')
        store_data[rssa_type] = monitor.data
        if render:
            generate_gif(rssa_type + '.gif', f'./src/pybullet-dynamics/SegWay_env/imgs/mm_evaluate/{rssa_type}/',
                        f'./src/pybullet-dynamics/SegWay_env/movies/mm_evaluate/', num_fig=num_steps)
    
    with open(log_path + '/SegWay_safe_control.pkl', 'wb') as file:
        pickle.dump(store_data, file)
    pkl_path = log_path + '/SegWay_safe_control.pkl'    
    return pkl_path, log_path
    
def get_rssa(rssa_type, env, safe_control_kwargs):
    param_dict = safe_control_kwargs['param_dict']
    if rssa_type == 'multiplicative_none' or rssa_type == 'additive_none':
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
        )
    elif rssa_type == 'multiplicative_mmrssa':
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
    
def draw_phi(data, rssa_types, truncate=-1):
    for rssa_type in rssa_types:
        values = np.array(data[rssa_type]['q'])[:truncate, 1]
        plt.plot(values, label=rssa_type)
        plt.xlabel('step')
    plt.ylabel('$\phi$')
    plt.legend()
    plt.plot(np.ones_like(values)*0.1, linestyle='--', c='k', linewidth=0.75)
    plt.savefig(log_path + '/phi.png')
    plt.show()
    plt.close()

def draw_u(data, rssa_types, truncate=-1):
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)
    axes = [ax1, ax2, ax3]
    axes = axes[:len(rssa_types)]
    for ax, rssa_type in zip(axes, rssa_types):
        u = np.array(data[rssa_type]['u'])[:truncate]
        u_ref = np.array(data[rssa_type]['u_ref'])[:truncate]
        ax.plot(u, label='u')
        ax.plot(u_ref, label='u_ref', linestyle='--', c='k', linewidth=0.75)
        ax.set_xlabel('step')
        ax.legend()
        ax.set_ylabel(rssa_type)

    plt.savefig(log_path + '/u.png')
    plt.show()
    plt.close()

if __name__ == '__main__':
    plt.rcParams['figure.dpi'] = 200  # 500

    additive_rssa_types = ['additive_mmrssa', 'gaussian_additive_mmrssa'] #, 'additive_none']
    multiplicative_rssa_types=['multiplicative_mmrssa', 'gaussian_multiplicative_mmrssa'] # , 'multiplicative_none']

    # rssa_types = additive_rssa_types
    rssa_types = multiplicative_rssa_types
    pkl_path, log_path = evaluate_in_MM_SegWay(rssa_types=rssa_types, num_steps=1000)
    with open(pkl_path, 'rb') as file:
        data = pickle.load(file)
    draw_phi(data, rssa_types)
    draw_u(data, rssa_types)
    