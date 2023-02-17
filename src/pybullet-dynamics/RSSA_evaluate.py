from matplotlib import pyplot as plt
import numpy as np
import yaml
from loguru import logger
import time
from datetime import datetime
import os
import shutil
import pickle

from RSSA_safety_index import SafetyIndex
from RSSA_gaussian import GaussianRSSA
from RSSA_convex import ConvexRSSA
from RSSA_utils import *

from SCARA_env.SCARA_parameter_learning import SCARAParameterLearningEnv
from SCARA_env.SCARA_safety_index_learning import SCARASafetyIndexLearningEnv
from SegWay_env.SegWay_parameter_learning import SegWayParameterLearningEnv
from SegWay_env.SegWay_safety_index_learning import SegWaySafetyIndexLearningEnv
from SCARA_env.SCARA_utils import draw_GP_confidence_interval

rssa_types = ['safety_index', 'gaussian_rssa', 'convex_rssa']
plt.rcParams['figure.dpi'] = 500

def evaluate_safe_control_in_SCARA(
    yaml_path = './src/pybullet-dynamics/SCARA_env/SCARA_params.yaml',
    log_root_path='./src/pybullet-dynamics/SCARA_env/log/',
):
    now = datetime.now()
    date_time = now.strftime('%Y-%m-%d__%H-%M-%S')
    log_path =log_root_path + date_time
    os.mkdir(log_path) 
    shutil.copy(src=yaml_path, dst=log_path + '/SCARA_params.yaml')
    logger.add(log_path + '/log.log')
    
    with open(yaml_path, 'r', encoding='utf-8') as file:
        file_data = file.read()
    SCARA_kwargs = yaml.load(file_data, Loader=yaml.FullLoader)
    robot_kwargs = SCARA_kwargs['robot_kwargs']
    param_learning_kwargs = SCARA_kwargs['param_learning_kwargs']
    safe_control_kwargs = SCARA_kwargs['safe_control_kwargs']
    track_kwargs = SCARA_kwargs['track_kwargs']
    
    env = SCARAParameterLearningEnv(
            dt=robot_kwargs['dt'],
            m_1=robot_kwargs['m_1'],
            l_1=robot_kwargs['l_1'],
            m_2=robot_kwargs['m_2'],
            l_2=robot_kwargs['l_2'],
            q_limit=robot_kwargs['q_limit'],
            dq_limit=robot_kwargs['dq_limit'],
            u_limit=robot_kwargs['u_limit'],
            use_online_adaptation=param_learning_kwargs['use_online_adaptation'],
            m_2_mean_init=param_learning_kwargs['m_2_mean_init'],
            m_2_std_init=param_learning_kwargs['m_2_std_init'],
        )   
    u_lim_max = np.max(robot_kwargs['u_limit']['high'])
    
    q_way_points = np.linspace(
        start=track_kwargs['q_start'],
        stop=track_kwargs['q_end'],
        num=track_kwargs['waypoint_num'],
    )
    
    q_way_points = np.concatenate((q_way_points, np.array([q_way_points[-1]]).repeat([100,], axis=0)))
    
    store_data = {}
    rssa_type_list = safe_control_kwargs['rssa_types']
    for rssa_type in rssa_type_list:
        env.reset()
        env.robot.q = np.asanyarray(track_kwargs['q_start'])
        env.robot.dq = np.asanyarray(track_kwargs['dq_start'])
        
        rssa = get_rssa(rssa_type, env, safe_control_kwargs)
        monitor = Monitor()
        for q_d in q_way_points:
            m_2_mean, m_2_std = env.param_pred()
            u = env.robot.computed_torque_control(q_d=q_d)
            for _ in range(track_kwargs['control_repeat_times']):
                if rssa is not None:
                    u = rssa.safe_control(u)
                env.step(u)
                if np.max(np.abs(u)) > u_lim_max:
                    logger.debug(f'rssa type: {rssa_type}, u_IF: {u}')
            monitor.update(
                q=env.robot.q,
                dq=env.robot.dq,
                p=env.robot.p,
                dp=env.robot.dp,
                u=u,
                dis_wall=env.wall_x - env.robot.p[0],
                m_2_mean=m_2_mean,
                m_2_std=m_2_std,
            )
            if rssa is not None:
                monitor.update(phi=rssa.phi)
        store_data[rssa_type] = monitor.data
    
    with open(log_path + '/SCARA_safe_control.pkl', 'wb') as file:
        pickle.dump(store_data, file)
    pkl_path = log_path + '/SCARA_safe_control.pkl'    
    return pkl_path, log_path
    

def evaluate_safe_control_in_SegWay(
    param_dict={'k_v': 1.0, 'beta': 0.0, 'eta': 0.05},
    yaml_path = './src/pybullet-dynamics/SegWay_env/SegWay_params.yaml',
    log_root_path='./src/pybullet-dynamics/SegWay_env/log/',
):
    now = datetime.now()
    date_time = now.strftime('%Y-%m-%d__%H-%M-%S')
    log_path =log_root_path + date_time
    os.mkdir(log_path) 
    shutil.copy(src=yaml_path, dst=log_path + '/SegWay_params.yaml')
    
    with open(yaml_path, 'r', encoding='utf-8') as file:
        file_data = file.read()
    SegWay_kwargs = yaml.load(file_data, Loader=yaml.FullLoader)
    robot_kwargs = SegWay_kwargs['robot_kwargs']
    param_learning_kwargs = SegWay_kwargs['param_learning_kwargs']
    safe_control_kwargs = SegWay_kwargs['safe_control_kwargs']
    speed_kwargs = SegWay_kwargs['speed_kwargs']
    
    env = SegWayParameterLearningEnv(
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
            use_online_adaptation=param_learning_kwargs['use_online_adaptation'],
            K_m_mean_init=param_learning_kwargs['K_m_mean_init'],
            K_m_std_init=param_learning_kwargs['K_m_std_init'],
        )   
    
    q_d = np.asanyarray(speed_kwargs['q_d'])
    dq_d = np.asanyarray(speed_kwargs['dq_d'])    
    store_data = {}
    rssa_type_list = safe_control_kwargs['rssa_types']
    for rssa_type in rssa_type_list:
        
        env.reset()
        env.robot.q = np.asanyarray(speed_kwargs['q_init'])
        
        rssa = get_rssa(rssa_type, env, safe_control_kwargs)
        monitor = Monitor()
        for _ in range(1500):
            K_m_mean, K_m_std = env.param_pred()
            u = env.robot.PD_control(q_d=q_d, dq_d=dq_d)
            if rssa is not None:
                u = rssa.safe_control(u)
            env.step(u)
            monitor.update(
                q=env.robot.q,
                dq=env.robot.dq,
                u=u,
                dis_a_limit=env.a_safe_limit['high'] - env.robot.q[1],
                K_m_mean=K_m_mean,
                K_m_std=K_m_std,
            )
        store_data[rssa_type] = monitor.data
    
    with open(log_path + '/SegWay_safe_control.pkl', 'wb') as file:
        pickle.dump(store_data, file)
    pkl_path = log_path + '/SegWay_safe_control.pkl'    
    return pkl_path, log_path
    
    
def get_rssa(rssa_type, env, safe_control_kwargs):
    param_dict = safe_control_kwargs['param_dict']
    if rssa_type == 'none':
        rssa = None
    elif rssa_type == 'safety_index':
        rssa = SafetyIndex(
            env=env,
            safety_index_params=param_dict,
            use_true_param=safe_control_kwargs['use_true_param'],
            gamma=safe_control_kwargs['gamma'],
            rho=safe_control_kwargs['rho'],
        )
    elif rssa_type == 'gaussian_rssa':
        rssa = GaussianRSSA(
            env=env,
            safety_index_params=param_dict,
            confidence_level=safe_control_kwargs['confidence_level'],
            sample_points_num=safe_control_kwargs['sample_points_num'],
            gamma=safe_control_kwargs['gamma'],
            fast_SegWay=safe_control_kwargs['fast_SegWay'],
        )
    elif rssa_type == 'convex_rssa':
        rssa = ConvexRSSA(
            env=env,
            safety_index_params=param_dict,
            sample_points_num=safe_control_kwargs['sample_points_num'],
            gamma=safe_control_kwargs['gamma'],
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
    # pkl_path, log_path = evaluate_safe_control_in_SCARA()
    pkl_path, log_path = evaluate_safe_control_in_SegWay()
    # with open(pkl_path, 'rb') as file:
    #     data = pickle.load(file)
    # draw_dis(data)
    # draw_CI(data)
    