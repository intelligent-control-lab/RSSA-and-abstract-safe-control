from matplotlib import pyplot as plt
import numpy as np
import yaml
from loguru import logger
import time
from datetime import datetime
import os
import shutil
import pickle
from copy import deepcopy

from RSSA_safety_index import SafetyIndex
from RSSA_gaussian import GaussianRSSA
from RSSA_convex import ConvexRSSA
from RSSA_evaluate import get_rssa
from RSSA_utils import *

from SCARA_env.SCARA_parameter_learning import SCARAParameterLearningEnv
from SCARA_env.SCARA_safety_index_learning import SCARASafetyIndexLearningEnv
from SegWay_env.SegWay_parameter_learning import SegWayParameterLearningEnv

def draw_FI_set(
    yaml_path = './src/pybullet-dynamics/SCARA_env/SCARA_params.yaml',
    log_root_path='./src/pybullet-dynamics/FI_set_log/',
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
    CBF_learning_kwargs = SCARA_kwargs['CBF_learning_kwargs']
    FI_sets_kwargs = SCARA_kwargs['FI_sets_kwargs']
    
    env = SCARAParameterLearningEnv(
            dt=robot_kwargs['dt'],
            m_1=robot_kwargs['m_1'],
            l_1=robot_kwargs['l_1'],
            m_2=robot_kwargs['m_2'],
            l_2=robot_kwargs['l_2'],
            q_limit=robot_kwargs['q_limit'],
            dq_limit=robot_kwargs['dq_limit'],
            u_limit=robot_kwargs['u_limit'],
            use_online_adaptation=False,
            m_2_mean_init=param_learning_kwargs['m_2_mean_init'],
            m_2_std_init=param_learning_kwargs['m_2_std_init'],
        )
    env.param_pred()
    convex_rssa = get_rssa('convex_rssa', env, safe_control_kwargs)
    
    q_sampled_per_dim = CBF_learning_kwargs['q_sampled_per_dim']
    dq_num = CBF_learning_kwargs['dq_num']
    param_dict = convex_rssa.safety_index_params
    offset = FI_sets_kwargs['offset']
    
    q_1_list = np.linspace(start=env.q_limit['low'][0], stop=env.q_limit['high'][0], num=q_sampled_per_dim)
    q_2_list = np.linspace(start=env.q_limit['low'][1], stop=env.q_limit['high'][1], num=q_sampled_per_dim)
    dq_list = np.random.uniform(low=env.dq_limit['low'], high=env.dq_limit['high'], size=(dq_num, 2))
    FI_set = np.zeros((len(q_1_list), len(q_2_list)))
    for i in range(len(q_1_list)):
        for j in range(len(q_2_list)):
            env.robot.q = np.asanyarray([q_1_list[i], q_2_list[j]])
            for dq in dq_list:
                env.robot.dq = dq
                phi = env.get_phi(safety_index_params=param_dict) - offset
                if phi >= 0:
                    FI_set[i, j] += 1
    
    if FI_sets_kwargs['if_binary']:
        FI_set[FI_set > 0] = -1
        FI_set += 1
        
    q_1_list, q_2_list = np.meshgrid(q_1_list, q_2_list)
    cb = plt.contourf(q_1_list, q_2_list, FI_set)
    plt.colorbar(cb)
    plt.savefig(log_path + '/FI_set.png')
    plt.close()
    
    with open(log_path + '/FI_sets.pkl', 'wb') as file:
        pickle.dump({'q_1': q_1_list, 'q_2': q_2_list, 'FI_set': FI_set}, file)
    
    pkl_path = log_path + '/FI_sets.pkl'
    return pkl_path, log_path



def compare_CR_GR_speed(
    sample_points_num=150,
):
    def run(env, ssa, step_length):
        logger.debug('Now test the time comsumption of ' + str(ssa.__class__))
        logger.debug(f'Numbers of steps: {step_length}')
        logger.debug(f'Sample number: {ssa.sample_points_num}')
        time_list = []
        
        env.reset()
        q_d = np.array([0, 0])
        dq_d = np.array([1, 0])
        for i in range(step_length):
            u = env.robot.PD_control(q_d, dq_d)
            start = time.time()
            u = ssa.safe_control(u)
            end = time.time()
            interval = end - start 
            # logger.debug(f'Time comsumption of one action: {interval} s')
            env.step(u)
            time_list.append(interval)
        
        total_time = np.sum(np.asanyarray(time_list))
        logger.debug(f'Total time comsumption: {total_time} s \n')
        return total_time
        
    env = SegWayParameterLearningEnv()
    gaussian_time = run(env, GaussianRSSA(env, sample_points_num=sample_points_num, fast_SegWay=True), 1000)
    convex_time = run(env, ConvexRSSA(env, sample_points_num=sample_points_num), 1000)
    
    return gaussian_time, convex_time


def get_max_phi_SegWay(
    K_m_std_init=0.3,
    a_init=0.0,
):
    logger.debug(f'K_m_std: {K_m_std_init}')
    logger.debug(f'a_init: {a_init}')
    env = SegWayParameterLearningEnv(use_online_adaptation=False, K_m_std_init=K_m_std_init)
    env.robot.q[0] = a_init
    ssa = GaussianRSSA(env, fast_SegWay=True)
    
    phi_list = []
    
    q_d = np.array([0, 0])
    dq_d = np.array([1, 0])
    for i in range(960):
        u = env.robot.PD_control(q_d, dq_d)
        u = ssa.safe_control(u)
        env.step(u)
        phi = env.get_phi(ssa.safety_index_params)
        phi_list.append(phi)
        # logger.debug(f'phi: {phi}')
    
    phi_list = np.asanyarray(phi_list)
    phi_max = np.max(phi_list)
    logger.debug(f'phi_max: {phi_max}\n\n')
    
    return phi_list, phi_max   


def get_max_phi_SCARA(
    m_2=1.0,
    m_2_std_init=0.3,
    u_limit={'low': [-20.0, -20.0], 'high': [20.0, 20.0]},
):
    logger.debug(f'm_2: {m_2}')
    env = SCARAParameterLearningEnv(m_2=m_2, use_online_adaptation=False, m_2_std_init=m_2_std_init)
    env.param_pred()
    ssa = ConvexRSSA(env, sample_points_num=50, safety_index_params={'alpha': 0.57, 'k_v': 2.15, 'beta': 0.072})
    
    phi_list = []
    
    q_way_points = np.linspace(start=[np.pi/2 - 0.05, 0.0], stop=[-np.pi/3, -np.pi], num=400)
    env.reset()

    for i, q_d in enumerate(q_way_points):
        u = env.robot.computed_torque_control(q_d=q_d)
        for _ in range(3):
            u_1 = ssa.safe_control(u)
            last_q = deepcopy(env.robot.q)
            last_dq = deepcopy(env.robot.dq)
            last_phi = env.get_phi(ssa.safety_index_params)
            env.step(u_1)
            phi = env.get_phi(ssa.safety_index_params)
            if phi > 0 and last_phi <= 0:
                env.robot.q = last_q
                env.robot.dq = last_dq
                u_1 = ssa.safe_control(u, force_rssa=True)
                env.step(u_1)
                phi = env.get_phi(ssa.safety_index_params)
            # print(phi)
            phi_list.append(phi)
            u = u_1
    
    phi_list = np.asanyarray(phi_list)
    phi_max = np.max(phi_list)
    logger.debug(f'phi_max: {phi_max}\n\n')
    
    return phi_list, phi_max   

def get_rho_SCARA(
    m_2=0.5,
    safety_index_params={'alpha': 1.0, 'k_v': 1.0, 'beta': 0.0},
):  
    log_path = add_log(log_root_path='./src/pybullet-dynamics/FI_set_log/')
    
    rho_list = []
    logger.debug(f'safety_index_params: {safety_index_params}')
    logger.debug(f'true m_2: {m_2}\n')
    
    env = SCARAParameterLearningEnv(use_online_adaptation=False, m_2=m_2)
    env.param_pred()
    for i in range(10000):
        env.robot.q = np.random.uniform(low=env.q_limit['low'], high=env.q_limit['high'])
        env.robot.dq = np.random.uniform(low=env.dq_limit['low'], high=env.dq_limit['high'])
        u = np.random.uniform(low=env.u_limit['low'], high=env.u_limit['high']).reshape(-1, 1)
        
        if env.detect_collision() or env.robot.p[0] <= 0.5:
            continue
        
        true_dot_Xr = env.f + env.g @ u
        true_m_2 = env.robot.m_2
        env.robot.m_2 = env.m_2_mean
        false_dot_Xr = env.f + env.g @ u
        env.robot.m_2 = true_m_2
        
        p_phi_p_Xr = env.get_p_phi_p_Xr(safety_index_params=safety_index_params)
        delta_dot_phi = p_phi_p_Xr @ (true_dot_Xr - false_dot_Xr)
        rho = np.linalg.norm(delta_dot_phi)
        rho_list.append(rho)
        logger.debug(f'Xr: {env.Xr}, rho: {rho}')
        
    rho_list = np.asarray(rho_list)
    rho_max = np.max(rho_list)
    logger.debug(f'rho_max: {rho_max}')
        
def get_M_dot_Xr_SCARA(
    m_2=0.005,
    safety_index_params={'alpha': 0.57, 'k_v': 2.15, 'beta': 0.072},
):  
    log_path = add_log(log_root_path='./src/pybullet-dynamics/FI_set_log/')
    
    M_dot_Xr_list = []
    logger.debug(f'safety_index_params: {safety_index_params}')
    logger.debug(f'true m_2: {m_2}\n')
    
    env = SCARAParameterLearningEnv(use_online_adaptation=False, m_2=m_2)
    env.param_pred()
    for i in range(10000):
        env.robot.q = np.random.uniform(low=env.q_limit['low'], high=env.q_limit['high'])
        env.robot.dq = np.random.uniform(low=env.dq_limit['low'], high=env.dq_limit['high'])
        u = np.random.uniform(low=env.u_limit['low'], high=env.u_limit['high']).reshape(-1, 1)
        
        if env.detect_collision() or env.robot.p[0] <= 0.5:
            continue
        
        true_dot_Xr = env.f + env.g @ u
        true_m_2 = env.robot.m_2
        env.robot.m_2 = 0.1
        false_dot_Xr = env.f + env.g @ u
        env.robot.m_2 = true_m_2
        
        delta_dot_Xr = true_dot_Xr - false_dot_Xr
        M_dot_Xr = np.linalg.norm(delta_dot_Xr)
        M_dot_Xr_list.append(M_dot_Xr)
        logger.debug(f'Xr: {env.Xr}, M_dot_Xr: {M_dot_Xr}')
    
    M_dot_Xr_list.sort()  
    plt.plot(M_dot_Xr_list)
    plt.savefig(log_path + '/M_dot_Xr_list.png')   
    M_dot_Xr_list = np.asarray(M_dot_Xr_list)
    select_idx = len(M_dot_Xr_list) // 10
    M_dot_Xr_max = M_dot_Xr_list[-select_idx]
    logger.debug(f'M_dot_Xr_max: {M_dot_Xr_max}')
        
    


if __name__ == '__main__':
    # draw_FI_set()
    
    # log_path = add_log(log_root_path='./src/pybullet-dynamics/compare_speed_log/')
    # num_list = [20, 50, 100, 150, 200]
    # gaussian_time_list = []
    # convex_time_list = []
    # for sample_points_num in num_list:
    #     gaussian_time, convex_time = compare_CR_GR_speed(sample_points_num=sample_points_num)
    #     gaussian_time_list.append(gaussian_time)
    #     convex_time_list.append(convex_time)
    # data = {
    #     'sample_num': num_list,
    #     'gaussian_time': gaussian_time_list,
    #     'convex_time': convex_time_list,
    # }
    # with open(log_path + '/time.pkl', 'wb') as file:
    #     pickle.dump(data, file)
    
    # log_path = add_log(log_root_path='./src/pybullet-dynamics/FI_set_log/')
    # # K_m_std_list = [0.3, 0.9, 1.6, 2.0, 2.1, 2.2, 2.3,  2.5, 3.0, 3.5]
    # K_m_std_list = [0.3, 0.9, 1.6, 2.0, 2.5, 2.6, 2.9, 3.0]
    # phi_max_list = []
    # for K_m_std_init in K_m_std_list:
    #     phi_list, phi_max = get_max_phi_SegWay(K_m_std_init=K_m_std_init, a_init=-0.5)
    #     phi_max_list.append(phi_max)
    # data = {
    #     'K_m_std': K_m_std_list,
    #     'phi_max': phi_max_list,
    # }
    # with open(log_path + '/phi_max_SegWay.pkl', 'wb') as file:
    #     pickle.dump(data, file)
        
    # log_path = add_log(log_root_path='./src/pybullet-dynamics/FI_set_log/')
    # # m_2_list = np.linspace(6e-3, 8e-3, 20)
    # m_2_list = [0.005, 0.1, 1.0, 1.9, 10.0]
    # phi_max_list = []
    # phi_list_list = []
    # for m_2 in m_2_list:
    #     phi_list, phi_max = get_max_phi_SCARA(m_2=m_2)
    #     phi_max_list.append(phi_max)
    #     phi_list_list.append(phi_list)
    # data = {
    #     'm_2': m_2_list,
    #     'phi_max': phi_max_list,
    #     'phi_list': phi_list_list,
    # }
    # with open(log_path + '/phi_max_SCARA.pkl', 'wb') as file:
    #     pickle.dump(data, file)
    
    # get_rho_SCARA()
    get_M_dot_Xr_SCARA()