from typing import Dict, List
from matplotlib import pyplot as plt
import numpy as np
import yaml
from loguru import logger
from datetime import datetime
import os
import shutil
from copy import copy
import pickle
from scipy.special import betainc, beta
from tqdm import tqdm

from RSSA_safety_index_learning import RSSASafetyIndexLearning
from SegWay_env.SegWay_multimodal_env import SegWayMultiplicativeNoiseEnv, SegWayAdditiveNoiseEnv
from MMRSSA_evaluate import get_rssa


class SegWayMMSafetyIndexLearning(RSSASafetyIndexLearning):
    def __init__(
        self,
        env: SegWayMultiplicativeNoiseEnv, 
        epoch, 
        elite_ratio,
        populate_num,
        init_sigma_ratio,
        noise_ratio,
        init_params: Dict,
        param_bounds: Dict[str, List],
        
        rssa_type,
        safe_control_kwargs: Dict,
        
        states_sampled_per_param=10000,
        iteration_limit=30000,
        reward_clip_ratio=0.1,
    ):
        super().__init__(env, epoch, elite_ratio, populate_num, init_sigma_ratio, noise_ratio, init_params, param_bounds)
        self.rssa_type = rssa_type
        self.rssa = get_rssa(rssa_type=rssa_type, env=env, safe_control_kwargs=safe_control_kwargs)
        self.states_sampled_per_param = states_sampled_per_param
        self.iteration_limit = iteration_limit
        self.reward_clip_ratio = reward_clip_ratio

        self.env: SegWayMultiplicativeNoiseEnv
        
    def step(self):
        params = copy(self.init_params)
        self.populate()
        rewards = []
        for i, data in tqdm(enumerate(self.population)):
            for key, x in zip(params.keys(), data):
                params[key] = x
            reward = self.evaluate_single_param(params)
            rewards.append(reward)
            logger.debug(f'param index: {i}, params: {params}, reward: {reward}')
        rewards = np.asanyarray(rewards)
        indices = np.argsort(-rewards) 
        best_members = self.population[indices[0:int(self.elite_ratio * self.populate_num)]]
        self.mu = np.mean(best_members, axis=0)
        self.sigma = np.cov(best_members.T) + self.noise

    def evaluate_single_param(self, param_dict: Dict):
        N_f = 0 # feasible
        N_i = 0 # infeasible
        self.rssa.safety_index_params = param_dict
        for i in range(self.iteration_limit):
            self.env.robot.q = np.random.uniform(low=self.env.q_limit['low'], high=self.env.q_limit['high'])
            self.env.robot.dq = np.random.uniform(low=self.env.dq_limit['low'], high=self.env.dq_limit['high'])
            phi = self.env.get_phi(safety_index_params=param_dict)
            # if np.abs(self.env.robot.q[1]) <= 0.1 or phi < 0:
            if phi < 0:
                continue
            else:
                if self.check_one_state():
                    N_f += 1
                else:
                    # print(self.env.robot.q, self.env.robot.dq)
                    N_i += 1
            
            if N_f + N_i == self.states_sampled_per_param:
                break
        
        ##### reward
        #  P(q>z) = 1- B(z;N_f+1,N_i+1)/B(N_f+1,N_i+1)
        # reward = 1 - betainc(N_f+1, N_i+1, 0.999)  # it seems that this is hard to optimize
        reward = N_f/(N_f+N_i)
        
        logger.debug(f'params: {param_dict}')
        logger.debug(f'total iterations: {i+1}, N_f: {N_f}, N_i: {N_i}, reward: {reward}')
        return reward
    
    def check_one_state(self):
        u_ref = np.zeros(1)
        self.rssa.safe_control(u_ref)
        return not self.rssa.if_infeasible

    def visualize(
        self, param_dict: Dict, 
        dq_sampled_per_dim=100, 
        q_num=100,
    ):  
        self.rssa.safety_index_params = param_dict
        dq_1_list = np.linspace(start=self.env.dq_limit['low'][0], stop=self.env.dq_limit['high'][0], num=dq_sampled_per_dim)
        dq_2_list = np.linspace(start=self.env.dq_limit['low'][1], stop=self.env.dq_limit['high'][1], num=dq_sampled_per_dim)
        heat_map = np.zeros((len(dq_1_list), len(dq_2_list)))
        for i in tqdm(range(len(dq_1_list))):
            for j in range(len(dq_2_list)):
                q_list = np.random.uniform(low=self.env.q_limit['low'], high=self.env.q_limit['high'], size=(q_num, 2)) # only q[1] is useful
                self.env.robot.dq = np.asanyarray([dq_1_list[i], dq_2_list[j]])
                penalty = 0
                for q in q_list:
                    self.env.robot.q = q
                    phi = self.env.get_phi(safety_index_params=param_dict)
                    if phi >= 0:
                        if not self.check_one_state():
                            logger.debug(f'q: {self.env.robot.q}, dq: {self.env.robot.dq}, phi: {phi}')
                            penalty += 1
                heat_map[i, j] = penalty
                logger.debug(f'dq: {self.env.robot.dq}, penalty: {penalty}')
        return dq_1_list, dq_2_list, heat_map
    

def MM_Learning(
    yaml_path = './src/pybullet-dynamics/SegWay_env/SegWay_multimodal_params.yaml',
    log_root_path='./src/pybullet-dynamics/SegWay_env/log/Safety_index_learning/',
    visualize_set='only_V',
):
    now = datetime.now()
    date_time = now.strftime('%Y-%m-%d__%H-%M-%S')
    log_path =log_root_path + date_time
    os.mkdir(log_path) 
    shutil.copy(src=yaml_path, dst=log_path + '/SegWay_multimodal_params.yaml')
    logger.add(log_path + '/log.log')
    
    with open(yaml_path, 'r', encoding='utf-8') as file:
        file_data = file.read()
    SegWay_kwargs = yaml.load(file_data, Loader=yaml.FullLoader)
    robot_kwargs = SegWay_kwargs['robot_kwargs']
    safe_control_kwargs = SegWay_kwargs['safe_control_kwargs']
    safety_index_learning_kwargs = SegWay_kwargs['safety_index_learning_kwargs']
    
    Env = SegWayAdditiveNoiseEnv if 'additive' in safety_index_learning_kwargs['rssa_type'] else SegWayMultiplicativeNoiseEnv

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

    MM_learn = SegWayMMSafetyIndexLearning(
        env=env,
        epoch=safety_index_learning_kwargs['epoch'],
        elite_ratio=safety_index_learning_kwargs['elite_ratio'],
        populate_num=safety_index_learning_kwargs['populate_num'],
        init_sigma_ratio=safety_index_learning_kwargs['init_sigma_ratio'],
        noise_ratio=safety_index_learning_kwargs['noise_ratio'],
        init_params=safety_index_learning_kwargs['init_param_dict'],
        param_bounds=safety_index_learning_kwargs['param_bounds'],
        
        rssa_type=safety_index_learning_kwargs['rssa_type'],
        safe_control_kwargs=safe_control_kwargs,
        
        states_sampled_per_param=safety_index_learning_kwargs['states_sampled_per_param'],
        iteration_limit=safety_index_learning_kwargs['iteration_limit'],
        reward_clip_ratio=safety_index_learning_kwargs['reward_clip_ratio']
    )
    
    # MM_learn.evaluate_single_param({'alpha': 1.0, 'k_v': 1.0, 'beta': 0.001})
    # MM_learn.evaluate_single_param({'alpha': 0.41299781574142935, 'k_v': 4.89561275104321, 'beta': 0.6751067447507083})
    # exit()

    visualize_set = safety_index_learning_kwargs['visualize_set']
    assert visualize_set == 'only_L' or visualize_set == 'only_V' or visualize_set == 'L_and_V'
    if visualize_set == 'only_L':
        MM_learn.learn()
    else: 
        if visualize_set == 'L_and_V':
            MM_learn.learn()
        param_dict = copy(MM_learn.init_params)
        for i, key in enumerate(param_dict.keys()):
            param_dict[key] = MM_learn.mu[i]
        dq_1_list, dq_2_list, heat_map = MM_learn.visualize(
            param_dict,
            dq_sampled_per_dim=safety_index_learning_kwargs['dq_sampled_per_dim'],
            q_num=safety_index_learning_kwargs['q_num']
        )
        with open(log_path + '/DR_heatmap.pkl', 'wb') as file:
                pickle.dump({'dq_1': dq_1_list, 'dq_2': dq_2_list, 'heat_map': heat_map}, file)
    
    pkl_path = log_path + '/DR_heatmap.pkl'
    return pkl_path, log_path
   
def draw_heatmap(data):
    q_ticks = [-1.5, 1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    dq_1 = data['dq_1']
    dq_2 = data['dq_2']
    penalty = data['heat_map']

    fig, ax = plt.subplots(figsize=(4, 10/3)) # set figure size

    c = ax.imshow(penalty, vmin = 0, vmax = 60,
                    extent =[dq_1.min(), dq_1.max(), dq_2.min(), dq_2.max()],
                        interpolation ='nearest', origin ='lower')
    ax.set_xlabel('$\mathrm{\dot{ p}\ (m/s)}$')
    ax.set_ylabel('$\mathrm{\dot{\\varphi}\ (rad/s)}$')
    
    cbar = fig.colorbar(c) 
    cbar.ax.set_ylabel('$\%$ no feasible control', rotation=-90, va='bottom')

    plt.tight_layout()
    plt.savefig(log_path + '/heatmap.png')


if __name__ == '__main__':
    # pkl_path, log_path = MM_Learning()
    pkl_path = '/home/liqian/RSSA-and-abstract-safe-control/src/pybullet-dynamics/SegWay_env/log/Safety_index_learning/phi_l/DR_heatmap.pkl'
    log_path = '/home/liqian/RSSA-and-abstract-safe-control/src/pybullet-dynamics/SegWay_env/log/Safety_index_learning/phi_l/'
    with open(pkl_path, 'rb') as file:
        data = pickle.load(file)
    draw_heatmap(data)