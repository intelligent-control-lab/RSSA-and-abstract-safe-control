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
        q_sampled_per_dim=100, 
        dq_num=100,
    ):  
        q_1_list = np.linspace(start=self.env.q_limit['low'][0], stop=self.env.q_limit['high'][0], num=q_sampled_per_dim)
        q_2_list = np.linspace(start=self.env.q_limit['low'][1], stop=self.env.q_limit['high'][1], num=q_sampled_per_dim)
        dq_list = np.random.uniform(low=self.env.dq_limit['low'], high=self.env.dq_limit['high'], size=(dq_num, 2))
        heat_map = np.zeros((len(q_1_list), len(q_2_list)))
        for i in range(len(q_1_list)):
            for j in range(len(q_2_list)):
                self.env.robot.q = np.asanyarray([q_1_list[i], q_2_list[j]])
                penalty = 0
                for dq in dq_list:
                    self.env.robot.dq = dq
                    phi = self.env.get_phi(safety_index_params=param_dict)
                    if not self.env.detect_collision() and phi >= 0:
                        if self.env.robot.p[0] > 0.5 and not self.check_one_state(phi, self.rssa_type):
                            logger.debug(f'x: {self.env.robot.p[0]}, dx: {self.env.robot.dp[0]}, phi: {phi}')
                            penalty += 1
                heat_map[i, j] = penalty
                logger.debug(f'q: {self.env.robot.q}, penalty: {penalty}')
        return q_1_list, q_2_list, heat_map
    

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
        q_1_list, q_2_list, heat_map = MM_learn.visualize(
            param_dict,
            q_sampled_per_dim=safety_index_learning_kwargs['q_sampled_per_dim'],
            dq_num=safety_index_learning_kwargs['dq_num']
        )
        with open(log_path + '/DR_heatmap.pkl', 'wb') as file:
                pickle.dump({'q_1': q_1_list, 'q_2': q_2_list, 'heat_map': heat_map}, file)
    
    pkl_path = log_path + '/DR_heatmap.pkl'
    return pkl_path, log_path
   
def draw_heatmap(data):
    q_ticks = [-1.5, 1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    q_1 = data['q_1']
    q_2 = data['q_2']
    penalty = data['heat_map']
    q_1, q_2 = np.meshgrid(q_1, q_2)
    # sns.set_theme()
    # ax = sns.heatmap(penalty, vmin=0, vmax=60)
    # xticks = range(0, len(q_ticks))
    # yticks = range(0, len(q_ticks))
    # ax.set_xticks(xticks)
    # ax.set_yticks(yticks)
    # ax.set_xticklabels(np.array(q_ticks))
    # ax.set_yticklabels(np.array(q_ticks))
    # plt.ylim(0,len(q_ticks))
    # plt.xlim(0,len(q_ticks))
    cb = plt.contourf(q_1, q_2, penalty, vmin=0, vmax=60)
    plt.colorbar(cb)
    plt.savefig(log_path + '/configration.png')
    
def cartesian_heatmap(data):
    q_1 = data['q_1']
    q_2 = data['q_2']
    penalty = data['heat_map']
    env = SegWayMultiplicativeNoiseEnv()
    
    cartesian_heatmap = []
    for i in range(len(q_1)):
        for j in range(len(q_2)):
            if penalty[i, j] != 0:
                env.robot.q = np.array([q_1[i], q_2[j]])
                cartesian_heatmap.append(env.robot.p)
    cartesian_heatmap = np.asanyarray(cartesian_heatmap)
    plt.figure(figsize=(5, 5))
    plt.scatter(cartesian_heatmap[:, 0], cartesian_heatmap[:, 1], alpha=0.1, s=8)
    plt.xlim(-2.0, 2.0)
    plt.plot([1.6, 1.6], [1.8, -1.8], linewidth=3)
    plt.savefig(log_path + '/cartesian.png')


if __name__ == '__main__':
    pkl_path, log_path = MM_Learning()
    with open(pkl_path, 'rb') as file:
        data = pickle.load(file)
    draw_heatmap(data)
    cartesian_heatmap(data)