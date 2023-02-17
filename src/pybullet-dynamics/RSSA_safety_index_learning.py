from copy import copy
from typing import Dict, List
from matplotlib import pyplot as plt
import numpy as np
from cvxopt import matrix, solvers
import yaml
from loguru import logger
import time
from datetime import datetime
import os
import shutil
import pickle

from SCARA_env.SCARA_safety_index_learning import SCARASafetyIndexLearningEnv
from SCARA_env.SCARA_parameter_learning import SCARAParameterLearningEnv
from SegWay_env.SegWay_safety_index_learning import SegWaySafetyIndexLearningEnv
from RSSA_evaluate import GaussianRSSA, ConvexRSSA, SafetyIndex, get_rssa

class RSSASafetyIndexLearning:
    def __init__(
        self,
        env: SCARASafetyIndexLearningEnv,
        epoch, 
        elite_ratio,
        populate_num,
        init_sigma_ratio,
        noise_ratio,
        init_params: Dict,
        param_bounds: Dict[str, List],
    ):
        self.env = env
        
        self.epoch = epoch 
        self.elite_ratio = elite_ratio
        self.populate_num = populate_num
        self.init_sigma_ratio = init_sigma_ratio
        self.noise_ratio = noise_ratio
        self.mu = self.init_params = init_params
        self.param_bounds = param_bounds
        
        for key_1, key_2 in zip(self.init_params.keys(), self.param_bounds.keys()):
            assert key_1 == key_2
        
        # initialize mu and sigma of params
        self.mu = np.array(list(init_params.values()))
        param_scales = [value[1] - value[0] for value in self.param_bounds.values()]
        param_scales = np.asanyarray(param_scales)
        self.sigma = np.diag((self.init_sigma_ratio * param_scales)**2)
        self.noise = np.diag((self.noise_ratio * param_scales)**2)
        self.param_lower_bounds = np.array([value[0] for value in self.param_bounds.values()])
        self.param_upper_bounds = np.array([value[1] for value in self.param_bounds.values()])
        
    def populate(self):
        self.population = np.random.multivariate_normal(self.mu, self.sigma, self.populate_num)
        self.population = np.clip(self.population, self.param_lower_bounds, self.param_upper_bounds)

    def step(self):
        params = copy(self.init_params)
        self.populate()
        rewards = []
        for i, data in enumerate(self.population):
            for key, x in zip(params.keys(), data):
                params[key] = x
            reward = self.env.evaluate_single_param(params)
            rewards.append(reward)
            logger.debug(f'param index: {i}, params: {params}, reward: {reward}')
        rewards = np.asanyarray(rewards)
        indices = np.argsort(-rewards) 
        best_members = self.population[indices[0:int(self.elite_ratio * self.populate_num)]]
        self.mu = np.mean(best_members, axis=0)
        self.sigma = np.cov(best_members.T) + self.noise
        
    def learn(self):
        for i in range(self.epoch):
            self.step()
            logger.debug(f'\n epoch: {i}, mu: {self.mu}, sigma: {self.sigma} \n')
            

'''
for DR-CBF learning in SCARA environment
'''           
class DRCBFLearning(RSSASafetyIndexLearning):
    def __init__(
        self,
        env: SCARAParameterLearningEnv, 
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
        
    def evaluate_single_param(self, param_dict: Dict):
        reward = 0
        count = 0
        for i in range(self.iteration_limit):
            self.env.robot.q = np.random.uniform(low=self.env.q_limit['low'], high=self.env.q_limit['high'])
            self.env.robot.dq = np.random.uniform(low=self.env.dq_limit['low'], high=self.env.dq_limit['high'])
            phi = self.env.get_phi(safety_index_params=param_dict)
            if self.env.robot.p[0] <= 0.5 or self.env.detect_collision() or phi < 0:
                continue
            else:
                if self.check_one_state(phi, self.rssa_type):
                    reward += 1
                count += 1
            
            if count == self.states_sampled_per_param:
                break
        
        # strengthen reward
        if count < self.iteration_limit * self.reward_clip_ratio:
            reward = 0.0
        else:
            reward = reward / count

        logger.debug(f'params: {param_dict}')
        logger.debug(f'total iterations: {i}, count: {count}, reward: {reward}')
        return reward
    
    def check_one_state(self, phi, rssa_type='convex_rssa'):
        if rssa_type == 'convex_rssa':
            return self.check_one_state_convex(phi)
        elif rssa_type == 'gaussian_rssa':
            return self.check_one_state_gaussian(phi)
        elif rssa_type == 'safety_index':
            return self.check_one_state_constant(phi)
        else:
            raise Exception('No such rssa type!')
    
    def check_one_state_convex(self, phi):
        A_hat_Uc, b_hat_Uc, _, _, _ = self.rssa.generate_convex_safety_con(phi)
        n = self.rssa.u_dim
        G = np.vstack([
            A_hat_Uc,
            np.eye(n),
            -np.eye(n),
        ])
        h = np.vstack([
            b_hat_Uc,
            np.asanyarray(self.env.u_limit['high']).reshape(-1, 1),
            -np.asanyarray(self.env.u_limit['low']).reshape(-1, 1),
        ])
        c = np.zeros((n, 1))
        sol_lp=solvers.lp(
            matrix(c),
            matrix(G),
            matrix(h),
        )
        if sol_lp['status'] is not 'optimal':
            return False
        return True
    
    def check_one_state_gaussian(self, phi):
        u_ref = np.zeros(2)
        self.rssa.safe_control(u_ref)
        return not self.rssa.if_infeasible
    
    def check_one_state_constant(self, phi):
        u_ref = np.zeros(2)
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
        
    
    
def DR_learning(
    yaml_path = './src/pybullet-dynamics/SCARA_env/SCARA_params.yaml',
    log_root_path='./src/pybullet-dynamics/safety_learning_log/log/',
    visualize_set='only_V',
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

    DR_learn = DRCBFLearning(
        env=env,
        epoch=CBF_learning_kwargs['epoch'],
        elite_ratio=CBF_learning_kwargs['elite_ratio'],
        populate_num=CBF_learning_kwargs['populate_num'],
        init_sigma_ratio=CBF_learning_kwargs['init_sigma_ratio'],
        noise_ratio=CBF_learning_kwargs['noise_ratio'],
        init_params=safe_control_kwargs['param_dict'],
        param_bounds=CBF_learning_kwargs['param_bounds'],
        
        rssa_type=CBF_learning_kwargs['rssa_type'],
        safe_control_kwargs=safe_control_kwargs,
        
        states_sampled_per_param=CBF_learning_kwargs['states_sampled_per_param'],
        iteration_limit=CBF_learning_kwargs['iteration_limit'],
        reward_clip_ratio=CBF_learning_kwargs['reward_clip_ratio']
    )
    
    visualize_set = CBF_learning_kwargs['visualize_set']
    assert visualize_set == 'only_L' or visualize_set == 'only_V' or visualize_set == 'L_and_V'
    if visualize_set == 'only_L':
        DR_learn.learn()
    else: 
        if visualize_set == 'L_and_V':
            DR_learn.learn()
        param_dict = copy(DR_learn.init_params)
        for i, key in enumerate(param_dict.keys()):
            param_dict[key] = DR_learn.mu[i]
        q_1_list, q_2_list, heat_map = DR_learn.visualize(
            param_dict,
            q_sampled_per_dim=CBF_learning_kwargs['q_sampled_per_dim'],
            dq_num=CBF_learning_kwargs['dq_num']
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
    env = SCARAParameterLearningEnv()
    
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
    # env = SCARASafetyIndexLearningEnv(
    #     states_sampled_per_param=10000,
    #     iteration_limit=30000,
    #     init_params={'alpha': 1.0, 'k_v': 1.0, 'beta': 0.0}
    # )
    # SI_learn = RSSASafetyIndexLearning(
    #     env=env,
    #     epoch=10,
    #     elite_ratio=0.1,
    #     populate_num=100,
    #     init_sigma_ratio=0.3,
    #     noise_ratio=0.01,
    #     init_params=env.init_params,
    #     param_bounds={'alpha': [1.0, 5.0], 'k_v': [0.1, 5.0], 'beta': [-1.0, 1.0]},
    # )
    # SI_learn.learn()
    
    # env = SCARAParameterLearningEnv(use_online_adaptation=False)
    # env.param_pred()
    # DR_learn = DRCBFLearning(
    #     env=env,
    #     epoch=10,
    #     elite_ratio=0.1,
    #     populate_num=100,
    #     init_sigma_ratio=0.3,
    #     noise_ratio=0.01,
    #     init_params={'alpha': 1.0, 'k_v': 1.0, 'beta': 0.0},
    #     param_bounds={'alpha': [1.0, 5.0], 'k_v': [0.1, 5.0], 'beta': [-1.0, 1.0]},
    #     sample_points_num=20,
    #     states_sampled_per_param=10000,
    #     iteration_limit=30000,
    # )
    # DR_learn.learn()
    
    pkl_path, log_path = DR_learning()
    with open(pkl_path, 'rb') as file:
        data = pickle.load(file)
    draw_heatmap(data)
    cartesian_heatmap(data)

