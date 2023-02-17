from datetime import datetime
from typing import Dict
import numpy as np
from scipy.spatial import ConvexHull
import warnings
from loguru import logger
import time
import os   

try:
    from panda_latent_env import PandaLatentEnv
    from panda_rod_utils import *
except:
    from panda_rod_env.panda_latent_env import PandaLatentEnv
    from panda_rod_env.panda_rod_utils import *

def turn_on_log():
    now = datetime.now()
    date_time = now.strftime('%Y-%m-%d__%H-%M-%S')
    log_root_path = './src/pybullet-dynamics/panda_rod_env/logs/latent/'
    log_path =log_root_path + date_time
    os.mkdir(log_path)
    logger.add(log_path + '/log.log')
    return log_path
    
class PandaLatentSILearningEnv(PandaLatentEnv):
    def __init__(
        self, 
        states_sampled_per_param=1000,
        iteration_limit=10000,
        init_params: Dict = {'alpha': 1.0, 'k_v': 30.0, 'beta': 0.0},
        gamma=0.1,
        reward_clip_ratio=0.1,
        
        render_flag=False, 
        goal_pose=[0.7, 0.3, 0.4], obstacle_pose=[0.6, 0.1, 0.5], tolerate_error=0.02, 
        d_min=0.05, z_limit={ 'low': [0, -1.0],'high': [0.8, 1.0] },
        log_path=None,
    ):
        super().__init__(
            render_flag=render_flag, 
            goal_pose=goal_pose, obstacle_pose=obstacle_pose, tolerate_error=tolerate_error, 
            d_min=d_min, z_limit=z_limit,
        )
        self.states_sampled_per_param = states_sampled_per_param
        self.iteration_limit = iteration_limit
        self.init_params = init_params
        self.gamma = gamma
        self.param_names = set(self.init_params.keys())
        self.reward_clip_ratio = reward_clip_ratio
        self.log_path = log_path
        
    def get_phi_z(self, safety_index_params: Dict, z, M):
        '''
        phi_z = d_min**a_1 - d**a_1 - a_2 * dot_d + a_3 / M + a_4
        '''
        SIP = safety_index_params
        phi_z =  self.d_min**SIP['a_1'] - z[0]**SIP['a_1'] - SIP['a_2'] * z[1] + SIP['a_3'] / M + SIP['a_4']
        return phi_z
    
    def get_p_phi_p_z(self, safety_index_params: Dict, z, M):
        SIP = safety_index_params
        p_phi_p_z = np.array([[-SIP['a_1'] * z[0]**(SIP['a_1'] - 1), -SIP['a_2']]])
        return p_phi_p_z
    
    def get_p_phi_p_M(self, safety_index_params: Dict, z, M):
        SIP = safety_index_params
        p_phi_p_M = -SIP['a_3'] / M**2
        return p_phi_p_M
    
    def revise_z(self, safety_index_params: Dict, z, M):
        SIP = safety_index_params
        z[1] = (self.d_min**SIP['a_1'] - z[0]**SIP['a_1'] + SIP['a_3'] / M + SIP['a_4']) / SIP['a_2']
        return z
    
    def evaluate_single_param(self, param_dict: Dict):
        logger.debug(f'param: {param_dict}')
        assert set(param_dict.keys()) == self.param_names
        # self.M_limit = self.get_M_range(safety_index_params=param_dict, sample_num=2)
        self.M_limit = {'low': 5.0, 'high': 80.0}
        z_list, M_list, dot_M_max = self.get_valid_z_and_M_list(safety_index_params=param_dict)
        reward = self.aux_evaluate(param_dict=param_dict, z_list=z_list, M_list=M_list, dot_M_max=dot_M_max)
        return reward
        
    def aux_evaluate(self, param_dict: Dict, z_list, M_list, dot_M_max):
        reward = 0
        count = 0
        for z, M in zip(z_list, M_list):
            phi_z = self.get_phi_z(safety_index_params=param_dict, z=z, M=M)
            p_phi_p_z = self.get_p_phi_p_z(safety_index_params=param_dict, z=z, M=M)
            p_phi_p_M = self.get_p_phi_p_M(safety_index_params=param_dict, z=z, M=M)
            f_z, g_z = self.get_f_z_and_g_z(z)
            c = -self.gamma * phi_z - (p_phi_p_z @ f_z).item() - np.abs(p_phi_p_M) * np.abs(dot_M_max)
            LgP = (p_phi_p_z @ g_z).item()
            if LgP > 0:
                v = -M
            else:
                v = M
            if LgP * v <= c:
                reward += 1
            else:
                print('No v solution!')
            count += 1
        
        # strengthen reward
        if count < self.iteration_limit * self.reward_clip_ratio:
            reward = 0.0
        else:
            reward = reward / count
        logger.debug(f'counts: {count}, reward: {reward}')
        return reward
        
    def get_valid_z_and_M_list(self, safety_index_params: Dict):
        valid_z_list = []
        valid_Xr_list = []
        valid_M_list = []
        valid_M_sample_list = []
        valid_dot_M_list = []
        for i in range(self.iteration_limit):
            z = np.random.uniform(low=self.z_limit['low'], high=self.z_limit['high'])
            M = np.random.uniform(low=0.0, high=self.M_limit['high'])
            # z[1] = (self.d_min**alpha - z[0]**alpha + beta) * M / k_v
            z = self.revise_z(safety_index_params=safety_index_params, z=z, M=M)
            if z[1] > self.z_limit['high'][1] or z[1] < self.z_limit['low'][1]:
                continue
            Xr, if_convergence = self.get_Xr_given_z(
                init_Xr=np.concatenate((self.robot.q_init, np.zeros(self.robot.dof))), 
                z_obj=z, 
                lr=5e-1, max_steps=100, eps=1e-2,
            )
            q = Xr[:self.robot.dof]
            dq = Xr[self.robot.dof:]
            if np.all(q >= self.q_limit['low']) and np.all(q <= self.q_limit['high']) and \
                np.all(dq >= self.dq_limit['low']) and np.all(dq <= self.dq_limit['high']) and \
                if_convergence:
                self.robot.set_joint_states(q, dq)
                self.update_latent_info(if_update_V_and_M=True, if_update_grad=False)
                M_sample = self.M
                if M_sample >= M:
                    u = self.u_vertices[np.random.randint(self.u_vertices.shape[0])]
                    self.step(u)
                    self.update_latent_info(if_update_V_and_M=True, if_update_grad=False)
                    dot_M = (self.M - M_sample) * 240
                    valid_z_list.append(z)
                    valid_Xr_list.append(Xr)
                    valid_M_list.append(M)
                    valid_M_sample_list.append(M_sample)
                    valid_dot_M_list.append(dot_M)
                    logger.debug(f'i: {i}, z: {z}, M: {M}, M_sample: {M_sample}, dot_M: {dot_M}')
        valid_dot_M_list = np.asanyarray(valid_dot_M_list)
        valid_dot_M_list = np.sort(valid_dot_M_list)
        dot_M_max = valid_dot_M_list[int(0.95 * len(valid_dot_M_list))]
        # dot_M_max = np.max(valid_dot_M_list)
        
        ### TODO: save data for debug
        if self.log_path is not None:
            data = {
                'valid_z_list': valid_z_list,
                'valid_Xr_list': valid_Xr_list,
                'valid_M_list': valid_M_list,
                'valid_M_sample_list': valid_M_sample_list,
                'valid_dot_M_list': valid_dot_M_list,
                'dot_M_max': dot_M_max,
            }
            with open(self.log_path + '/data.pkl', 'wb') as file:
                pickle.dump(data, file) 
        ### END TODO
        
        return valid_z_list, valid_M_list, dot_M_max
    
        
        

if __name__ == '__main__':
    log_path = None
    log_path = turn_on_log()
    env = PandaLatentSILearningEnv(
        d_min=0.05, 
        iteration_limit=1000, 
        log_path=log_path,
        init_params={'a_1': 1.0, 'a_2': 10.0, 'a_3': 10.0, 'a_4': 0.01},
    )
    env.evaluate_single_param(param_dict=env.init_params)
    
    # with open('./src/pybullet-dynamics/panda_rod_env/logs/latent/2022-08-05__18-26-53/data.pkl', 'rb') as file:
    #     data = pickle.load(file)
    # env.aux_evaluate(
    #     param_dict={'a_1': 1.0, 'a_2': 1.0, 'a_3': 20.0, 'a_4': 0.01},
    #     z_list=data['valid_z_list'], M_list=data['valid_M_list'], dot_M_max=data['dot_M_max'],
    # )    
        