from typing import Dict
import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

try:
    from SCARA_env import SCARAEnv
    from SCARA_utils import *
except:
    from SCARA_env.SCARA_env import SCARAEnv
    from SCARA_env.SCARA_utils import *

solvers.options['show_progress'] = False

class SCARASafetyIndexLearningEnv(SCARAEnv):
    '''
    This class provide interfaces for safety index learning in SCARA enviroment.

    '''
    def __init__(
        self,
        states_sampled_per_param=1000,
        iteration_limit=10000,
        init_params: Dict = {'alpha': 1.0, 'k_v': 1.0, 'beta': 0.0},
        gamma=0.01, 

        dt=1/240,
        m_1=1.0, l_1=1.0,
        m_2=1.0, l_2=1.0,
        q_limit={'low': [-np.pi, -np.pi/2], 'high': [np.pi, np.pi]},
        dq_limit={'low': [-2.0, -2.0], 'high': [2.0, 2.0]},
        u_limit={'low': [-20.0, -20.0], 'high': [20.0, 20.0]},
    ):
        super().__init__(dt, m_1, l_1, m_2, l_2, q_limit, dq_limit, u_limit)
        
        self.states_sampled_per_param = states_sampled_per_param
        self.iteration_limit = iteration_limit
        self.init_params = init_params
        self.gamma = gamma

        self.param_names = set(self.init_params.keys())

        self.get_u_constraints()

    ### Interface for safety index learning
    def evaluate_single_param(self, param_dict: Dict):
        '''
        For a single parameter: (alpha, k_v, beta), we evaluate it by sampling states in the whole state space.
        For each state x, we check that if phi(x) >= 0 hold, whether there exists a u, s.t. dot{phi(x)} < -gamma * phi(x) 

        This is equal to solve multiple LP's: for every state x, the object is minimize (p_phi_p_Xr @ g) @ u. Constraints: u \in U_{lim}
        
        After getting u_{min} from the LP, we check whether p_phi_p_Xr @ (f + g @ u_{lim}) < -gamma * phi holds.
        If the above condition holds, reward += 1.
        '''
        assert set(param_dict.keys()) == self.param_names

        reward = 0
        count = 0
        for i in range(self.iteration_limit):
            self.robot.q = np.random.uniform(low=self.q_limit['low'], high=self.q_limit['high'])
            self.robot.dq = np.random.uniform(low=self.dq_limit['low'], high=self.dq_limit['high'])
            phi = self.get_phi(safety_index_params=param_dict)
            if self.detect_collision() or phi < 0:
                continue
            else:
                p_phi_p_Xr = self.get_p_phi_p_Xr(safety_index_params=param_dict)
                if not np.max(np.abs(p_phi_p_Xr)) < 1e10:   # deal with the situation where p_phi_p_Xr is NAN
                    continue
                if self.check_one_state(phi, p_phi_p_Xr):
                    reward += 1
                count += 1
            
            if count == self.states_sampled_per_param:
                break
        
        # strengthen reward
        if count < self.iteration_limit * 0.3:
            reward = 0.0
        else:
            reward = reward / count

        print(f'total iterations: {i}, count: {count}')
        return reward
    
    ### Interface for safety index learning
    def visualize(
        self, param_dict: Dict, 
        q_sampled_per_dim, dq_num,
        img_name, 
        save_path='./src/pybullet-dynamics/SCARA_env/imgs/safety_index_learning/',
    ):
        q_1_list = np.linspace(start=self.q_limit['low'][0], stop=self.q_limit['high'][0], num=q_sampled_per_dim)
        q_2_list = np.linspace(start=self.q_limit['low'][1], stop=self.q_limit['high'][1], num=q_sampled_per_dim)
        dq_list = np.random.uniform(low=self.dq_limit['low'], high=self.dq_limit['high'], size=(dq_num, 2))
        heat_map = np.zeros((len(q_1_list), len(q_2_list)))
        for i in range(len(q_1_list)):
            for j in range(len(q_2_list)):
                self.robot.q = np.asanyarray([q_1_list[i], q_2_list[j]])
                penalty = 0
                for dq in dq_list:
                    self.robot.dq = dq
                    phi = self.get_phi(safety_index_params=param_dict)
                    if not self.detect_collision() and phi >= 0:
                        p_phi_p_Xr = self.get_p_phi_p_Xr(safety_index_params=param_dict)
                        if np.max(np.abs(p_phi_p_Xr)) < 1e10:   # deal with the situation where p_phi_p_Xr is NAN
                            if not self.check_one_state(phi, p_phi_p_Xr):
                                print(f'x: {self.robot.p[0]}, dx: {self.robot.dp[0]}, phi: {phi}')
                                penalty += 1
                heat_map[i, j] = penalty
                print(f'q: {self.robot.q}, penalty: {penalty}')
        
        # draw heatmap
        plt.imshow(heat_map)
        plt.xlabel('q_1')
        plt.ylabel('q_2')
        # plt.xticks(np.linspace(self.q_limit['low'][0], self.q_limit['high'][0], num=3))
        # plt.yticks(np.linspace(self.q_limit['low'][1], self.q_limit['high'][1], num=3))
        plt.title('SCARA configuration space')
        plt.savefig(save_path + img_name)
    
    def check_one_state(self, phi, p_phi_p_Xr):
        f = self.f
        g = self.g
        u_min = self.find_u_min(p_phi_p_Xr @ g)
        dot_phi = p_phi_p_Xr @ (f + g @ u_min)
        if dot_phi.item() < -self.gamma * phi:
            return True
        else:
            return False

    def find_u_min(self, c):
        # find max c.T @ x   subject to A @ x < b:
        c = c.reshape(-1, 1)
        c = c / np.max(np.abs(c)) if np.max(np.abs(c)) > 1e-3 else c # resize c to avoid too big objective, which may lead to infeasibility (a bug of cvxopt)
        sol_lp=solvers.lp(
            matrix(c),
            matrix(self.A_u_min),
            matrix(self.b_u_min)
        )

        assert sol_lp['x'] is not None
        return np.vstack(np.array(sol_lp['x']))
                
    def get_u_constraints(self):
        n = len(self.u_limit['high'])
        self.A_u_min = np.vstack([
            np.eye(n),
            -np.eye(n),
        ])
        self.b_u_min = np.vstack([
            np.asanyarray(self.u_limit['high']).reshape(-1, 1),
            -np.asanyarray(self.u_limit['low']).reshape(-1, 1),
        ])



if __name__ == '__main__':
    env = SCARASafetyIndexLearningEnv(
        init_params={'alpha': 1.0, 'k_v': 0.5, 'beta': 0.939}
    )
    # reward = env.evaluate_single_param(env.init_params)
    # print(reward)
    env.visualize(
        param_dict=env.init_params,
        q_sampled_per_dim=20,
        dq_num=100,
        # img_name='test_original.jpg',
        img_name='test_improve.jpg',
    )
    