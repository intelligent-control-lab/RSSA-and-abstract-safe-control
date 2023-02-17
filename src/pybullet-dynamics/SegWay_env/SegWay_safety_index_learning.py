from typing import Dict
import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

try:
    from SegWay_env import SegWayEnv
    from SegWay_utils import *
except:
    from SegWay_env.SegWay_env import SegWayEnv
    from SegWay_env.SegWay_utils import *

solvers.options['show_progress'] = False

class SegWaySafetyIndexLearningEnv(SegWayEnv):
    '''
    This class provide interfaces for safety index learning in SegWay enviroment.

    '''
    def __init__(
        self,
        states_sampled_per_param=1000,
        iteration_limit=10000,
        init_params: Dict = {'k_v': 1.0, 'beta': 0.0},
        gamma=0.01, 

        dt=1/240,
        K_m=2.524, 
        K_b=0.189,
        m_0=52.710, m=44.798, J_0=5.108,
        L=0.169, l=0.75, R=0.195,
        g=9.81,
        q_limit={'low': [-1.0, -np.pi/2], 'high': [1.0, np.pi/2]},
        dq_limit={'low': [-5.0, -5.0], 'high': [5.0, 5.0]},
        u_limit={'low': [-10.0], 'high': [10.0]},
        a_safe_limit={'low': -0.1, 'high': 0.1},
    ):
        super().__init__(
            dt,
            K_m,
            K_b,
            m_0, m, J_0,
            L, l, R,
            g,
            q_limit, dq_limit, u_limit, a_safe_limit,
        )
        
        self.states_sampled_per_param = states_sampled_per_param
        self.iteration_limit = iteration_limit
        self.init_params = init_params
        self.gamma = gamma

        self.param_names = set(self.init_params.keys())

        self.get_u_constraints()

    ### Interface for safety index learning
    def evaluate_single_param(self, param_dict: Dict):
        '''
        For a single parameter: (k_v, beta), we evaluate it by sampling states in the whole state space.
        For each state x, we check that if phi(x) >= 0 hold, whether there exists a u, s.t. dot{phi(x)} < -gamma * phi(x) 

        This is equal to solve multiple LP's: for every state x, the object is minimize (p_phi_p_Xr @ g) @ u. Constraints: u \in U_{lim}
        
        After getting u_{min} from the LP, we check whether p_phi_p_Xr @ (f + g @ u_{lim}) < -gamma * phi holds.
        If the above condition holds, reward += 1.
        '''
        assert set(param_dict.keys()) == self.param_names

        reward = 0
        count = 0
        for i in range(self.iteration_limit):
            a = np.random.uniform(low=self.a_safe_limit['low'], high=self.a_safe_limit['high'])
            self.robot.q[1] = a
            self.robot.dq = np.random.uniform(low=self.dq_limit['low'], high=self.dq_limit['high'])
            phi = self.get_phi(safety_index_params=param_dict)
            if phi < 0:
                continue
            else:
                p_phi_p_Xr = self.get_p_phi_p_Xr(safety_index_params=param_dict)
                if self.check_one_state(phi, p_phi_p_Xr):
                    reward += 1
                count += 1
            
            if count == self.states_sampled_per_param:
                break
            
        print(f'total iterations: {i}, count: {count}')
        return reward
    
    ### Interface for safety index learning
    def visualize(
        self, param_dict: Dict, 
        q_sampled_per_dim, dq_num,
        img_name, 
        save_path='./src/pybullet-dynamics/SegWay_env/imgs/safety_index_learning/',
    ):
        a_list = np.linspace(start=self.a_safe_limit['low'], stop=self.a_safe_limit['high'], num=q_sampled_per_dim)
        dq_list = np.random.uniform(low=self.dq_limit['low'], high=self.dq_limit['high'], size=(dq_num, 2))
        penalty_list = []
        for i in range(len(a_list)):
            penalty = 0
            for dq in dq_list:
                self.robot.q[1] = a_list[i]
                self.robot.dq = dq
                phi = self.get_phi(safety_index_params=param_dict)
                if phi >= 0:
                    p_phi_p_Xr = self.get_p_phi_p_Xr(safety_index_params=param_dict)
                    if not self.check_one_state(phi, p_phi_p_Xr):
                        print(f'x: {self.robot.p[0]}, dx: {self.robot.dp[0]}, phi: {phi}')
                        penalty += 1
            penalty_list.append(penalty)
            print(f'a: {self.robot.q[1]}, penalty: {penalty}')
        
        # draw penalty_list
        plt.plot(penalty_list)
        plt.xlabel('a')
        plt.ylabel('penalty')
        # plt.xticks(np.linspace(self.q_limit['low'][0], self.q_limit['high'][0], num=3))
        # plt.yticks(np.linspace(self.q_limit['low'][1], self.q_limit['high'][1], num=3))
        plt.title('SegWay configuration space')
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
    env = SegWaySafetyIndexLearningEnv(
        # init_params={'k_v': 3.55, 'beta': 0.53}
    )
    reward = env.evaluate_single_param(env.init_params)
    print(reward)
    env.visualize(
        param_dict=env.init_params,
        q_sampled_per_dim=100,
        dq_num=100,
        img_name='test_original.jpg',
        # img_name='test_improve.jpg',
    )