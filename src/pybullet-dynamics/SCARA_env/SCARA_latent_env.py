from typing import Dict
import numpy as np
from numpy import sin, cos
from scipy.spatial import ConvexHull

try:
    from SCARA_safety_index_learning  import SCARASafetyIndexLearningEnv
    from SCARA_utils import *
except:
    from SCARA_env.SCARA_safety_index_learning import SCARASafetyIndexLearningEnv
    from SCARA_env.SCARA_utils import *
    
class SCARALatentEnv(SCARASafetyIndexLearningEnv):
    '''
    z = [x_{eff}, \dot{x}_{eff}].T, v = [\ddot{x}_{eff}]
    '''
    def __init__(
        self, 
        states_sampled_per_param=3000,
        iteration_limit=10000,
        init_params: Dict = {'alpha': 1.0, 'k_v': 1.0, 'beta': 0.0},
        gamma=0.01,
        dt=1 / 240, m_1=1, l_1=1, m_2=1, l_2=1, 
        q_limit={ 'low': [-np.pi / 2, -np.pi / 2],'high': [np.pi / 2, np.pi / 2] }, 
        dq_limit={ 'low': [-2, -2],'high': [2, 2] }, 
        u_limit={ 'low': [-20, -20],'high': [20, 20] },
        max_dot_M_sample_num=10000,
    ):
        super().__init__(states_sampled_per_param, iteration_limit, init_params, gamma, dt,  m_1, l_1, m_2, l_2, q_limit, dq_limit, u_limit)
        
        u_max = self.u_limit['high']
        u_min = self.u_limit['low']
        self.u_vertices = []
        for indices in exhaust(len(u_max)):
            u = np.zeros_like(u_max)
            for i in range(len(u_max)):
                u[i] = u_max[i] if indices[i] else u_min[i]
            self.u_vertices.append(u)
        self.u_vertices = np.asanyarray(self.u_vertices)
        hull = ConvexHull(self.u_vertices)
        vertice_order = hull.vertices
        self.u_vertices = self.u_vertices[vertice_order]
        
        self.max_dot_M = self.get_max_dot_M(max_dot_M_sample_num)
        
    @property
    def f_z(self):
        f_z = np.zeros((2, 1))
        f_z[0, 0] = self.z[1]
        return f_z
    
    @property
    def g_z(self):
        g_z = np.zeros((2, 1))
        g_z[1, 0] = 1.0
        return g_z
    
    @property
    def C_x(self):
        '''
        v = C_x @ \bar{u}, where \bar{u} = [1.0, u].T
        '''
        q_1, q_2 = self.robot.q
        dq_1, dq_2 = self.robot.dq
        l_1 = self.robot.l_1
        l_2 = self.robot.l_2
        
        K = np.zeros((1, 2))
        K[0, 0] = (-l_1 * sin(q_1) - l_2 * sin(q_1 + q_2))
        K[0, 1] = (-l_2 * sin(q_1 + q_2))
        M_inv = np.linalg.inv(self.robot.M)
        H = self.robot.H.reshape(-1, 1)
        
        C_x = np.zeros((1, 3))
        C_x[:, 1:] = K @ M_inv 
        C_x[0, 0] = (-l_1 * cos(q_1) - l_2 * cos(q_1 + q_2)) * dq_1**2 + (-l_2 * cos(q_1 + q_2)) * dq_1 * dq_2 + \
                    (-l_2 * cos(q_1 + q_2)) * dq_2**2 + (-l_2 * cos(q_1 + q_2)) * dq_1 * dq_2 + \
                    (-K @ M_inv @ H).item()
        # C_x[0, 0] = 0.0
        return C_x
    
    @property
    def V_x(self):
        # return vertices of V_x's convex hull
        V_x_vertices = []
        C_x = self.C_x
        for u in self.u_vertices:
            v = self.get_v(u, C_x)
            V_x_vertices.append(v)
        V_x_vertices = np.asanyarray(V_x_vertices)
        return V_x_vertices
    
    def get_z(self):
        x_eff = self.robot.p[0]
        dot_x_eff = self.robot.dp[0]
        return np.array([x_eff, dot_x_eff])
    
    def set_z(self, z=None):
        if z is None:
            self.z = self.get_z()
        else:
            self.z = z
    
    def get_M_x(self):
        '''
        M_x = min ||v||, \forall v \in V_x. In SCARA, v is of onr dimension. 
        '''
        V_x_vertices = self.V_x
        V_x_vertices = np.squeeze(V_x_vertices)
        V_x_vertices.sort()
        V_x_max = V_x_vertices[-1]
        V_x_min = V_x_vertices[0]
        if V_x_max > 0 and V_x_min < 0:
            return V_x_max if V_x_max + V_x_min < 0 else -V_x_min
        else:
            raise Exception('V_x is empty!')
    
    def set_M_x(self, M_x=None):
        if M_x is None:
            self.M_x = self.get_M_x()
        else:
            self.M_x = M_x
    
    def get_max_dot_M(self, sample_num=10000):
        max_dot_M = -np.inf
        dot_M_list = []
        M_list = []
        v_max_list = []
        v_min_list = []
        Xr_list = []
        self.M_range = {'low': np.inf, 'high': -np.inf}
        for i in range(sample_num):
            self.robot.q = np.random.uniform(low=self.q_limit['low'], high=self.q_limit['high'])
            self.robot.dq = np.random.uniform(low=self.dq_limit['low'], high=self.dq_limit['high'])
            # u = self.u_vertices[np.random.choice(len(self.u_vertices))]
            u = np.random.uniform(low=self.u_limit['low'], high=self.u_limit['high'])
            
            if self.detect_collision():
                continue
            
            V_x_vertices = self.V_x
            V_x_vertices = np.squeeze(V_x_vertices)
            V_x_vertices.sort()
            v_x_max = V_x_vertices[-1]
            v_x_min = V_x_vertices[0]
            v_max_list.append(v_x_max)
            v_min_list.append(v_x_min)
            
            try:
                M_x_1 = self.get_M_x()
                self.step(u)
                M_x_2 = self.get_M_x()
                M_max = max(M_x_1, M_x_2)
                M_min = min(M_x_1, M_x_2)
                if M_max > self.M_range['high']:
                    self.M_range['high'] = M_max
                if M_min < self.M_range['low']:
                    self.M_range['low'] = M_min
            except:
                print(self.Xr)
                print('No M_x solution!')
                continue 
            dot_M = np.abs(M_x_2 - M_x_1) / self.robot.dt
            max_dot_M = dot_M if dot_M > max_dot_M else max_dot_M
            
            if dot_M > 400:
                Xr_list.append(self.Xr)
            
            if v_x_max < 1.0 or v_x_min > -1.0:
                print(self.Xr)
                print(f'v_max: {v_x_max}, v_min: {v_x_min}')
            
            dot_M_list.append(dot_M)
            M_list.append(M_max)
                
        quick_hist(dot_M_list, img_name='dot_M.png', xlabel='dot_M', ylabel='sample_num')
        quick_hist(M_list, img_name='M.png', xlabel='M', ylabel='sample_num')
        quick_scatter(v_max_list, v_min_list, img_name='v_limits.png', xlabel='v_max', ylabel='v_min')
        Xr_list = np.asanyarray(Xr_list)
        quick_scatter(Xr_list[:, 0], Xr_list[:, 1], img_name='big_q.png', xlabel='q_1', ylabel='q_2')
        quick_scatter(Xr_list[:, 2], Xr_list[:, 3], img_name='big_dq.png', xlabel='dq_1', ylabel='dq_2')
        
        return max_dot_M
    
    def get_v(self, u, C_x=None):
        if C_x is None:
            C_x = self.C_x
        u_bar = np.concatenate(([1.0], u)).reshape(-1, 1)
        v = (C_x @ u_bar).item()
        return np.array([v])
    
    def get_phi_z(self, safety_index_params: Dict, z=None, M_x=None):
        '''
        phi = -x_wall**alpha + x_eff**alpha + k_v * dot{x_eff} / M_x + beta
        '''
        alpha = safety_index_params['alpha']
        beta = safety_index_params['beta']
        k_v = safety_index_params['k_v']
        self.set_z(z)
        self.set_M_x(M_x)
        
        x_eff, dot_x_eff = self.z
        # phi = -self.wall_x**alpha + x_eff**alpha + k_v * dot_x_eff / self.M_x + beta
        phi = -self.wall_x**alpha + x_eff**alpha + k_v * dot_x_eff + beta
        return phi
    
    def get_p_phi_p_z(self, safety_index_params: Dict):
        # call it after self.get_phi_z()
        alpha = safety_index_params['alpha']
        k_v = safety_index_params['k_v']
        p_phi_p_z = np.zeros((1, 2))
        p_phi_p_z[0, 0] = alpha * self.z[0]**(alpha - 1)
        p_phi_p_z[0, 1] = k_v / self.M_x
        return p_phi_p_z
    
    def get_p_phi_p_M_x(self, safety_index_params: Dict):
        # call it after self.get_phi_z()
        k_v = safety_index_params['k_v']
        return -k_v * self.z[1] / self.M_x**2
    
    ### Interface for safety index learning
    def evaluate_single_param(
        self, 
        param_dict: Dict,
        z_limit={ 'low': [0.5, -3.0],'high': [1.5, 3.0] }, 
    ):
        '''
        For a single parameter: (alpha, k_v, beta), we evaluate it by sampling states in the whole state space.
        We check whether p_phi_p_z @ (f + g @ v) + |p_phi_p_Mx| * dot_Mx_max < -gamma * phi holds.
        If the above condition holds, reward += 1.
        '''
        assert set(param_dict.keys()) == self.param_names

        reward = 0
        count = 0
        for i in range(self.iteration_limit):
            z = np.random.uniform(low=z_limit['low'], high=z_limit['high'])
            # M_x = np.random.uniform(low=self.M_range['low'], high=self.M_range['high'])
            phi = self.get_phi_z(safety_index_params=param_dict, z=z, M_x=None)
            if self.detect_collision() or phi < 0:
                continue
            else:
                p_phi_p_z = self.get_p_phi_p_z(safety_index_params=param_dict)
                p_phi_p_M_x = self.get_p_phi_p_M_x(safety_index_params=param_dict)
                if self.check_one_state(phi, p_phi_p_z, p_phi_p_M_x):
                    reward += 1
                count += 1
            
            if count == self.states_sampled_per_param:
                break
        
        # strengthen reward
        if count < self.iteration_limit * 0.01:
            reward = 0.0
        else:
            reward = reward / count

        print(f'total iterations: {i}, count: {count}, reward: {reward}')
        return reward
    
    def check_one_state(self, phi, p_phi_p_z, p_phi_p_M_x):
        f_z = self.f_z
        g_z = self.g_z
        v_min = self.find_u_min(p_phi_p_z @ g_z)
        v_min = np.asanyarray([[v_min]])
        dot_phi = p_phi_p_z @ (f_z + g_z @ v_min) + np.abs(p_phi_p_M_x) * self.max_dot_M
        if dot_phi.item() < -self.gamma * phi:
            return True
        else:
            return False
        
    def find_u_min(self, c):
        # c is just a scalar
        c = c.item()
        if c >= 0:
            return -self.M_x
        else:
            return self.M_x
    
    
    
    
if __name__ == '__main__':
    # env = SCARALatentEnv(dt=1/1000)
    # print(env.max_dot_M)
    # for _ in range(1000):
    #     env.robot.q = np.random.uniform(low=env.q_limit['low'], high=env.q_limit['high'])
    #     env.robot.dq = np.random.uniform(low=env.dq_limit['low'], high=env.dq_limit['high'])
    #     u = np.random.uniform(low=env.u_limit['low'], high=env.u_limit['high'])
        
    #     v_the = env.get_v(u)[0]
    #     # v_the_C_x = (env.C_x @ np.concatenate(([1.0], u)).reshape(-1, 1)).item()
    #     dot_x_eff_1 = env.robot.dp[0]
    #     env.step(u)
    #     dot_x_eff_2 = env.robot.dp[0]
    #     v_exp = (dot_x_eff_2 - dot_x_eff_1) / env.robot.dt
    #     print(f'v_the: {v_the}, v_exp: {v_exp}, diff_ratio: {(v_the-v_exp) / np.abs(v_the)}')
    #     # print(f'v_the: {v_the}, v_exp: {v_the_C_x}, diff_ratio: {(v_the-v_the_C_x) / np.abs(v_the)}')
    
    env = SCARALatentEnv(dt=1/1000)
    env.max_dot_M = 0.0
    env.evaluate_single_param(param_dict=env.init_params)
    