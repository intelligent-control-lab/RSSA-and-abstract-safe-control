from typing import Dict
import numpy as np
import gym
import matplotlib.pyplot as plt

try:
    from SegWay_model import SegWayModel
except:
    from SegWay_env.SegWay_model import SegWayModel

class SegWayEnv(gym.Env):
    def __init__(
        self,
        dt=1/240,
        K_m=2.524, 
        K_b=0.189,
        m_0=52.710, m=44.798, J_0=5.108,
        L=0.169, l=0.75, R=0.195,
        g=9.81,
        q_limit={'low': [-100.0, -np.pi/2], 'high': [100.0, np.pi/2]},
        dq_limit={'low': [-5.0, -5.0], 'high': [5.0, 5.0]},
        u_limit={'low': [-20.0], 'high': [20.0]},
        a_safe_limit={'low': -0.1, 'high': 0.1},
    ):
        super().__init__()
        self.robot = SegWayModel(
            dt,
            K_m, 
            K_b,
            m_0, m, J_0,
            L, l, R,
            g,
        )

        self.q_limit = q_limit
        self.dq_limit = dq_limit
        self.u_limit = u_limit
        self.a_safe_limit = a_safe_limit

        self.reset()

    ### Interface for safe control
    def get_phi(self, safety_index_params: Dict):
        '''
        phi = -a_safe_max + a + k_v * da + beta
        '''
        k_v = safety_index_params['k_v']
        beta = safety_index_params['beta']
        a_safe_max = self.a_safe_limit['high']
        a = self.robot.q[1]
        da = self.robot.dq[1]
        return -a_safe_max + a + k_v * da + beta

    ### Interface for safe control
    def get_p_phi_p_Xr(self, safety_index_params: Dict):
        '''
        p_phi_p_Xr = [1.0, 0.0, k_v, 0.0]
        '''
        k_v = safety_index_params['k_v']
        p_phi_p_Xr = np.zeros((1, 4))
        p_phi_p_Xr[0, 1] = 1.0
        p_phi_p_Xr[0, 3] = k_v
        return p_phi_p_Xr
    
    def get_phi_another(self, safety_index_params: Dict):
        k_v = safety_index_params['k_v']
        beta = safety_index_params['beta']
        a_safe_min = self.a_safe_limit['low']
        a = self.robot.q[1]
        da = self.robot.dq[1]
        return a_safe_min - a - k_v * da + beta
    
    def get_p_phi_p_Xr_another(self, safety_index_params: Dict):
        k_v = safety_index_params['k_v']
        p_phi_p_Xr = np.zeros((1, 4))
        p_phi_p_Xr[0, 1] = -1.0
        p_phi_p_Xr[0, 3] = -k_v
        return p_phi_p_Xr

    @property
    def Xr(self):
        # Xr: (q, dq)
        return np.concatenate((self.robot.q, self.robot.dq))

    @property
    def f(self):
        M_inv = np.linalg.pinv(self.robot.M)
        f = np.zeros(4)
        f[:2] = self.robot.dq
        f[2:] = -M_inv @ self.robot.H
        f = f.reshape(-1, 1) # f.shape: (4, 1)
        return f
    
    @property
    def g(self):
        M_inv = np.linalg.pinv(self.robot.M)
        g = np.zeros((4, 1))
        g[2:, 0] = M_inv @ self.robot.B   # g.shape: (4, 1)
        return g

    def step(self, u):
        u = np.clip(u, self.u_limit['low'], self.u_limit['high'])
        self.robot.step(u)
        return self.Xr, None, None, None

    def reset(self):
        self.robot.q = np.zeros(2)
        self.robot.dq = np.zeros(2)

    def render(
        self,
        mode='rgb',
        x_lim=[-1.0, 6.0], y_lim=[-0.2, 1.8],
        figsize=(50.0, 5.0),
        save_path='./src/pybullet-dynamics/SegWay_env/imgs/env_test/',
        img_name='test.jpg',
        
    ):
        p_1 = [self.robot.q[0], self.robot.R]
        p_2 = self.robot.p
        
        fig = plt.figure()
        # plt.xlim(x_lim)
        # plt.ylim(y_lim)

        ax = fig.add_subplot(111)
        ax.plot([p_1[0], p_2[0]], [p_1[1], p_2[1]], c='r') # draw frame
        circle = plt.Circle(p_1, self.robot.R, color='b', fill=False) # draw wheel
        ax.add_patch(circle)
        ax.plot([x_lim[0], x_lim[1]], [0.0, 0.0], c='k') # draw ground
        plt.axis('equal')
        plt.savefig(save_path + img_name)
        plt.close()



if __name__ == '__main__':
    env = SegWayEnv()

    q_d = np.array([0, 0])
    dq_d = np.array([1, 0])
    for i in range(960):
        u = env.robot.PD_control(q_d, dq_d)
        env.step(u)
        
        # print(env.f)
        # print(env.g)
        env.render(img_name=str(i) + '.jpg')