from typing import Dict
import numpy as np
import gym
import matplotlib.pyplot as plt

try:
    from SCARA_model import SCARAModel
except:
    from SCARA_env.SCARA_model import SCARAModel

class SCARAEnv(gym.Env):
    def __init__(
        self,
        dt=1/240,
        m_1=1.0, l_1=1.0,
        m_2=1.0, l_2=1.0,
        q_limit={'low': [-np.pi/2, -np.pi/2], 'high': [np.pi/2, np.pi/2]},
        dq_limit={'low': [-5.0, -5.0], 'high': [5.0, 5.0]},
        u_limit={'low': [-20.0, -20.0], 'high': [20.0, 20.0]},
    ):
        super().__init__()
        self.robot = SCARAModel(
            dt,
            m_1, l_1,
            m_2, l_2,
        )

        self.q_limit = q_limit
        self.dq_limit = dq_limit
        self.u_limit = u_limit

        self.reset()
        self.set_wall()

    ### Interface for safe control
    def get_phi(self, safety_index_params: Dict):
        '''
        phi = -wall_x**alpha + x_eff**alpha + k_v * dot{x_eff} + beta
        '''
        alpha = safety_index_params['alpha']
        beta = safety_index_params['beta']
        k_v = safety_index_params['k_v']
        if self.robot.p[0] <= 0:
            return -1.0
        return -self.wall_x**alpha + self.robot.p[0]**alpha + k_v * self.robot.dp[0] + beta

    ### Interface for safe control
    def get_p_phi_p_Xr(self, safety_index_params: Dict):
        '''
        p_phi_p_Mr = [alpha * x_eff**(alpha - 1), 0.0, k_v, 0.0]
        '''
        alpha = safety_index_params['alpha']
        k_v = safety_index_params['k_v']
        if self.robot.p[0] <= 0:
            return np.array([[1.0, 0.0, k_v, 0.0]])
        p_phi_p_Mr = np.array([
            [alpha * self.robot.p[0]**(alpha - 1), 0.0, k_v, 0.0]
        ])
        p_phi_p_Xr = p_phi_p_Mr @ self.p_Mr_p_Xr    # p_phi_p_Xr.shape: (1, 4)
        return p_phi_p_Xr

    @property
    def Xr(self):
        # Xr: (q, dq)
        return np.concatenate((self.robot.q, self.robot.dq))
    
    @property
    def Mr(self):
        # Mr: (p, dp)
        return np.concatenate((self.robot.p, self.robot.dp))

    @property
    def p_Mr_p_Xr(self):
        p_Mr_p_Xr = np.zeros((4, 4))
        p_Mr_p_Xr[:2, :2] = p_Mr_p_Xr[2:, 2:] = self.robot.J
        p_Mr_p_Xr[2:, :2] = self.robot.p_dp_p_q
        return p_Mr_p_Xr

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
        g = np.zeros((4, 2))
        g[2:, :] = M_inv    # g.shape: (4, 2)
        return g

    def get_disturbance(self, F_ext):
        # dot_Xr = f(Xr) + g(Xr) @ u + disturbance
        M_inv = np.linalg.pinv(self.robot.M)
        disturbance = np.zeros(4)
        disturbance[2:] = M_inv @ self.robot.J.T @ F_ext
        return disturbance

    def get_F_ext(self):
        return np.zeros(2)

    def step(self, u):
        u = np.clip(u, self.u_limit['low'], self.u_limit['high'])
        F_ext = self.get_F_ext()
        self.robot.step(u, F_ext)
        return self.Xr, None, None, None

    def reset(self):
        self.robot.q = np.array([np.pi/2, 0.0])
        self.robot.dq = np.zeros(2)

    def set_wall(
        self,
        x=1.5, 
        y_min=-1.5, y_max=1.5,
    ):
        self.wall_x = x
        self.wall_y_min = y_min
        self.wall_y_max = y_max

    def detect_collision(self, eps=-0.1):
        return self.robot.p[0] + eps > self.wall_x

    def render(
        self,
        mode='rgb',
        x_lim=[-2.5, 2.5], y_lim=[-2.5, 2.5],
        figsize=(5.0, 5.0),
        save_path='./src/pybullet-dynamics/SCARA_env/imgs/way_points/',
        img_name='test.jpg',
        
    ):
        p_1 = [0.0, 0.0]
        p_2 = [self.robot.l_1 * np.cos(self.robot.q[0]), self.robot.l_1 * np.sin(self.robot.q[0])]
        p_3 = self.robot.p
        
        plt.figure(figsize=figsize)
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.plot([p_1[0], p_2[0]], [p_1[1], p_2[1]], c='r')
        plt.plot([p_2[0], p_3[0]], [p_2[1], p_3[1]], c='r')
        plt.plot([self.wall_x, self.wall_x], [self.wall_y_min, self.wall_y_max], c='b') # draw the wall
        plt.savefig(save_path + img_name)
        plt.close()


if __name__ == '__main__':
    env = SCARAEnv(
        dt=1/240,
        m_1=1.0, l_1=1.0,
        m_2=1.0, l_2=1.0,
    )
    
    env.robot.q = np.array([0.9, -0.2])
    env.render(save_path='./src/pybullet-dynamics/SCARA_env/imgs/points/',)

    # q_way_points = np.linspace(start=[np.pi/2 - 0.05, 0.0], stop=[-np.pi/3, -np.pi], num=400)

    # for i, q_d in enumerate(q_way_points):
    #     u = env.robot.computed_torque_control(q_d=q_d)
    #     # print(u)
    #     # print(q_d - env.robot.q)
    #     for _ in range(3):
    #         env.step(u)
    #         print(env.Xr)
    #     # env.render(img_name=str(i) + '.jpg')

    # env.robot.dt = 1e-3
    # Mr_1 = env.Mr
    # Xr_1 = env.Xr
    # p_Mr_p_Xr = env.p_Mr_p_Xr
    # env.step(u)
    # Mr_2 = env.Mr
    # Xr_2 = env.Xr
    # d_Xr = Xr_2 - Xr_1
    # d_Mr = Mr_2 - Mr_1
    # print(d_Xr)
    # print(d_Mr)
    # print(d_Mr - p_Mr_p_Xr @ d_Xr)

    # # for i in range(240):
    # #     env.step(np.zeros(2))
    # #     env.render(
    # #         save_path='./src/pybullet-dynamics/SCARA_env/imgs/F_ext_test_1/',
    # #         img_name=str(i) + '.jpg',
    # #     )



