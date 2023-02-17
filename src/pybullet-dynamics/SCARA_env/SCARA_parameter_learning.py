import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, RBF

try:
    from SCARA_env import SCARAEnv
    from SCARA_utils import *
except:
    from SCARA_env.SCARA_env import SCARAEnv
    from SCARA_env.SCARA_utils import *

class SCARAParameterLearningEnv(SCARAEnv):
    def __init__(
        self,
        use_online_adaptation=True,
        m_2_mean_init=1.0,
        m_2_std_init=0.1,
        
        dt=1/240,
        m_1=1.0, l_1=1.0,
        m_2=1.0, l_2=1.0,
        q_limit={'low': [-np.pi/2, -np.pi/2], 'high': [np.pi/2, np.pi/2]},
        dq_limit={'low': [-2.0, -2.0], 'high': [2.0, 2.0]},
        u_limit={'low': [-5.0, -5.0], 'high': [5.0, 5.0]},
    ):
        super().__init__(dt, m_1, l_1, m_2, l_2, q_limit, dq_limit, u_limit)

        # for online adaptation part 
        self.use_online_adaptation = use_online_adaptation
        self.first_predict = True
        self.m_2_mean_init = m_2_mean_init
        self.m_2_std_init = m_2_std_init

    ### Interface for safe control
    def param_pred(self):
        '''
        Here we assume the underlying parameter follows a Gaussian process. 
        In SCARA, the mean and std of m_2 will be returned.
        
        In online adaptation part, using EKF:
        - predict:
            - m_2_predict = m_2_update
            - _, R_k**0.5 = GP_predict()
            - P_{k| k - 1} = P_{k - 1| k - 1}
        - update:
            - m_2_obs = get_m_2_data()
            - S_k = P_{k| k - 1} + R_k
            - K_k = P_{k| k - 1} * S_k^{-1} 
            - m_2_update = m_2_predict + K_k * (m_2_obs - m_2_predict)
            - P_{k| k} = (1 - K_k) * P_{k| k - 1}
        '''
        if not self.use_online_adaptation:
            self.m_2_mean = self.m_2_mean_init
            self.m_2_std = self.m_2_std_init
            return self.m_2_mean, self.m_2_std
        
        # for online adpatation
        R_std = self.m_2_std_init
        if self.first_predict:
            self.first_predict = False
            self.m_2_mean = self.m_2_mean_init
            self.m_2_std = R_std
            return self.m_2_mean, R_std
        else:
            # predict part
            m_2_predict = self.m_2_mean
            P = self.m_2_std ** 2
            std_predict = self.m_2_std
            
            # update part
            f_obs = self.f[2:, :]
            f_predict = self.get_f_data(m_2_predict)
            p_f_p_m_2 = self.get_p_f_p_m_2(m_2_predict)
            S = p_f_p_m_2 @ p_f_p_m_2.T * P + R_std**2 * np.eye(2)
            # S = p_f_p_m_2 @ p_f_p_m_2.T * P
            K = P * p_f_p_m_2.T @ np.linalg.inv(S)
            m_2_update = m_2_predict + (K @ (f_obs - f_predict)).item()
            P = (1 - (K @ p_f_p_m_2).item()) * P
            self.m_2_mean = m_2_update
            self.m_2_std = P**0.5
            
            return m_2_predict, std_predict
    
    ### Interface for safe control
    def sample_f_g_points(self, points_num=10):
        '''
        Sample f(theta | x) and g(theta | x), where theta | x is Gaussian \\
        In SCARA, theta is m_2 
        '''
        mean = self.m_2_mean
        std = self.m_2_std
        true_m_2 = self.robot.m_2
        f_points = []
        g_points = []
        
        cnt = 0
        while cnt < points_num:
            m_2 = np.random.randn() * std + mean
            if m_2 <= 0:    # truncated Gaussian
                continue
            self.robot.m_2 = m_2
            f_points.append(self.f)
            g_points.append(self.g)
            cnt += 1
        self.robot.m_2 = true_m_2
        return f_points, g_points
    
    ### Interface for safe control
    def set_param(self, param):
        self.robot.m_2 = param
        
    ### Interface for safe control
    def get_param(self):
        return self.robot.m_2
    
    def get_f_data(self, m_2):
        true_m_2 = self.robot.m_2
        self.robot.m_2 = m_2
        f = self.f[2:, :]
        self.robot.m_2 = true_m_2
        return f
    
    def get_p_f_p_m_2(self, m_2):
        true_m_2 = self.robot.m_2
        self.robot.m_2 = m_2
        m_1 = self.robot.m_1
        a_1, b_1, c_1, d_1, a_2, b_2, c_2, d_2 = self.get_f_coefs()
        p_f_p_m_2 = np.zeros(2)
        p_f_p_m_2[0] = (b_1 * c_1 - d_1 * a_1) * m_1 / (c_1 * m_1 + d_1 * m_2)**2
        p_f_p_m_2[1] = (b_2 * c_2 - d_2 * a_2) * m_1 / (c_2 * m_1 + d_2 * m_2)**2
        self.robot.m_2 = true_m_2
        p_f_p_m_2 = p_f_p_m_2.reshape(-1, 1)
        return p_f_p_m_2
        
    def get_f_coefs(self):
        '''
        Coeficients of f(m_1, m_2): \\
        f_1 = \frac{a_1 * m_1 + b_1 * m_2}{c_1 * m_1 + d_1 * m_2}   \\
        f_2 = \frac{a_2 * m_1 + b_2 * m_2}{c_2 * m_1 + d_2 * m_2}   \\
        Where f_1 and f_2 are f[2:]
        '''
        # for ease of notation, rename some of the parameters
        l_1 = self.robot.l_1
        l_2 = self.robot.l_2
        q_1 = self.robot.q[0]
        q_2 = self.robot.q[1]
        dq_1 = self.robot.dq[0]
        dq_2 = self.robot.dq[1]

        # a_1
        a_1 = 0

        # b_1
        tmp_1 = dq_1**2 * (3 * l_1 * cos(q_2) + 2 * l_2)
        tmp_2 = 2 * dq_2 * l_2 * (2 * dq_1 + dq_2)
        b_1 = 3 * sin(q_2) * (tmp_1 + tmp_2)

        # c_1
        c_1 = 4 * l_1

        # d_1
        d_1 = (12 - 9 * cos(q_2)**2) * l_1

        # a_2 
        a_2 = -6 * dq_1**2 * l_1**2 * sin(q_2)

        # b_2
        tmp_1 = 2 * dq_1**2 * (3 * l_1**2 +  3 * l_1 * l_2 * cos(q_2) + l_2**2)
        tmp_2 = dq_2 * l_2 * (6 * dq_1 * l_1 * cos(q_2) + 4 * dq_1 * l_2 + 3 * dq_2 * l_1 * cos(q_2) + 2 * dq_2 * l_2)
        b_2 = -3 * sin(q_2) * (tmp_1 + tmp_2)

        # c_2
        c_2 = 4 * l_1 * l_2

        # d_2 
        d_2 = (12 - 9 * cos(q_2)**2) * l_1 * l_2

        return a_1, b_1, c_1, d_1, a_2, b_2, c_2, d_2

    def get_m_2_data(self):
        f_1, f_2 = self.f[2:, 0] + self.get_disturbance(self.get_F_ext())[2:]
        a_1, b_1, c_1, d_1, a_2, b_2, c_2, d_2 = self.get_f_coefs()
        m_1 = self.robot.m_1

        # deal with the situation when (b_1 - d_1 * f_1) or (b_2 - d_2 * f_2) is zero
        try:
            m_2 = (c_1 * f_1 - a_1) / (b_1 - d_1 * f_1) * m_1
        except Exception:
            try:
                m_2 = (c_2 * f_2 - a_2) / (b_2 - d_2 * f_2) * m_1
            except Exception:
                return None

        if not m_2 < 1e10:
            m_2 = self.robot.m_2
        return m_2

    def reset(self):
        self.robot.q = np.array([np.pi/2, 0.0])
        self.robot.dq = np.zeros(2)
        self.first_predict = True

    

if __name__ == '__main__':
    env = SCARAParameterLearningEnv(
        dt=1/240,
        m_1=1.0, l_1=1.0,
        m_2=1.0, l_2=1.0,
    )

    # env.robot.set_joint_states([-1.5, 1.0], [2.0, -2.0])
    # env.robot.m_1 = 1
    # m_1 = env.robot.m_1
    # a_1, b_1, c_1, d_1, a_2, b_2, c_2, d_2 = env.get_f_coefs()
    # mean, std = env.param_pred()
    # m_2_min = mean - 3 * std
    # m_2_max = mean + 3 * std
    # g_1_list = []
    # g_2_list = []
    # g_3_list = []
    # g_4_list = []
    # m_2_list = np.linspace(m_2_min, m_2_max, 200)
    # for m_2 in np.linspace(m_2_min, m_2_max, 200):
    #     env.robot.m_2 = m_2
    #     g = env.g
    #     g_1_list.append(g[2, 0])
    #     g_2_list.append(g[2, 1])
    #     g_3_list.append(g[3, 0])
    #     g_4_list.append(g[3, 1])
        
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot(g_1_list, g_2_list, g_3_list)
    # plt.show()

    env = SCARAParameterLearningEnv(m_2=0.1, use_online_adaptation=True)
    env.reset()

    m_2_mean_list = []
    m_2_std_list = []
    q_way_points = np.linspace(start=[np.pi/2 - 0.05, 0.0], stop=[-np.pi/3, -np.pi], num=400)
    for i, q_d in enumerate(q_way_points):
        u = env.robot.computed_torque_control(q_d=q_d)
        env.step(u)
        # print(env.Xr[-1])
        m_2_mean, m_2_std = env.param_pred()
        if i % 3 == 0:
            print(m_2_mean, m_2_std)
            m_2_mean_list.append(m_2_mean)
            m_2_std_list.append(m_2_std)

    draw_GP_confidence_interval(mean=m_2_mean_list, std=m_2_std_list, y_name='m_2_pred', img_name='wo_OA.jpg')