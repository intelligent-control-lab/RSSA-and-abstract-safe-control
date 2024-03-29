import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, RBF
import scipy.stats

try:
    from SegWay_env import SegWayEnv
    from SegWay_utils import *
except:
    from SegWay_env.SegWay_env import SegWayEnv
    from SegWay_env.SegWay_utils import *

class SegWayParameterLearningEnv(SegWayEnv):
    def __init__(
        self,
        use_online_adaptation=True,
        K_m_mean_init=2.524,
        K_m_std_init=0.3,
        
        dt=1/240,
        K_m=2.524, 
        K_b=0.189,
        m_0=52.710, m=44.798, J_0=5.108,
        L=0.169, l=0.75, R=0.195,
        g=9.81,
        q_limit={'low': [-1.0, -np.pi/2], 'high': [1.0, np.pi/2]},
        dq_limit={'low': [-5.0, -2.0], 'high': [5.0, 2.0]},
        u_limit={'low': [-20.0], 'high': [20.0]},
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
        
        # for online adaptation part 
        self.use_online_adaptation = use_online_adaptation
        self.first_predict = True
        self.K_m_mean_init = K_m_mean_init
        self.K_m_std_init = K_m_std_init

    ### Interface for safe control
    def param_pred(self):
        '''
        Here we assume the underlying parameter follows a Gaussian process. 
        In SegWay, the mean and std of K_m will be returned.
        
        In online adaptation part, using EKF:
        - predict:
            - K_m_predict = K_m_update
            - _, R_k**0.5 = GP_predict()
            - P_{k| k - 1} = P_{k - 1| k - 1}
        - update:
            - K_m_obs = get_K_m_data()
            - S_k = P_{k| k - 1} + R_k
            - K_k = P_{k| k - 1} * S_k^{-1} 
            - K_m_update = K_m_predict + K_k * (K_m_obs - K_m_predict)
            - P_{k| k} = (1 - K_k) * P_{k| k - 1}
        '''
        if not self.use_online_adaptation:
            self.K_m_mean = self.K_m_mean_init
            self.K_m_std = self.K_m_std_init
            return self.K_m_mean, self.K_m_std
        
        # for online adpatation
        R_std = self.K_m_std_init
        if self.first_predict:
            self.first_predict = False
            self.K_m_mean = self.K_m_mean_init
            self.K_m_std = R_std
            return self.K_m_mean, R_std
        else:
            # predict part
            K_m_predict = self.K_m_mean
            P = self.K_m_std ** 2
            std_predict = self.K_m_std
            
            # update part
            f_obs = self.f[2:, :]
            f_predict = self.get_f_data(K_m_predict)
            p_f_p_K_m = self.get_p_f_p_K_m(K_m_predict)
            S = p_f_p_K_m @ p_f_p_K_m.T * P + R_std**2 * np.eye(2)
            # S = p_f_p_K_m @ p_f_p_K_m.T * P
            K = P * p_f_p_K_m.T @ np.linalg.inv(S)
            K_m_update = K_m_predict + (K @ (f_obs - f_predict)).item()
            P = (1 - (K @ p_f_p_K_m).item()) * P
            self.K_m_mean = K_m_update
            self.K_m_std = P**0.5
            
            return K_m_predict, std_predict
    
    ### Interface for safe control
    def sample_f_g_points(self, points_num=10):
        '''
        Sample f(theta | x) and g(theta | x), where theta | x is Gaussian 

        In SCARA, theta is K_m(x)
        '''
        mean, std = self.param_pred()
        true_K_m = self.robot.K_m
        f_points = []
        g_points = []
        for _ in range(points_num):
            K_m = np.random.randn() * std + mean
            self.robot.K_m = K_m
            f_points.append(self.f)
            g_points.append(self.g)

        self.robot.K_m = true_K_m
        return f_points, g_points
    
    ### Interface for safe control
    def set_param(self, param):
        self.robot.K_m = param
        
    ### Interface for safe control
    def get_param(self):
        return self.robot.K_m
    
    def get_f_data(self, K_m):
        true_K_m = self.robot.K_m
        self.robot.K_m = K_m
        f = self.f[2:, :]
        self.robot.K_m = true_K_m
        return f
    
    def get_p_f_p_K_m(self, K_m):
        # p_f_p_K_m = -M_inv @ D * K_b / R
        true_K_m = self.robot.K_m
        self.robot.K_m = K_m
        M_inv, _, D = self.get_f_coefs()
        p_f_p_K_m = -M_inv @ D * self.robot.K_b / self.robot.R
        self.robot.K_m = true_K_m
        p_f_p_K_m = p_f_p_K_m.reshape(-1, 1)
        return p_f_p_K_m

    def get_f_coefs(self):
        '''
        Coeficients of f(b_t(K_m)): f = -M_inv @ (C + D * b_t)

        where f is env.f[2:]
        '''
        M_inv = np.linalg.pinv(self.robot.M)

        _, a = self.robot.q
        dr, da = self.robot.dq
        m = self.robot.m
        L = self.robot.L
        g = self.robot.g
        R = self.robot.R

        C = np.zeros(2)
        C[0] = -m * L * sin(a) * da**2
        C[1] = -m * g * L * sin(a) 

        D = np.zeros(2)
        D[0] = (dr - R * da) / R
        D[1] = -(dr - R * da)

        return M_inv, C, D

    def get_K_m_data(self):
        f = np.squeeze(self.f)[2:]

        M_inv, C, D = self.get_f_coefs()
        x = -f - M_inv @ C
        y = M_inv @ D * self.robot.K_b / self.robot.R

        K_m = x[1] / y[1]
        return K_m

    def reset(self):
        self.robot.q = np.zeros(2)
        self.robot.dq = np.zeros(2)
        self.first_predict = True
        
    def fast_gaussian_uncertain_bound(self, p_phi_p_Xr, confidence_level):
        true_K_m = self.robot.K_m
        self.robot.K_m = self.K_m_mean_init
        f_mu = self.f
        g_mu = self.g
        self.robot.K_m = true_K_m
        
        # get LfP_max
        LfP_mu = (p_phi_p_Xr @ f_mu).item()
        M_inv, C, D = self.get_f_coefs()
        f_trans = np.zeros((4, 1))
        f_trans[2:, 0] = -M_inv @ D * self.robot.K_b / self.robot.R
        LfP_trans = (p_phi_p_Xr @ f_trans).item()
        LfP_cov = self.K_m_std_init**2 * LfP_trans**2
        LfP_max = scipy.stats.norm.isf(confidence_level) * np.sqrt(LfP_cov) + LfP_mu
        
        # get LgP_mu and LgP_cov
        LgP_mu = p_phi_p_Xr @ g_mu
        g_trans = np.zeros((4, 1))
        g_trans[2:, 0] = M_inv @ np.array([1/self.robot.R, -1]) 
        LgP_trans = (p_phi_p_Xr @ g_trans).item()
        LgP_cov = self.K_m_std_init**2 * LgP_trans**2
        
        # get L
        L = np.array([[np.sqrt(LgP_cov)]])
        
        return LfP_max, LgP_mu, L
        


if __name__ == '__main__':
    env = SegWayParameterLearningEnv(use_online_adaptation=False)
    env.GP_K_m_spatial_fit()
    env.reset()

    K_m_mean_list = []
    K_m_std_list = []
    q_d = np.array([0, 0])
    dq_d = np.array([1, 0])
    for i in range(960):
        u = env.robot.PD_control(q_d, dq_d)
        env.step(u)
        # print(env.Xr[-1])
        if i == 600:
            pass
        K_m_mean, K_m_std = env.param_pred()
        K_m_mean_list.append(K_m_mean)
        K_m_std_list.append(K_m_std)

    draw_GP_confidence_interval(mean=K_m_mean_list, std=K_m_std_list, y_name='K_m_pred', img_name='wo_OA.jpg')