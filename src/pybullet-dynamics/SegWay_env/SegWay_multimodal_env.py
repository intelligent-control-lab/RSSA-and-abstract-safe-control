from typing import Dict
import numpy as np

try:
    from SegWay_env import SegWayEnv
    from SegWay_utils import *
except:
    from SegWay_env.SegWay_env import SegWayEnv
    from SegWay_env.SegWay_utils import *

class SegWayAdditiveNoiseEnv(SegWayEnv):
    def __init__(
        self,
        modal_params = [[0.3, np.array([0.1]*4).reshape(-1,1), np.eye(4, 4)*0.2], 
                        [0.7, np.array([-0.2]*4).reshape(-1,1), np.eye(4, 4)*0.2]],  
        # [[modal_1_ratio, modal_1_mu, modal_1_sigma], [modal_2], ...] where modal_1_mu (4,1) modal_1_sigma (4, 4) 
        
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
        
        self.modal_params = modal_params
    
    def get_phi(self, safety_index_params: Dict):
        '''
        phi = -a_safe_max ** alpha + |a| ** alpha + k_v * sign(a) * da + beta
        '''
        alpha = safety_index_params['alpha']
        k_v = safety_index_params['k_v']
        beta = safety_index_params['beta']
        a_safe_max = self.a_safe_limit['high']
        a = self.robot.q[1]
        da = self.robot.dq[1]
        return -a_safe_max ** alpha + np.abs(a) ** alpha + k_v * np.sign(a) * da + beta

    def get_p_phi_p_Xr(self, safety_index_params: Dict):
        '''
        p_phi_p_Xr = [0.0, alpha * |a|^(alpha-1) * sign(a), 0.0, k_v * sign(a)]
        '''
        alpha = safety_index_params['alpha']
        k_v = safety_index_params['k_v']
        a = self.robot.q[1]
        # da = self.robot.dq[1]
        p_phi_p_Xr = np.zeros((1, 4))
        p_phi_p_Xr[0, 1] = alpha * np.abs(a) ** (alpha-1) * np.sign(a)
        p_phi_p_Xr[0, 3] = k_v * np.sign(a)
        return p_phi_p_Xr

    ### Interface for safe control
    def sample_f_points(self, points_num=10):
        '''
        Additive noise
        '''
        f_points = []
        weights = [modal_param[0] for modal_param in self.modal_params]
        mus = [modal_param[1] for modal_param in self.modal_params]
        sigmas = [modal_param[2] for modal_param in self.modal_params]
        for _ in range(points_num):
            f = self.f
            modal_index = np.random.choice(a=len(weights), p=weights)
            noise = np.random.multivariate_normal(np.squeeze(mus[modal_index]), sigmas[modal_index], size=1)
            f_points.append(f + noise.reshape(-1,1))

        return f_points
    
    def get_true_model_params(self):
        return self.modal_params

    def reset(self):
        self.robot.q = np.zeros(2)
        self.robot.dq = np.zeros(2)
        self.time=0
        

class SegWayMultiplicativeNoiseEnv(SegWayAdditiveNoiseEnv):
    def __init__(
        self,
        K_m_modal_params = [[0.3, 2.3, 0.1], 
                        [0.7, 2.6, 0.2]],  
        # [[modal_1_ratio, modal_1_mu, modal_1_sigma], [modal_2], ...] where modal_1_mu is scalar, modal_1_sigma is scalar 
        
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
            None,
            dt,
            K_m,
            K_b,
            m_0, m, J_0,
            L, l, R,
            g,
            q_limit, dq_limit, u_limit, a_safe_limit,
        )
        
        # for online adaptation part 
        self.K_m_modal_params = K_m_modal_params
    
    ### Interface for safe control
    def sample_f_g_points(self, points_num=10):
        '''
        sample multimodal f and g
        '''
        true_K_m = self.robot.K_m
        f_points = []
        g_points = []
        weights = [modal_param[0] for modal_param in self.K_m_modal_params]
        mus = [modal_param[1] for modal_param in self.K_m_modal_params]
        sigmas = [modal_param[2] for modal_param in self.K_m_modal_params]
        for _ in range(points_num):
            modal_index = np.random.choice(a=len(weights), p=weights)
            K_m = np.random.normal(mus[modal_index], sigmas[modal_index], size=1)
            self.robot.K_m = K_m
            f_points.append(self.f)
            g_points.append(self.g)

        self.robot.K_m = true_K_m
        return f_points, g_points
    
    def compute_f_g_modal_parameters(self):
        '''
        Compute f g modal parameters using dynamic models with out sampling.
        _____________
        return:
        modal_params is a list. modal_1_params = modal_params[0], modal_1_params is a dict, keys: 'weight', 'f_mu', 'f_sigma', 'g_mu', 'g_sigma'
        '''
        weights = [modal_param[0] for modal_param in self.K_m_modal_params]
        mus = [modal_param[1] for modal_param in self.K_m_modal_params]
        sigmas = [modal_param[2] for modal_param in self.K_m_modal_params]

        modal_params = []
        for weight, mu, sigma in zip(weights, mus, sigmas):
            modal_param = dict()
            true_K_m = self.robot.K_m
            self.robot.K_m = mu
            f_mu = self.f
            g_mu = self.g
            self.robot.K_m = true_K_m
            
            M_inv, C, D = self.get_f_coefs()

            # get Cov(f)
            f_trans = np.zeros((4, 1))
            f_trans[2:, 0] = -M_inv @ D * self.robot.K_b / self.robot.R
            f_cov = sigma**2 * f_trans @ f_trans.T

            # get Cov(g)
            g_trans = np.zeros((4, 1))
            g_trans[2:, 0] = M_inv @ np.array([1/self.robot.R, -1])
            g_cov = sigma**2 * g_trans @ g_trans.T

            modal_param['weight'] = weight
            modal_param['f_mu'] = f_mu
            modal_param['f_sigma'] = f_cov
            modal_param['g_mu'] = g_mu
            modal_param['g_sigma'] = g_cov

            modal_params.append(modal_param)

        return modal_params

    
    def get_true_model_params(self):
        return self.K_m_modal_params

    def reset(self):
        self.robot.q = np.zeros(2)
        self.robot.dq = np.zeros(2)
        self.time=0

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
        C[0] = -m * L * np.sin(a) * da**2
        C[1] = -m * g * L * np.sin(a) 

        D = np.zeros(2)
        D[0] = (dr - R * da) / R
        D[1] = -(dr - R * da)

        return M_inv, C, D
