from matplotlib import pyplot as plt
import numpy as np
import scipy.stats
from scipy.linalg import block_diag
import cvxpy as cp
from cvxopt import matrix, solvers
from loguru import logger
from tqdm import tqdm

from SegWay_env.SegWay_multimodal_env import SegWayAdditiveNoiseEnv
from SegWay_env.SegWay_utils import *
from RSSA_safety_index import SafetyIndex

class MMAddRSSA(SafetyIndex):
    def __init__(
        self,
        env: SegWayAdditiveNoiseEnv,
        safety_index_params={
            'alpha': 1.0,
            'k_v': 1.0,
            'beta': 0.0,
        },
        sampling=False,
        sample_points_num=10,
        gamma=0.1,
        epsilon_0=0.05,
        epsilon_f=0.01
    ):
        super().__init__(env, safety_index_params)
        self.sample_points_num = sample_points_num
        self.x_dim = self.env.Xr.shape[0]
        self.u_dim = self.env.g.shape[1]
        self.gamma = gamma
        self.sampling = sampling # similar to fast_SegWay in GaussianRSSA

        self.epsilon_0=epsilon_0
        self.epsilon_f=epsilon_f

        self.env: SegWayAdditiveNoiseEnv

    def get_rho(self, sigma):
        delta = cp.Variable(self.env.f.shape) # (4, 1) in Segway

        prob = cp.Problem(
            cp.Maximize(self.env.get_p_phi_p_Xr(self.safety_index_params) @ delta),
            [cp.quad_form(delta, np.linalg.inv(sigma)) <= 1]
        )
        prob.solve()
        return prob.value

    def get_optimal_o(self, modal_params, l_k1=0, r_k1=10):
        p_phi_p_Xr = self.env.get_p_phi_p_Xr(self.safety_index_params)
        f = self.env.f
        modal_1_weight, modal_1_mu, modal_1_sigma = modal_params[0]
        while r_k1 - l_k1 >= self.epsilon_0:
            k_1 = (l_k1 + r_k1)/2
            o_1 = -p_phi_p_Xr @ (f + modal_1_mu) - k_1 * self.get_rho(modal_1_sigma)
            k_list = [k_1]
            for i in range(len(modal_params)-1):
                weight, mu, sigma = modal_params[i+1]
                k_list.append((-p_phi_p_Xr@(f+mu)-o_1)/self.get_rho(sigma))
            
            sum=0
            for i in range(len(modal_params)):
                sum += modal_params[i][0] * scipy.stats.norm.cdf(k_list[i])
            
            if sum > 1 - self.epsilon_f:
                r_k1 = k_1
            else:
                l_k1 = k_1
        
        # TODO: 1. use numpy vector to replace loop; 2. precompute rhos and store them

        optimal_o = -p_phi_p_Xr @ (f + modal_1_mu) - k_1 * self.get_rho(modal_1_sigma)
        return optimal_o
    
    def predict_f_modal_parameters(self):
        '''
        Get modal parameters of (f + d) by fitting sampled points or acquiring true values directly
        '''
        if self.sampling:
            raise NotImplementedError
        else:
            return self.env.get_true_model_params()

    def safe_control(self, u_ref):
        # The problem we try to solve is:
        # || u - u_ref || 
        # subject to
        #               G u <= h
        #       

        # And in case there is no solution that satisfies the safety constraints,
        # we add an extra dim to u as a slack variable. Therefore, the objective becomes:
        
        # || u - u_ref || + 10000 * ||u_slack||
        # (u - u_ref) P_obj (u - u_ref)     # P_obj = I,  P_obj[n+1, n+1] = 100000
        
        # And the constraints becomes:
        # G @ u < h is composed by 3 parts:
        #       a. grad_phi @ g @ u <= -gamma(phi) + o_optimal (RHS)
        #       b. control limit: u_min <  u <  u_max
        #       c. u_slack > 0,  the slack variable must be positive
        modal_params_pred = self.predict_f_modal_parameters()

        n = self.u_dim
        P = np.eye(n + 1).astype(float)
        P[-1, -1] = 10000
        q = np.vstack([-u_ref.reshape(-1, 1), np.zeros((1, 1))]).astype(float)

        phi = self.env.get_phi(self.safety_index_params)
        p_phi_p_Xr = self.env.get_p_phi_p_Xr(self.safety_index_params)

        if phi > 0:
            grad_phi_mul_g = p_phi_p_Xr @ self.env.g
            RHS = -self.gamma * phi + self.get_optimal_o(modal_params_pred)
        else:
            grad_phi_mul_g = np.zeros((1, n))
            RHS = 1
        
        A_slack = np.zeros((1, n + 1))
        A_slack[0, -1] = -1
        b_slack = np.zeros((1, 1))
        G = np.vstack([
            np.hstack([grad_phi_mul_g, -np.ones((grad_phi_mul_g.shape[0], 1))]),
            np.eye(n, n + 1),
            -np.eye(n, n + 1),
            A_slack,
        ])
        h = np.vstack([
            RHS,
            np.asanyarray(self.env.u_limit['high']).reshape(-1, 1),
            -np.asanyarray(self.env.u_limit['low']).reshape(-1, 1),
            b_slack,
        ])

        sol_obj = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
        u = np.squeeze(sol_obj['x'])
        if np.abs(u[-1]) > 1e-2:
            # logger.debug(f'u_FI in MMAddRSSA: {u}')
            self.if_infeasible = True
        else:
            self.if_infeasible = False
        u = u[:-1]
        
        return u

if __name__ == '__main__':
    env = SegWayAdditiveNoiseEnv()
    env.reset()
    ssa = MMAddRSSA(env, 
                    safety_index_params={
                        'alpha': 1.0,
                        'k_v': 1.0,
                        'beta': 0.001,
                    },
                    sampling=False)
    
    q_d = np.array([0, 0])
    dq_d = np.array([1, 0])
    for i in tqdm(range(960)):
        u = env.robot.PD_control(q_d, dq_d)
        u = ssa.safe_control(u)
        # print(env.Xr)
        env.step(u)
        env.render(img_name=str(i) + '.jpg', save_path='./src/pybullet-dynamics/SegWay_env/imgs/mm_add_rssa/')

    generate_gif('mm_add_rssa.gif', './src/pybullet-dynamics/SegWay_env/imgs/mm_add_rssa/', duration=0.01)

    # safety_index_params={
    #     'alpha': 1.0,
    #     'k_v': 1.0,
    #     'beta': 0.001,
    # }

    # env.robot.K_m = 1.9
    # env.robot.q =  np.array([0, -0.2559])
    # env.robot.dq = np.array([-4.91, -0.733])
    # phi = env.get_phi(safety_index_params)
    # p_phi_p_Xr = env.get_p_phi_p_Xr(safety_index_params)
    # print(f'phi= {phi}')
    # print(f'left={p_phi_p_Xr @ env.g}')
    # print(f'right={-0.1*phi - p_phi_p_Xr @ env.f}')
    # print(f'u {(-0.1*phi - p_phi_p_Xr @ env.f)/(p_phi_p_Xr @ env.g)}')
    # print(ssa.safe_control(np.array([0])))
    

    # plt.plot(a_list)
    # plt.savefig('./src/pybullet-dynamics/SegWay_env/imgs/parameter_learning/a.jpg')
    # plt.plot(K_m_true_list, label='true K_m')
    # plt.legend()
    # # plt.close()
    # draw_GP_confidence_interval(mean=K_m_mean_list, std=K_m_std_list, y_name='K_m_pred', img_name='test.jpg')