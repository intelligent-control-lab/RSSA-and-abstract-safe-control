from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.linalg import block_diag
from cvxopt import matrix, solvers
import cvxpy as cp
from loguru import logger
from tqdm import tqdm

from SegWay_env.SegWay_multimodal_env import SegWayAdditiveNoiseEnv
from RSSA_safety_index import SafetyIndex
from SegWay_env.SegWay_utils import *

solvers.options['show_progress'] = False

class GaussianAddRSSA(SafetyIndex):
    '''
    modified from GaussianRSSA for comparison in MMRSSA
    '''
    def __init__(
        self,
        env: SegWayAdditiveNoiseEnv,
        safety_index_params={
            'alpha': 1.0,
            'k_v': 1.0,
            'beta': 0.0,
        },
        p_gaussian = 0.999,
        sample_points_num=300,
        gamma=0.1,
        fast_SegWay=False,
        debug=False
    ):
        super().__init__(env, safety_index_params)
        self.p_gaussian = p_gaussian
        self.sample_points_num = sample_points_num
        self.x_dim = self.env.Xr.shape[0]
        self.u_dim = self.env.g.shape[1]
        self.gamma = gamma
        self.fast_SegWay = fast_SegWay
        self.sampling = True  # Only sampling method is implemented 
        self.debug = debug
        self.env: SegWayAdditiveNoiseEnv
        
    def predict_f_gaussian_parameters(self):
        '''
        Get unimodal gaussian parameters by fitting sampled f&g points or computing using dynamic models (the environment has to support this)
        __________________
        return
        gaussian_param is a dict whose keys are 'f_mu', 'f_sigma'
        '''
        gaussian_param = dict()
        if self.sampling:
            # sampling method
            f_points = self.env.sample_f_points(points_num=self.sample_points_num)
            f_points = np.array(f_points).reshape(self.sample_points_num, -1)
            
            gaussian_param['f_mu'] = np.mean(f_points, axis=0).reshape(-1, 1)    # shape: (x_dim, 1)
            gaussian_param['f_sigma'] = np.cov(f_points.T)

            if self.debug:
                fig, ax = plt.subplots()
                ax.plot(f_points[:, 2], f_points[:, 3], 'o')

                mean = gaussian_param['f_mu'][2:]
                sigma = gaussian_param['f_sigma'][2:,2:]

                sqrt_cov = np.sqrt(np.linalg.eigvals(sigma))
                width, height = stats.norm.ppf((self.p_gaussian + 1)/2) * 2 * sqrt_cov

                # Calculate rotation angle
                evec = np.linalg.eigh(sigma)[1]
                angle = np.arctan2(evec[0,1], evec[0,0]) * 180 / np.pi

                # Plot ellipse
                from matplotlib.patches import Ellipse
                ax = plt.gca()
                ax.add_patch(Ellipse(mean, width=width, height=height, angle=angle,
                                    facecolor='none', edgecolor='r'))

                plt.show()

        else:
            # TODO: compute unimodal gaussian parameters directly from multimodal parameters
            raise NotImplementedError
        
        return gaussian_param

    def get_rho(self, sigma):
        '''
        get sigma worse case of grad phi @ f
        '''
        delta = cp.Variable(self.env.f.shape) # (4, 1) in Segway
        p_phi_p_Xr = self.env.get_p_phi_p_Xr(self.safety_index_params)

        sigma += 0.00001 * np.eye(sigma.shape[0])
        prob = cp.Problem(
            cp.Maximize(p_phi_p_Xr @ delta),
            [cp.quad_form(delta, np.linalg.inv(sigma)) <= 1]
        )
        prob.solve()
        return prob.value

    def get_stable_L(self, Q: np.ndarray):
        # Q = L @ L.T
        w, v = np.linalg.eig(Q)
        w[np.abs(w) < 1e-10] = 0.0
        w_sqrt = np.sqrt(w)
        L = v @ np.diag(w_sqrt)
        L = np.real(L)
        L[np.abs(L) < 1e-10] = 0.0
        return L
    
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
        gaussian_param = self.predict_f_gaussian_parameters()
        f_mu = gaussian_param['f_mu']
        f_sigma = gaussian_param['f_sigma']

        n = self.u_dim
        P = np.eye(n + 1).astype(float)
        P[-1, -1] = 10000
        q = np.vstack([-u_ref.reshape(-1, 1), np.zeros((1, 1))]).astype(float)

        phi = self.env.get_phi(self.safety_index_params)
        p_phi_p_Xr = self.env.get_p_phi_p_Xr(self.safety_index_params)

        if phi > 0:
            grad_phi_mul_g = p_phi_p_Xr @ self.env.g
            RHS = -self.gamma * phi - p_phi_p_Xr @ f_mu - stats.norm.ppf((self.p_gaussian+1)/2) * np.sqrt(p_phi_p_Xr @ f_sigma @ p_phi_p_Xr.T)
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
        if np.abs(u[-1]) > 1e-3:
            logger.debug(f'u_FI in GaussianAddRSSA: {u}')
        u = u[:-1]

        return u
        

if __name__ == '__main__':

    env = SegWayAdditiveNoiseEnv()
    env.reset()
    ssa = GaussianAddRSSA(env, fast_SegWay=False, debug=True)

    q_d = np.array([0, 0])
    dq_d = np.array([1, 0])
    for i in tqdm(range(960)):
        u = env.robot.PD_control(q_d, dq_d)
        u = ssa.safe_control(u)
        # print(env.Xr)
        env.step(u)
        env.render(img_name=str(i) + '.jpg', save_path='./src/pybullet-dynamics/SegWay_env/imgs/mm_gaussian_rssa/')

    generate_gif('mm_gaussian_rssa.gif', './src/pybullet-dynamics/SegWay_env/imgs/mm_gaussian_rssa/', duration=0.01)