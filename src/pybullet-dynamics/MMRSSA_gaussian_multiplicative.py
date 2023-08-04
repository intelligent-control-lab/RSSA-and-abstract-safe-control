from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.linalg import block_diag
from cvxopt import matrix, solvers
import cvxpy as cp
from loguru import logger
from tqdm import tqdm

from SegWay_env.SegWay_multimodal_env import SegWayMultiplicativeNoiseEnv
from RSSA_safety_index import SafetyIndex
from SegWay_env.SegWay_utils import *

solvers.options['show_progress'] = False

class GaussianMulRSSA(SafetyIndex):
    '''
    modified from GaussianRSSA for comparison in MMRSSA
    '''
    def __init__(
        self,
        env: SegWayMultiplicativeNoiseEnv,
        safety_index_params={
            'alpha': 1.0,
            'k_v': 1.0,
            'beta': 0.0,
        },
        p_gaussian = 0.99,
        sample_points_num=100,
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
        self.debug=debug
        
    def predict_f_g_gaussian_parameters(self):
        '''
        Get unimodal gaussian parameters by fitting sampled f&g points or computing using dynamic models (the environment has to support this)
        __________________
        return
        gaussian_param is a dict whose keys are 'f_mu', 'f_sigma', 'g_mu', 'g_sigma'
        '''
        gaussian_param = dict()
        if self.sampling:
            # sampling method
            f_points, g_points = self.env.sample_f_g_points(points_num=self.sample_points_num)
            f_points = np.array(f_points).reshape(self.sample_points_num, -1)
            g_points_flat = np.array(g_points).reshape(self.sample_points_num, -1, order='F')   # g is expanded column first with order='F'
            
            gaussian_param['f_mu'] = np.mean(f_points, axis=0).reshape(-1, 1)    # shape: (x_dim, 1)
            gaussian_param['f_sigma'] = np.cov(f_points.T)
            gaussian_param['g_mu'] = np.mean(g_points_flat, axis=0).reshape(self.x_dim, self.u_dim)    # shape: (x_dim, u_dim)
            gaussian_param['g_sigma'] = np.cov(g_points_flat.T)

            if self.debug:
                fig, ax = plt.subplots()
                ax.plot(f_points[:, 2], f_points[:, 3], 'o')

                mean = np.mean(f_points[:, 2:], axis=0).reshape(-1, 1)
                sigma = np.cov(f_points[:, 2:].T)

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

    def generate_gaussian_safety_con(self, phi, gaussian_param):
        '''
        || L.T @ u || <= c @ u + d

        d of the two methods are not exactly equal
        Generate differentiable L and d as well.
        '''
        p_phi_p_Xr = self.env.get_p_phi_p_Xr(self.safety_index_params)
        f_mu = gaussian_param['f_mu']
        f_sigma = gaussian_param['f_sigma']
        g_mu = gaussian_param['g_mu']
        g_sigma = gaussian_param['g_sigma']

        norm_ppf = stats.norm.ppf((np.sqrt(self.p_gaussian)+1)/2)
        chi2_ppf = stats.chi2.ppf(np.sqrt(self.p_gaussian), self.u_dim)

        if self.fast_SegWay:
            '''
            This method can be used only in SegWay env which has u_dim = 1
            There is no optimization problem when computing LfP_max
            '''

            # get LfP_max
            LfP_mu = (p_phi_p_Xr @ f_mu).item()
            LfP_cov = p_phi_p_Xr @ f_sigma @ p_phi_p_Xr.T

            LfP_max = norm_ppf * np.sqrt(LfP_cov) + LfP_mu
            
            # get LgP_mu and LgP_cov
            LgP_mu = p_phi_p_Xr @ g_mu
            LgP_cov = p_phi_p_Xr @ g_sigma @ p_phi_p_Xr.T
            
            # get L
            L = np.array(np.sqrt(chi2_ppf * LgP_cov))  # add chi2 compare to RSSA_gaussian

            c = -LgP_mu.T
            d = -LfP_max - self.gamma * phi
            d = d.item()

        else:
            '''
            General method 
            '''
            rho = self.get_rho(f_sigma)

            c = -(p_phi_p_Xr @ g_mu).T
            d = - self.gamma * phi - p_phi_p_Xr @ f_mu - norm_ppf * rho
            d = d.item()

            ##### compute L
            LgP_sigma = p_phi_p_Xr @ g_sigma @ p_phi_p_Xr.T  # TODO: what if u_dim > 1

            try:
                L_LgP = np.linalg.cholesky(LgP_sigma)
                # L = self.get_stable_L(chi2_ppf * LgP_sigma) # TODO: Why use this?
                L = np.sqrt(chi2_ppf) * L_LgP
            except Exception:
                LgP_sigma = LgP_sigma + 1e-5 * np.eye(LgP_sigma.shape[0])
                L_LgP = np.linalg.cholesky(LgP_sigma)
                L = np.sqrt(chi2_ppf) * L_LgP
                print('unsatble LgP_sigma!')
            
        return L, c, d
    
    def safe_control(self, u_ref):
        # The problem we try to solve is:
        # || u - u_ref || 
        # subject to
        #               A u <= b
        #       || L.T u || <= d @ u + e

        # And in case there is no solution that satisfies the safety constraints,
        # we add an extra dim to u as a slack variable. Therefore, the objective becomes:
        
        # || u - u_ref || + 10000 * ||u_slack||
        # (u - u_ref) P_obj (u - u_ref)     # P_obj = I,  P_obj[n+1, n+1] = 100000
        
        # And the constraints becomes:
        # 1. A @ u < b is composed by two parts:
        #       a. control limit: u_min <  u <  u_max
        #       b. u_slack > 0,  the slack variable must be positive
        # 2. || L.T u || <= c @ u + d + u_slack
        
        # fit gaussian distribution
        gaussian_param = self.predict_f_g_gaussian_parameters()

        # P_obj
        P_obj = np.eye(self.u_dim + 1).astype(float)
        P_obj[-1, -1] = 10000
        
        # add slack variable to L, c, d
        self.phi = self.env.get_phi(self.safety_index_params)
        if self.phi > 0:
            L, c, d = self.generate_gaussian_safety_con(self.phi, gaussian_param)
            Ls = [block_diag(L, np.zeros((1, 1)))]
            cs = [np.vstack([c, np.ones((1, 1))])]
            ds = [d] 
        else:
            Ls = []
            cs = []
            ds = []
        
        # A:  A_U,  A_slack
        A_slack = np.zeros((1, self.u_dim + 1))
        A_slack[0, -1] = -1
        b_slack = np.zeros((1, 1))
        A = np.vstack([
            np.eye(self.u_dim, self.u_dim + 1),
            -np.eye(self.u_dim, self.u_dim + 1),
            A_slack,
        ]).astype(float)
        b = np.vstack([
            np.asanyarray(self.env.u_limit['high']).reshape(-1, 1),
            -np.asanyarray(self.env.u_limit['low']).reshape(-1, 1),
            b_slack,
        ])

        u_ref = u_ref.reshape((-1, 1))
        u_ref = np.vstack([u_ref, np.zeros(1)])
        u = self.solve_qcqp(u_ref, A, b, P_obj, Ls, cs, ds)
        u = np.squeeze(u)
        # print(u[-1])
        if np.abs(u[-1]) > 1e-3:
            logger.debug(f'u_FI in GaussianMulRSSA: {u}')
            self.if_infeasible = True
        else:
            self.if_infeasible = False
        u = u[:-1]
        return u
    
    def solve_qcqp(self, u_ref, A, b, P_obj, Ls, cs, ds):
        # obj:   (u-u_ref).T @ P @ (u-u_ref)
        # s.t.  A @ u <= b
        #       || L_i.T @ u || <= cs_i @ u + ds_i
        
        # let du = u - u_ref
        # obj:   du.T @ P @ du
        # s.t.  A @ du <= -A @ u_ref + b
        #       || L_i.T @ u || <= cs_i @ u + ds_i
        
        # convert to SOCP

        # We first add an extra dimension for du to as the objective, then
        # minimize_{u}  mask.T @ du   # [0, ..., 0, 1].T @ du,   the last dimension corresponds to the objective
        #               || PL @ du ||  <= mask.T @ du
        #               0 <= -a_i @ du - a_i @ u_ref - b_i
        #               || L_i.T @ du + L_i.T @ u_ref || <= cs_i @ u + cs_i @ u_ref + ds_i
        #               where PL = block_diag (the cholesky decomposition of P, [0])
        
        # Define and solve the CVXPY problem.
        n = len(u_ref) + 1
        du = cp.Variable(n)
        mask = np.zeros((n, 1))
        mask[-1, 0] = 1

        A = np.hstack([A, np.zeros((A.shape[0], 1))])
        PL = np.linalg.cholesky(P_obj)
        PL = block_diag(PL, np.zeros((1, 1)))

        Ls = [block_diag(L, np.zeros((1, 1))) for L in Ls]
        cs = [np.vstack([c, np.zeros((1, 1))]) for c in cs]
        cs = [np.squeeze(c) for c in cs]
        u_ref = np.vstack([u_ref, np.zeros((1,1))])

        # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
        # # || PL @ du ||  <= mask.T @ du
        soc_constraints = [cp.SOC(mask.T @ du, PL @ du)]
        # || L_i.T @ du + L_i.T @ u_ref || <= cs_i @ u + cs_i @ u_ref + ds_i
        soc_constraints += [cp.SOC(
            cs[i] @ du + (cs[i] @ u_ref).item() + ds[i],
            Ls[i].T @ du + np.squeeze(Ls[i].T @ u_ref)
        ) for i in range(len(Ls))]
        # 0 <= -a_i @ du - a_i @ u_ref - b_i
        soc_constraints += [cp.SOC(
            -A[i, :] @ du - A[i, :] @ u_ref + b[i, :],
            np.zeros((1, n)) @ du
        ) for i in range(b.shape[0])]
        
        prob = cp.Problem(cp.Minimize(mask.T @ du), soc_constraints)
        prob.solve()

        return u_ref[:-1] + np.vstack(du.value[:-1])

if __name__ == '__main__':

    env = SegWayMultiplicativeNoiseEnv()
    env.reset()
    ssa = GaussianMulRSSA(env, fast_SegWay=True, debug=True)

    q_d = np.array([0, 0])
    dq_d = np.array([1, 0])
    for i in tqdm(range(960)):
        u = env.robot.PD_control(q_d, dq_d)
        u = ssa.safe_control(u)
        # print(env.Xr)
        env.step(u)
        env.render(img_name=str(i) + '.jpg', save_path='./src/pybullet-dynamics/SegWay_env/imgs/mm_gaussian_rssa/')

    generate_gif('mm_gaussian_rssa.gif', './src/pybullet-dynamics/SegWay_env/imgs/mm_gaussian_rssa/', duration=0.01)