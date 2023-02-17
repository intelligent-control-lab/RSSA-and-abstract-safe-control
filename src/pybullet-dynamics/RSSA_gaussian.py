from matplotlib import pyplot as plt
import numpy as np
import scipy.stats
from scipy.linalg import block_diag
from cvxopt import matrix, solvers
import cvxpy as cp
from loguru import logger

from SCARA_env.SCARA_parameter_learning import SCARAParameterLearningEnv
from SegWay_env.SegWay_parameter_learning import SegWayParameterLearningEnv
from SegWay_env.SegWay_utils import *
from RSSA_safety_index import SafetyIndex

solvers.options['show_progress'] = False

class GaussianRSSA(SafetyIndex):
    def __init__(
        self,
        env: SegWayParameterLearningEnv,
        safety_index_params={
            'alpha': 1.0,
            'k_v': 1.0,
            'beta': 0.0,
        },
        confidence_level=0.01,
        sample_points_num=10,
        gamma=0.1,
        fast_SegWay=False,
    ):
        super().__init__(env, safety_index_params)
        self.confidence_level = confidence_level
        self.sample_points_num = sample_points_num
        self.x_dim = self.env.Xr.shape[0]
        self.u_dim = self.env.g.shape[1]
        self.gamma = gamma
        self.fast_SegWay = fast_SegWay
        
    def get_stable_L(self, Q: np.ndarray):
        # Q = L @ L.T
        w, v = np.linalg.eig(Q)
        w[np.abs(w) < 1e-10] = 0.0
        w_sqrt = np.sqrt(w)
        L = v @ np.diag(w_sqrt)
        L = np.real(L)
        L[np.abs(L) < 1e-10] = 0.0
        return L
    
    def get_gaussian_f_g(self):
        f_points, g_points = self.env.sample_f_g_points(self.sample_points_num)
        f_points = np.array(f_points).reshape(self.sample_points_num, -1)
        g_points = np.array(g_points).reshape(self.sample_points_num, -1)   # g is expanded row first!!!
        
        f_mu = np.mean(f_points, axis=0).reshape(-1, 1)    # shape: (x_dim, 1)
        f_sigma = np.cov(f_points.T)
        g_mu = np.mean(g_points, axis=0).reshape(self.x_dim, self.u_dim)    # shape: (x_dim, u_dim)
        g_sigma = np.cov(g_points.T)
        return f_mu, f_sigma, g_mu, g_sigma
        
    def find_max_qclp(self, c, mu, sigma):
        # max c.T @ x
        # subject to  (x-mu).T @ Sgima^{-1} @ (x-mu) <= chi2
        #
        # Sigma = L @ L.T
        # Let y = L^{-1} @ (x-mu)
        #
        # we will convert it to a cvxpy SOCP:
        # max c.T @ (L @ y + mu) = c.T @ L @ y + c.T @ mu
        # subject to || y ||<= sqrt(chi2)
        n = len(c)
        chi2 = scipy.stats.chi2.isf(self.confidence_level, n)

        # TODO: let sigma be positive definite
        try:
            # L = np.linalg.cholesky(sigma)
            L = self.get_stable_L(sigma)
        except Exception:
            sigma = sigma + 1e-5 * np.eye(n)
            L = np.linalg.cholesky(sigma)
            # print('unsatble sigma!')
        # END TODO

        y = cp.Variable(n)

        # We use cp.SOC(sqrt(chi2), y) to create the SOC constraint ||y||_2 <= sqrt(chi2).
        prob = cp.Problem(
            cp.Maximize(c.T @ L @ y),
            [cp.SOC(np.sqrt(chi2), y)]
        )
        prob.solve()
        x = (L @ y.value).reshape(-1, 1) + mu
        assert x is not None
        return x
    
    def get_gaussian_grad_phi(self):
        p_phi_p_Xr = self.env.get_p_phi_p_Xr(self.safety_index_params)
        
        f_mu, f_sigma, g_mu, g_sigma = self.get_gaussian_f_g()

        max_f = self.find_max_qclp(p_phi_p_Xr.T, f_mu, f_sigma)
        max_LfP = p_phi_p_Xr @ max_f

        LgP_mu = p_phi_p_Xr @ g_mu
        LgP_sigma = np.zeros((self.u_dim, self.u_dim))
        # gx is expanded row first!!!
        # so the indices should be mapped
        for i in range(self.u_dim):
            for j in range(self.u_dim):
                i_idx = [self.u_dim*k+i for k in range(self.x_dim)]
                j_idx = [self.u_dim*k+j for k in range(self.x_dim)]
                LgP_sigma[i, j] = p_phi_p_Xr @ g_sigma[i_idx, :][:, j_idx] @ p_phi_p_Xr.T
        
        return max_LfP, LgP_mu, LgP_sigma
    
    def generate_gaussian_safety_con(self, phi):
        max_LfP, LgP_mu, LgP_sigma = self.get_gaussian_grad_phi()
        chi2 = scipy.stats.chi2.isf(self.confidence_level, self.u_dim)

        # || L.T @ u || <= c @ u + d
        try:
            # L = np.linalg.cholesky(chi2 * LgP_sigma)
            L = self.get_stable_L(chi2 * LgP_sigma)
        except Exception:
            LgP_sigma = LgP_sigma + 1e-5 * np.eye(LgP_sigma.shape[0])
            L = np.linalg.cholesky(chi2 * LgP_sigma)
            print('unsatble LgP_sigma!')

        c = -LgP_mu.T
        d = -max_LfP.item() - self.gamma * phi

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
        
        # P_obj
        P_obj = np.eye(self.u_dim + 1).astype(float)
        P_obj[-1, -1] = 10000
        
        # add slack variable to L, c, d
        self.phi = self.env.get_phi(self.safety_index_params)
        if self.phi > 0:
            if self.fast_SegWay:
                L, c, d = self.fast_generate_gaussian_safety_con_SegWay(self.phi)
            else:
                L, c, d = self.generate_gaussian_safety_con(self.phi)
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
        if np.abs(u[-1]) > 1e-1:
            # logger.debug(f'u_FI in GaussianRSSA: {u}')
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
    
    def fast_generate_gaussian_safety_con_SegWay(self, phi):
        p_phi_p_Xr = self.env.get_p_phi_p_Xr(self.safety_index_params)
        max_LfP, LgP_mu, L = self.env.fast_gaussian_uncertain_bound(p_phi_p_Xr, self.confidence_level)
        
        c = -LgP_mu.T
        d = -max_LfP - self.gamma * phi
        
        return L, c, d

if __name__ == '__main__':
    env = SCARAParameterLearningEnv(m_2=0.5, use_online_adaptation=False, m_2_std_init=0.3)
    env.reset()
    env.param_pred()
    ssa = GaussianRSSA(env)
    # env = SegWayParameterLearningEnv()
    # env.reset()
    # ssa = GaussianRSSA(env, fast_SegWay=True)

    q_way_points = np.linspace(start=[np.pi/2 - 0.05, 0.0], stop=[-np.pi/3, -np.pi], num=400)
    for i, q_d in enumerate(q_way_points):
        u = env.robot.computed_torque_control(q_d=q_d)
        print(env.Xr)
        for _ in range(3):
            u = ssa.safe_control(u)
            env.step(u)
            if env.detect_collision():
                print('COLLISION!')
            phi = env.get_phi(ssa.safety_index_params)
            # print(phi)
    #     env.render(img_name=str(i) + '.jpg', save_path='./src/pybullet-dynamics/SCARA_env/imgs/gaussian_rssa/')
    
    # q_d = np.array([0, 0])
    # dq_d = np.array([1, 0])
    # for i in range(960):
    #     u = env.robot.PD_control(q_d, dq_d)
    #     u = ssa.safe_control(u)
    #     print(env.Xr)
    #     env.step(u)
        # K_m_mean_list.append(K_m_mean)
        # K_m_std_list.append(K_m_std)
        # K_m_true_list.append(env.robot.K_m)

    # plt.plot(a_list)
    # plt.savefig('./src/pybullet-dynamics/SegWay_env/imgs/parameter_learning/a.jpg')
    # plt.plot(K_m_true_list, label='true K_m')
    # plt.legend()
    # # plt.close()
    # draw_GP_confidence_interval(mean=K_m_mean_list, std=K_m_std_list, y_name='K_m_pred', img_name='test.jpg')