import time
import numpy as np
import scipy.stats
from scipy.linalg import block_diag
from cvxopt import matrix, solvers
import cvxpy as cp

from panda_rod_ssa import PandaRodSSA
from panda_rod_env import PandaRodEnv
from panda_rod_utils import compare, video_record, Monitor

solvers.options['show_progress'] = False

monitor = Monitor()
# monitor.start()

class PandaRodRSSA(PandaRodSSA):
    def __init__(
        self, 
        env, 
        prefix, 
        d_min=0.05,
        eta=1, 
        k_v=1, 
        lamda=0.5, 
        use_gt_dynamics=False,
        n_ensemble=10,
        num_layer=3,
        hidden_dim=256,
    ):
        super().__init__(env, prefix, d_min, eta, k_v, lamda, use_gt_dynamics, n_ensemble, num_layer, hidden_dim)

    def nn_gaussian_f_g(self, Xr):
        fxs, gxs =self.ensemble_inference(Xr)
        fx_mu = np.mean(fxs, axis=0)
        fx_sigma = np.cov(fxs.T)
        gx_mu = np.mean(gxs, axis=0)    # gx is expanded row first!!!
        gx_sigma = np.cov(gxs.T)
        return fx_mu, fx_sigma, gx_mu, gx_sigma

    def find_max_qclp(self, c, mu, sigma, p):
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
        chi2 = scipy.stats.chi2.isf(p, n)

        # TODO: let sigma be positive definite
        sigma = sigma + 0.01 * np.eye(n)
        try:
            L = np.linalg.cholesky(sigma)
        except Exception:
            sigma = sigma + 0.01 * np.eye(n)
            L = np.linalg.cholesky(sigma)
            print('unsatble fx_sigma!')
        # END TODO

        y = cp.Variable(n)

        # We use cp.SOC(sqrt(chi2), y) to create the SOC constraint ||y||_2 <= sqrt(chi2).
        prob = cp.Problem(
            cp.Maximize(c.T @ L @ y),
            [cp.SOC(np.sqrt(chi2), y)]
        )
        prob.solve()
        x = L @ y.value + mu
        assert x is not None
        return x
    
    def gaussian_grad_phi(self, Xr, Mh):
        d, _, p_d_p_Xr, p_dot_d_p_Xr = self.compute_grad(Mh)
        p_phi_p_Xr = -2 * d * p_d_p_Xr - self.k_v * p_dot_d_p_Xr
        
        fx_mu, fx_sigma, gx_mu, gx_sigma = self.nn_gaussian_f_g(Xr)

        max_fx = self.find_max_qclp(p_phi_p_Xr.T, fx_mu, fx_sigma, p=0.01)
        max_LfP = p_phi_p_Xr @ max_fx

        gx_mu = gx_mu.reshape(self.x_dim, self.u_dim)
        LgP_mu = p_phi_p_Xr @ gx_mu
        LgP_sigma = np.zeros((self.u_dim, self.u_dim))
        # gx is expanded row first!!!
        # so the indices should be mapped
        for i in range(self.u_dim):
            for j in range(self.u_dim):
                i_idx = [self.u_dim*k+i for k in range(self.x_dim)]
                j_idx = [self.u_dim*k+j for k in range(self.x_dim)]
                LgP_sigma[i, j] = p_phi_p_Xr @ gx_sigma[i_idx, :][:, j_idx] @ p_phi_p_Xr.T
        
        return max_LfP, LgP_mu, LgP_sigma
    
    def generate_gaussian_safety_con(self, Xr, Mh, p=0.01):
        print('UNSAFE!')
        Xr = np.vstack(Xr)
        Mh = np.vstack(Mh)

        max_LfP, LgP_mu, LgP_sigma = self.gaussian_grad_phi(Xr, Mh)
        chi2 = scipy.stats.chi2.isf(p, self.u_dim)

        # || L.T @ u || <= c u + d
        try:
            L = np.linalg.cholesky(chi2 * LgP_sigma)
        except Exception:
            LgP_sigma = LgP_sigma + 0.01 * np.eye(LgP_sigma.shape[0])
            L = np.linalg.cholesky(chi2 * LgP_sigma)
            print('unsatble LgP_sigma!')

        c = -LgP_mu.T
        d = -max_LfP.item() - self.eta - self.lamda

        return L, c, d
    
    def gaussian_safe_control(self, uref):
        ''' safe control
        Input:
            uref: reference control 
        '''
        # The problem we try to solve is:
        # || u - uref || 
        # subject to
        #               A u <= b
        #       || L.T u || <= d @ u + e

        # And in case there is no solution that satisfies the safety constraints,
        # we add an extra dim to u as a slack variable. Therefore, the objective becomes:
        
        # || u - uref || + 1000*||u_slack||
        # (u - uref) P_obj (u - uref)     # P_obj = I,  P_obj[n+1,n+1] = 100000
        
        # And the constraints becomes:
        # 1. A @ u < b is composed by two parts:
        #       a. control limit: u_min <  u <  u_max
        #       b. u_slack > 0,  the slack variable must be positive
        # 2. || L.T u || <= d @ u + e + u_slack, the set U_c

        Xr = self.env.Xr
        cons = [self.generate_gaussian_safety_con(Xr, Mh) for Mh in self.env.obstacles if self.phi(Mh) > 0]

        # P_obj
        P_obj = np.eye(self.u_dim + 1).astype(float)
        P_obj[-1, -1] = 1e+10

        # Ls, cs, ds
        if len(cons) > 0:
            Ls = [block_diag(con[0], np.zeros((1, 1))) for con in cons]
            cs = [np.vstack([con[1], np.ones(1)]) for con in cons]
            ds = [con[2] for con in cons] 
        else:
            Ls = []
            cs = []
            ds = []
        
        # A:  A_U,  A_slack
        A_slack = np.zeros((1, self.u_dim + 1))
        A_slack[0, -1] = -1
        b_slack = np.zeros((1, 1))
        A = np.vstack([
            -np.eye(self.u_dim, self.u_dim + 1),
            np.eye(self.u_dim, self.u_dim + 1),
            A_slack,
        ]).astype(float)
        b = np.vstack([
            self.env.max_u.reshape((-1, 1)),
            self.env.max_u.reshape((-1, 1)),
            b_slack,
        ])

        uref = uref.reshape((-1, 1))
        uref = np.vstack([uref, np.zeros(1)])
        u = self.solve_qcqp(uref, A, b, P_obj, Ls, cs, ds)
        # print(u)
        u = u[:-1]
        return np.vstack(np.array(u))

    def solve_qcqp(self, uref, A, b, P_obj, Ls, cs, ds):
        # obj:   (u-uref).T @ P @ (u-uref)
        # s.t.  A @ u <= b
        #       || L_i.T @ u || <= cs_i @ u + ds_i
        
        # let du = u - uref
        # obj:   du.T @ P @ du
        # s.t.  A @ du <= -A @ uref + b
        #       || L_i.T @ u || <= cs_i @ u + ds_i
        
        # convert to SOCP

        # We first add an extra dimension for du to as the objective, then
        # minimize_{u}  mask.T @ du   # [0, ..., 0, 1].T @ du,   the last dimension corresponds to the objective
        #               || PL @ du ||  <= mask.T @ du
        #               0 <= -a_i @ du - a_i @ uref - b_i
        #               || L_i.T @ du + L_i.T @ uref || <= cs_i @ u + cs_i @ uref + ds_i
        #               where PL = block_diag (the cholesky decomposition of P, [0] )
        
        # Define and solve the CVXPY problem.
        n = len(uref) + 1
        du = cp.Variable(n)
        mask = np.zeros((n, 1))
        mask[-1] = 1

        A = np.hstack([A, np.zeros((A.shape[0], 1))])
        PL = np.linalg.cholesky(P_obj)
        PL = block_diag(PL, np.zeros((1, 1)))

        Ls = [block_diag(L, np.zeros((1, 1))) for L in Ls]
        cs = [np.vstack([d, np.zeros((1, 1))]) for d in cs]
        cs = [np.squeeze(d) for d in cs]
        uref = np.vstack([uref, np.zeros((1,1))])

        # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
        # # || PL @ du ||  <= mask.T @ du
        soc_constraints = [cp.SOC(mask.T @ du, PL @ du)]
        # || L_i.T @ du + L_i.T @ uref || <= cs_i @ u + cs_i @ uref + ds_i
        soc_constraints += [cp.SOC(
            cs[i] @ du + (cs[i] @ uref).item() + ds[i],
            Ls[i].T @ du + np.squeeze(Ls[i].T @ uref)
        ) for i in range(len(Ls))]
        # 0 <= -a_i @ du - a_i @ uref - b_i
        soc_constraints += [cp.SOC(
            -A[i, :] @ du - A[i, :] @ uref + b[i, :],
            np.zeros((1, n)) @ du
        ) for i in range(b.shape[0])]
        
        prob = cp.Problem(cp.Minimize(mask.T @ du), soc_constraints)
        prob.solve()

        return uref[:-1] + np.vstack(du.value[:-1])

    def compute_monitor_data(self, Xr, Mh):
        Xr = np.vstack(Xr)
        Mh = np.vstack(Mh)

        _, LgP_nn_mu, LgP_nn_sigma = self.gaussian_grad_phi(Xr, Mh)
        f_nn, f_nn_sigma, g_nn, g_nn_sigma = self.nn_gaussian_f_g(Xr)

        self.f_nn = f_nn
        self.f_nn_sigma = f_nn_sigma
        self.g_nn = g_nn
        self.g_nn_sigma = g_nn_sigma

        d, _, p_d_p_Xr, p_dot_d_p_Xr = self.compute_grad(Mh)
        p_phi_p_Xr = -2 * d * p_d_p_Xr - self.k_v * p_dot_d_p_Xr
        LfP_nn_mu = (p_phi_p_Xr @ f_nn).item()
        LfP_nn_sigma = (p_phi_p_Xr @ f_nn_sigma @ p_phi_p_Xr.T).item()

        self.LfP_nn = LfP_nn_mu
        self.LfP_nn_sigma = LfP_nn_sigma
        self.LgP_nn = LgP_nn_mu
        self.LgP_nn_sigma = LgP_nn_sigma

        self.LfP = (p_phi_p_Xr @ self.env.f).item()
        self.LgP = p_phi_p_Xr @ self.env.g


if __name__ == '__main__':
    # env = PandaRodEnv(render_flag=False, goal_pose=[0.8, 0.1, 0.5], obstacle_pose=[0.55, 0.0, 0.4])
    env = PandaRodEnv(render_flag=True, goal_pose=[0.8, 0.1, 0.5], obstacle_pose=[0.55, 0.0, 0.4])
    rssa = PandaRodRSSA(env, prefix='./src/pybullet-dynamics/panda_rod_env/model/env_data_test_bn_1_small/')

    monitor.update(ssa_args=rssa.__dict__)
    monitor.update(env_args=env.__dict__)

    images = []
    env.empty_step()
    for i in range(960):
        print(i)

        u = env.compute_underactuated_torque()
        u = rssa.gaussian_safe_control(u)
        # print(u)
        _, _, _, _ = env.step(u)

        img = rssa.render_image()
        images.append(img)

        rssa.compute_monitor_data(env.Xr, env.Mh)
        monitor.update(
            f=env.f, g=env.g, dot_Xr=env.dot_Xr, 
            f_nn=rssa.f_nn, g_nn=rssa.g_nn, u=u, f_nn_sigma=rssa.f_nn_sigma, g_nn_sigma=rssa.g_nn_sigma,
            LfP_nn=rssa.LfP_nn, LfP_nn_sigma=rssa.LfP_nn_sigma, LgP_nn=rssa.LgP_nn, LgP_nn_sigma=rssa.LgP_nn_sigma,
            LfP=rssa.LfP, LgP=rssa.LgP, 
            d=rssa.d, dot_d=rssa.dot_d, Phi=rssa.Phi,
        )

        print(f'd: {rssa.d}, dot_d: {rssa.dot_d}, Phi: {rssa.Phi}')

        if env.if_success():
            print('--SUCCESS!--')
            break
            
        time.sleep(0.05)
    
    video_record('./src/pybullet-dynamics/panda_rod_env/movies/rssa_change_mass.mp4', images)
    monitor.close()
    


