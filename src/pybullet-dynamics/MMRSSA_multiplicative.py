from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.linalg import block_diag
from cvxopt import matrix, solvers
import cvxpy as cp
from loguru import logger
from tqdm import tqdm
import copy
import torch
from scipy.optimize import minimize, LinearConstraint

from SegWay_env.SegWay_multimodal_env import SegWayMultiplicativeNoiseEnv
from SegWay_env.SegWay_utils import *
from RSSA_safety_index import SafetyIndex
from MMRSSA_utils import DifferentiableChi2ppf

solvers.options['show_progress'] = False

class MMMulRSSA(SafetyIndex):
    def __init__(
        self,
        env: SegWayMultiplicativeNoiseEnv,
        safety_index_params={
            'alpha': 1.0,
            'k_v': 1.0,
            'beta': 0.0,
        },
        p_init=[0.999, 0.999],
        gamma=0.1,
        sampling=False,
        sample_points_num=10,
        epsilon_0=0.1,
        epsilon_f=0.01,
        max_gd_iterations = 50,
        alpha = 0.002,
        beta = 0.01,
        fast_SegWay = False, 
        debug=False
    ):
        super().__init__(env, safety_index_params)
        self.p_init = p_init  # p_i in paper
        self.sample_points_num = sample_points_num
        self.x_dim = self.env.Xr.shape[0]
        self.u_dim = self.env.g.shape[1]
        self.gamma = gamma
        self.sampling = sampling

        self.epsilon_0 = epsilon_0  # gradient descent termination condition
        self.epsilon_f = epsilon_f
        self.max_gd_iterations = max_gd_iterations  # max gd iterations
        self.alpha = alpha # step size in gd
        self.beta = beta # step size in gd for outer lagrangian multipliers

        self.fast_SegWay = fast_SegWay

        self.debug = debug

        self.env: SegWayMultiplicativeNoiseEnv

    def predict_f_g_modal_parameters(self):
        '''
        Get modal parameters by fitting sampled f&g points or computing using dynamic models (the environment has to support this)
        __________________
        return
        modal_params is a list. modal_1_params = modal_params[0], modal_1_params is a dict, keys: 'weight', 'f_mu', 'f_sigma', 'g_mu', 'g_sigma'
        '''
        if self.sampling:
            f_points, g_points = self.env.sample_f_g_points(points_num=100)
            f_points = np.array(f_points).reshape(self.sample_points_num, -1)
            g_points_flat = np.array(g_points).reshape(self.sample_points_num, -1, order='F')   # g is expanded column first with order='F'
            
            # TODO: Compute multimodal guassian parameters. Notice that f and g are coupled.
            raise NotImplementedError
            # f_mu = np.mean(f_points, axis=0).reshape(-1, 1)    # shape: (x_dim, 1)
            # f_sigma = np.cov(f_points.T)
            # g_mu = np.mean(g_points, axis=0).reshape(self.x_dim, self.u_dim)    # shape: (x_dim, u_dim)
            # g_sigma = np.cov(g_points.T)
        else:
            modal_params = self.env.compute_f_g_modal_parameters()
        
        return modal_params

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

    def generate_gaussian_safety_con(self, phi, p_i, diff_p_i: torch.Tensor, modal_param):
        '''
        || L.T @ u || <= c @ u + d

        d of the two methods are not exactly equal
        Generate differentiable L and d as well.
        '''
        p_phi_p_Xr = self.env.get_p_phi_p_Xr(self.safety_index_params)
        f_mu = modal_param['f_mu']
        f_sigma = modal_param['f_sigma']
        g_mu = modal_param['g_mu']
        g_sigma = modal_param['g_sigma']

        norm_ppf = stats.norm.ppf((np.sqrt(p_i)+1)/2)
        chi2_ppf = stats.chi2.ppf(np.sqrt(p_i), self.u_dim)
        normal = torch.distributions.Normal(0, 1)
        diff_norm_ppf = normal.icdf((torch.sqrt(diff_p_i)+1)/2)
        diff_chi2_ppf = DifferentiableChi2ppf.apply(torch.sqrt(diff_p_i), self.u_dim)

        if self.fast_SegWay:
            '''
            This method can be used only in SegWay env which has u_dim = 1
            There is no optimization problem when computing LfP_max
            '''

            # get LfP_max
            LfP_mu = (p_phi_p_Xr @ f_mu).item()
            LfP_cov = p_phi_p_Xr @ f_sigma @ p_phi_p_Xr.T

            LfP_max = norm_ppf * np.sqrt(LfP_cov) + LfP_mu
            diff_LfP_max = diff_norm_ppf * torch.from_numpy(np.sqrt(LfP_cov)) + LfP_mu
            
            # get LgP_mu and LgP_cov
            LgP_mu = p_phi_p_Xr @ g_mu
            LgP_cov = p_phi_p_Xr @ g_sigma @ p_phi_p_Xr.T
            
            # get L
            L = np.array(np.sqrt(chi2_ppf * LgP_cov))  # add chi2 compare to RSSA_gaussian

            diff_L = torch.sqrt(diff_chi2_ppf * LfP_cov.item())

            c = -LgP_mu.T
            d = -LfP_max - self.gamma * phi
            d = d.item()

            diff_d = -diff_LfP_max - self.gamma * phi
            diff_d.squeeze()
            
        else:
            '''
            General method 
            '''
            rho = self.get_rho(f_sigma)

            c = -(p_phi_p_Xr @ g_mu).T
            d = - self.gamma * phi - p_phi_p_Xr @ f_mu - norm_ppf * rho
            d = d.item()

            diff_d = torch.from_numpy(- self.gamma * phi - p_phi_p_Xr @ f_mu) - diff_norm_ppf * rho
            diff_d.squeeze()

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
                exit()
            
            diff_L = torch.sqrt(diff_chi2_ppf) * torch.from_numpy(L_LgP)

        return L, c, d, diff_L, diff_d
    
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
        
        # initialize p = {p_i}
        p = np.ones_like(self.p_init) * (1-self.epsilon_f) + 0.001

        # P_obj
        P_obj = np.eye(self.u_dim + 1).astype(float)
        P_obj[-1, -1] = 10000
        
        phi = self.env.get_phi(self.safety_index_params)
        modal_params_pred = self.predict_f_g_modal_parameters()

        self.modal_params_pred = modal_params_pred  # for drawing

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

        if self.debug:
            p_list = []
            value_for_grad_list = []
            opt_value_list = []


        self.obj_grad = None
        self.constraint_grad = None
        def safe_control_with_p(p):
            diff_p = torch.tensor(p, requires_grad=True) # differentiable p for gradient computation using pytorch

            self.safety_conditions = []   # for drawing

            # add slack variable to L, c, d
            Ls=[]
            cs=[]
            ds=[]
            diff_Ls = []
            diff_ds = []
            if phi > 0:
                for p_i, diff_p_i, modal_param in zip(p, diff_p, modal_params_pred):
                    L, c, d, diff_L, diff_d = self.generate_gaussian_safety_con(phi, p_i, diff_p_i, modal_param)
                    self.safety_conditions.append({'L': L.item(), 'c':c.item(), 'd':d})   # for drawing
                    Ls.append(block_diag(L, np.zeros((1, 1))))
                    cs.append(np.vstack([c, np.ones((1, 1))]))
                    ds.append(d)
                    diff_Ls.append(torch.block_diag(diff_L, torch.zeros((1, 1), dtype=torch.float64)))
                    diff_ds.append(diff_d)
            else:
                pass
            
            u, opt_value, value_for_grad = self.solve_qcqp(copy.deepcopy(u_ref), A, b, P_obj, Ls, cs, ds, diff_Ls, diff_ds)
            u = np.squeeze(u)
            # logger.debug(f'u[-1] = {u[-1]}')
            # logger.debug(f'opt_value = {opt_value}')
            if np.abs(u[-1]) > 1e-2:
                # logger.debug(f'u_FI in MMMulRSSA: {u}')
                self.if_infeasible = True
            else:
                self.if_infeasible = False
            u = u[:-1]

            grad_numpy = None
            # update p
            if len(Ls) > 0:
                value_for_grad.backward()
                grad_numpy = diff_p.grad.numpy()

                if self.debug:
                    p_list.append(p)
                    value_for_grad_list.append(value_for_grad.item())
                    opt_value_list.append(opt_value)
            
            return u, opt_value, grad_numpy

        if phi > 0:
            num_modal = len(self.p_init)
            weights = np.array([modal_param['weight'] for modal_param in modal_params_pred])

            # Objective function
            def objective(p):
                u, value, grad = safe_control_with_p(p)
                return value, grad

            # Constraint function
            cons = []

            def main_constraint(p):
                return np.dot(weights, p) - (1-self.epsilon_f)
            
            cons.append({'type': 'ineq', 'fun': main_constraint, 'jac': lambda x: weights, 'hess': lambda x: np.zeros(num_modal, num_modal)})

            A_ = np.eye(num_modal)
            lb = np.zeros(num_modal)
            rb = np.ones(num_modal)*0.999999
            p_limit_cons = LinearConstraint(A_, lb, rb)

            cons.append(p_limit_cons)

            bounds = [[0.000001, 0.999999]] * num_modal

            res = minimize(objective, self.p_init, jac=True, constraints=cons, bounds=bounds,  method='SLSQP', tol=1e-4, options={'disp': False, 'maxiter': 10})
            # res = minimize(objective, self.p_init, jac=True, constraints=cons, bounds=bounds,  method='trust-constr', tol=1e-4, options={'disp': False, 'initial_tr_radius': 0.1})

            opt_value = res.fun
            optimal_p = res.x
            self.p_init = optimal_p  # update initial p
            u, _, _= safe_control_with_p(optimal_p)

            self.optimal_p = optimal_p  # for drawing

            if self.debug:
                print(f'optimal value: {opt_value}')
                print(f'optimal p : {optimal_p}')
                print(f'constraint: {np.dot(weights, optimal_p) - (1-self.epsilon_f)}')

        else:
            u, _, _ = safe_control_with_p(p)


        if self.debug and p_list:
            ax1 = plt.subplot(131)
            p1, p2 = zip(*p_list)
            ax1.plot(p1, p2, 'o-')
            ax1.plot(p1[0], p2[0], 'o', color='red') 
            ax1.set_xlabel('p1')
            ax1.set_ylabel('p2')

            ax2 = plt.subplot(132)
            ax2.plot(value_for_grad_list)
            ax2.set_ylabel('value for grad') 

            ax3 = plt.subplot(133)
            ax3.plot(opt_value_list)
            ax3.set_ylabel('opt value')

            plt.show()

        ###########################
        #  Plot landscape
        if False:
            x = np.linspace(0.0, 1, 20)
            y = np.linspace(0.0, 1, 20)
            X, Y = np.meshgrid(x, y)
            opt_value_list = np.empty(X.shape)
            value_for_grad_list = np.empty(X.shape)

            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    p=[X[i,j], Y[i,j]]
                    '''
                    gradient descent step: min d({pi}) + lambdas[0] * ((1-epsilon_f)-sum(weight*p_i)) + sum(lambdas[ ]*(p_i-1)) + sum(lambdas[]*(-p_i))
                    '''
                    # add slack variable to L, c, d
                    Ls=[]
                    cs=[]
                    ds=[]
                    diff_Ls = []
                    diff_ds = []
                    if phi > 0:
                        for p_i, diff_p_i, modal_param in zip(p, diff_p, modal_params_pred):
                            L, c, d, diff_L, diff_d = self.generate_gaussian_safety_con(phi, p_i, diff_p_i, modal_param)
                            Ls.append(block_diag(L, np.zeros((1, 1))))
                            cs.append(np.vstack([c, np.ones((1, 1))]))
                            ds.append(d)
                            diff_Ls.append(torch.block_diag(diff_L, torch.zeros((1, 1), dtype=torch.float64)))
                            diff_ds.append(diff_d)
                    else:
                        pass
                    
                    u, opt_value, value_for_grad = self.solve_qcqp(copy.deepcopy(u_ref), A, b, P_obj, Ls, cs, ds, diff_Ls, diff_ds)
                    u = np.squeeze(u)
                    u = u[:-1]

                    if len(Ls) > 0:
                        weights = torch.tensor([modal_param['weight'] for modal_param in modal_params_pred])
                        modal_dim = len(Ls)
                        value_for_grad += lambdas[0]*((1 - self.epsilon_f) - torch.dot(weights, diff_p))

                        opt_value_list[i,j]=opt_value
                        value_for_grad_list[i,j]=value_for_grad.item()
            
            if Ls:
                fig, (ax1, ax2) = plt.subplots(1, 2)
                CS = ax1.contourf(X, Y, opt_value_list)
                ax1.clabel(CS, inline=True, fontsize=10)

                CS = ax2.contourf(X, Y, value_for_grad_list)
                ax2.clabel(CS, inline=True, fontsize=10)

                plt.show()

        return u
    
    def solve_qcqp(self, u_ref, A, b, P_obj, Ls, cs, ds, diff_Ls, diff_ds):
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

        diff_Ls = [torch.block_diag(L, torch.zeros((1, 1))) for L in diff_Ls]

        # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
        # || L_i.T @ du + L_i.T @ u_ref || <= cs_i @ u + cs_i @ u_ref + ds_i
        soc_constraints = []
        soc_constraints += [cp.SOC(
            cs[i] @ du + (cs[i] @ u_ref).item() + ds[i],
            Ls[i].T @ du + np.squeeze(Ls[i].T @ u_ref)
        ) for i in range(len(Ls))]
        # # || PL @ du ||  <= mask.T @ du
        soc_constraints += [cp.SOC(mask.T @ du, PL @ du)]
        # 0 <= -a_i @ du - a_i @ u_ref - b_i
        soc_constraints += [cp.SOC(
            -A[i, :] @ du - A[i, :] @ u_ref + b[i, :],
            np.zeros((1, n)) @ du
        ) for i in range(b.shape[0])]
        
        prob = cp.Problem(cp.Minimize(mask.T @ du), soc_constraints)
        prob.solve()

        value_for_grad = 0
        if len(Ls) > 0:
            '''
            compute gradient of optimal value w.r.t. p_i using the envelope theorem
            value_for_grad += sum_i(dual_value*(|| L(p)@du || - d(p)))
            '''
            for i in range(len(Ls)):
                dual_value = torch.from_numpy(soc_constraints[i].dual_value[0])
                value_for_grad += dual_value * (torch.norm(diff_Ls[i] @ torch.from_numpy(du.value), p=2) - diff_ds[i])

        return u_ref[:-1] + np.vstack(du.value[:-1]), prob.value, value_for_grad

if __name__ == '__main__':

    env = SegWayMultiplicativeNoiseEnv()
    env.reset()
    ssa = MMMulRSSA(env, 
                    safety_index_params={
                        'alpha': 1.0,
                        'k_v': 1.0,
                        'beta': 0.001,
                    },
                    sampling=False,
                    fast_SegWay=False,
                    debug=False)
    
    q_d = np.array([0, 0])
    dq_d = np.array([1, 0])
    # for i in tqdm(range(960)):
    #     u = env.robot.PD_control(q_d, dq_d)
    #     u = ssa.safe_control(u)
    #     # print(env.Xr)
    #     env.step(u)
    #     env.render(img_name=str(i) + '.jpg', save_path='./src/pybullet-dynamics/SegWay_env/imgs/mm_mul_rssa/')

    # generate_gif('mm_mul_rssa.gif', './src/pybullet-dynamics/SegWay_env/imgs/mm_mul_rssa/', duration=0.01)


    ######## for debug
    safety_index_params={
        'alpha': 1.0,
        'k_v': 1.0,
        'beta': 0.001,
    }

    env.robot.K_m = 2.2
    env.robot.q =  np.array([0, -0.721])
    env.robot.dq = np.array([-2.184, -3.41])
    phi = env.get_phi(safety_index_params)
    p_phi_p_Xr = env.get_p_phi_p_Xr(safety_index_params)
    print(f'phi= {phi}')
    print(f'left={p_phi_p_Xr @ env.g}')
    print(f'right={-0.1*phi - p_phi_p_Xr @ env.f}')
    print(f'u {(-0.1*phi - p_phi_p_Xr @ env.f)/(p_phi_p_Xr @ env.g)}')
    print(ssa.safe_control(np.array([0])))