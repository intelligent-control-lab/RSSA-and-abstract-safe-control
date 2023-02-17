import os
from re import A
import numpy as np
import gym
import pandas as pd
import progressbar
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import gym_dynamics

from cvxopt import matrix, solvers
from utils import make_gif
import torch
from dynamics_model import FC
import scipy as sp
import scipy.stats
from scipy.linalg import block_diag
import cvxpy as cp
import warnings
warnings.filterwarnings('ignore')
solvers.options['show_progress'] = False

class SafetyIndex():
    def __init__(self, env, d_min, eta):
        self.coe = [0, 2, 5]

        self.env = env
        self.max_u = env.max_u
        self.dt = env.dt

        self.d_min = d_min
        self.eta = eta
    
    def load_nn_ensemble(self, prefix, n_ensemble, num_layer, hidden_dim, epoch):
        u_dim = 2
        x_dim = 4
        self.models = [FC(num_layer, x_dim, hidden_dim, x_dim+u_dim*x_dim) for i in range(n_ensemble)]
        for i in range(n_ensemble):
            load_path = "../model/"+prefix+str(i)+"/epoch_1000.pth"
            self.models[i].load_state_dict(torch.load(load_path))
    
    def ensemble_inference(self, x):
        u_dim = 2
        x_dim = 4
        n = len(self.models)
        fxs = np.zeros((n, x_dim))
        gxs = np.zeros((n, x_dim*u_dim))
        with torch.no_grad():
            
            inputs = torch.tensor(x).reshape(1,x_dim).float()
            
            for i in range(n):
                outputs = self.models[i].forward(inputs)
                fxs[i,:] = outputs[:,:x_dim].numpy()
                gxs[i,:] = outputs[:,x_dim:].numpy()

        return fxs, gxs
    
    def set_params(self, params):
        self.coe = params

    def phi(self, x, o):
        dp = x[[0,1]] - o[[0,1]]
        d = np.linalg.norm(dp)
        d = max(d, 1e-6)
        vx = x[2] * np.cos(x[3]) - o[2]
        vy = x[2] * np.sin(x[3]) - o[3]
        dv = np.array([vx, vy]) - o[[2,3]]
        dot_d = dp.T @ dv / d
        
        phi = self.d_min - d**self.coe[1] - self.coe[2] * dot_d

        return phi

    def compute_grad(self, x, o):
        Mr = np.vstack([x[0,0], x[1,0], x[2,0] * np.cos(x[3,0]), x[2,0] * np.sin(x[3,0])])
        Mh = np.vstack(o)

        dim = np.shape(Mr)[0] // 2
        p_idx = np.arange(dim)
        v_idx = p_idx + dim

        d = np.linalg.norm(Mr[p_idx] - Mh[p_idx])
        d = max(d, 1e-6)
        
        # sgn = -1 if np.asscalar((Mr[[0,1],0] - Mh[[0,1],0]).T * (Mr[[2,3],0] - Mh[[2,3],0])) < 0 else 1
        # dot_d = sgn * sqrt((Mr[2,0] - Mh[2,0])**2 + (Mr[3,0] - Mh[3,0])**2)

        dM = Mr - Mh

        dp = np.vstack(dM[p_idx,[0]])
        dv = np.vstack(dM[v_idx,[0]])

        #dot_d is the component of velocity lies in the dp direction
        dot_d = dp.T @ dv / d
        
        p_Mr_p_Xr = np.matrix([[1,0,0,0], [0,1,0,0], [0,0,np.cos(x[3,0]),-x[2,0]*np.sin(x[3,0])], [0,0,np.sin(x[3,0]),x[2,0]*np.cos(x[3,0])]])
        p_Mh_p_Xh = np.eye(len(o))
        
        p_dot_d_p_dp = dv / d - (dp.T @ dv) * dp / (d**3)
        p_dot_d_p_dv = dp / d
        
        p_dp_p_Mr = np.hstack([np.eye(dim), np.zeros((dim,dim))])
        p_dp_p_Mh = -p_dp_p_Mr

        p_dv_p_Mr = np.hstack([np.zeros((dim,dim)), np.eye(dim)])
        p_dv_p_Mh = -p_dv_p_Mr

        p_dot_d_p_Mr = p_dp_p_Mr.T @ p_dot_d_p_dp + p_dv_p_Mr.T @ p_dot_d_p_dv
        p_dot_d_p_Mh = p_dp_p_Mh.T @ p_dot_d_p_dp + p_dv_p_Mh.T @ p_dot_d_p_dv
        
        p_dot_d_p_Xr = p_Mr_p_Xr.T @ p_dot_d_p_Mr
        p_dot_d_p_Xh = p_Mh_p_Xh.T @ p_dot_d_p_Mh


        d = 1e-3 if d == 0 else d
        dot_d = 1e-3 if dot_d == 0 else dot_d

        p_d_p_Mr = np.vstack([ dp / d, np.zeros((dim,1))])
        p_d_p_Mh = np.vstack([-dp / d, np.zeros((dim,1))])
        p_d_p_Xr = p_Mr_p_Xr.T @ p_d_p_Mr
        p_d_p_Xh = p_Mh_p_Xh.T @ p_d_p_Mh

        return d, dot_d, dp, dv, p_d_p_Xr, p_dot_d_p_Xr, p_d_p_Mh, p_dot_d_p_Mh

class SSA(SafetyIndex):
    
    def __init__(self, env, d_min, eta, use_gt_dynamics=True):
        super(SSA, self).__init__(env, d_min, eta)
        self.use_gt_dynamics = use_gt_dynamics
        if not self.use_gt_dynamics:
            self.load_nn_ensemble("uncertain-unicycle-ensemble-FC4-100-[1, 2, 3]-", 10, 4, 100, 1000)

    def nn_fg(self, x):
        fxs, gxs = self.ensemble_inference(x)
        fx_mu = np.mean(fxs, axis=0).reshape(-1,1)
        gx_mu = np.mean(gxs, axis=0).reshape(-1,len(x)).T
        return fx_mu, gx_mu

    def grad_phi(self, x, o):
        
        d, dot_d, dp, dv, p_d_p_Xr, p_dot_d_p_Xr, p_d_p_Mh, p_dot_d_p_Mh = self.compute_grad(x, o)

        p_phi_p_Xr = - self.coe[1] * d**(self.coe[1]-1) * p_d_p_Xr - self.coe[2] * p_dot_d_p_Xr

        p_phi_p_Mh = - self.coe[1] * d**(self.coe[1]-1) * p_d_p_Mh - self.coe[2] * p_dot_d_p_Mh
        
        fx, gx = (self.env.f(x, perturb=False), self.env.g(x, perturb=False)) if self.use_gt_dynamics else self.nn_fg(x)
        dot_Mh = np.vstack([o[2], o[3], 0, 0])
        LfP = p_phi_p_Xr.T @ fx + p_phi_p_Mh.T @ dot_Mh
        LgP = p_phi_p_Xr.T @ gx
        return LfP, LgP

    def generate_safety_con(self, x, o):
        x = np.vstack(x)
        o = np.vstack(o)

        p = self.phi(x, o)
        LfP, LgP = self.grad_phi(x, o)
        
        # a*u <= b
        a = LgP
        # b = -LfP - self.eta * p
        b = -LfP - self.eta

        
        if p < 0:
            return np.zeros_like(a), np.ones_like(b)
        else:
            return a, b

    def safe_control(self, x, o, uref):
        ''' safe control
        Input:
            uref: reference control 
            x: state 
        '''
        # if the next state is unsafe, then trigger the safety control 
        ###### TODO ######## 

        # solve QP 
        # Compute the control constraints
        # Get f(x), g(x); note it's a hack for scalar u

        # compute the QP 
        # objective: 0.5*u*P*u + q*u

        w = 1000.0
        P = np.eye(len(uref)).astype(float)
        q = np.vstack(-uref).astype(float)
        # q = np.vstack(np.zeros(len(uref))).astype(float)
        
        # constraint
        # A u <= b  
        # u <= umax 
        # -u <= umax
        # c >= 0, c is the slack variable 

        n = len(uref)
        
        Abs = [self.generate_safety_con(x, o) for o in self.env.obstacles if self.phi(x,o) > 0]
        if self.env.wall_bounded:
            Abs = Abs + [self.generate_safety_con(x, o) for o in self.env.wall_obs]
        A = np.vstack([Ab[0] for Ab in Abs]) if len(Abs) > 0 else np.zeros((1,n))
        b = np.vstack([Ab[1] for Ab in Abs]) if len(Abs) > 0 else np.ones((1,1))

        
        A_slack = np.zeros((1,n+1))
        A_slack[0,-1] = -1
        b_slack = np.zeros((1,1))
        G = np.vstack([np.hstack([A, -np.ones((A.shape[0],1))]), np.eye(n,n+1), -np.eye(n,n+1), A_slack]).astype(float)
        h = np.vstack([b, self.env.max_u, self.env.max_u, b_slack])
        
        phis = [self.phi(x,o) for o in self.env.obstacles]
        
        P = np.eye(len(uref)+1).astype(float)
        P[-1,-1] = w
        q = np.vstack([-uref, [0]]).astype(float)
        
        sol_obj = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
        u = np.array(sol_obj['x'])
        u = u[:-1]
        return u

class RSSA(SSA):
    
    def __init__(self, env, d_min, eta, use_gt_uncertainty=False, use_convex=True):
        super(SSA, self).__init__(env, d_min, eta)
        self.use_gt_uncertainty = use_gt_uncertainty
        self.use_convex = use_convex

        if not self.use_gt_uncertainty:
            self.load_nn_ensemble("uncertain-unicycle-ensemble-FC4-100-[1, 2, 3]-", 10, 4, 100, 1000)

    def find_max_lp(self, c, A, b):
        #find max c.T @ x   subject to A x < b:
        # print("c")
        # print(c)
        c = np.reshape(c, (-1,1))
        c = c / np.max(np.abs(c)) if np.max(np.abs(c)) > 1e-3 else c # resize c to avoid too big objective, which may lead to infeasibility (a bug of cvxopt)
        sol_lp=solvers.lp(matrix(-c),matrix(A),matrix(b))

        print("c")
        print(c)
        print("A")
        print(A)
        print("b")
        print(b)
        print("sol_lp")
        print(sol_lp)

        if sol_lp['x'] is None:
            return None
        
        return np.vstack(np.array(sol_lp['x']))

    def points_2_vrep_hrep(self, points):
        hull = scipy.spatial.ConvexHull(points)
        vertices = points[hull.vertices]
        center = np.vstack(np.mean(vertices, axis=0))
        A = hull.equations[:,:-1]
        b = hull.equations[:,-1]
        print("center")
        print(center)
        # print(np.shape(center))
        # print(np.shape(vertices))
        # print(np.shape(A))
        # print(np.shape(b))
        print("A")
        print(A)
        print("b")
        print(b)
        for i in range(len(b)):
            if A[i,:] @ center > b[i]:
                # A[i,:] = -A[i,:]
                b[i] = -b[i]
            print("A[i,:] @ center > b[i]")
            print(A[i,:])
            print(center)
            print(A[i,:] @ center)
            print(b[i])
        return vertices, A, b

    def nn_convex_f_g(self, x):
        fxs, gxs = self.ensemble_inference(x)
        fx_hull_vertices, A_fx, b_fx = self.points_2_vrep_hrep(fxs)
        # return fx_hull_vertices, A_fx, b_fx, fx_hull_vertices, A_fx, b_fx
        gx_hull_vertices, A_gx, b_gx = self.points_2_vrep_hrep(gxs)
        # /print("nn convex")
        
        # fx_hull = scipy.spatial.ConvexHull(fxs[:,[1,2]])
        # plt.figure()
        # scipy.spatial.convex_hull_plot_2d(fx_hull)
        # plt.show()
        # plt.pause(3)

        return fx_hull_vertices, A_fx, b_fx, gx_hull_vertices, A_gx, b_gx
    
    def gt_convex_f_g(self, x):
        fx_hull_vertices, A_fx, b_fx = self.env.f_hull(x)
        gx_hull_vertices, A_gx, b_gx = self.env.g_hull(x)
        return fx_hull_vertices, A_fx, b_fx, gx_hull_vertices, A_gx, b_gx

    def nn_gaussian_f_g(self, x):
        fxs, gxs = self.ensemble_inference(x)
        fx_mu = np.mean(fxs, axis=0)
        fx_sigma = np.cov(fxs.T)
        gx_mu = np.mean(gxs, axis=0)  # gx is expanded row first. [g00, g01, g10, g11, g20, g21 ...]
        gx_sigma = np.cov(gxs.T)
        return fx_mu, fx_sigma, gx_mu, gx_sigma
    
    def gt_gaussian_f_g(self, x):
        fx_mu, fx_sigma = self.env.f_gaussian(x)
        gx_mu, gx_sigma = self.env.g_gaussian(x)
        return fx_mu, fx_sigma, gx_mu, gx_sigma

    def convex_grad_phi(self, x, o):
        
        d, dot_d, dp, dv, p_d_p_Xr, p_dot_d_p_Xr, p_d_p_Mh, p_dot_d_p_Mh = self.compute_grad(x, o)
        p_phi_p_Xr = - self.coe[1] * d**(self.coe[1]-1) * p_d_p_Xr - self.coe[2] * p_dot_d_p_Xr
        p_phi_p_Mh = - self.coe[1] * d**(self.coe[1]-1) * p_d_p_Mh - self.coe[2] * p_dot_d_p_Mh
        fx_hull_vertices, A_fx, b_fx, gx_hull_vertices, A_gx, b_gx = self.gt_convex_f_g(x) if self.use_gt_uncertainty else self.nn_convex_f_g(x)
        fx, gx = (self.env.f(x, perturb=True), self.env.g(x, perturb=True))
        dot_Mh = np.vstack([o[2], o[3], 0, 0])
        
        max_fx = self.find_max_lp(p_phi_p_Xr, A_fx, b_fx)
        max_LfP = p_phi_p_Xr.T @ max_fx + p_phi_p_Mh.T @ dot_Mh

        gx_hull_vertices = [gx.reshape((-1, len(x))).T for gx in gx_hull_vertices]
        LgP_hull_vertices = [p_phi_p_Xr.T @ gx for gx in gx_hull_vertices]
        
        return max_LfP, LgP_hull_vertices, A_gx, b_gx, p_phi_p_Xr

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
        chi2 = scipy.stats.chi2.ppf(p,n)
        L = np.linalg.cholesky(sigma)
        y = cp.Variable(n)
        # We use cp.SOC(sqrt(chi2), y) to create the SOC constraint ||y||_2 <= sqrt(chi2).
        prob = cp.Problem(cp.Minimize(c.T @ L @ y), [cp.SOC(np.sqrt(chi2), y)])
        prob.solve()
        x = L @ y.value + mu
        return x

    def gaussian_grad_phi(self, x, o):
        
        d, dot_d, dp, dv, p_d_p_Xr, p_dot_d_p_Xr, p_d_p_Mh, p_dot_d_p_Mh = self.compute_grad(x, o)
        p_phi_p_Xr = - self.coe[1] * d**(self.coe[1]-1) * p_d_p_Xr - self.coe[2] * p_dot_d_p_Xr
        p_phi_p_Mh = - self.coe[1] * d**(self.coe[1]-1) * p_d_p_Mh - self.coe[2] * p_dot_d_p_Mh
        
        fx_mu, fx_sigma, gx_mu, gx_sigma = self.nn_gaussian_f_g(x)
        n_x = len(x)
        n_u = len(gx_mu) // n_x
        

        dot_Mh = np.vstack([o[2], o[3], 0, 0])
        max_fx = self.find_max_qclp(p_phi_p_Xr, fx_mu, fx_sigma, 0.01)

        assert max_fx is not None
        
        gx_mu = gx_mu.reshape((len(x), -1))  # gx_mu is expanded row (u dim) first. [g00, g01, g10, g11, g20, g21 ...]
        
        max_LfP = p_phi_p_Xr.T @ fx_mu
        LgP_mu = p_phi_p_Xr.T @ gx_mu

        LgP_sigma = np.zeros((n_u, n_u))
        for i in range(n_u):
            for j in range(n_u):
                i_idx = [n_u*k+i for k in range(n_x)]
                j_idx = [n_u*k+j for k in range(n_x)]
                LgP_sigma[i,j] = p_phi_p_Xr.T @ gx_sigma[i_idx,:][:,j_idx] @ p_phi_p_Xr

        return max_LfP, LgP_mu, LgP_sigma
    
    def generate_convex_safety_con(self, x, o):
        x = np.vstack(x)
        o = np.vstack(o)
        
        max_LfP, LgP_hull_vertices, A_gx, b_gx, p_phi_p_Xr = self.convex_grad_phi(x, o)

        A_hat_Uc = np.vstack(LgP_hull_vertices)
        b_hat_Uc = np.vstack([-max_LfP - self.eta for i in range(len(LgP_hull_vertices))])
        
        return A_hat_Uc, b_hat_Uc, A_gx, b_gx, p_phi_p_Xr

    def generate_gaussian_safety_con(self, x, o, p=0.01):
        x = np.vstack(x)
        o = np.vstack(o)
        
        max_LfP, LgP_mu, LgP_sigma = self.gaussian_grad_phi(x, o)
        n = np.shape(x)[0]
        chi2 = scipy.stats.chi2.ppf(p,n)

        # || L u || <= c u + d
        L = np.linalg.cholesky(chi2 * LgP_sigma)
        c = -LgP_mu.T
        d = (-max_LfP - self.eta).item()
        
        return L, c, d

    def gaussian_safe_control(self, x, o, uref):
        ''' safe control
        Input:
            uref: reference control 
            x: state 
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

        n = len(uref)
        
        cons = [self.generate_gaussian_safety_con(x, o) for o in self.env.obstacles if self.phi(x,o) > 0]
        if self.env.wall_bounded:
            cons = cons + [self.generate_gaussian_safety_con(x, o) for o in self.env.wall_obs if self.phi(x,o) > 0]
        
        P_obj = np.eye(len(uref)+1).astype(float)
        P_obj[-1,-1] = 10000.

        Ls = [block_diag(con[0], np.zeros((1,1))) for con in cons] if len(cons) > 0 else [] # 
        cs = [np.vstack([con[1],np.ones(1)]) for con in cons] if len(cons) > 0 else [] # 
        ds = [con[2] for con in cons] if len(cons) > 0 else [] # 
        
        # A:  A_U,  A_slack
        A_slack = np.zeros((1,n+1))
        A_slack[0,-1] = -1
        b_slack = np.zeros((1,1))
        A = np.vstack([np.eye(n,n+1), -np.eye(n,n+1), A_slack]).astype(float)
        b = np.vstack([self.env.max_u, self.env.max_u, b_slack])

        uref = np.vstack([uref,np.zeros(1)])
        u = self.solve_qcqp(uref, A, b, P_obj, Ls, cs, ds)
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
        #               || a_i @ du || <= -a_i @ uref + b
        #               || L_i.T @ du + L_i.T @ uref || <= cs_i @ u + cs_i @ uref + ds_i
        #               where PL = block_diag (the cholesky decomposition of P, [0] )
        
        # Define and solve the CVXPY problem.
        n = len(uref) + 1
        du = cp.Variable(n)
        mask = np.zeros((n,1))
        mask[-1] = 1
        
        A = np.hstack([A, np.zeros((A.shape[0],1))])
        PL = np.linalg.cholesky(P_obj)
        PL = block_diag(PL, np.zeros((1,1)))

        Ls = [block_diag(L, np.zeros((1,1))) for L in Ls]
        cs = [np.vstack([d, np.zeros((1,1))]) for d in cs]
        cs = [np.squeeze(d) for d in cs]
        uref = np.vstack([uref, np.zeros((1,1))])

        # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
        soc_constraints = [cp.SOC(-A[i,:] @ du -A[i,:] @ uref + b[i], np.zeros((1,n)) @ du) for i in range(len(b))]#    0 <= -A_i' @ du + b_i'
        soc_constraints += [cp.SOC(mask.T @ du, PL @ du)] #|| PL @ du ||  <= mask.T @ du
        soc_constraints += [cp.SOC(cs[i] @ du + (cs[i] @ uref).item() + ds[i], Ls[i].T @ du + np.squeeze(Ls[i].T @ uref)) for i in range(len(Ls))]

        prob = cp.Problem(cp.Minimize(mask.T @ du), soc_constraints)
        
        prob.solve()

        return uref[:-1] + np.vstack(du.value[:-1])

    def convex_safe_control(self, x, o, uref):
        ''' safe control
        Input:
            uref: reference control 
            x: state 
        '''

        w = 1000.0
        P = np.eye(len(uref)).astype(float)
        q = np.vstack(-uref).astype(float)

        n = len(uref)

        # safety constraints: A * u < b
        # First part: LgP_i * u < b,  an overapproximation of U_c
        # Second part: u < u_lim,  control limit
        # Third part: lambda > 0,  the slack variable must be positive
        # Abs = [self.generate_safety_con(x, o) for o in self.env.obstacles if self.phi(x, o) > 0]
        # Abs = Abs + [self.generate_safety_con(x, o) for o in self.env.wall_obs if self.phi(x, o) > 0]
        Abs = [self.generate_convex_safety_con(x, o) for o in self.env.obstacles if self.phi(x,o) > 0]
        if self.env.wall_bounded:
            Abs = Abs + [self.generate_convex_safety_con(x, o) for o in self.env.wall_obs if self.phi(x,o) > 0]
        A_hat_Uc = np.vstack([Ab[0] for Ab in Abs]) if len(Abs) > 0 else np.zeros((1,n)) # LgP_i * u < b,  the set \hat U_c
        b_hat_Uc = np.vstack([Ab[1] for Ab in Abs]) if len(Abs) > 0 else np.ones((1,1))
        
        A_slack = np.zeros((1,n+1))
        A_slack[0,-1] = -1
        b_slack = np.zeros((1,1))

        G = np.vstack([np.hstack([A_hat_Uc, -np.ones((A_hat_Uc.shape[0],1))]), np.eye(n,n+1), -np.eye(n,n+1), A_slack]).astype(float)
        h = np.vstack([b_hat_Uc, self.env.max_u, self.env.max_u, b_slack])
        
        P = np.eye(len(uref)+1).astype(float)
        P[-1,-1] = w
        q = np.vstack([-uref, [0]]).astype(float)
        
        sol_qp = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
        u = np.vstack(sol_qp['x'])
        u = u[:-1]
        
        for Ab in Abs: # project u to LgP of all phis
            b_hat_Uc = Ab[1]
            c = b_hat_Uc[0] # - max_LfP - eta (b_hat_Uc = (- max_LfP - eta) * np.ones(n))
            A_gx = Ab[2]
            b_gx = Ab[3]
            p_phi_p_Xr = Ab[4]
            grad_phi_u = np.vstack(p_phi_p_Xr) * np.hstack(u)
            grad_phi_u = grad_phi_u.T.reshape(-1)
            max_g = self.find_max_lp(grad_phi_u, A_gx, b_gx)
            max_LgP = max_g.reshape((-1, len(x))) @ p_phi_p_Xr
            
            # we need to make sure max_LgP.T @ u < - max_LfP - eta

            if max_LgP is None: # no robust safe control exists for this phi
                continue # TODO: we need to switch to safest index here
            
            if u.T @ max_LgP > c:
                u = u * c / (u.T @ max_LgP)
                u = self.env.filt_action(u)

        return np.vstack(np.array(u))

        
    def safe_control(self, x, o, uref):
        if self.use_convex:
            return self.convex_safe_control(x, o, uref)
        else:
            return self.gaussian_safe_control(x, o, uref)

def evaluate(uncertainty=1, use_rssa=True, render=False):
    
    # env = gym.make('Uncertain-Unicycle-v0')
    env = gym.make('Uncertain-Unicycle-Hit-v0')

    # state, reward, done, info = env.reset(uncertainty=uncertainty, obs_num=5)
    state, reward, done, info = env.reset(state=[-3,-2,0,np.pi/4], goal = [5,5], obs=[0,0.1,0,0], uncertainty=uncertainty, obs_radius=2)
    step = 0
    
    np.random.seed(0)
    obs_r = env.obs_radius + 0
    eta = 0.1
    ssa = RSSA(env, obs_r, eta, use_gt_uncertainty=False, use_convex=False) if use_rssa else SSA(env, obs_r, eta, use_gt_dynamics=False)
    # ssa.test_qcqp()
    # exit()
    for step in range(200):
        u_ref = env.compute_u_ref()
        u = ssa.safe_control(state, info["obs_state"], u_ref)
        
        state, reward, done, info = env.step(u)

        if done:
            break
            env.reset(state=[-3,-2,0,np.pi/4], goal = [5,5], obs=[0,0.1,0,0], uncertainty=uncertainty, obs_radius=2)

        if render:
            phis = [ssa.phi(state,o) for o in ssa.env.obstacles]
            # if np.any(np.array(phis) > 0):
            #     input()
            env.render(phis=phis, save=True)

if __name__ == "__main__":
    evaluate(uncertainty=5, use_rssa=True, render=True)
    # evaluate(uncertainty=5, use_rssa=False, render=True)
    # evaluate(2)
    # for uncertainty in np.arange(0,6,1):
        # print("uncertainty = ", uncertainty)
    #     evaluate(uncertainty)