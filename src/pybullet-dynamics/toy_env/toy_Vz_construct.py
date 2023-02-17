import pybullet as p
import numpy as np
from scipy.linalg import null_space
import warnings
import copy

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from pyhull.halfspace import Halfspace, HalfspaceIntersection
from scipy.spatial import ConvexHull
from cvxopt import matrix, solvers

from toy_model import ToyModel
from toy_utils import *

class ToyVzConstruction:
    '''
    Given z = (p, dp), (p is the position of end-effector, dp is the velocity of end-effector)\\
    reconstruct X(z) := { x=(q, dq) | x \in X(z) },
    and construct V(z) := \cap _{x \in X(z)} { v(x, u) | u \in U }
    '''
    def __init__(self, physics_client_id):
        self.model = ToyModel(physics_client_id=physics_client_id)

        # get all the limitations
        self.dof = self.model.dof
        self.q_max = self.model.q_max
        self.q_min = self.model.q_min
        self.dq_max = self.model.velocity_limit
        self.dq_min = -self.model.velocity_limit
        self.u_max = self.model.torque_limit
        self.u_min = -self.model.torque_limit

        # prepare for the half-spaces of VIK
        self.dq_limit_halfspaces = []
        for i in range(self.dof):
            normal = np.zeros_like(self.dq_max)
            normal[i] = 1.0
            self.dq_limit_halfspaces.append(Halfspace(normal, -self.dq_max[i]))
            self.dq_limit_halfspaces.append(Halfspace(-normal, self.dq_min[i]))
        
        # prepare for LCQP in VIK construction
        self.G = np.vstack((np.eye(self.dof), -np.eye(self.dof)))
        self.h = np.concatenate((self.dq_max, -self.dq_min))
        self.u_vertices = []
        id = 0
        max_id = 2**self.dof - 1    # find the vertices of U
        while id <= max_id:
            tmp_id = id
            u = np.zeros_like(self.u_min)
            for i in range(self.dof):
                flag = tmp_id % 2
                tmp_id = tmp_id // 2
                u[i] = self.u_max[i] if flag else self.u_min[i]
            id = id + 1
            self.u_vertices.append(u)
        self.u_vertices = np.asanyarray(self.u_vertices)

    def sample_ddp_set_given_q_dq(self, q, dq, sample_size=1000):
        '''
        Calculate DDP_1 := \cup _{u \in U} { ddp(q, dq, u) },
        through sampling u uniformly in U
        '''
        ddp_sample = []
        for _ in range(sample_size):
            u = np.random.uniform(low=self.u_min, high=self.u_max)
            ddp = self.model.calculate_end_eff_acceleration(q, dq, u)
            ddp_sample.append(ddp)
        ddp_sample = np.asanyarray(ddp_sample)
        u_sample = np.asanyarray(u_sample)
        return ddp_sample

    def vertices_ddp_set_given_q_dq(self, q, dq):
        '''
        Calculate DDP_1 := \cup _{u \in U} { ddp(q, dq, u) },
        through mapping the vertices of U into vertices of DDP_1.\\
        U is supposed to be a hyper-rectangle.
        '''
        ddp_vertices = []
        for u in self.u_vertices:
            ddp = self.model.calculate_end_eff_acceleration(q, dq, u)
            ddp_vertices.append(ddp)
        ddp_vertices = np.asanyarray(ddp_vertices)
        return ddp_vertices

    def sample_q_set_given_p(
        self, p, sample_size=1000,
        lr=1e-1, max_steps=100, eps=1e-3,
    ):
        '''
        Calculate IK(p) := { q | FK(q) = p }
        '''
        def IK(init_q, lr, max_steps, eps):
            q = init_q
            calc_p = self.model.kinematics_chain.get_computed_end_eff_pose(q)
            count = 0
            while count < max_steps and np.linalg.norm(p - calc_p) > eps:
                count += 1
                calc_p = self.model.kinematics_chain.get_computed_end_eff_pose(q)
                J = self.model.get_jacobian(q)  
                grad = 2 * (calc_p - p) @ J
                q = q - lr * grad
            if count > max_steps:
                return None
            else:
                return q

        q_sample = []
        for _ in range(sample_size):
            init_q = np.random.uniform(low=self.q_min, high=self.q_max)
            q = IK(init_q=init_q, lr=lr, max_steps=max_steps, eps=eps)
            if q is not None:
                q_sample.append(q)
        q_sample = np.asanyarray(q_sample)
        return q_sample

    def sample_dq_set_given_dp_q(
        self,
        dp, q,
        sample_size=20,
    ):
        '''
        Calculate VIK(dp, q) := { dq | J(q) @ dq = dp and dq_min < dq < dq_max }:\\
            1. get the vertices of the polyhedron VIK(dp, q):\\
                a. get unconstrained linear subspace: { J(q) @ dq = dp }\\
                b. it is the intersection of the subspace and {dq_min < dq < dq_max}\\
            2. sample points inside it \\
        WARNING: this function can only be used when dim(q) = 3 and dim(q) = 2 and J(q) is full row rank
        '''
        J = self.model.get_jacobian(q)
        special_solution = np.linalg.pinv(J) @ dp
        Null_J = np.squeeze(null_space(J))

        def check_in_box(param):
            nonlocal count, param_list
            vec = Null_J * param + special_solution
            print(vec)
            if np.all(vec <= self.dq_max) and np.all(vec >= self.dq_min):
                count = count - 1
                param_list.append(param)
                return True
            else:
                return False

        # 1.b. get the vertices of the intersection
        sorted_ids = np.argsort(-np.abs(Null_J))
        count = 2
        param_list = []
        for id in sorted_ids:
            param_max = (self.dq_max[id] - special_solution[id]) / Null_J[id]
            if check_in_box(param_max) and count == 0:
                break
            param_min = (self.dq_min[id] - special_solution[id]) / Null_J[id]
            if check_in_box(param_min) and count == 0:
                break
        if len(param_list) < 2:
            warnings.warn('VIK set is empty!')
            return None

        param_list.sort()   
        dq_sample = []
        for _ in range(sample_size):
            param = np.random.uniform(low=param_list[0], high=param_list[1])
            dq = Null_J * param + special_solution
            dq_sample.append(dq)
        dq_sample = np.asanyarray(dq_sample)

        dq_limit = []
        for param in param_list:
            dq = Null_J * param + special_solution
            dq_limit.append(dq)
        return dq_sample, dq_limit

    def fit_ddp_when_q_is_fixed(self, q, dp, data_size=20):
        '''
        When q and u is fixed, ddp is quadratic of dq: \\
        ddp = p_{J(q) @ dq}_p_q @ dq + J(q) @ M(q)^(-1) @ (u - C(q, dq) - g(q)) \\
        set u = 0: \\
        ddp_0 = p_{J(q) @ dq}_p_q @ dq - J(q) @ M(q)^(-1) @ (C(q, dq) + g(q)) \\
        Fit this function with ddp_0[i] = 0.5 * dq.T @ P[i] @ dq + Q[i].T @ dq + c[i] 
        '''
        u = np.zeros_like(self.u_max)
        J = self.model.get_jacobian(q)
        special_solution = np.linalg.pinv(J) @ dp
        sigma = max(np.sqrt(np.linalg.norm(special_solution)), 1.0)

        dq_data = np.zeros((data_size, len(q)))
        ddp_0_data = np.zeros((data_size, len(dp)))
        for i in range(data_size):
            dq = special_solution + np.random.randn(len(q)) * sigma
            ddp_0 = self.model.calculate_end_eff_acceleration(q, dq, u)
            dq_data[i, :] = dq 
            ddp_0_data[i, :] = ddp_0

        poly = PolynomialFeatures(degree=2)
        X = poly.fit_transform(dq_data)
        params = []
        for k in range(len(dp)):
            y = ddp_0_data[:, k]
            regression_model = LinearRegression(fit_intercept=False)
            regression_model.fit(X, y)
            param = regression_model.coef_
            const_term = param[0]
            linear_term = param[1: self.dof+1]
            quad_term_tmp = param[self.dof+1: ]
            quad_term = np.zeros((self.dof, self.dof))
            j = 0
            for i in range(self.dof):
                quad_term[i, i:] = quad_term_tmp[j: j+self.dof-i]
                j = j + self.dof - i
            quad_term = quad_term + quad_term.T
            params.append({'P': quad_term, 'Q': linear_term, 'c': const_term})
        
        return params
        
    def vertices_dq_set_given_dp_q(
        self, 
        dp, q,
        halfspace_layered_degree=1e-5,  
        round_level=2,
    ):
        '''
        Calculate VIK(dp, q) := { dq | J(q) @ dq = dp and dq_min < dq < dq_max }:\\
            1. First, given u = 0, fit ddp_0[i] = 0.5 * dq.T @ P[i] @ dq + Q[i].T @ dq + c[i] \\
            2. Second, calculate argmax{ ddp_0[i](dq) } and argmin{ ddp_0[i](dq) }  \\
            3. Third, use the dq's found in 2. to calculate the intersection of V(q, dq)'s: \\
                a. get vertices of ddp_0 \\
                b. get intersections of ddp_0 + J(q) @ M(q)^(-1) @ { u | u \in U } \\
        Paramters:
            halfspace_layered_degree: ensure a.T @ x = b could be relaxed as a.T @ x <= b + eps \cap a.T @ x >= b - eps 
            round_level: used to remove the redundancy in vertices of VIK
        '''
        # construct vertices of VIK(dp, q) when only considering dq limits
        J = self.model.get_jacobian(q)
        G = np.vstack((self.G, J, -J))  # get feasible point
        h = np.concatenate((self.h, dp, -dp))
        c = np.zeros(self.dof)
        sol_lp = solvers.lp(
            c=matrix(c.reshape(-1, 1)),
            G=matrix(G),
            h=matrix(h.reshape(-1, 1)),
        )
        feasible_point = np.squeeze(sol_lp['x'])
        if feasible_point is None:
            warnings.warn('VIK seems empty!')
            return None
        VIK_halfspaces = copy.copy(self.dq_limit_halfspaces)
        for k in range(J.shape[0]):
            VIK_halfspaces.append(Halfspace(J[k], -dp[k] - halfspace_layered_degree))
            VIK_halfspaces.append(Halfspace(-J[k], dp[k] - halfspace_layered_degree))
        halfspace_intersection = HalfspaceIntersection(VIK_halfspaces, feasible_point)
        VIK_vertices = halfspace_intersection.vertices
        
        # remove the redundancy in VIK(dp, q) when only considering dq limits
        dic = {}
        for vertex in VIK_vertices:
            key = tuple(np.around(vertex, decimals=round_level))
            try:
                tmp = dic[key]
                dic[key] = {
                    'num': tmp['num'] + 1,
                    'dq': (tmp['num'] * tmp['dq'] + vertex) / (tmp['num'] + 1)
                }
            except KeyError:
                dic[key] = {
                    'num': 1,
                    'dq': vertex
                }
        VIK_vertices = [val['dq'] for val in dic.values()]

        # solve QP with linear constraints
        params = self.fit_ddp_when_q_is_fixed(q, dp)
        for k in range(len(dp)):
            P = params[k]['P']
            eign_values = np.linalg.eigvals(P)
            if np.all(eign_values > 0):
                print('Positive deifinte!')
                sol_qp = solvers.qp(
                    P=matrix(params[k]['P']),
                    q=matrix(params[k]['Q'].reshape(-1, 1)),
                    G=matrix(self.G),
                    h=matrix(self.h.reshape(-1, 1)),
                    A=matrix(J),
                    b=matrix(dp.reshape(-1, 1)),
                )
                dq = np.squeeze(sol_qp['x'])
                VIK_vertices.append(dq)
            elif np.all(eign_values < 0):
                print('Negative definite!')
                sol_qp = solvers.qp(
                    P=-matrix(params[k]['P']),
                    q=-matrix(params[k]['Q'].reshape(-1, 1)),
                    G=matrix(self.G),
                    h=matrix(self.h.reshape(-1, 1)),
                    A=matrix(J),
                    b=matrix(dp.reshape(-1, 1)),
                )
                dq = np.squeeze(sol_qp['x'])
                VIK_vertices.append(dq)
            else:
                print('Indefinite!')
                
        # construct convex hull for ddp_0
        ddp_0_vertices = []
        u = np.zeros_like(self.u_max)
        for dq_vertex in VIK_vertices:
            ddp_0 = self.model.calculate_end_eff_acceleration(q, dq_vertex, u)
            ddp_0_vertices.append(ddp_0)
        ddp_0_vertices = np.asanyarray(ddp_0_vertices)
        ddp_0_convexhull = ConvexHull(ddp_0_vertices)
        ddp_0_convexhull_vertices = ddp_0_vertices[ddp_0_convexhull.vertices]
        
        # get V(z) from ddp_0
        Vz_equations = None
        M, _, _ = self.model.calculate_dynamic_matrices(q, np.zeros_like(self.dq_max))
        M_inv = np.linalg.pinv(M)
        for ddp_0 in ddp_0_convexhull_vertices:
            ddp_vertices = []
            for u in self.u_vertices:
                ddp = ddp_0 + J @ M_inv @ u
                ddp_vertices.append(ddp)
            ddp_vertices = np.asanyarray(ddp_vertices)
            ddp_convexhull = ConvexHull(ddp_vertices)
            if Vz_equations is None:
                Vz_equations = ddp_convexhull.equations
            else:
                Vz_equations = np.concatenate((Vz_equations, ddp_convexhull.equations))
        G = Vz_equations[:, :-1]  # get feasible point
        h = -Vz_equations[:, -1]
        c = np.zeros_like(dp)
        sol_lp = solvers.lp(
            c=matrix(c.reshape(-1, 1)),
            G=matrix(G),
            h=matrix(h.reshape(-1, 1)),
        )
        feasible_point = np.squeeze(sol_lp['x'])
        if feasible_point is None:
            warnings.warn('VIK seems empty!')
            return None
        Vz_halfspaces = []
        for i in range(Vz_equations.shape[0]):
            Vz_halfspaces.append(Halfspace(normal=Vz_equations[i, :-1], offset=Vz_equations[i, -1]))
        Vz_halfspaces_intersection = HalfspaceIntersection(Vz_halfspaces, feasible_point)
        return Vz_halfspaces_intersection.vertices



if __name__ == '__main__':
    physics_client_id = p.connect(p.DIRECT)
    Vz_solver = ToyVzConstruction(physics_client_id=physics_client_id)

    q = np.array([0.38, 1.03, 0.32])
    dq = (Vz_solver.dq_max + Vz_solver.dq_min) / 2 + 0.1
    X = np.array([1.0, -1.0])
    V = np.array([0.17, -0.34])
    # DDP_1 = Vz_solver.sample_ddp_set_given_q_dq(q, dq)
    # draw_sample_points(DDP_1, 'DDP_1.jpg')
    # draw_sample_points(U, 'U.jpg')
    # DDP_1 = Vz_solver.vertices_ddp_set_given_q_dq(q, dq)
    # draw_2d_polyhedron(DDP_1, fig_name='DDP_1_polyhedron.jpg')
    
    # X = np.array([1.0, -2.0])
    # IK_p = Vz_solver.sample_q_set_given_p(X)
    # draw_sample_points(IK_p, 'IK_p.jpg', save_path='./src/pybullet-dynamics/toy_env/imgs/IK_p_and_DDP_1_vertices/', show_flag=True)

    # for i, q in enumerate(IK_p):
    #     print(i)
    #     dq = np.zeros_like(q)
    #     DDP_1 = Vz_solver.sample_ddp_set_given_q_dq(q, dq, sample_size=100)
    #     # DDP_1 = Vz_solver.vertices_ddp_set_given_q_dq(q, dq)
    #     draw_sample_points(DDP_1, fig_name=str(i) + '.jpg', 
    #                     save_path='./src/pybullet-dynamics/toy_env/imgs/IK_p_and_DDP_1_vertices/', 
    #                     close_flag=False)

    VIK, dq_limit = Vz_solver.sample_dq_set_given_dp_q(V, q, sample_size=20)
    # draw_sample_points(VIK, 'VIK.jpg', show_flag=True)
    for i, dq in enumerate(VIK):
        print(i)
        DDP_2 = Vz_solver.vertices_ddp_set_given_q_dq(q, dq)
        draw_2d_polyhedron(DDP_2, fig_name=str(i) + '.jpg', 
                        save_path='./src/pybullet-dynamics/toy_env/imgs/VIK_and_DDP_2_sample/', 
                        close_flag=False)
    
    # for j, dq in enumerate(dq_limit):
    #     DDP_2 = Vz_solver.vertices_ddp_set_given_q_dq(q, dq)
    #     draw_2d_polyhedron(DDP_2, fig_name=str(i + j + 1) + '.jpg', 
    #                     save_path='./src/pybullet-dynamics/toy_env/imgs/VIK_and_DDP_2_sample/', 
    #                     c='r', alpha=0.8,
    #                     close_flag=False) 

    Vz_q_fixed_vertices = Vz_solver.vertices_dq_set_given_dp_q(V, q)
    draw_2d_polyhedron(Vz_q_fixed_vertices, 'Vz_q_fixed.jpg', c='r', alpha=1)

    # for i in range(100):
    #     print(i)
    #     q = np.random.uniform(-1.57, 1.57, size=3)
    #     V = np.random.randn(2)
    #     Vz_solver.vertices_dq_set_given_dp_q(V, q)

    # X_start = np.array([1.0, -1.0])
    # X_end = np.array([-1.0, -2.0])
    # X_list = np.linspace(start=X_start, stop=X_end, num=100)
    # for i, X in enumerate(X_list):
    #     print(i)
    #     IK_p = Vz_solver.sample_q_set_given_p(X, sample_size=100)
    #     draw_sample_points(IK_p, fig_name=str(i) + '.jpg', 
    #                     save_path='./src/pybullet-dynamics/toy_env/imgs/IK_p_with_different_p/', 
    #                     alpha=0.6,
    #                     close_flag=True, show_flag=False)
    