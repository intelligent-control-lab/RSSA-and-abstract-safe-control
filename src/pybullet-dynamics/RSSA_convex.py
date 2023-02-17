import numpy as np
import scipy
from scipy.linalg import block_diag, null_space
from scipy.spatial import ConvexHull
from cvxopt import matrix, solvers
import cvxpy as cp
from sklearn.decomposition import PCA
import pickle
from loguru import logger

from SCARA_env.SCARA_parameter_learning import SCARAParameterLearningEnv
from SegWay_env.SegWay_parameter_learning import SegWayParameterLearningEnv
from RSSA_safety_index import SafetyIndex
from RSSA_utils import *

solvers.options['show_progress'] = False
monitor = Monitor()

class ConvexRSSA(SafetyIndex):
    def __init__(
        self,
        env: SCARAParameterLearningEnv,
        safety_index_params={
            'alpha': 1.0,
            'k_v': 1.0,
            'beta': 0.0,
        },
        sample_points_num=50,
        gamma=0.1,
    ):
        super().__init__(env, safety_index_params)
        self.sample_points_num = sample_points_num
        self.gamma = gamma
        self.x_dim = self.env.Xr.shape[0]
        self.u_dim = self.env.g.shape[1]

    def find_max_lp(self, c, A, b):
        # find max c.T @ x   subject to A @ x < b:
        c = c.reshape(-1, 1)
        b = b.reshape(-1, 1)
        c = c / np.max(np.abs(c)) if np.max(np.abs(c)) > 1e-3 else c # resize c to avoid too big objective, which may lead to infeasibility (a bug of cvxopt)
        sol_lp=solvers.lp(
            matrix(-c),
            matrix(A),
            matrix(b)
        )

        assert sol_lp['x'] is not None
        return np.vstack(np.array(sol_lp['x']))

    def use_pca_to_points(self, points, tolerance=1e-10):
        num, dim = points.shape
        center = np.mean(points, axis=0)
        points = points - center
        pca = PCA(n_components=dim)
        pca.fit(points)
        trans_matrix = []
        max_ratio = pca.explained_variance_ratio_[0]
        for i, ratio in enumerate(pca.explained_variance_ratio_):
            if ratio > max_ratio * tolerance:
                trans_matrix.append(pca.components_[i])
            else:
                break
        trans_matrix = np.asanyarray(trans_matrix)  # trans_matrix's shape: (low_dim, high_dim)
        points = points @ trans_matrix.T
        return points, center, trans_matrix

    def points_2_vrep_hrep(self, points):
        # points should be of the shape (num, dim)
        num, dim = points.shape

        # when dim = 1, do not use convexhull method:
        if dim == 1:
            points = np.squeeze(points)
            points = np.sort(points)
            vertices = np.asanyarray([[points[0]], [points[-1]]])
            A = np.zeros((2, 1))
            b = np.zeros(2)
            A[0, 0] = 1.0
            b[0] = points[-1]
            A[1, 0] = -1.0
            b[1] = -points[0]

        # when dim > 1:
        else:
            hull = ConvexHull(points)
            vertices = points[hull.vertices]
            center = np.mean(vertices, axis=0)
            A = hull.equations[:, :-1]
            b = -hull.equations[:, -1]
            for i in range(len(b)):
                if A[i, :] @ center > b[i]:
                    A[i, :] = -A[i, :]
                    b[i] = -b[i]
                    
        return vertices, A, b

    def get_convex_f_g(self):
        f_points, g_points = self.env.sample_f_g_points(self.sample_points_num)
        f_points = np.array(f_points).reshape(self.sample_points_num, -1)
        g_points = np.array(g_points).reshape(self.sample_points_num, -1)   # g is expanded row first!!!

        f_pca_points, self.f_pca_center, self.f_pca_trans_matrix = self.use_pca_to_points(f_points)
        self.f_pca_hull_vertices, A_f_pca, b_f_pca = self.points_2_vrep_hrep(f_pca_points)
        
        g_pca_points, self.g_pca_center, self.g_pca_trans_matrix = self.use_pca_to_points(g_points)
        self.g_pca_hull_vertices, A_g_pca, b_g_pca = self.points_2_vrep_hrep(g_pca_points)

        return self.f_pca_hull_vertices, A_f_pca, b_f_pca, self.g_pca_hull_vertices, A_g_pca, b_g_pca

    def get_convex_grad_phi(self):
        p_phi_p_Xr = self.env.get_p_phi_p_Xr(self.safety_index_params)
        
        f_pca_hull_vertices, A_f_pca, b_f_pca, g_pca_hull_vertices, A_g_pca, b_g_pca = self.get_convex_f_g()

        # get max_LfP: max p_phi_p_Xr @ (f_center + f_trans_matrix.T @ f_pca)
        try:
            max_f_pca = self.find_max_lp(p_phi_p_Xr @ self.f_pca_trans_matrix.T, A_f_pca, b_f_pca)
        except:
            max_f_pca = f_pca_hull_vertices[0].reshape(-1, 1)
            print('unstable f hull!')
        max_LfP = p_phi_p_Xr @ (self.f_pca_center.reshape(-1, 1) + self.f_pca_trans_matrix.T @ max_f_pca)

        # recover vertices of g hull from vertices of g_pca hull
        g_hull_vertices = g_pca_hull_vertices @ self.g_pca_trans_matrix + self.g_pca_center
        g_hull_vertices = g_hull_vertices.reshape(-1, self.x_dim, self.u_dim)
        self.LgP_hull_vertices = [p_phi_p_Xr @ g for g in g_hull_vertices]

        return max_LfP, self.LgP_hull_vertices, A_g_pca, b_g_pca, p_phi_p_Xr

    def generate_convex_safety_con(self, phi):
        max_LfP, LgP_hull_vertices, A_g_pca, b_g_pca, p_phi_p_Xr = self.get_convex_grad_phi()
        A_hat_Uc = np.vstack(LgP_hull_vertices)
        self.c = -max_LfP - self.gamma * phi
        b_hat_Uc = np.vstack([self.c for _ in range(len(LgP_hull_vertices))])
        return A_hat_Uc, b_hat_Uc, A_g_pca, b_g_pca, p_phi_p_Xr

    def safe_control(self, u_ref, force_rssa=False):
        # object
        n = self.u_dim
        P = np.eye(n + 1).astype(float)
        P[-1, -1] = 10000
        q = np.vstack([-u_ref.reshape(-1, 1), np.zeros((1, 1))]).astype(float)

        self.phi = self.env.get_phi(self.safety_index_params)
        try:
            Ab = self.generate_convex_safety_con(self.phi)
            self.analyze_dynamics() # for uncertainty analysis
        except:
            Ab = None
        
        if self.phi > 0 or force_rssa:
            if Ab is None:
                Ab = self.generate_convex_safety_con(self.phi)
            Abs = [Ab]
            A_hat_Uc = Ab[0]
            b_hat_Uc = Ab[1]
        else:
            Abs = []
            A_hat_Uc = np.zeros((1, n))
            b_hat_Uc = np.ones((1, 1))
        
        A_slack = np.zeros((1, n + 1))
        A_slack[0, -1] = -1
        b_slack = np.zeros((1, 1))
        G = np.vstack([
            np.hstack([A_hat_Uc, -np.ones((A_hat_Uc.shape[0], 1))]),
            np.eye(n, n + 1),
            -np.eye(n, n + 1),
            A_slack,
        ])
        h = np.vstack([
            b_hat_Uc,
            np.asanyarray(self.env.u_limit['high']).reshape(-1, 1),
            -np.asanyarray(self.env.u_limit['low']).reshape(-1, 1),
            b_slack,
        ])

        sol_obj = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
        u = np.squeeze(sol_obj['x'])
        if np.abs(u[-1]) > 1e-3:
            logger.debug(f'u_FI in ConvexRSSA: {u}')
        u = u[:-1]
        

        # project u to LgP
        for Ab in Abs:
            c = Ab[1][0]
            A_g_pca = Ab[2]
            b_g_pca = Ab[3]
            p_phi_p_Xr = Ab[4]
            grad_phi_mul_u = p_phi_p_Xr.reshape(-1, 1) @ u.reshape(1, -1)
            grad_phi_mul_u = grad_phi_mul_u.reshape(-1, 1)
            
            try:
                grad_phi_mul_u_mul_trans_matrix = self.g_pca_trans_matrix @ grad_phi_mul_u 
                max_g_pca = self.find_max_lp(grad_phi_mul_u_mul_trans_matrix, A_g_pca, b_g_pca)
                max_g = self.g_pca_center.reshape(-1, 1) + self.g_pca_trans_matrix.T @ max_g_pca
                max_g = max_g.reshape(self.x_dim, self.u_dim)
            except:
                print('unstable g hull!')
                return u

            max_LgP = p_phi_p_Xr @ max_g
            max_LgP = np.squeeze(max_LgP)

            c = c - np.dot(grad_phi_mul_u.reshape(-1,), self.g_pca_center)
            if np.dot(max_LgP, u) > c:
                u = u * c / np.dot(max_LgP, u)
        
        return u
    
    # just for uncertainty bound analysis in SCRCA environment
    def analyze_dynamics(self, m_2=0.5):
        true_param = self.env.get_param()
        self.env.set_param(m_2)
        self.f_pca_fake = self.f_pca_trans_matrix @ (self.env.f - self.f_pca_center.reshape(-1, 1))
        self.g_pca_fake = self.g_pca_trans_matrix @ (self.env.g.reshape(-1, 1) - self.g_pca_center.reshape(-1, 1))
        self.env.set_param(true_param)


if __name__ == '__main__':
    # env = SCARAParameterLearningEnv(m_1=0.2, m_2=1.0, use_online_adaptation=False, m_2_std_init=0.3)
    # ssa = ConvexRSSA(env, safety_index_params={'alpha': 1.0, 'k_v': 1.0, 'beta': 0.0, 'eta':0.05}, sample_points_num=50)

    # env.reset()
    # q_way_points = np.linspace(start=[np.pi/2 - 0.05, 0.0], stop=[-np.pi/3, -np.pi], num=400)
    # for i, q_d in enumerate(q_way_points):
    #     env.param_pred()
    #     u = env.robot.computed_torque_control(q_d=q_d)
    #     for _ in range(3):
    #         u = ssa.safe_control(u)
    #         env.step(u)
    #     monitor.update(
    #         Xr=env.Xr,
    #         f_pca_center=ssa.f_pca_center, 
    #         f_pca_trans_matrix=ssa.f_pca_trans_matrix, 
    #         f_pca_hull_vertices=ssa.f_pca_hull_vertices,
    #         g_pca_center=ssa.g_pca_center, 
    #         g_pca_trans_matrix=ssa.g_pca_trans_matrix, 
    #         g_pca_hull_vertices=ssa.g_pca_hull_vertices,
    #         f_pca_fake=ssa.f_pca_fake,
    #         g_pca_fake=ssa.g_pca_fake,
    #         c=ssa.c,
    #         LgP_hull_vertices=ssa.LgP_hull_vertices,
    #     )
            
    #     # env.render(img_name=str(i) + '.jpg', save_path='./src/pybullet-dynamics/SCARA_env/imgs/convex_rssa/')
    # with open('./src/pybullet-dynamics/safety_learning_log/f_g_pca.pkl', 'wb') as file:
    #     pickle.dump(monitor.data, file)
    
    env = SegWayParameterLearningEnv()
    env.reset()
    ssa = ConvexRSSA(env)
    
    q_d = np.array([0, 0])
    dq_d = np.array([1, 0])
    for i in range(960):
        u = env.robot.PD_control(q_d, dq_d)
        u = ssa.safe_control(u)
        print(env.Xr)
        env.step(u)