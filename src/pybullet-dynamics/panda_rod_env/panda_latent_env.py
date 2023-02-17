from typing import Dict
import numpy as np
from scipy.spatial import ConvexHull
import warnings
from loguru import logger
from scipy.optimize import linprog

try:
    from panda_rod_env import PandaRodEnv
    from panda_rod_utils import *
except:
    from panda_rod_env.panda_rod_env import PandaRodEnv
    from panda_rod_env.panda_rod_utils import *

class PandaLatentEnv(PandaRodEnv):
    '''
    Franka Panda environment for Latent Space Safety Index project.
    '''
    def __init__(
        self, 
        render_flag=False, 
        robot_file_path='./src/pybullet-dynamics/panda_rod_env/urdf/panda_without_rod.urdf',
        dof=7,
        end_eff_id=7,
        goal_pose=[0.7, 0.3, 0.4], 
        obstacle_pose=[0.6, 0.1, 0.5], 
        tolerate_error=0.02,
        
        d_min=0.05,
        z_limit={'low': [0.0, -5.0], 'high': [0.8, 5.0]},
        if_task=True,
    ):
        super().__init__(
            render_flag=render_flag, 
            robot_file_path=robot_file_path,
            goal_pose=goal_pose, 
            obstacle_pose=obstacle_pose, 
            tolerate_error=tolerate_error,
            dof=dof,
            end_eff_id=end_eff_id,
            if_task=if_task,
        )
        if not self.if_task:
            return 
        
        self.d_min = d_min
        self.z_limit = z_limit
        
        # get q, dq, u's limits from robot's urdf: just trying to be consistent with other environrments
        self.q_limit = {'low': self.robot.q_min, 'high': self.robot.q_max}
        self.dq_limit = {'low': -self.robot.velocity_limit, 'high': self.robot.velocity_limit}
        self.u_limit = {'low': -self.robot.torque_limit, 'high': self.robot.torque_limit}
        
        # get vertices of U_lim
        self.get_U_lim_vertices()
        
        # get dimension of z and v
        self.update_latent_info(if_update_V_and_M=False)
        self.v_dim = len(self.get_v(self.u_limit['low']))
        self.z_dim = len(np.squeeze(self.z))
        
        # get Q_u for a^T @ (pinv_C^T @ Q_u @ pinv_C) @ a <= 1
        self.Q_u = np.diag(1 / self.max_u)
        
        # used when getting dot_M_max_per_state
        self.u_bound = [pair for pair in zip(self.u_limit['low'], self.u_limit['high'])]
        
    ### Interface for latent safety index learning
    def get_phi_z(self, safety_index_params: Dict, z, M):
        '''
        phi = d_min**alpha - d**alpha - k_v * dot{d} / M + beta.
        '''
        alpha = safety_index_params['alpha']
        beta = safety_index_params['beta']
        k_v = safety_index_params['k_v']
        phi_z = self.d_min**alpha - z[0]**alpha - k_v * z[1] / M + beta
        return phi_z

    ### Interface for latent safety index learning
    def get_p_phi_p_z(self, safety_index_params: Dict, z, M):
        '''
        p_phi_p_z = [-alpha * d**(alpha - 1), -k_v / M].
        '''
        alpha = safety_index_params['alpha']
        k_v = safety_index_params['k_v']
        p_phi_p_z = np.array([[-alpha * z[0]**(alpha - 1), -k_v / M]])
        return p_phi_p_z
    
    ### Interface for latent safety index learning
    def get_p_phi_p_M(self, safety_index_params: Dict, z, M):
        alpha = safety_index_params['alpha']
        k_v = safety_index_params['k_v']
        p_phi_p_M = k_v * z[1] / M**2
        return p_phi_p_M
    
    def get_f_z_and_g_z(self, z):
        f_z = np.zeros((2, 1))
        f_z[0, 0] = z[1]
        g_z = np.zeros((2, 1))
        g_z[1, 0] = 1.0
        return f_z, g_z
        
    ### Interface for latent safety index learning
    def get_M_range(
        self,
        safety_index_params: Dict, 
        sample_num=100,
        lr=5e-1, max_steps=100, eps=1e-2,
    ):
        Xr_list = self.sample_Xr_at_phi_boundary(
            safety_index_params=safety_index_params, sample_num=sample_num, 
            lr=lr, max_steps=max_steps, eps=eps,
        )
        M_list = []
        M_next_list = []
        for Xr in Xr_list:
            q = Xr[:self.robot.dof]
            dq = Xr[self.robot.dof:]
            self.robot.set_joint_states(q, dq)
            self.update_latent_info(if_update_grad=False)
            M_list.append(self.M)
            
            # take a step to get dot_M
            u = self.u_vertices[np.random.randint(self.u_vertices.shape[0])]
            self.step(u)
            self.update_latent_info(if_update_grad=False)
            M_next_list.append(self.M)
        
        # get dot_M_max
        M_list = np.asanyarray(M_list)
        M_next_list = np.asanyarray(M_next_list)
        dot_M = (M_next_list - M_list) * 240
        self.dot_M_max = np.max(dot_M)
        
        logger.debug(f'M_list: {M_list}')
        M_list.sort()
        M_range = {'low': M_list[0], 'high': M_list[-1]}
        return M_range

    ### Interface for latent safety index learning
    def update_latent_info(self, if_update_V_and_M=True, if_update_grad=False):
        self.update_z()
        self.update_z_dynamics()
        self.update_a_and_C()
        if if_update_V_and_M:
            self.update_V()
            self.update_M()
        if if_update_grad:
            self.update_p_z_p_Xr()
      
    def update_z(self):
        '''
        z = [d, \dot{d}], where d = ||dM_p||, \dot{d} = dM_p.T @ dM_v / d.
        '''
        self.dM_p = self.robot.get_end_eff_pose() - self.obstacle_pose
        self.dM_v = self.robot.get_end_eff_velocity()
        self.d = np.linalg.norm(self.dM_p)
        self.dot_d = np.dot(self.dM_p, self.dM_v) / self.d
        self.z = np.array([self.d, self.dot_d]).reshape(-1, 1)
        
    def update_p_z_p_Xr(self):
        '''
        Given a z, if we want to find a Xr, then we should use gradient descending method.
        - p_z_p_Xr = p_z_p_Mr @ p_Mr_p_Xr.
        '''
        p_Mr_p_Xr = self.p_Mr_p_Xr
        p_z_p_Mr = np.zeros((2, 6))
        p_z_p_Mr[0, :3] = p_z_p_Mr[1, 3:] = self.dM_p / self.d
        p_z_p_Mr[1, :3] = self.dM_v / self.d - np.dot(self.dM_p, self.dM_v) * self.dM_p / self.d**3
        self.p_z_p_Xr = p_z_p_Mr @ p_Mr_p_Xr    
        
    def update_z_dynamics(self):
        '''
        get f_z and g_z.
        - IMPORTANT: this method should be called after self.update_z().
        '''
        self.f_z, self.g_z = self.get_f_z_and_g_z(self.z)
        
    def update_a_and_C(self):
        '''
        v = a(x) + C(x) @ u. Where:
        - a(x) = ||dM_v||^2 / d - (dM_p.T @ dM_v)^2 / d^3 + 1 / d * dM_p.T @ (p_V_p_q @ dq - p_V_p_dq @ M_inv @ (g + C))
        - C(x) = 1 / d * dM_p.T @ p_V_p_dq @ M_inv
        - IMPORTANT: this method should be called after self.update_z().
        '''
        q, dq = self.robot.get_joint_states()
        M, g, Cor = self.robot.calculate_dynamic_matrices(q, dq)
        M_inv = np.linalg.inv(M)
        p_V_p_q, p_V_p_dq = self.robot.kinematics_chain.get_VFK_grad(q, dq)
        self.a = np.linalg.norm(self.dM_v)**2 / self.d - np.dot(self.dM_p, self.dM_v)**2 / self.d**3 + \
            1 / self.d * np.dot(self.dM_p, p_V_p_q @ dq - p_V_p_dq @ M_inv @ (g + Cor))
        self.C = 1 / self.d * self.dM_p.reshape(1, -1) @ p_V_p_dq @ M_inv
        self.C = np.squeeze(self.C)
        
    def update_V(self):
        '''
        V = ConvexHull(a + C @ u, u \in U_lim). 
        - In this case, v is of one dimension. Thus, we just need to reserve its two end points.
        - IMPORTANT: this method should be called after self.update_a_and_C().
        '''
        V_vertices = []
        for u in self.u_vertices:
            v = self.get_v(u)
            V_vertices.append(v)
        V_vertices = np.asanyarray(V_vertices)
        V_vertices = np.squeeze(V_vertices)
        V_vertices.sort()
        self.V = np.array([V_vertices[0], V_vertices[-1]])
        
    def get_v(self, u):
        '''
        v = \ddot{d}, where \ddot{d} = ||dM_v||^2 / d - (dM_p.T @ dM_v)^2 / d^3 + (dM_p.T @ dM_a) / d = a(x) + C(x) @ u.
        - Note that dM_a cannot be obtained from pybullet. We should calculate it by ourselves. 
        - IMPORTANT: this method should be called after self.update_a_and_C().
        '''
        v = self.a + np.dot(self.C, u)
        return np.array([v])
    
    def update_M(self):
        '''
        M = min(-V[0], V[1]).
        - IMPORTANT: this method should be called after self.update_V().
        '''
        self.M = min(-self.V[0], self.V[1])
    
    def get_U_lim_vertices(self):
        u_max = self.u_limit['high']
        u_min = self.u_limit['low']
        self.u_vertices = []
        for indices in exhaust(len(u_max)):
            u = np.zeros_like(u_max)
            for i in range(len(u_max)):
                u[i] = u_max[i] if indices[i] else u_min[i]
            self.u_vertices.append(u)
        self.u_vertices = np.asanyarray(self.u_vertices)
        hull = ConvexHull(self.u_vertices)
        vertice_order = hull.vertices
        self.u_vertices = self.u_vertices[vertice_order]
    
    def step(self, u):
        u = np.clip(u, a_min=self.u_limit['low'], a_max=self.u_limit['high'])
        return super().step(u)
    
    def sample_z_at_phi_boundary(self, safety_index_params: Dict, sample_num):
        '''
        sample z's satisfying phi(z) = 0.
        '''
        alpha = safety_index_params['alpha']
        beta = safety_index_params['beta']
        k_v = safety_index_params['k_v']
        d_list = np.linspace(self.z_limit['low'][0], self.z_limit['high'][0], sample_num)
        z_list = []
        for d in d_list:
            c = self.d_min**alpha - d**alpha + beta
            dot_d = c / k_v
            if dot_d >= self.z_limit['low'][1] and dot_d <= self.z_limit['high'][1]:
                z_list.append([d, dot_d])
        z_list = np.array(z_list)
        logger.debug(f'z_list: {z_list}')
        return z_list
    
    def sample_Xr_at_phi_boundary(
        self, 
        safety_index_params: Dict, 
        sample_num=100,
        lr=5e-1, max_steps=100, eps=1e-2,
    ):      
        z_list = self.sample_z_at_phi_boundary(safety_index_params=safety_index_params, sample_num=sample_num)
        Xr_list = []
        for z in z_list:
            # init_q = np.random.uniform(low=self.q_limit['low'], high=self.q_limit['high'])
            # init_dq = np.random.uniform(low=self.dq_limit['low'], high=self.dq_limit['high'])
            # init_Xr = np.concatenate((init_q, init_dq))
            init_Xr = np.concatenate((self.robot.q_init, np.zeros(self.robot.dof)))
            Xr, if_convergence = self.get_Xr_given_z(init_Xr=init_Xr, z_obj=z, lr=lr, max_steps=max_steps, eps=eps)
            q = Xr[:self.robot.dof]
            dq = Xr[self.robot.dof:]
            if np.all(q >= self.q_limit['low']) and np.all(q <= self.q_limit['high']) and \
                np.all(dq >= self.dq_limit['low']) and np.all(dq <= self.dq_limit['high']) and \
                if_convergence:
                Xr_list.append(Xr)
        Xr_list = np.asanyarray(Xr_list)
        return Xr_list
                
    def get_Xr_given_z(self, init_Xr, z_obj, lr, max_steps, eps):
        Xr = np.squeeze(init_Xr)
        count = 0
        if_convergence = True
        while count < max_steps:
            count += 1
            q = Xr[:self.robot.dof]
            dq = Xr[self.robot.dof:]
            self.robot.set_joint_states(q, dq)
            self.update_latent_info(if_update_V_and_M=False, if_update_grad=True)
            z_calc = np.squeeze(self.z)
            print(np.linalg.norm(z_calc - z_obj))
            if np.linalg.norm(z_calc - z_obj) <= eps:
                break
            grad = 2 * (z_calc - z_obj) @ self.p_z_p_Xr
            Xr = Xr - lr * grad
        if np.linalg.norm(z_calc - z_obj) > eps:
            warnings.warn('No Xr solution!')
            if_convergence = False
        return Xr, if_convergence
    
    def render_image(self):
        image_1 = self.render(
            height=512, width=512,
            cam_dist=2,
            camera_target_position=[0, 0.5, 0],
            cam_yaw=30, cam_pitch=-30, cam_roll=0, 
        )
        image_2 = self.render(
            height=512, width=512,
            cam_dist=2,
            camera_target_position=[0, 0, 0],
            cam_yaw=120, cam_pitch=-50, cam_roll=0,
        )
        image = np.concatenate((image_1, image_2), axis=1)
        # visually split two perspectives
        image[:, 512, :] = 0
        # cv2.imwrite('./src/pybullet-dynamics/panda_rod_env/imgs/test.jpg', image)
        return image
    
    def get_dot_M_max_per_state(
        self,
        delta_x=1e-4,
    ):
        '''
        get dot_M_max_per_state = max (p_M_p_x @ dot_x) = max (p_M_p_x @ (f_x + g_x @ u))
        - p_M_p_x is generated with numerical method: 
            - (p_M_p_x)_i = (M(x + [0, 0, ..., delta_x, ..., 0].T) - M(x)) / delta_x 
        '''
        q, dq = self.robot.get_joint_states()
        M = self.M
        p_M_p_x = []
        for i in range(self.robot.dof):
            q_new = np.array(q, copy=True) 
            q_new[i] += delta_x
            self.robot.set_joint_states(q_new, dq)
            self.update_latent_info()
            M_new = self.M
            p_M_p_x_i = (M_new - M) / delta_x
            p_M_p_x.append(p_M_p_x_i)
        for i in range(self.robot.dof):
            dq_new = np.array(dq, copy=True) 
            dq_new[i] += delta_x
            self.robot.set_joint_states(q, dq_new)
            self.update_latent_info()
            M_new = self.M
            p_M_p_x_i = (M_new - M) / delta_x
            p_M_p_x.append(p_M_p_x_i)
        p_M_p_x = np.asanyarray(p_M_p_x).reshape(1, -1)
        self.robot.set_joint_states(q, dq)
        
        f_x = self.f
        g_x = self.g
        c = np.squeeze(p_M_p_x @ g_x)
        res = linprog(c, bounds=self.u_bound)
        u_min = res['x'].reshape(-1, 1)
        res = linprog(-c, bounds=self.u_bound)
        u_max = res['x'].reshape(-1, 1)
        dot_M_min = (p_M_p_x @ (f_x + g_x @ u_min)).item()
        dot_M_max = (p_M_p_x @ (f_x + g_x @ u_max)).item()
        dot_M_max_per_state = max(abs(dot_M_min), abs(dot_M_max))
        return dot_M_max_per_state
    
    

if __name__ == '__main__':
    env = PandaLatentEnv(
        render_flag=False, 
        goal_pose=[0.7, 0.3, 0.4], 
        obstacle_pose=[0.45, 0.1, 0.55], 
    )
    u = env.compute_naive_torque()
    env.step(u)
    q, dq = env.robot.get_joint_states()
    env.robot.set_joint_states(q, dq + 1e-3)
    env.update_latent_info()
    env.get_dot_M_max_per_state()
    
    images = []
    for i in range(int(1e6)):
        print(i)
        # u = env.compute_naive_torque()
        # env.step(u)
        env.update_latent_info()
        
        # q, dq = env.robot.get_joint_states()
        # print(dq)
        # env.update_latent_info(if_update_grad=True)
        # print(env.M)
        
        # rgb = env.render_image()
        # images.append(rgb)
        
    # video_record(movie_name='./src/pybullet-dynamics/panda_rod_env/movies/latent_example/test.mp4', images=images)
        # cv2.imwrite('./src/pybullet-dynamics/panda_rod_env/imgs/latent_example/test.png', rgb)
        
    #     env.robot.set_joint_states(q + 0.1, dq + 0.1)
    #     env.update_latent_info(if_update_grad=True)
    #     print(env.M)
    
    # M_min = np.inf
    # for i in range(1000):
    #     q = np.random.uniform(low=env.q_limit['low'], high=env.q_limit['high'])
    #     dq = np.random.uniform(low=env.dq_limit['low'], high=env.dq_limit['high'])
    #     env.robot.set_joint_states(q, dq)
    #     try:
    #         env.update_latent_info()
    #         if env.M < M_min:
    #             M_min = env.M
    #             print(f'z: {np.squeeze(env.z)}, V: {env.V}, M_min: {M_min}')
    #             print(f'eff_pose: {env.robot.get_end_eff_pose()}')
    #             print(f'q: {q}')
    #             print(f'dq: {dq}')
    #     except:
    #         pass        
        
        
    
        
        
        
        
        