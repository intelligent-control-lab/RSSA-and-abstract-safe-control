from random import sample
from typing import Dict
import warnings
import torch as th
from torch import sin, cos, tan, arccos
import numpy as np
from numpy import pi
from torch import nn
from torch.distributions import uniform
import torch.nn.functional as F

try:
    from fly_inv_pend_env import FlyingInvertedPendulumEnv
    from fly_inv_pend_utils import *
except:
    from fly_inv_pend_env.fly_inv_pend_env import FlyingInvertedPendulumEnv
    from fly_inv_pend_env.fly_inv_pend_utils import *
    
class FlyingInvertedPendulumLatentEnv(FlyingInvertedPendulumEnv):
    def __init__(
        self, 
        device, 
        env_num=100, 
        dt=1 / 240, 
        g=9.81, 
        m=0.8, J_x=0.005, J_y=0.005, J_z=0.009, l=1.5, k_1=4, k_2=0.05, m_p=0.04, L_p=0.03, 
        box_ang_vel_limit=20, 
        delta_max=0.5,
        z_limit={'low': [0.1, -5.0],'high': [pi / 3, 5.0]},
    ):  
        self.z_dim = 2
        self.z_lim = {
            'low': th.tensor(z_limit['low'], device=device),
            'high': th.tensor(z_limit['high'], device=device),
        }
        super().__init__(
            device=device, 
            env_num=env_num, 
            dt=dt, 
            g=g, 
            m=m, J_x=J_x, J_y=J_y, J_z=J_z, l=l, k_1=k_1, k_2=k_2, m_p=m_p, L_p=L_p, 
            box_ang_vel_limit=box_ang_vel_limit, 
            delta_max=delta_max,
        )
        
    ### Interface for latent safety index learning
    def get_phi_z(
        self, 
        z: th.Tensor, M: th.Tensor,
        safety_index_params: Dict,
    ):
        '''
        Phi = -delta_max**a_1 + delta**a_1 + a_2 * \dot{delta} / M + a_3
        - delta = arccos(cos(theta) * cos(phi))
        '''
        a_1 = safety_index_params['a_1']
        a_2 = safety_index_params['a_2']
        a_3 = safety_index_params['a_3']
        Phi = -self.delta_max**a_1 + z[:, 0]**a_1 + a_2 * z[:, 1] / M + a_3
        return Phi
    
    ### Interface for latent safety index learning
    def get_p_phi_p_z(
        self, 
        z: th.Tensor, M: th.Tensor,
        safety_index_params: Dict,
    ):  
        a_1 = safety_index_params['a_1']
        a_2 = safety_index_params['a_2']
        p_phi_p_z = th.zeros((z.shape[0], 2), device=self.device)
        p_phi_p_z[:, 0] = a_1 * z[:, 0]**(a_1 - 1) 
        p_phi_p_z[:, 1] = a_2 / M
        return p_phi_p_z[:, None, :]
    
    ### Interface for latent safety index learning
    def get_p_phi_p_M(
        self, 
        z: th.Tensor, M: th.Tensor,
        safety_index_params: Dict,
    ):
        a_2 = safety_index_params['a_2']
        p_phi_p_M = -a_2 / M**2 * z[:, 0]
        return p_phi_p_M
        
    def reset(self, x_init: th.Tensor = None, if_update_V_and_M=True):
        super().reset(x_init)
        self.update_latent_info(if_update_V_and_M=if_update_V_and_M)
        
    def step(self, u: th.Tensor, if_update_V_and_M=True):
        super().step(u)
        self.update_latent_info(if_update_V_and_M=if_update_V_and_M)
        
    def update_latent_info(self, if_update_V_and_M=True):
        # IMPORTANT: call it after self.update_dynamics()
        self.update_z()
        self.update_p_z_p_x()
        self.update_z_dynamics()
        self.update_a_and_C()
        if if_update_V_and_M:
            self.update_V()
            self.update_M()
        
    def update_z(self):
        self.z = self.get_z(self.x)
        
    def update_p_z_p_x(self):
        self.p_z_p_x = self.get_p_z_p_x(self.x)
        
    def update_z_dynamics(self):
        # IMPORTANT: call it after self.update_z()
        self.f_z, self.g_z = self.get_f_z_and_g_z(self.z)
        
    def update_a_and_C(self):
        '''
        v = \ddot{delta} = a(x) + C(x) @ u
        - abbreviate [phi, theta] as p. 
        - define \ddot{delta} = b(p, dp) + A(p, dp) @ ddp and ddp = f_p + g_p @ u
        - a(x) = b(p, dp) + A(p, dp) @ f_p, C(x) = A(p, dp) @ g_p
        '''
        # IMPORTANT: call it after self.update_z() and self.update_p_z_p_x() and self.update_dynamics()
        dot_phi = self.x[:, self.XD['dphi']]
        dot_theta = self.x[:, self.XD['dtheta']]
        p_dot_delta_p_phi = self.p_z_p_x[:, 1, self.XD['phi']]
        p_dot_delta_p_theta = self.p_z_p_x[:, 1, self.XD['theta']]
        p_dot_delta_p_dot_phi = self.p_z_p_x[:, 1, self.XD['dphi']]
        p_dot_delta_p_dot_theta = self.p_z_p_x[:, 1, self.XD['dtheta']]
        
        # b and A
        b = p_dot_delta_p_phi * dot_phi + p_dot_delta_p_theta * dot_theta
        A = th.vstack([p_dot_delta_p_dot_phi, p_dot_delta_p_dot_theta]).T
        
        # f_p and g_p
        f_p = self.f_x[:, [self.XD['dphi'], self.XD['dtheta']], :]
        g_p = self.g_x[:, [self.XD['dphi'], self.XD['dtheta']], :]
        
        # a and C
        self.a = b + (A[:, None, :] @ f_p).squeeze_()
        self.C = A[:, None, :] @ g_p
        
    def update_V(self):
        # IMPORTANT: call it after self.update_a_and_C()
        V_vertices = self.a[:, None] + (self.C[:, None, :, :] @ self.u_lim_vertices[:, :, None]).squeeze_()
        V_vertices, _ = th.sort(V_vertices)
        self.V = V_vertices[:, [0, -1]]
        
    def update_M(self):
        # IMPORTANT: call it after self.update_V()
        self.M_x = th.min(-self.V[:, 0], self.V[:, 1])
        
    def get_v(self, u: th.Tensor):
        if len(u.shape) > 2:
            return self.a + (self.C @ u).squeeze_()
        else:
            return self.a + (self.C @ u[:, :, None]).squeeze_()
        
    def get_f_z_and_g_z(self, z: th.Tensor):
        batch_size = z.shape[0]
        f_z = th.zeros((batch_size, 2, 1), device=self.device)
        f_z[:, 0, 0] = z[:, 1]
        g_z = th.zeros((batch_size, 2, 1), device=self.device)
        g_z[:, 1, 0] = 1.0
        return f_z, g_z
        
    def get_z(self, x_fake: th.Tensor):
        '''
        get z from x_fake. z = [delta, \dot{delta}], where delta = arccos(cos(phi) * cos(theta))
        - \dot{delta} = (sin(phi) * cos(theta) * \dot{phi} + cos(phi) * sin(theta) * \dot{theta}) / sin(delta)
        '''
        phi = x_fake[:, self.XD['phi']]
        theta = x_fake[:, self.XD['theta']]
        dot_phi = x_fake[:, self.XD['dphi']]
        dot_theta = x_fake[:, self.XD['dtheta']]
        delta = arccos(cos(phi) * cos(theta))
        dot_delta = (sin(phi) * cos(theta) * dot_phi + cos(phi) * sin(theta) * dot_theta) / sin(delta)
        z = th.vstack([delta, dot_delta]).T
        return z
         
    def get_p_z_p_x(self, x_fake: th.Tensor):
        phi = x_fake[:, self.XD['phi']]
        theta = x_fake[:, self.XD['theta']]
        dot_phi = x_fake[:, self.XD['dphi']]
        dot_theta = x_fake[:, self.XD['dtheta']]
        delta = arccos(cos(phi) * cos(theta))
        
        p_dot_delta_p_dot_phi = p_delta_p_phi = sin(phi) / sin(delta) * cos(theta)
        p_dot_delta_p_dot_theta = p_delta_p_theta = sin(theta) / sin(delta) * cos(phi)
        p_dot_delta_p_phi = (-dot_theta * sin(phi) + dot_phi * sin(theta) * cos(delta)) * sin(theta) / sin(delta)**3
        p_dot_delta_p_theta = (-dot_phi * sin(theta) + dot_theta * sin(phi) * cos(delta)) * sin(phi) / sin(delta)**3
        
        p_z_p_x = th.zeros((x_fake.shape[0], self.z_dim, self.x_dim), device=self.device)
        p_z_p_x[:, 0, self.XD['phi']] = p_delta_p_phi
        p_z_p_x[:, 0, self.XD['theta']] = p_delta_p_theta
        p_z_p_x[:, 1, self.XD['phi']] = p_dot_delta_p_phi
        p_z_p_x[:, 1, self.XD['theta']] = p_dot_delta_p_theta
        p_z_p_x[:, 1, self.XD['dphi']] = p_dot_delta_p_dot_phi
        p_z_p_x[:, 1, self.XD['dtheta']] = p_dot_delta_p_dot_theta
        return p_z_p_x
    
    def get_x_given_z(
        self, 
        init_x: th.Tensor, z_obj: th.Tensor, 
        lr, max_steps, eps,
    ):
        assert init_x.shape[0] == z_obj.shape[0] and init_x.device == z_obj.device
        count = 0
        x = init_x
        while count < max_steps:
            count += 1
            p_z_p_x = self.get_p_z_p_x(x)
            z_calc = self.get_z(x)
            grad = 2 * (z_calc - z_obj)[:, None, :] @ p_z_p_x
            x = x - lr * grad.squeeze_()
        losses = th.linalg.norm(z_calc - z_obj, dim=1)
        if_convergence = ~(losses > eps)
        non_convergence_num = th.count_nonzero(~if_convergence)
        print(f'There {non_convergence_num} z cannot find a suitable x.')
        return x, if_convergence
    
    def sample_z_at_phi_boundary(self, safety_index_params, sample_num):
        a_1 = safety_index_params['a_1']
        a_2 = safety_index_params['a_2']
        a_3 = safety_index_params['a_3']
        z_obj = np.linspace(
            start=self.z_lim['low'].cpu().numpy(),
            stop=self.z_lim['high'].cpu().numpy(),
            num=sample_num,
        )
        z_obj = th.from_numpy(z_obj).to(self.device)
        z_obj[:, 1] = 1 / a_2 * (self.delta_max**a_1 - z_obj[:, 0]**a_1 - a_3)
        z_obj = z_obj[(z_obj[:, 1] <= self.z_lim['high'][1]) & (z_obj[:, 1] >= self.z_lim['low'][1])]   
        z_obj = z_obj[z_obj[:, 0] <= self.delta_max]
        return z_obj
    
    def sample_x_at_phi_boundary(
        self, 
        safety_index_params: Dict, 
        sample_num=100,
        lr=5e-1, max_steps=100, eps=1e-2,
    ):
        z_obj = self.sample_z_at_phi_boundary(safety_index_params=safety_index_params, sample_num=sample_num)
        x_init = th.zeros((z_obj.shape[0], self.x_dim), device=self.device) + 1e-2
        x, if_convergence = self.get_x_given_z(init_x=x_init, z_obj=z_obj, lr=lr, max_steps=max_steps, eps=eps)
        return x, if_convergence
        
    
if __name__ == '__main__':
    device = 'cuda'
    env = FlyingInvertedPendulumLatentEnv(device=device, L_p=3.0, dt=1e-3, env_num=10) 
    # x, if_convergence = env.sample_x_at_phi_boundary(
    #     safety_index_params={'a_1': 1.0, 'a_2': 0.2, 'a_3': 0.0},
    #     sample_num=1000,
    # )  
    # env.reset(x_init=x)
    
    sample_num = 10
    u_ref = th.zeros((sample_num, env.u_dim), device=device)
    v_ref = th.zeros((sample_num, 1), device=device)
    x_init = th.ones((sample_num, 10), device=device) * 0.1
    for i in range(100):
        z_obj = uniform.Uniform(low=env.z_lim['low'], high=env.z_lim['high']).sample(th.Size((sample_num, )))
        # z_obj = th.zeros((100, 2), device=device) + 0.1
        x, if_convergence = env.get_x_given_z(x_init, z_obj, lr=0.1, max_steps=100, eps=1e-2)
        x = x[if_convergence]
        u = u_ref[if_convergence]
        v = v_ref[if_convergence]
        env.reset(x_init=x, if_update_V_and_M=True)
        z = env.z
        if_safe = (z[:, 0] <= env.delta_max)
        # data = {
        #     'x': x,
        #     'f_x': env.f_x,
        #     'g_x': env.g_x,
        #     'u_ref': u,
        #     'z': z,
        #     'M': F.relu(env.M_x[:, None]) + 1e-5,
        #     'f_z': env.f_z,
        #     'g_z': env.g_z,
        #     'v_ref': v,
        #     'safe_mask': if_safe,
        #     'unsafe_mask': ~if_safe,
        # }
        # th.save(data, f'./src/pybullet-dynamics/fly_inv_pend_env/data/latent_neural_cbf_data/{i}.pth')
    
    # x_init = uniform.Uniform(low=env.x_lim['low'], high=env.x_lim['high']).sample(th.Size((100, )))
    # env.reset(x_init=x_init)
    # print(env.M_x)
    # print(env.f_x[0])
    # print(env.g_x[0])
        
          