from decimal import getcontext
from typing import Dict
import torch as th
from torch import sin, cos, tan
import numpy as np
from numpy import pi
from torch import nn
import gym

try:
    from fly_inv_pend_utils import *
except:
    from fly_inv_pend_env.fly_inv_pend_utils import *

class FlyingInvertedPendulumEnv(gym.Env):
    '''
    Dynamics of flying inverted pendulum model. 
    Implemented in torch to support parallization and auto differetiation.
    '''
    def __init__(
        self,
        device,
        env_num=100,
        dt=1e-3,
        g=9.81,
        m=0.8, 
        J_x=5e-3,
        J_y=5e-3,
        J_z=9e-3,
        l=1.5,
        k_1=4.0, k_2=0.05,
        m_p=0.04,
        L_p=0.03,
        box_ang_vel_limit=20.0,
        
        delta_max=0.5,
    ):
        super().__init__()
        self.device = device
        self.dt = dt
        self.g = g
        self.m = m
        self.J_x = J_x
        self.J_y = J_y
        self.J_z = J_z
        self.l = l
        self.k_1 = k_1
        self.k_2 = k_2
        self.m_p = m_p
        self.L_p = L_p
        self.M = self.m + self.m_p
        self.J_inv = th.tensor(np.diag([(1.0 / self.J_x), (1.0 / self.J_y), (1.0 / self.J_z)]), device=self.device).float()
        self.delta_max = delta_max
        
        # set state names. Excluded x, y, z
        self.x_dim = 10
        self.u_dim = 4
        self.env_num = env_num
        self.x = th.zeros((self.env_num, self.x_dim))
        self.state_names = [
            'gamma', 'beta', 'alpha', 'phi', 'theta',   # q
            'dgamma', 'dbeta', 'dalpha', 'dphi', 'dtheta',  # dq
        ]
        self.XD = dict(zip(self.state_names, range(self.x_dim)))    # state index dict
        
        # set state limits
        x_max = th.tensor([
            pi / 3, pi / 3, pi, pi / 3, pi / 3,
            box_ang_vel_limit, box_ang_vel_limit, box_ang_vel_limit, box_ang_vel_limit, box_ang_vel_limit,
        ], device=self.device)
        self.x_lim = {'low': -x_max, 'high': x_max}
        
        # set u limits
        self.get_u_lim()
    
    ### Interface for safe control  
    def get_phi_x(self, safety_index_params: Dict):
        '''
        Phi = -delta_max**a_1 + delta**a_1 + a_2 * \dot{delta} + a_3
        - delta = arccos(cos(theta) * cos(phi))
        '''
        pass
    
    ### Interface for safe control  
    def get_p_phi_x_p_x(self, safety_index_params: Dict):
        pass
        
    def reset(self, x_init: th.Tensor = None):
        if x_init is not None:
            self.env_num = x_init.shape[0] 
            self.x = x_init.to(self.device)
        else:
            self.x = th.zeros((self.env_num, self.x_dim), device=self.device)
        self.update_dynamics(x=self.x)
        
    def step(self, u: th.Tensor):
        assert u.shape == (self.env_num, self.u_dim)
        dot_x = self.f_x + self.g_x @ u[:, :, None]
        self.x = self.x + dot_x.squeeze_(dim=-1) * self.dt
        self.update_dynamics(x=self.x)
        
    def get_u_lim(self):
        '''
        u = control_Mat @ motor_input - [M * g, 0, 0, 0]
        limit of motor_input: [0, 0, 0, 0] to [1, 1, 1, 1]
        '''
        k_1 = 4.0
        k_2 = 0.05
        l = 0.3 / 2
        self.motor_input_limit = {
            'low': th.tensor([0, 0, 0, 0], device=self.device).float(),
            'high': th.tensor([1, 1, 1, 1], device=self.device).float(),
        }
        self.control_Mat = th.tensor([
            [k_1, k_1, k_1, k_1],
            [0, -l * k_1, 0, l * k_1],
            [l * k_1, 0, -l * k_1, 0],
            [-k_2, k_2, -k_2, k_2],
        ], device=self.device)
        motor_input_vertices = th.tensor(exhaust(4, neg_val=-1), device=self.device).float()
        const_vec = th.tensor([self.M * self.g, 0, 0, 0], device=self.device)
        self.u_lim_vertices = (self.control_Mat @ motor_input_vertices[:, :, None]).squeeze_() - const_vec
        self.u_lim_A, self.u_lim_b = get_constraints(self.u_lim_vertices.cpu().numpy())
        simple_u_high = []
        simple_u_low = []
        for i in range(self.u_dim):
            u_component = self.u_lim_vertices[:, i]
            u_component, _ = th.sort(u_component)
            simple_u_low.append(u_component[0])
            simple_u_high.append(u_component[-1])
        simple_u_high = th.stack(simple_u_high)
        simple_u_low = th.stack(simple_u_low)
        self.u_simplified_lim = {'low': simple_u_low, 'high': simple_u_high}
        
    def update_dynamics(self, x: th.Tensor):
        '''
        Pre-computations for f(x), g(x) and z(x). 
        - shape of x: (batch_size, x_dim)
        - define: [ddgamma, ddbeta, ddalpha] = drone_mat @ [tau_gamma, tau_beta, tau_alpha]
        - define: [ddphi, ddtheta] = pendulum_mat @ (F + mg) + pendulum_vec
        
        IMPORTANT: u is arranged as [F, tau_gamma, tau_beta, tau_alpha]!
        - f = [dgamma, dbeta, dalpha, dphi, dtheta, 0, pendulum_mat @ mg + pendulum_vec]
        - g = [[0, 0],
               [0, drone_mat],
               [pendulum_mat, 0]]
        '''
        if len(x.shape) > 2:
            x.squeeze_(dim=-1)
        gamma = x[:, self.XD['gamma']]
        beta = x[:, self.XD['beta']]
        alpha = x[:, self.XD['alpha']]
        phi = x[:, self.XD['phi']]
        theta = x[:, self.XD['theta']]
        dphi = x[:, self.XD['dphi']]
        dtheta = x[:, self.XD['dtheta']]
        
        batch_size = x.shape[0]

        # R
        R = th.zeros((batch_size, 3, 3), device=self.device) 
        R[:, 0, 0] = cos(alpha) * cos(beta)
        R[:, 0, 1] = cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma)
        R[:, 0, 2] = cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma)
        R[:, 1, 0] = sin(alpha) * cos(beta)
        R[:, 1, 1] = sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma)
        R[:, 1, 2] = sin(alpha) * sin(beta) * cos(gamma) - cos(alpha) * sin(gamma)
        R[:, 2, 0] = -sin(beta)
        R[:, 2, 1] = cos(beta) * sin(gamma)
        R[:, 2, 2] = cos(beta) * cos(gamma)

        # k_v
        k_x = R[:, 0, 2]
        k_y = R[:, 1, 2]
        k_z = R[:, 2, 2]
        
        # drone_mat
        drone_mat = R @ self.J_inv
        
        # pendulum_mat
        pendulum_mat = th.zeros((batch_size, 2, 1), device=self.device)
        pendulum_mat[:, 0, 0] = 3.0 * (k_y * cos(phi) + k_z * sin(phi)) / (2 * self.M * self.L_p * cos(theta))
        pendulum_mat[:, 1, 0] = 3.0 * (-k_x * cos(theta) - k_y * sin(phi) * sin(theta) + k_z * cos(phi) * sin(theta)) / (2.0 * self.M * self.L_p)
        
        # pendulum_vec
        pendulum_vec = th.zeros((batch_size, 2, 1), device=self.device)
        pendulum_vec[:, 0, 0] = 2 * dtheta * dphi * tan(theta)
        pendulum_vec[:, 1, 0] = -th.square(dphi) * sin(theta) * cos(theta)
        
        # f_x
        self.f_x = th.zeros((batch_size, self.x_dim, 1), device=self.device)
        self.f_x[:, :self.x_dim // 2] = x[:, self.x_dim // 2:, None]
        self.f_x[:, [self.XD['dphi'], self.XD['dtheta']]] = pendulum_mat * self.M * self.g + pendulum_vec
        
        # g
        self.g_x = th.zeros((batch_size, self.x_dim, self.u_dim), device=self.device)
        self.g_x[:, [self.XD['dgamma'], self.XD['dbeta'], self.XD['dalpha']], 1:] = drone_mat
        self.g_x[:, [self.XD['dphi'], self.XD['dtheta']], :1] = pendulum_mat
        
    
        
        
if __name__ == '__main__':
    device = 'cuda'
    env = FlyingInvertedPendulumEnv(device=device, dt=1e-3, env_num=1)
    x = th.ones((1, 10, 1), device=device) * 0.001
    x = th.zeros_like(x)
    env.reset(x_init=x)
    
    # robot.update_dynamics(x)
    
    for i in range(960):
        u = th.zeros((env.env_num, env.u_dim)).to(device) + 0.01
        env.step(u)
        if i % 20 == 0:
            print(env.x[0])
          
        
        
        
        
        
        
        