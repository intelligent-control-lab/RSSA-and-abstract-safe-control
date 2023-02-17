from typing import Dict
import torch as th
import numpy as np
from numpy import pi
from torch.distributions import uniform
import pickle

try:
    from fly_inv_pend_latent_env import FlyingInvertedPendulumLatentEnv
except:
    from fly_inv_pend_env.fly_inv_pend_latent_env import FlyingInvertedPendulumLatentEnv
    
class FlyingInvertedPendulumLatentSILearningEnv(FlyingInvertedPendulumLatentEnv):
    def __init__(
        self, 
        device, 
        env_num=100, 
        dt=1 / 240, 
        g=9.81, 
        m=0.8, J_x=0.005, J_y=0.005, J_z=0.009, l=1.5, k_1=4, k_2=0.05, m_p=0.04, L_p=3.0, 
        box_ang_vel_limit=20, 
        delta_max=0.5,
        z_limit={'low': [0.1, -5.0],'high': [pi / 3, 5.0]},
        
        states_sampled_per_param=10000,
        iteration_limit=10000,
        init_params: Dict = {'a_1': 1.0, 'a_2': 0.2, 'a_3': 0.01},
        gamma=0.1,
        reward_clip_ratio=0.1,
    ):
        super().__init__(
            device=device,
            env_num=env_num, 
            dt=dt, 
            g=g, 
            m=m, J_x=J_x, J_y=J_y, J_z=J_z, l=l, k_1=k_1, k_2=k_2, m_p=m_p, L_p=L_p, 
            box_ang_vel_limit=box_ang_vel_limit, 
            delta_max=delta_max,
            z_limit=z_limit
        )
        self.states_sampled_per_param = states_sampled_per_param
        self.iteration_limit = iteration_limit
        self.init_params = init_params
        self.gamma = gamma
        self.param_names = set(self.init_params.keys())
        self.reward_clip_ratio = reward_clip_ratio
        self.z_lim['high'][0] = self.delta_max
        self.z_dim = 2
        self.v_dim = 1
        self.dot_M_max = 1.0
        # self.z_lim['low'][1] = 0.0
        
    def get_phi_0_z(self, z: th.Tensor, M: th.Tensor):
        phi_0 = z[:, 0] - self.delta_max
        return phi_0[:, None]
    
    def get_p_phi_0_p_z(self, z: th.Tensor, M: th.Tensor):
        p_phi_0_p_z = th.zeros_like(z)
        p_phi_0_p_z[:, 0] = 1.0
        return p_phi_0_p_z[:, None, :]
        
    def get_p_phi_0_p_M(self, z: th.Tensor, M: th.Tensor):
        return th.zeros_like(M)[:, :, None]
        
    def get_phi_z(self, z: th.Tensor, M: th.Tensor, safety_index_params: Dict):
        # Phi = -delta_max + delta + a * e^dot_delta / M
        a = safety_index_params['a']
        Phi = -self.delta_max + z[:, 0] + a * th.exp(z[:, 1]) / M
        return Phi
    
    def get_p_phi_p_z(self, z: th.Tensor, M: th.Tensor, safety_index_params: Dict):
        a = safety_index_params['a']
        p_phi_p_z = th.zeros((z.shape[0], 2), device=self.device)
        p_phi_p_z[:, 0] = 1.0
        p_phi_p_z[:, 1] = a * th.exp(z[:, 1]) / M
        return p_phi_p_z[:, None, :]
    
    def get_p_phi_p_M(self, z: th.Tensor, M: th.Tensor, safety_index_params: Dict):
        a = safety_index_params['a']
        p_phi_p_M = -a * th.exp(z[:, 1]) / M**2
        return p_phi_p_M
    
    def revise_z(self, z: th.Tensor, M: th.Tensor, safety_index_params: Dict):
        a = safety_index_params['a']
        z[:, 1] = th.log((self.delta_max - z[:, 0]) * M / a)
        return z
        
    def evaluate_single_param(self, param_dict: Dict):
        assert set(param_dict.keys()) == self.param_names
        
        # get sampled M_range
        # x_boundary, if_convergence = self.sample_x_at_phi_boundary(safety_index_params=param_dict, sample_num=100)
        # self.reset(x_boundary)
        # M_boundary = self.M_x
        # x_boundary = x_boundary[M_boundary < 1e10]
        # M_boundary = M_boundary[M_boundary < 1e10]
        # self.reset(x_boundary)
        # M_boundary, _ = th.sort(M_boundary)
        # M_min = M_boundary[0]
        # M_max = M_boundary[-1]
        # self.M_lim = {'low': M_min, 'high': M_max}
        self.M_lim = {'low': th.tensor(0.0, device=self.device), 'high': th.tensor(2.0, device=self.device)}
        
        # sample z, M to get phi, p_phi_p_z, and p_phi_p_M
        z, M, dot_M_max = self.sample_z_and_M_at_phi_boundary(safety_index_params=param_dict, sample_num=self.states_sampled_per_param)
        phi = self.get_phi_z(z=z, M=M, safety_index_params=param_dict)
        z = z[phi >= 0]
        M = M[phi >= 0]
        phi = phi[phi >= 0]
        if (z.shape[0] / self.states_sampled_per_param) <= self.reward_clip_ratio:
            return 0.0
        p_phi_p_z = self.get_p_phi_p_z(z=z, M=M, safety_index_params=param_dict)
        p_phi_p_M = self.get_p_phi_p_M(z=z, M=M, safety_index_params=param_dict)
        
        # get rewards
        f_z, g_z = self.get_f_z_and_g_z(z=z)
        LfP = (p_phi_p_z @ f_z).squeeze_()
        LgP = (p_phi_p_z @ g_z).squeeze_()
        c = -self.gamma * phi - LfP - th.abs(p_phi_p_M) * dot_M_max
        # c = -self.gamma * phi - LfP
        LgP_mul_v_1 = LgP * -M
        LgP_mul_v_2 = LgP * M
        LgP_mul_v_min = th.min(LgP_mul_v_1, LgP_mul_v_2)
        if_feasible = (LgP_mul_v_min <= c)
        reward = th.count_nonzero(if_feasible) / z.shape[0]
        return reward.item()     
    
    def sample_z_and_M_at_phi_boundary(self, safety_index_params: Dict, sample_num):
        z = uniform.Uniform(low=self.z_lim['low'], high=self.z_lim['high']).sample(th.Size((sample_num, )))
        M = uniform.Uniform(low=th.zeros_like(self.M_lim['low']), high=self.M_lim['high']).sample(th.Size((sample_num, )))
        z = self.revise_z(safety_index_params=safety_index_params, z=z, M=M)
        
        # clip z_obj
        if_clip = (z[:, 1] <= self.z_lim['high'][1]) & (z[:, 1] >= self.z_lim['low'][1])
        z = z[if_clip]
        M = M[if_clip]
        
        # get dot_M_max near the phi=0 boundary
        x_sample, if_convergence = self.get_x_given_z(
            init_x=th.ones((z.shape[0], self.x_dim), device=self.device) * 0.1,
            z_obj=z,
            lr=5e-1, max_steps=100, eps=1e-2,
        )
        z = z[if_convergence]
        M = M[if_convergence]      
        self.reset(x_sample)
        M_sample = self.M_x
        if_M_x_ge_M = (M_sample >= M) 
        z = z[if_M_x_ge_M]
        M = M[if_M_x_ge_M]
        M_sample = M_sample[if_M_x_ge_M]
        x_sample = x_sample[if_M_x_ge_M]
        self.reset(x_sample)
        u_indices = th.randint(low=0, high=self.u_lim_vertices.shape[0], size=(self.env_num, ))
        u = self.u_lim_vertices[u_indices]
        self.step(u)
        dot_M = (self.M_x - M_sample) / self.dt
        dot_M_max = th.max(dot_M).item()
        
        # data = {
        #     'z': z,
        #     'M': M,
        #     'M_sample': M_sample,
        #     'u': u,
        #     'x': x_sample,
        #     'dot_M': dot_M,
        # }
        # with open('./src/pybullet-dynamics/fly_inv_pend_env/data/sample_z.pkl', 'wb') as file:
        #     pickle.dump(data, file)
            
        self.dot_M_max = 1.0
        
        return z, M, self.dot_M_max
        
    def latent_space_safe_control(
        self, 
        safety_index_params: Dict,
        u_ref: th.Tensor,
        dot_M_max = 1.0,
    ):
        v = self.get_v(u=u_ref)
        phi = self.get_phi_z(z=self.z, M=self.M_x, safety_index_params=safety_index_params)
        p_phi_p_z = self.get_p_phi_p_z(z=self.z, M=self.M_x, safety_index_params=safety_index_params)
        p_phi_p_M = self.get_p_phi_p_M(z=self.z, M=self.M_x, safety_index_params=safety_index_params)
        if th.all(phi > 0):
            dot_phi_z = p_phi_p_z @ (self.f_z + self.g_z @ v[:, None, None])
            dot_phi_M = th.abs(p_phi_p_M) * dot_M_max
            dot_phi = dot_phi_z.squeeze_() + dot_phi_M
            if th.all(dot_phi > -self.gamma * phi):
                u_vertices_num = self.u_lim_vertices.shape[0]
                u = th.rand(size=(self.env_num, u_vertices_num), device=self.device)
                tmp = th.sum(u, dim=-1, keepdim=True)
                u = u / tmp
                u = u @ self.u_lim_vertices
                v = self.get_v(u=u)
                dot_phi_z = p_phi_p_z @ (self.f_z + self.g_z @ v[:, None, None])
                dot_phi = dot_phi_z.squeeze_() + dot_phi_M
                feasible_indices = (dot_phi <= -self.gamma * phi)
                feasible_u = u[feasible_indices]
                u_ref = u_ref[feasible_indices]
                u_dis = th.norm(feasible_u - u_ref, dim=-1)
                min_idx = th.argmin(u_dis)
                u_safe = feasible_u[min_idx].repeat((self.env_num, 1))
        else:
            u_safe = u_ref
        return u_safe
    

if __name__ == '__main__':
    env = FlyingInvertedPendulumLatentSILearningEnv(
        device='cuda', 
        dt=1e-3,
        init_params={'a': 6e-1},
        L_p=3.0,
    )
    # reward = env.evaluate_single_param(env.init_params)
    # print(reward)
    
    z = th.tensor([0.2, 0.0], device=env.device).repeat((env.env_num, 1))
    init_x = th.zeros((env.env_num, env.x_dim), device=env.device) + 1e-2
    init_x, if_convergence = env.get_x_given_z(init_x=init_x, z_obj=z, lr=5e-1, max_steps=100, eps=1e-2)
    env.reset(x_init=init_x)
    u_ref = th.zeros(size=(env.env_num, env.u_dim), device=env.device)
    u_safe = env.latent_space_safe_control(safety_index_params=env.init_params, u_ref=u_ref)
        
        