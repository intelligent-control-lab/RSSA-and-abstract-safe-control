from typing import Dict
import yaml
import torch as th
import torch.nn as nn
from functorch import make_functional, make_functional_with_buffers, jacrev, vmap

from humanoid_nn_models import HumanoidZModel, HumanoidVModel

def generate_z_func(root_path):
    with open(root_path + 'z_train.yaml') as file:
        cfg = yaml.load(file)
    model = HumanoidZModel(**cfg['model'])
    model.load_state_dict(th.load(root_path + 'best_model.pth'))
    model.to(device='cuda')
    model.eval()
    z_func, z_params, z_buffers = make_functional_with_buffers(model, disable_autograd_tracking=True)
    return z_func, z_params, z_buffers

def generate_v_func(root_path):
    with open(root_path + 'v_train.yaml') as file:
        cfg = yaml.load(file)
    model = HumanoidVModel(**cfg['model'])
    model.load_state_dict(th.load(root_path + 'best_model.pth'))
    model.eval()
    model.to(device='cuda')
    v_func, v_params, v_buffers = make_functional_with_buffers(model, disable_autograd_tracking=True)
    return v_func, v_params, v_buffers


class HumanoidLatentSISolver:
    def __init__(
        self,
        z_model_root_path='./src/pybullet-dynamics/humanoid_env/train_z_log/2022-08-06__14-22-33/', 
        v_model_root_path='./src/pybullet-dynamics/humanoid_env/train_v_log/2022-08-06__14-20-29/', 
        init_params={'a_1': 1.0, 'a_2': 1.0, 'a_3': 0.01},
        z_min=1.2,
        device='cuda',
        gamma=0.01,
    ):
        self.z_func, self.z_params, self.z_buffers = generate_z_func(root_path=z_model_root_path)
        self.v_func, self.v_params, self.v_buffers = generate_v_func(root_path=v_model_root_path)
        self.init_params = init_params
        self.z_min = z_min
        self.device = device
        self.gamma = gamma
        
    def get_phi_z(self, safety_index_params: Dict, z: th.Tensor):
        '''
        phi_z = z_min**a_1 - z[0]**a[1] - a_2 * z[1] + a_3
        '''
        S = safety_index_params
        return self.z_min**S['a_1'] - z[:, 0]**S['a_1'] - S['a_2'] * z[:, 1] + S['a_3']
    
    def get_p_phi_p_z(self, safety_index_params: Dict, z: th.Tensor):
        S = safety_index_params
        p_phi_p_z = th.zeros((z.shape[0], 2), device=self.device)
        p_phi_p_z[:, 0] = -S['a_1'] * z[:, 0]**(S['a_1'] - 1) 
        p_phi_p_z[:, 1] = -S['a_2']
        return p_phi_p_z[:, None, :]
    
    def calc_safe_u(
        self, 
        safety_index_params: Dict, 
        x: th.Tensor, u: th.Tensor,
        lr=1e-3, max_steps=100, eps=1e-2,
    ):
        z = self.get_z(x)
        v = self.get_v(x, u)
        phi = self.get_phi_z(safety_index_params, z)
        
        p_phi_p_z = self.get_p_phi_p_z(safety_index_params, z)
        f_z, g_z = self.get_f_z_and_g_z(z)
        LfP = (p_phi_p_z @ f_z).squeeze_()
        LgP = (p_phi_p_z @ g_z).squeeze_()
        unsafe_indices = (phi > 0) & ((LfP + LgP * v) > -self.gamma * phi)
        unsafe_x = x[unsafe_indices]
        unsafe_u = u[unsafe_indices]
        unsafe_z = z[unsafe_indices]
        unsafe_phi = phi[unsafe_indices]
        unsafe_LfP = LfP[unsafe_indices]
        unsafe_LgP = LgP[unsafe_indices]
        critical_v = (-unsafe_phi * self.gamma - unsafe_LfP) / unsafe_LgP
        if len(critical_v) > 0:
            critical_u, if_convergence = self.get_u_given_v_and_x(
                u_ref=unsafe_u, x=unsafe_x, v=critical_v,
                lr=lr, max_steps=max_steps, eps=eps,
            )
            u[unsafe_indices] = critical_u
        return u
        
    def get_f_z_and_g_z(self, z: th.Tensor):
        batch_size = z.shape[0]
        f_z = th.zeros((batch_size, 2, 1), device=self.device)
        f_z[:, 0, 0] = z[:, 1]
        g_z = th.zeros((batch_size, 2, 1), device=self.device)
        g_z[:, 1, 0] = 1.0
        return f_z, g_z
    
    def get_p_z_p_x(self, x: th.Tensor):
        p_z_p_x_batch = vmap(
                            jacrev(func=self.z_func, argnums=2), in_dims=(None, None, 0)
                        )(self.z_params, self.z_buffers, x[:, None, :])
        return p_z_p_x_batch.squeeze_()
    
    def get_z(self, x: th.Tensor):
        z_batch = self.z_func(self.z_params, self.z_buffers, x)
        return z_batch
    
    def get_p_v_p_u(self, x: th.Tensor, u: th.Tensor):
        p_v_p_u_batch = vmap(
                            jacrev(func=self.v_func, argnums=3), in_dims=(None, None, 0, 0)
                        )(self.v_params, self.v_buffers, x[:, None, :], u[:, None, :])
        return p_v_p_u_batch.squeeze_().reshape(-1, 21)
    
    def get_v(self, x: th.Tensor, u: th.Tensor):
        v_batch = self.v_func(self.v_params, self.v_buffers, x, u)
        return v_batch
    
    def get_u_given_v_and_x(
        self, 
        u_ref: th.Tensor, x: th.Tensor, v: th.Tensor,
        lr, max_steps, eps,
    ):
        count = 0
        u = u_ref
        while count < max_steps:
            count += 1
            p_v_p_u = self.get_p_v_p_u(x, u)
            v_calc = self.get_v(x, u)
            grad = 2 * (v_calc - v)[:, None, None] @ p_v_p_u[:, None, :]
            u = u - lr * grad.squeeze_()
            losses = th.linalg.norm((v_calc - v)[:, None], dim=1)
        losses = th.linalg.norm((v_calc - v)[:, None], dim=1)
        if_convergence = ~(losses > eps)
        # non_convergence_num = th.count_nonzero(~if_convergence)
        print(losses[~if_convergence])
        return u, if_convergence
    
    def calc_u_safe_using_sampling_method(
        self, 
        u_ref, 
        env,
        safety_index_params: Dict,
        
        # for sampling method
        init_radius=0.01,
        max_expanding_steps=7,
        max_shrinking_steps=10,
    ):  
        
        root_states, dof_states = env.get_states()
        z = env.get_z()
        print(z[0])
        f_z, g_z = self.get_f_z_and_g_z(z)
        phi_z = self.get_phi_z(safety_index_params=safety_index_params, z=z)
        p_phi_p_z = self.get_p_phi_p_z(safety_index_params=safety_index_params, z=z)
        LfP = p_phi_p_z @ f_z
        LgP = p_phi_p_z @ g_z
        if th.all(phi_z < 0):
            return u_ref
        
        # import ipdb; ipdb.set_trace()
        def get_dot_phi(u, z=z, LfP=LfP, LgP=LgP):
            env.step(u)
            z_next = env.get_z()
            v = (z_next[:, 1] - z[:, 1]) * 60
            dot_phi = LfP + LgP @ v[:, None, None]
            # print(z_next[0])
            env.reset_manual(root_states=root_states, dof_states=dof_states)
            z = env.get_z()
            # print(z[0])
            return dot_phi.squeeze_()
        
        dot_phi_ref = get_dot_phi(u_ref)
        if th.all(dot_phi_ref < -self.gamma * phi_z):
            return u_ref
        
        # use Box-Mueller algorithm to sample points uniformly in a u_dim sphere
        points = th.randn(size=u_ref.shape, device=self.device)
        points = points / th.sum(points, dim=-1, keepdim=True)
        
        # expanding iteration
        i = 0
        delta_u =  init_radius * points
        while i < max_expanding_steps:
            u_fake = u_ref + delta_u
            dot_phi_fake = get_dot_phi(u_fake)
            print(dot_phi_fake[0])
            if th.any(dot_phi_fake < -self.gamma * phi_z):
                break
            else:
                i += 1
                delta_u *= 2
                
        if i == max_expanding_steps:
            raise Exception('No infeasible u!')
        
        # shrinking iteration
        right = delta_u
        left = delta_u / 2
        middle = (right + left) / 2
        i = 0
        while i < max_shrinking_steps:
            u_fake = u_ref + middle
            dot_phi_fake = get_dot_phi(u_fake)
            if th.any(dot_phi_fake < -self.gamma * phi_z):
                right = middle
            else:
                left = middle
            middle = (right + left) / 2    
            i += 1
        
        # pickle one u
        delta_u = middle[th.argmin(dot_phi_fake + self.gamma * phi_z)]
        u_safe = u_ref + delta_u[None, :]
        return u_safe
        
        

if __name__ == '__main__':
    solver = HumanoidLatentSISolver(
        z_model_root_path='./train_z_log/2022-08-06__14-22-33/', 
        v_model_root_path='./train_v_log/2022-08-06__14-20-29/', 
        init_params={'a_1': 1.0, 'a_2': 1.0, 'a_3': 0.01},
    )
    loss = nn.MSELoss()

    # z_data =  th.load('./data/z_data/0.pth')
    # x = z_data['x']
    # z = z_data['z']
    # z_pred = solver.get_z(x)
    # print(loss(z_pred, z))
    # p_x_p_z = solver.get_p_z_p_x(x)
    # print(p_x_p_z.shape)
    
    v_data = th.load('./data/v_data/10.pth')
    x = v_data['x']
    u = v_data['u']
    v = v_data['v']
    # v_pred = solver.get_v(x, u)
    # print(loss(v, v_pred))
    # p_v_p_u = solver.get_p_v_p_u(x, u)
    # print(p_v_p_u.shape)
    
    # u_calc, if_convergence = solver.get_u_given_v_and_x(
    #     u_ref=u, x=x, v=v,
    #     lr=1e-3, max_steps=100, eps=1e-2,
    # )
    # print(u_calc[~if_convergence])
    
    solver.calc_safe_u(
        safety_index_params=solver.init_params, 
        x=x, u=u
    )