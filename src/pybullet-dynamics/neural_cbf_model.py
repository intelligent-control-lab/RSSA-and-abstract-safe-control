from collections import OrderedDict
from typing import List, Tuple
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from functorch import make_functional, make_functional_with_buffers, jacrev, vmap

try:
    from fly_inv_pend_env.fly_inv_pend_SI_learning_env import FlyingInvertedPendulumLatentSILearningEnv
except:
    from fly_inv_pend_SI_learning_env import FlyingInvertedPendulumLatentSILearningEnv

class PhiModel(nn.Module):
    def __init__(
        self, 
        x_dim,
        phi_hidden_dim=256,
        phi_num_layer=2,
        device='cuda',
    ) -> None:
        super().__init__()
        self.device = device
        self.x_dim = x_dim
        self.phi_hidden_dim = phi_hidden_dim
        self.phi_nn_model = self.get_phi_network(
            input_dim=x_dim * 3, hidden_dim=phi_hidden_dim, num_layer=phi_num_layer,
        )
        
    def get_phi_network(self, input_dim, hidden_dim, num_layer):
        phi_layers = OrderedDict()
        phi_layers['input_linear'] = nn.Linear(input_dim, hidden_dim)
        phi_layers['input_activation'] = nn.Tanh()
        for i in range(num_layer):
            phi_layers[f'layer_{i}_linear'] = nn.Linear(hidden_dim, hidden_dim)
            phi_layers[f'layer_{i}_activation'] = nn.Tanh()
        phi_layers['output_linear'] = nn.Linear(hidden_dim, 1)
        phi_nn_model = nn.Sequential(phi_layers)
        phi_nn_model.to(self.device)
        return phi_nn_model
    
    def forward(self, x: th.Tensor):
        x = th.cat((x, th.cos(x), th.sin(x)), dim=-1)
        phi = self.phi_nn_model(x)
        return phi
    

class NeuralCBFModel:
    def __init__(
        self, 
        env: FlyingInvertedPendulumLatentSILearningEnv,
        device='cuda',
        gamma=0.1,
        relaxation_penalty=5000.0,
        phi_hidden_dim=256,
        phi_num_layer=2,
    ):  
        self.env = env
        self.device = device
        self.gamma = gamma
        self.relaxation_penalty = relaxation_penalty
        self.origin_space_differentiable_qp_solver, self.origin_space_param_names_dict = self.get_origin_space_differentiable_qp_solver()
        self.phi_fmodel, self.phi_params, self.phi_buffers = self.get_phi_fmodel(
            x_dim=env.x_dim, phi_hidden_dim=phi_hidden_dim, phi_num_layer=phi_num_layer,
        )
        
    def get_origin_space_differentiable_qp_solver(self):
        u_var = cp.Variable(self.env.u_dim)
        relaxation_var = cp.Variable(1, nonneg=True)
        vars = [u_var, relaxation_var]
        
        phi_param = cp.Parameter(1, nonneg=True)
        LfP_param = cp.Parameter(1)
        LgP_param = cp.Parameter(self.env.u_dim)
        u_ref_param = cp.Parameter(self.env.u_dim)
        params = [phi_param, LfP_param, LgP_param, u_ref_param]
        param_names = ['phi', 'LfP', 'LgP', 'u_ref']
        param_names_map = dict(zip(param_names, range(4)))
        
        constraints = []
        constraints.append(
            LfP_param + LgP_param @ u_var + self.gamma * phi_param - relaxation_var <= 0
        )
        constraints.append(self.env.u_lim_A @ u_var <= self.env.u_lim_b)
        
        objective_expression = cp.sum_squares(u_var - u_ref_param) + cp.multiply(self.relaxation_penalty, relaxation_var)
        objective = cp.Minimize(objective_expression)
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()
        differentiable_qp_solver =  CvxpyLayer(
            problem, variables=vars, parameters=params
        )
        return differentiable_qp_solver, param_names_map
    
    def get_phi_fmodel(self, x_dim, phi_hidden_dim=256, phi_num_layer=2):
        phi_nn_model = PhiModel(x_dim, phi_hidden_dim, phi_num_layer)
        phi_fmodel, phi_params, phi_buffers = make_functional_with_buffers(phi_nn_model, disable_autograd_tracking=False)
        return phi_fmodel, phi_params, phi_buffers
    
    def get_phi_x(self, x: th.Tensor) -> th.Tensor:
        phi =  self.phi_fmodel(self.phi_params, self.phi_buffers, x)
        return phi
    
    def get_p_phi_p_x(self, x: th.Tensor) -> th.Tensor:
        p_phi_p_x = vmap(jacrev(func=self.phi_fmodel, argnums=2), in_dims=(None, None, 0))(self.phi_params, self.phi_buffers, x)
        return p_phi_p_x
    
    def solve_origin_state_SI_qp(
        self, 
        phi: th.Tensor, LfP: th.Tensor, LgP: th.Tensor, u_ref: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor]:
        params = [phi, LfP, LgP, u_ref]
        result = self.origin_space_differentiable_qp_solver(
            *params,
            solver_args={'max_iters': 50000},
        )
        u = result[0] 
        relaxation = th.hstack(result[1:])
        return u, relaxation
    
    def boundary_loss(
        self,
        phi: th.Tensor,
        safe_mask: th.Tensor, unsafe_mask: th.Tensor,
    ) -> List[Tuple[str, th.Tensor]]:
        eps = 1e-2
        loss = []
        
        phi_safe = phi[safe_mask]
        safe_violation = F.relu(eps + phi_safe)
        safe_phi_term = 1e2 * safe_violation.mean()
        loss.append(('SI safe region term', safe_phi_term))
        
        phi_unsafe = phi[unsafe_mask]
        unsafe_violation = F.relu(eps - phi_unsafe)
        unsafe_phi_term = 1e2 * unsafe_violation.mean()
        loss.append(('SI unsafe region term', unsafe_phi_term))
        return loss
    
    def descent_loss(
        self,
        phi: th.Tensor, LfP: th.Tensor, LgP: th.Tensor, u_ref: th.Tensor,
    ) -> List[Tuple[str, th.Tensor]]:
        loss = []
        u_qp, qp_relaxation = self.solve_origin_state_SI_qp(phi, LfP, LgP, u_ref)
        qp_relaxation_loss = qp_relaxation.mean()
        loss.append(('QP relaxation', qp_relaxation_loss))
        return loss
    
    def forward(
        self, 
        x: th.Tensor, f_x: th.Tensor, g_x: th.Tensor, u_ref: th.Tensor,
        safe_mask: th.Tensor, unsafe_mask: th.Tensor,
    ) -> th.Tensor:
        phi = self.get_phi_x(x)
        p_phi_p_x = self.get_p_phi_p_x(x)
        LfP = (p_phi_p_x @ f_x).reshape(-1, 1)
        LgP = (p_phi_p_x @ g_x).reshape(-1, self.env.u_dim)
        
        component_losses = {}
        component_losses.update(self.boundary_loss(phi, safe_mask, unsafe_mask))
        component_losses.update(self.descent_loss(phi, LfP, LgP, u_ref))
        total_loss = 0.0
        for _, loss_value in component_losses.items():
            if not th.isnan(loss_value):
                total_loss += loss_value
        losses = {'loss': total_loss, **component_losses}
        return losses
        
if __name__ == '__main__':
    env = FlyingInvertedPendulumLatentSILearningEnv(
        device='cuda', 
        dt=1e-3,
        init_params={'a': 6e-1},
        L_p=3.0,
    )
    neural_cbf = NeuralCBFModel(env=env, device='cuda')
    x = th.zeros(env.env_num, env.x_dim).cuda() + 1e-2
    f_x = th.zeros((env.env_num, env.x_dim, 1)).cuda()
    g_x = th.ones((env.env_num, env.x_dim, env.u_dim)).cuda()
    u_ref = th.zeros((env.env_num, env.u_dim)).cuda()
    safe_mask = th.randint(low=0, high=2, size=(env.env_num, )).cuda().bool()
    unsafe_mask = th.randint(low=0, high=2, size=(env.env_num, )).cuda().bool()
    losses = neural_cbf.forward(x, f_x, g_x, u_ref, safe_mask, unsafe_mask)
    print(losses)
        
        
