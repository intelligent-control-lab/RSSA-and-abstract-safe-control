from collections import OrderedDict
from typing import List, Tuple
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from functorch import make_functional, make_functional_with_buffers, jacrev, vmap
import wandb

try:
    from fly_inv_pend_env.fly_inv_pend_SI_learning_env import FlyingInvertedPendulumLatentSILearningEnv
except:
    from fly_inv_pend_SI_learning_env import FlyingInvertedPendulumLatentSILearningEnv

class LatentPhiModel(nn.Module):
    def __init__(
        self, 
        z_dim,
        phi_hidden_dim=256,
        phi_num_layer=2,
        device='cuda',
    ) -> None:
        super().__init__()
        self.device = device
        self.z_dim = z_dim
        self.phi_hidden_dim = phi_hidden_dim
        self.phi_num_layer = phi_num_layer
        self.phi_nn_model = self.get_phi_network(
            input_dim=z_dim * 3 + 1, hidden_dim=phi_hidden_dim, num_layer=phi_num_layer,
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
    
    def forward(self, z: th.Tensor, M: th.Tensor):
        z = th.cat((z, th.cos(z), th.sin(z)), dim=-1)
        input = th.cat((z, M), dim=-1)
        phi = self.phi_nn_model(input)
        return phi


class LatentNeuralCBFModel:
    def __init__(
        self, 
        env: FlyingInvertedPendulumLatentSILearningEnv, 
        device='cuda', 
        gamma=0.1, 
        relaxation_penalty=5000,
        phi_hidden_dim=256, phi_num_layer=2,
    ):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.relaxation_penalty = relaxation_penalty
        self.dot_M_max = env.dot_M_max
        self.latent_space_differentiable_qp_solver, self.latent_space_param_names_dict = self.get_latent_space_differentiable_qp_solver()
        self.phi_fmodel, self.phi_params, self.phi_buffers = self.get_phi_fmodel(
            z_dim=env.z_dim, phi_hidden_dim=phi_hidden_dim, phi_num_layer=phi_num_layer,
        )
        
    def get_latent_space_differentiable_qp_solver(self):
        v_var = cp.Variable(self.env.v_dim)
        relaxation_var = cp.Variable(1, nonneg=True)
        vars = [v_var, relaxation_var]
        
        phi_param = cp.Parameter(1)
        LfP_param = cp.Parameter(1)
        LgP_param = cp.Parameter(1)
        p_phi_p_M_param = cp.Parameter(1, nonneg=True)
        v_ref_param = cp.Parameter(1)
        M_param = cp.Parameter(1, nonneg=True)
        params = [phi_param, LfP_param, LgP_param, p_phi_p_M_param, v_ref_param, M_param]
        param_names = ['phi', 'LfP', 'LgP', 'p_phi_p_M', 'v_ref', 'M']
        param_names_map = dict(zip(param_names, range(len(params))))
        
        constraints = []
        constraints.append(
            LfP_param + LgP_param @ v_var + self.gamma * phi_param + p_phi_p_M_param * self.dot_M_max - relaxation_var <= 0
        )
        constraints.append(v_var <= M_param)
        constraints.append(v_var >= -M_param)
        
        objective_expression = cp.sum_squares(v_var - v_ref_param) + cp.multiply(self.relaxation_penalty, relaxation_var)
        objective = cp.Minimize(objective_expression)
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()
        differentiable_qp_solver =  CvxpyLayer(
            problem, variables=vars, parameters=params
        )
        return differentiable_qp_solver, param_names_map
    
    def get_phi_fmodel(self, z_dim, phi_hidden_dim=256, phi_num_layer=2):
        phi_nn_model = LatentPhiModel(z_dim, phi_hidden_dim, phi_num_layer)
        phi_fmodel, phi_params, phi_buffers = make_functional_with_buffers(phi_nn_model, disable_autograd_tracking=False)
        return phi_fmodel, phi_params, phi_buffers
    
    def get_phi_z(self, z: th.Tensor, M: th.Tensor) -> th.Tensor:
        phi =  self.phi_fmodel(self.phi_params, self.phi_buffers, z, M)
        return phi
    
    def get_p_phi_p_z_and_M(self, z: th.Tensor, M: th.Tensor) -> th.Tensor:
        p_phi_p_z, p_phi_p_M = vmap(jacrev(func=self.phi_fmodel, argnums=(2, 3)), in_dims=(None, None, 0, 0))(self.phi_params, self.phi_buffers, z, M)
        return p_phi_p_z, th.abs(p_phi_p_M)
    
    def get_phi_0_z(self, z: th.Tensor, M: th.Tensor) -> th.Tensor:
        return self.env.get_phi_0_z(z, M)
    
    def get_p_phi_0_p_z_and_M(self, z: th.Tensor, M: th.Tensor) -> th.Tensor:
        return self.env.get_p_phi_0_p_z(z, M), self.env.get_p_phi_0_p_M(z, M)
    
    def solve_latent_state_SI_qp(
        self, 
        phi: th.Tensor, LfP: th.Tensor, LgP: th.Tensor, p_phi_p_M: th.Tensor, v_ref: th.Tensor, M: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor]:
        params = [phi, LfP, LgP, p_phi_p_M, v_ref, M]
        result = self.latent_space_differentiable_qp_solver(
            *params,
            solver_args={'max_iters': 50000},
        )
        v = result[0] 
        relaxation = th.hstack(result[1:])
        return v, relaxation
    
    def boundary_loss(
        self,
        phi: th.Tensor,
        safe_mask: th.Tensor, unsafe_mask: th.Tensor,
    ) -> List[Tuple[str, th.Tensor]]:
        eps = 1e-2
        loss = []
        
        phi_safe = phi[safe_mask]
        safe_violation = F.relu(eps + phi_safe)
        safe_phi_term = safe_violation.mean()
        loss.append(('SI safe region term', 0.1 * safe_phi_term))
        
        phi_unsafe = phi[unsafe_mask]
        unsafe_violation = F.relu(eps - phi_unsafe)
        unsafe_phi_term = unsafe_violation.mean()
        loss.append(('SI unsafe region term', unsafe_phi_term))
        return loss
    
    def descent_loss(
        self,
        phi: th.Tensor, LfP: th.Tensor, LgP: th.Tensor, p_phi_p_M: th.Tensor, v_ref: th.Tensor, M: th.Tensor,
    ) -> List[Tuple[str, th.Tensor]]:
        loss = []
        v_qp, qp_relaxation = self.solve_latent_state_SI_qp(phi, LfP, LgP, p_phi_p_M, v_ref, M)
        qp_relaxation_loss = qp_relaxation.mean()
        loss.append(('QP relaxation', 2 * qp_relaxation_loss))
        return loss
    
    def auxiliary_loss(
        self,
        phi: th.Tensor, phi_0: th.Tensor,
    ):
        loss = []
        phi_delta = F.relu(phi_0 - phi)
        loss.append(('auxiliary term', phi_delta.mean()))
        return loss
    
    def forward(
        self, 
        z: th.Tensor, f_z: th.Tensor, g_z: th.Tensor, v_ref: th.Tensor, M: th.Tensor,
        safe_mask: th.Tensor, unsafe_mask: th.Tensor,
    ) -> th.Tensor:
        phi = self.get_phi_z(z, M)
        p_phi_p_z, p_phi_p_M = self.get_p_phi_p_z_and_M(z, M)
        phi_0 = self.get_phi_0_z(z, M)
        p_phi_0_p_z, p_phi_0_p_M = self.get_p_phi_0_p_z_and_M(z, M)
        
        # if_larger = (phi >= phi_0)
        # wandb.log({'debug/if_larger': th.count_nonzero(if_larger)})
        
        # phi = th.where(if_larger, phi, phi_0)
        # p_phi_p_z = th.where(if_larger[:, :, None], p_phi_p_z, p_phi_0_p_z)
        # p_phi_p_M = th.where(if_larger[:, :, None], p_phi_p_M, p_phi_0_p_M)
        
        LfP = (p_phi_p_z @ f_z).reshape(-1, 1)
        LgP = (p_phi_p_z @ g_z).reshape(-1, 1)
        p_phi_p_M = p_phi_p_M.reshape(-1, 1)
        
        component_losses = {}
        component_losses.update(self.boundary_loss(phi, safe_mask, unsafe_mask))
        component_losses.update(self.descent_loss(phi, LfP, LgP, p_phi_p_M, v_ref, M))
        component_losses.update(self.auxiliary_loss(phi, phi_0))
        total_loss = 0.0
        for _, loss_value in component_losses.items():
            if not th.isnan(loss_value):
                total_loss += loss_value
        losses = {'loss': total_loss, **component_losses}
        return losses
    
    
if __name__ == '__main__':
    # x = th.ones((10, 2))
    # y = th.ones((10, 1))
    # def func(x, y):
    #     return th.cat((x, y), dim=-1)
    # p_z_p_x, p_z_p_y = vmap(jacrev(func=func, argnums=(0, 1)), in_dims=(0, 0))(x, y)
    # print(p_z_p_x.shape)
    # print(p_z_p_y.shape)
    
    env = FlyingInvertedPendulumLatentSILearningEnv(
        device='cuda', 
        dt=1e-3,
        init_params={'a': 6e-1},
        L_p=3.0,
    )
    neural_cbf = LatentNeuralCBFModel(env=env, device='cuda')
    z = th.zeros(env.env_num, env.z_dim).cuda() + 1e-2
    f_z = th.ones((env.env_num, env.z_dim, 1)).cuda()
    g_z = th.ones((env.env_num, env.z_dim, 1)).cuda()
    v_ref = th.zeros((env.env_num, 1)).cuda()
    M = th.ones((env.env_num, 1)).cuda() * 0.01
    safe_mask = th.randint(low=0, high=2, size=(env.env_num, )).cuda().bool()
    unsafe_mask = th.randint(low=0, high=2, size=(env.env_num, )).cuda().bool()
    losses = neural_cbf.forward(z, f_z, g_z, v_ref, M, safe_mask, unsafe_mask)
    print(losses)
    loss = losses['loss']
    loss.backward()