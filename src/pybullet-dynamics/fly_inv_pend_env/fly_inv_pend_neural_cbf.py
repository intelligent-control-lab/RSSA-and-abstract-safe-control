from typing import Dict, List
import torch as th
import sys
import yaml
from torch.distributions import uniform
print(sys.path)
sys.path.append('/home/zhux/robust-safe-set/src/pybullet-dynamics')

from neural_cbf_model import NeuralCBFModel
from latent_neural_cbf_model import LatentNeuralCBFModel
from fly_inv_pend_SI_learning_env import FlyingInvertedPendulumLatentSILearningEnv

def get_env_and_model(
    log_root_path='./src/pybullet-dynamics/fly_inv_pend_env/latent_neural_cbf_log/2022-08-14__14-51-34/',
    data_path='./src/pybullet-dynamics/fly_inv_pend_env/data/latent_neural_cbf_data/',
):
    with open(log_root_path + 'fly_inv_pend_latent_neural_cbf.yaml', 'r') as file:
        cfg = yaml.load(file)
    env = FlyingInvertedPendulumLatentSILearningEnv(**cfg['robot_kwargs'])
    model = LatentNeuralCBFModel(env=env, **cfg['model_kwargs'])
    model.phi_params = th.load(log_root_path + 'best_model.pth')
    return env, model

def evaluate_latent_neural_cbf(
    data_indices,
    log_root_path='./src/pybullet-dynamics/fly_inv_pend_env/latent_neural_cbf_log/2022-08-26__14-29-46/',
    data_path='./src/pybullet-dynamics/fly_inv_pend_env/data/latent_neural_cbf_data/',
):
    # for safety index finetune
    infeasible_data: Dict[str, List[th.Tensor]] = {
        'z': [], 'M': [], 'f_z': [], 'g_z': [], 'v_ref': [], 'safe_mask': [], 'unsafe_mask': [],
    }
    
    env, model = get_env_and_model(log_root_path=log_root_path, data_path=data_path)
    reward_list = []
    for data_idx in data_indices:
        data: Dict[str, th.Tensor] = th.load(data_path + str(data_idx) + '.pth') 
        z = data['z']; M = data['M']; f_z = data['f_z']; g_z = data['g_z']
        phi = model.get_phi_z(z=z, M=M)
        p_phi_p_z, p_phi_p_M = model.get_p_phi_p_z_and_M(z=z, M=M)
        LfP = (p_phi_p_z @ f_z).squeeze_()
        LgP = (p_phi_p_z @ g_z).squeeze_()
        c = -model.gamma * phi.squeeze_() - LfP - p_phi_p_M.squeeze_() * model.dot_M_max
        LgP_mul_v_1 = LgP * -M.squeeze_()
        LgP_mul_v_2 = LgP * M.squeeze_()
        LgP_mul_v_min = th.min(LgP_mul_v_1, LgP_mul_v_2)
        if_feasible = (LgP_mul_v_min <= c)
        reward = th.count_nonzero(if_feasible) / z.shape[0]
        reward_list.append(reward.item())
        
        # for safety index finetune
        if_infeasible = ~if_feasible
        for key in infeasible_data.keys():
            infeasible_data[key].append(data[key][if_infeasible])
    
    # for safety index finetune
    for key in infeasible_data.keys():
        infeasible_data[key] = th.cat(infeasible_data[key])
        print(infeasible_data[key].shape)
    infeasible_data['M'] = infeasible_data['M'][:, None]
    # th.save(infeasible_data, './src/pybullet-dynamics/fly_inv_pend_env/data/latent_neural_cbf_data/infeasible.pth')
        
    return reward_list


def latent_neural_safe_control(
    env: FlyingInvertedPendulumLatentSILearningEnv,
    model: LatentNeuralCBFModel,
    u_ref: th.Tensor = None,
    u_sample_num=10000,
):
    '''
    Only support one environment!
    '''
    assert env.env_num == 1
    if u_ref is None:
        u_ref = th.zeros(size=(env.env_num, env.u_dim), device=env.device)
    z = env.z
    M = env.M_x[:, None]
    f_z, g_z = env.f_z, env.g_z
    v_ref = env.get_v(u_ref)
    phi = model.get_phi_z(z=z, M=M)
    p_phi_p_z, p_phi_p_M = model.get_p_phi_p_z_and_M(z=z, M=M)
    LfP = (p_phi_p_z @ f_z).squeeze_()
    LgP = (p_phi_p_z @ g_z).squeeze_()
    c = -model.gamma * phi.squeeze_() - LfP - p_phi_p_M.squeeze_() * model.dot_M_max
    LgP_mul_v_ref = LgP * v_ref
    if LgP_mul_v_ref <= c:
        return u_ref
    
    u = uniform.Uniform(low=env.u_simplified_lim['low'], high=env.u_simplified_lim['high']).sample(th.Size((u_sample_num, )))
    v = env.get_v(u)
    LgP_mul_v = LgP * v
    if_feasible = (LgP_mul_v <= c)
    v_feasible = v[if_feasible]
    u_feasible = u[if_feasible]
    
    return 

if __name__ == '__main__':
    # data_indices = list(range(10))
    # evaluate_latent_neural_cbf(data_indices=data_indices)

    env, model = get_env_and_model(
        log_root_path='./src/pybullet-dynamics/fly_inv_pend_env/latent_neural_cbf_log/2022-08-14__14-51-34/',
    )
    x_init = th.ones((1, 10), device=env.device) * 0.1
    z_init = th.tensor([[0.4, 2.0]], device=env.device)
    x_init, if_convergence = env.get_x_given_z(x_init, z_init, lr=0.1, max_steps=100, eps=1e-2)
    env.reset(x_init=x_init)
    latent_neural_safe_control(env, model)


