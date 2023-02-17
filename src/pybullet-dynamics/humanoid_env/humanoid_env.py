from copy import deepcopy
from datetime import datetime
import time
import isaacgym
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import os
import cv2
import hydra
import yaml
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import gym
import torch as th
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
print(sys.argv)

from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

from humanoid_latent_SI_solver import HumanoidLatentSISolver
from humanoid_utils import *

@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    global if_return, AGENT
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, get_rlgames_env_creator
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import model_builder
    from isaacgymenvs.learning import amp_continuous
    from isaacgymenvs.learning import amp_players
    from isaacgymenvs.learning import amp_models
    from isaacgymenvs.learning import amp_network_builder
    import isaacgymenvs

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.wandb_name}_{time_str}"
    
    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    rank = int(os.getenv("LOCAL_RANK", "0"))
    if cfg.multi_gpu:
        # torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
        cfg.sim_device = f'cuda:{rank}'
        cfg.rl_device = f'cuda:{rank}'

    # sets seed. if seed is -1 will pick a random one
    cfg.seed += rank
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=rank)

    def create_env_thunk(**kwargs):
        # import ipdb; ipdb.set_trace()
        envs = isaacgymenvs.make(
            cfg.seed, 
            cfg.task_name, 
            cfg.task.env.numEnvs, 
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            False,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        if cfg.capture_video:
            envs.is_vector_env = True
            envs = gym.wrappers.RecordVideo(
                envs,
                f"videos/{run_name}",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        return envs

    # register the rl-games adapter to use inside the runner
    vecenv.register('RLGPU',
                    lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': create_env_thunk,
    })

    # register new AMP network builder and agent
    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
        runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
        model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
        model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())

        return runner

    rlg_config_dict = omegaconf_to_dict(cfg.train)

    # convert CLI arguments into dictionory
    # create runner and set the settings
    runner = build_runner(RLGPUAlgoObserver())
    runner.load(rlg_config_dict)
    runner.reset()

    # dump config dict
    experiment_dir = os.path.join('runs', cfg.train.params.config.name)
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    rank = 0
    runner.run({
        'train': not cfg.test,
        'play': cfg.test,
        'checkpoint' : cfg.checkpoint,
        'sigma' : None
    },
    if_return=if_return)
    try:
        AGENT = runner.player
    except:
        AGENT = runner.agent


if_return = True
AGENT = None
class HumanoidEnv:
    def __init__(self):
        try:
            launch_rlg_hydra()
        except:
            self.z_dim = 2
            self.v_dim = 1
            self.dot_M_max = 100
            return
        self.agent = AGENT
        self.device = self.agent.env.device
        self.dof = self.agent.env.num_dof
        self.num_envs = self.agent.env.num_envs
        self.env_ids = th.arange(self.num_envs, device=self.device)
    
    def step(self, u):
        obses, r, done, info = self.agent.env_step(self.agent.env, u)
        return obses
    
    def reset(self):
        obses = self.agent.env_reset(self.agent.env)
        batch_size = 1
        batch_size = self.agent.get_batch_size(obses, batch_size)
        return obses
    
    def render(self, mode='rgb_array'):
        rgb = self.agent.env.render(mode=mode)
        return rgb
    
    def get_z(self):
        pos_z = self.agent.env.root_states[:, 2].detach().clone()
        vel_z = self.agent.env.root_states[:, 9].detach().clone()
        z = th.hstack([pos_z[:, None], vel_z[:, None]])
        return z
        
    def calc_u_ref(self, obses):
        return self.agent.get_action(obses, True)
    
    def get_states(self):
        root_states = self.agent.env.root_states.detach().clone()
        dof_states = th.cat((self.agent.env.dof_pos.detach().clone(), self.agent.env.dof_vel.detach().clone()), dim=-1)
        return root_states, dof_states
    
    def reset_manual(self, root_states: th.Tensor, dof_states: th.Tensor):
        self.agent.env.root_states[self.env_ids] = root_states[self.env_ids]
        # self.agent.env.dof_pos = tensor_clamp(
        #     self.agent.env.initial_dof_pos + dof_states[:, :self.dof], 
        #     self.agent.env.dof_limits_lower, self.agent.env.dof_limits_upper
        # )
        self.agent.env.dof_pos[self.env_ids] = dof_states[:, :self.dof][self.env_ids]
        self.agent.env.dof_vel[self.env_ids] = dof_states[:, self.dof:][self.env_ids]
        self.agent.env.gym.set_actor_root_state_tensor(
            self.agent.env.sim, 
            gymtorch.unwrap_tensor(self.agent.env.root_states)
        )
        self.agent.env.gym.set_dof_state_tensor(
            self.agent.env.sim,
            gymtorch.unwrap_tensor(self.agent.env.dof_state)
        )

        
    

if __name__ == '__main__':
    ssa = HumanoidLatentSISolver()
    env = HumanoidEnv()
    collect_M(env, ssa)
    exit(0)
    
    root_states, dof_states = env.get_states()
    env.reset_manual(root_states=root_states, dof_states=dof_states)
    
    obses = env.reset()
    u = env.calc_u_ref(obses)
    obses = env.step(u)
    u = ssa.calc_u_safe_using_sampling_method(u_ref=u, env=env, safety_index_params=ssa.init_params)
    # rgb = env.render()
    pos_z_list = []
    vel_z_list = []
    for i in range(480):
        print(i)
        u = env.calc_u_ref(obses)
        # u = ssa.calc_u_safe_using_sampling_method(u_ref=u, env=env, safety_index_params=ssa.init_params)
        # u = ssa.calc_safe_u(
        #     safety_index_params=ssa.init_params,
        #     x=obses, u=u,
        # )
        obses = env.step(u)
        # print('u: ', u[0])
        # print('obs: ', obses[0])
        if i % 50 == 49:
            root_states, dof_states = env.get_states()
            env.reset_manual(root_states=root_states + 0.1, dof_states=dof_states + 0.1)
            for j in range(10):
                env.step(u)
            env.reset_manual(root_states=root_states, dof_states=dof_states)
        
        # rgb = env.render()
        # cv2.imwrite(f'./imgs/ssa/{i}.png', rgb)
        
    #     pos_z = env.get_z()[0, 0].cpu().numpy()
    #     vel_z = env.get_z()[0, 1].cpu().numpy()
    #     pos_z_list.append(pos_z)
    #     vel_z_list.append(vel_z)
    # vel_z_fake_list = np.diff(pos_z_list) * 60
    # plt.plot(pos_z_list)
    # # plt.plot(vel_z_list)
    # # plt.plot(vel_z_fake_list)
    # plt.savefig('./imgs/z/z_list_origin.png')
    # plt.close()

   