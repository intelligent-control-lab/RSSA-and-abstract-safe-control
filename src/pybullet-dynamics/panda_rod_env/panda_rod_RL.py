import numpy as np
from gym import spaces
import time
import datetime
import pickle

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy

from stable_baselines3.common.vec_env import SubprocVecEnv

from panda_rod_env import PandaRodEnv

class RewardModel:
    def __init__(
        self,
        goal_pose,
        goal_allowable_error,
        sparse_reward_coff=500,
        dense_distance_reward_coff=0.02,
        dense_time_reward_coff=-0.01,
        dense_action_encourage_speed_coff=0.01,
    ):
        self.goal_pose = np.asanyarray(goal_pose)
        self.goal_allowable_error = goal_allowable_error

        self.sparse_reward_coff = sparse_reward_coff
        self.dense_distance_reward_coff = dense_distance_reward_coff
        self.dense_time_reward_coff = dense_time_reward_coff
        self.dense_action_encourage_speed_coff = dense_action_encourage_speed_coff
    
    def if_success(self, end_eff_pose):
        self.distance = np.linalg.norm(
            end_eff_pose - self.goal_pose
        )
        if self.distance < self.goal_allowable_error:
            return True
        else:
            return False
    
    def get_reward(self, end_eff_velocity, success_flag):
        reward = 0
        if success_flag:
            reward += self.sparse_reward_coff
        reward += self.dense_distance_reward_coff / (self.goal_allowable_error + self.distance)
        reward += self.dense_time_reward_coff
        reward += self.dense_action_encourage_speed_coff * np.linalg.norm(end_eff_velocity)
        return reward
    
    def get_contact_punishment(self):
        return -10

class PandaRodEnvRL(PandaRodEnv):
    def __init__(
        self, 
        render_flag=False, 
        max_steps=960, 
        goal_pose=[0.8, 0.2, 0.5], 
        goal_orientation=[1, 0, 0, 0], 
        obstacle_pose=[0.56, 0, 0.6],
        goal_allowable_error=0.02,
        random_init_pose=True,
    ):
        super().__init__(render_flag, max_steps, goal_pose, goal_orientation, obstacle_pose)

        self.create_observation_space()
        self.create_action_space()

        self.reward_model = RewardModel(self.goal_pose, goal_allowable_error)

        self.random_init_pose = random_init_pose
        if self.random_init_pose:
            self.workspace_limit = {
                'x': [0.4, 0.8],
                'y': [-0.1, 0.1],
                'z': [0.4, 0.6],
            }

        self.reset()
    
    def step(self, action):
        self.current_step += 1

        target_torque = action * self.action_coff
        self.target_torque = target_torque
        if self.actuated_dof < self.robot.dof and len(target_torque) < self.robot.dof:
            zero_torque = np.zeros((self.robot.dof - self.actuated_dof, ))
            target_torque = np.concatenate((target_torque, zero_torque))
        self.robot.set_target_torques(target_torque)
        self.pybullet.stepSimulation(physicsClientId=self.physics_client_id)

        obs = self.get_observation()

        ### TODO: save raw data for learned dynamics
        if not self.robot.get_contact_info():
            self.save_nn_data()
        ### END TODO

        if self.robot.get_contact_info():
            reward = self.reward_model.get_contact_punishment()
            done = True
            success_flag = False
            print('---CONTACT!---')
        else:
            end_eff_pose = self.robot.get_end_eff_pose()
            end_eff_velocity = self.robot.get_end_eff_velocity()

            success_flag = self.reward_model.if_success(end_eff_pose)
            reward = self.reward_model.get_reward(end_eff_velocity, success_flag)
            if success_flag or self.current_step == self.max_steps:
                done = True
            else:
                done = False

        info = {
            'is_success': success_flag
        }

        if success_flag:
            print('---REACH THE GOAL!---')
        
        return obs, reward, done, info

    def reset(self):
        self.current_step = 0
        self.last_Xr = None

        if self.random_init_pose:
            # random_init_x = self.np_random.uniform(
            #     low=self.workspace_limit['x'][0],
            #     high=self.workspace_limit['x'][1],
            # )
            # random_init_y = self.np_random.uniform(
            #     low=self.workspace_limit['y'][0],
            #     high=self.workspace_limit['y'][1],
            # )
            # random_init_z = self.np_random.uniform(
            #     low=self.workspace_limit['z'][0],
            #     high=self.workspace_limit['z'][1],
            # )
            random_init_x = self.np_random.choice(self.workspace_limit['x'])
            random_init_y = self.np_random.choice(self.workspace_limit['y'])
            random_init_z = self.np_random.choice(self.workspace_limit['z'])
            random_init_pose = [random_init_x, random_init_y, random_init_z]
            # print(random_init_pose)
            self.robot.q_init = self.robot.solve_inverse_kinematics(random_init_pose)

        self.robot.reset()
        obs = self.get_observation()
        return obs

    def create_observation_space(self):
        '''
        observation incorporates two parts:
        1. q: (self.dof, )
        2. dq: (self.dof, )
        '''
        self.observation_space = spaces.Box(
            low=np.concatenate((self.robot.q_min, -self.robot.velocity_limit)),
            high=np.concatenate((self.robot.q_max, self.robot.velocity_limit)),
            dtype='float32'
        )
    
    def create_action_space(self):
        '''
        action just one part:
        1. acturated tau: (self.dof - self.actuated_dof, )
        normalize action space to [-1.0, 1.0]
        because the action limits is already symmetric, we just need one side
        '''
        self.action_space = spaces.Box(
            low=-np.ones_like(self.robot.torque_limit[:self.actuated_dof]),
            high=np.ones_like(self.robot.torque_limit[:self.actuated_dof]),
            dtype='float32'
        )
        self.action_coff = self.robot.torque_limit[:self.actuated_dof]
    
    def get_observation(self):
        q, dq = self.robot.get_joint_states()
        return np.concatenate((q, dq))

    def save_nn_data(self):
        raw_data = {
            'x': self.Xr,
            'dot_x': self.dot_Xr,
            'u': self.target_torque,
        }
        file_name = './src/pybullet-dynamics/panda_rod_env/data/nn_raw_data/' + str(time.time()) + '.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(raw_data, file)

    
def train_PPO():
    exp_name = datetime.datetime.now().strftime('%y%m%d_%H%M')
    
    env = make_vec_env(
        env_id=PandaRodEnvRL,
        n_envs=8,
        seed=0,
        monitor_dir=f'./src/pybullet-dynamics/panda_rod_env/logs/models/{exp_name}/monitor',
        vec_env_cls=SubprocVecEnv
    )
    eval_env = make_vec_env(
        env_id=PandaRodEnvRL,
        seed=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=f'./src/pybullet-dynamics/panda_rod_env/logs/models/{exp_name}/ckps/'
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'./src/pybullet-dynamics/panda_rod_env/logs/models/{exp_name}/best_model',
        log_path=f'./src/pybullet-dynamics/panda_rod_env/logs/models/{exp_name}/results', 
        eval_freq=50000,
        render=False,
        n_eval_episodes=5,
        verbose=2
    )
    callback = CallbackList([checkpoint_callback, eval_callback])
    
    model = PPO(
        MlpPolicy,
        env,
        verbose=2,
        seed=2,
        batch_size=8192,
        n_steps=8192,
        n_epochs=10,
        learning_rate=1e-3,
        tensorboard_log=f'./src/pybullet-dynamics/panda_rod_env/logs/tensorboard'
    )
    model.learn(
        total_timesteps=1e6,
        callback=callback,
        reset_num_timesteps=True,
        tb_log_name=exp_name
    )



if __name__ == '__main__':
    train_PPO()

    env = PandaRodEnvRL(render_flag=True, random_init_pose=True)
    while True:
        pass
    model = PPO.load('./src/pybullet-dynamics/panda_rod_env/logs/models/220425_1536/ckps/rl_model_10000000_steps.zip')  

        


