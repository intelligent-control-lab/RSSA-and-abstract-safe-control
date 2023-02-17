import pickle
import pybullet as pb
from pybullet_utils import bullet_client

import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
import torch as th
import time

try:
    from panda_rod_model import PandaRodModel
    from panda_rod_utils import *
except:
    from panda_rod_env.panda_rod_model import PandaRodModel
    from panda_rod_env.panda_rod_utils import *


class PandaRodEnv(gym.Env):
    def __init__(
        self,
        render_flag=False,
        robot_file_path='./src/pybullet-dynamics/panda_rod_env/urdf/panda_without_rod.urdf',
        dof=7,
        end_eff_id=7,
        max_steps=960,
        goal_pose=[0.8, 0.3, 0.5],
        goal_orientation=[1.0, 0.0, 0.0, 0.0],
        obstacle_pose=[0.6, 0.1, 0.5],
        tolerate_error=0.05,
        random_init_pose=False,
        store_nn_data_path='./src/pybullet-dynamics/panda_rod_env/data/nn_raw_data_env_1/',
        use_real_dot_xr=True,
        change_rod_mass=False,
        rod_mass=[0.1, 0.4],
        if_task=True,
    ):
        self.render_flag = render_flag
        if self.render_flag:
            p = bullet_client.BulletClient(connection_mode=pb.SHARED_MEMORY)
            self.physics_client_id = p._client
            if self.physics_client_id < 0:
                p = bullet_client.BulletClient(connection_mode=pb.GUI)
                self.physics_client_id = p._client
            p.resetDebugVisualizerCamera(
                cameraDistance=3.20,
                cameraYaw=124.80,
                cameraPitch=-62.20,
                cameraTargetPosition=[-0.48, -0.39, -0.66],
                physicsClientId=self.physics_client_id
            )
                
        else:
            p = bullet_client.BulletClient(connection_mode=pb.DIRECT)
            self.physics_client_id = p._client
        
        self.pybullet = p
        self.pybullet.setGravity(0, 0, -9.8, physicsClientId=self.physics_client_id)

        self.max_steps = max_steps
        self.current_step = 0
        self.dT = 1 / 240
        self.tolerate_error = tolerate_error
        self.use_real_dot_xr = use_real_dot_xr

        self.robot = PandaRodModel(
            physics_client_id=self.physics_client_id, robot_file_path=robot_file_path,
            dof=dof, end_eff_id=end_eff_id,
        )
        self.actuated_dof = self.robot.dof
        self.B = np.zeros((self.robot.dof, self.actuated_dof))
        self.B[:self.actuated_dof, :self.actuated_dof] = np.eye(self.actuated_dof)
        self.max_u = self.robot.torque_limit[:self.actuated_dof]

        self.if_task = if_task
        if if_task:
            self.goal_pose = np.asanyarray(goal_pose)
            self.goal_orientation = np.asanyarray(goal_orientation)
            self.goal = ObjectModel(cartesian_pos=self.goal_pose, physics_client_id=self.physics_client_id)
            self.obstacle_pose = np.asanyarray(obstacle_pose)
            self.obstacle = ObjectModel(cartesian_pos=self.obstacle_pose, physics_client_id=self.physics_client_id, rgba=[1, 1, 0, 0.5])

            # for usage of SSA, wrap obstacle's Cartesian pose in a list
            self.obstacles = [self.Mh]

        self.seed()

        # for data collection
        self.random_init_pose = random_init_pose
        if self.random_init_pose:
            self.workspace_limit = {
                'x': [0.4, 0.9],
                'y': [-0.3, 0.3],
                'z': [0.4, 0.8],
            }
        self.store_nn_data_path = store_nn_data_path
        
        # to cause disturbance / uncertainty in f and g, change the mass of end rod
        self.change_rod_mass = change_rod_mass
        if self.change_rod_mass:
            for i, mass in enumerate(rod_mass):
                self.robot.change_rod_mass(link_id=self.robot.jac_id-1-i, mass=mass)
        self.rod_mass = []
        for i in range(2):
            self.rod_mass.append(self.robot.get_rod_mass(link_id=self.robot.jac_id-1-i))

        self.reset()  

    def step(self, target_torque):
        target_torque = np.squeeze(target_torque)
        self.current_step += 1
        if self.actuated_dof < self.robot.dof and len(target_torque) < self.robot.dof:
            zero_torque = np.zeros((self.robot.dof - self.actuated_dof, ))
            target_torque = np.concatenate((target_torque, zero_torque))
        # print(target_torque)
        q, _ = self.robot.get_joint_states()
        self.robot.set_target_torques(target_torque)
        self.pybullet.stepSimulation(physicsClientId=self.physics_client_id) 

        obs = self.get_observation()
        done = None
        info = None
        reward = None

        return obs, reward, done, info

    def empty_step(self):
        self.pybullet.stepSimulation(physicsClientId=self.physics_client_id)
    
    def reset(self):
        self.current_step = 0
        self.last_Xr = None

        if self.random_init_pose:
            random_init_x = self.np_random.uniform(
                low=self.workspace_limit['x'][0],
                high=self.workspace_limit['x'][1],
            )
            random_init_y = self.np_random.uniform(
                low=self.workspace_limit['y'][0],
                high=self.workspace_limit['y'][1],
            )
            random_init_z = self.np_random.uniform(
                low=self.workspace_limit['z'][0],
                high=self.workspace_limit['z'][1],
            )
            random_init_pose = [random_init_x, random_init_y, random_init_z]
            self.robot.q_init = self.robot.solve_inverse_kinematics(random_init_pose)

        self.robot.reset()
        obs = self.get_observation()
        return obs

    def render(
        self,
        mode='rgb',
        height=960, width=960,
        cam_dist=4.28,
        camera_target_position=[0.0, 0.0, 2.0],
        cam_yaw=89.2, cam_pitch=-21.8, cam_roll=0,
        near_val=0.1, far_val=100.0
    ):
        view_matrix = self.pybullet.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera_target_position,
            distance=cam_dist,
            yaw=cam_yaw, pitch=cam_pitch, roll=cam_roll,
            upAxisIndex=2,
            physicsClientId=self.physics_client_id
        )
        proj_matrix = self.pybullet.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(width) / height,
            nearVal=near_val, farVal=far_val,
            physicsClientId=self.physics_client_id
        )
        _, _, rgb, _, _ = self.pybullet.getCameraImage(
            width=width, height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=self.pybullet.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.physics_client_id
        )
        rgb = np.array(rgb, dtype=np.uint8).reshape((height, width, 4))
        rgb = rgb[:, :, :3]
        rgb = bgr_to_rgb(rgb)
        return rgb
        
    def seed(self, seed=1):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_observation(self):
        q, dq = self.robot.get_joint_states()
        obs = np.concatenate((q, dq))
        return obs

    def compute_naive_torque(
        self,
        mode='computed_torque_control',
    ):
        if mode == 'operational_space_control':
            tau = self.robot.operational_space_control(self.goal_pose, self.goal_orientation)

        elif mode == 'computed_torque_control':
            target_q = self.robot.solve_inverse_kinematics(self.goal_pose, self.goal_orientation)
            target_dq = [0.0] * self.robot.dof
            tau = self.robot.computed_torque_control(target_q, target_dq)

        elif mode == 'pd_control':
            target_q = self.robot.solve_inverse_kinematics(self.goal_pose, self.goal_orientation)
            target_dq = [0.0] * self.robot.dof
            tau = self.robot.pd_control(target_q, target_dq)

        else:
            raise NameError(f'{mode} is not implemented!')
        return tau

    def compute_underactuated_torque(self):
        tau_full_dof = self.compute_naive_torque().reshape((-1, 1))
        # dot_Xr_target = self.f + self.get_g_full_dof() @ tau_full_dof
        # tau = np.linalg.lstsq(self.g, dot_Xr_target - self.f, rcond=None)[0].reshape((-1, ))
        tau = tau_full_dof[:self.actuated_dof, 0]
        # tau = np.clip(tau, a_min=-self.max_u, a_max=self.max_u)
        self.target_torque = tau
        return tau

    def if_success(self):
        end_eff_pose = np.asanyarray(self.robot.get_end_eff_pose())
        target_pose = np.asanyarray(self.goal_pose)
        if np.linalg.norm(end_eff_pose - target_pose) < self.tolerate_error:
            return True
        return False
    
    def save_nn_data(self):
        raw_data = {
            'x': self.Xr,
            'dot_x': self.dot_Xr,
            'u': self.target_torque,
        }
        file_name = self.store_nn_data_path + str(time.time()) + '.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(raw_data, file)
    
    def get_g_full_dof(self):
        q, dq = self.robot.get_joint_states()
        M, _, _ = self.robot.calculate_dynamic_matrices(q, dq)
        M_inv = np.linalg.pinv(M)
        g_full_dof = np.zeros((2 * self.robot.dof, self.robot.dof))
        g_full_dof[self.robot.dof:, :] = M_inv
        return g_full_dof
    
    def compute_true_f_and_g(self, q, dq):
        M, g, C = self.robot.calculate_dynamic_matrices(q, dq)
        M_inv = np.linalg.pinv(M)

        f = np.zeros((2 * self.robot.dof, ))
        f[:self.robot.dof] = dq
        f[self.robot.dof:] = -M_inv @ (g + C)
        f = f.reshape((2 * self.robot.dof, 1))

        g_full_dof = np.zeros((2 * self.robot.dof, self.robot.dof))
        g_full_dof[self.robot.dof:, :] = M_inv
        g = g_full_dof @ self.B

        return f, g

    @property
    def f(self):
        q, dq = self.robot.get_joint_states()
        M, g, C = self.robot.calculate_dynamic_matrices(q, dq)
        M_inv = np.linalg.pinv(M)
        f = np.zeros((2 * self.robot.dof, ))
        f[:self.robot.dof] = dq
        f[self.robot.dof:] = -M_inv @ (g + C)
        f = f.reshape((2 * self.robot.dof, 1))
        return f
    
    @property
    def g(self):
        # g = g_full_dof @ B
        g_full_dof = self.get_g_full_dof()
        g = g_full_dof @ self.B
        return g

    @property
    def Xr(self):
        q, dq = self.robot.get_joint_states()
        return np.concatenate((q, dq)).reshape((-1, 1))

    @property
    def dot_Xr(self):
        if self.use_real_dot_xr:
            dot_Xr =  self.f + self.g @ self.target_torque.reshape((-1, 1))
            dot_Xr = np.squeeze(dot_Xr)
            return dot_Xr
        else:
            if self.last_Xr is None:
                self.last_Xr = self.Xr
                return np.zeros_like(self.last_Xr)
            else:
                dot_Xr = (self.Xr - self.last_Xr) / self.dT
                self.last_Xr = self.Xr
                return dot_Xr
            
    @property
    def Mr(self):
        end_eff_pos = self.robot.get_end_eff_pose()
        end_eff_vel = self.robot.get_end_eff_velocity()
        return np.concatenate((end_eff_pos, end_eff_vel)).reshape((-1, 1))
    
    @property
    def p_Mr_p_Xr(self):
        q, dq = self.robot.get_joint_states()
        return self.robot.get_p_Mr_p_Xr(q, dq)

    @property
    def Mh(self):
        obstacle_pos = np.asanyarray(self.obstacle_pose)
        obstacle_vel = np.array([0.0, 0.0, 0.0])
        return np.concatenate((obstacle_pos, obstacle_vel)).reshape((-1, 1))

    

if __name__ == '__main__':
    count = 0
    # env = PandaRodEnv(render_flag=False, goal_pose=[0.65, 0.1, 0.5], obstacle_pose=[0.5, 0.0, 0.4])
    env = PandaRodEnv(render_flag=True, goal_pose=[0.65, 0.1, 0.5], obstacle_pose=[0.5, 0.0, 0.4])
    for i in range(int(960)):
        # tau = env.compute_naive_torque()
        # obs, reward, done, info = env.step(tau)
        # time.sleep(0.1)
        print(i)
        count += 1
        u = env.compute_underactuated_torque()
        env.step(u)

        # q, dq = env.robot.get_joint_states()
        # q = th.tensor(q).repeat(3, 1).float()
        # dq = th.tensor(dq).repeat(3, 1).float()
        # u = th.tensor(u).repeat(3, 1).float()
        # Xr = th.tensor(to_np(env.Xr)).repeat(3, 1).float()

        # f, g = env.robot.dynamics_chain.get_f_and_g(q, dq)
        # dot_Xr = env.robot.dynamics_chain(Xr, u)

        # compare(dot_Xr[1], env.dot_Xr)
        # compare(f[1], env.f)
        # compare(g[2], env.g)


        # print(to_np(env.Xr)[9:])
        # print(to_np(u))
        # print(env.robot.get_contact_info())

        # if not env.robot.get_contact_info():
        #     env.save_nn_data()
        # else:
        #     print('--CONTACT!--')
        #     env.reset()
        #     count = 0

        # if count == 500:
        #     print('--TIMEOUT!--')
        #     env.reset()
        #     count = 0

        if env.if_success():
            print('--SUCCESS!--')
            break
            env.reset()
            count = 0
        
        time.sleep(0.1)



