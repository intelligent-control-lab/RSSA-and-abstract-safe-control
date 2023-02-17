import numpy as np
import time
import datetime
import pickle

from panda_rod_env import PandaRodEnv

class PandaRodDataCollection:
    def __init__(
        self,
        workspace_limit={'x': [0.3, 0.8], 'y': [-0.3, 0.3], 'z': [0.3, 0.7]},
        store_nn_data_path='./src/pybullet-dynamics/panda_rod_env/data/raw_data_diff/',
        total_steps=int(1e6),
        max_inner_step=960,
        tolerate_error=0.05,
    ):
        self.workspace_limit = workspace_limit
        self.env = PandaRodEnv(store_nn_data_path=store_nn_data_path)
        self.max_inner_step = max_inner_step
        self.total_steps=total_steps
        self.tolerate_error = tolerate_error
        self.store_nn_data_path = store_nn_data_path

        self.current_inner_step = 0

        self.restart()
    
    def restart(self):
        self.current_inner_step = 0

        # initial pose of the robot
        random_init_x = self.env.np_random.uniform(
                low=self.workspace_limit['x'][0],
                high=self.workspace_limit['x'][1],
            )
        random_init_y = self.env.np_random.uniform(
            low=self.workspace_limit['y'][0],
            high=self.workspace_limit['y'][1],
        )
        random_init_z = self.env.np_random.uniform(
            low=self.workspace_limit['z'][0],
            high=self.workspace_limit['z'][1],
        )
        random_init_pose = [random_init_x, random_init_y, random_init_z]
        self.init_pos = np.asanyarray(random_init_pose)

        self.env.robot.q_init = self.env.robot.solve_inverse_kinematics(random_init_pose)
        self.env.robot.reset()

        # position of the goal
        random_goal_x = self.env.np_random.uniform(
                low=self.workspace_limit['x'][0],
                high=self.workspace_limit['x'][1],
            )
        random_goal_y = self.env.np_random.uniform(
            low=self.workspace_limit['y'][0],
            high=self.workspace_limit['y'][1],
        )
        random_goal_z = self.env.np_random.uniform(
            low=self.workspace_limit['z'][0],
            high=self.workspace_limit['z'][1],
        )
        self.env.goal_pose = self.goal_pos = np.array([random_goal_x, random_goal_y, random_goal_z])

    def data_collect(self):
        for _ in range(self.total_steps):
            self.current_inner_step += 1

            u = self.env.compute_underactuated_torque()
            self.env.step(u)

            if self.env.robot.get_contact_info():
                print('--COLLISON!--')
                self.restart()
            elif self.env.if_success():
                print('--SUCCESS!--')
                self.restart()
            elif self.current_inner_step >= self.max_inner_step:
                print('--TIMEOUT!--')
                self.restart()
            else:
                self.env.save_nn_data()


class PandaRodDataCollectionRandom:
    def __init__(
        self, 
        workspace_limit={'x': [0.3, 0.8], 'y': [-0.3, 0.3], 'z': [0.3, 0.7]},
        store_nn_data_path='./src/pybullet-dynamics/panda_rod_env/data/raw_data_diff/',
        store_nn_data_path_random='./src/pybullet-dynamics/panda_rod_env/data/nn_raw_data_diff_1/',
        store_nn_data_path_zero='./src/pybullet-dynamics/panda_rod_env/data/nn_raw_data_env_3/zero/',
        total_steps=int(1e6),
        max_inner_step=960,
        tolerate_error=0.05,
        zero_torque_rate=0,
    ):
        self.workspace_limit = workspace_limit
        self.env = PandaRodEnv()
        self.max_inner_step = max_inner_step
        self.total_steps=total_steps
        self.tolerate_error = tolerate_error
        self.zero_torque_rate = zero_torque_rate
        self.random_path = store_nn_data_path_random
        self.zero_path = store_nn_data_path_zero

        self.store_nn_data_path = store_nn_data_path

        self.current_inner_step = 0
        self.restart()
        

    def restart(self):
        self.current_inner_step = 0

        # initial pose of the robot
        random_init_x = self.env.np_random.uniform(
                low=self.workspace_limit['x'][0],
                high=self.workspace_limit['x'][1],
            )
        random_init_y = self.env.np_random.uniform(
            low=self.workspace_limit['y'][0],
            high=self.workspace_limit['y'][1],
        )
        random_init_z = self.env.np_random.uniform(
            low=self.workspace_limit['z'][0],
            high=self.workspace_limit['z'][1],
        )
        random_init_pose = [random_init_x, random_init_y, random_init_z]
        self.init_pos = np.asanyarray(random_init_pose)

        self.env.robot.q_init = self.env.robot.solve_inverse_kinematics(random_init_pose)
        self.env.robot.reset()

    def get_random_u(self):
        random_u = self.env.np_random.uniform(
            low=-self.env.max_u,
            high=self.env.max_u
        )
        return random_u
    
    def data_collect(self):
        for _ in range(self.total_steps):
            self.current_inner_step += 1

            if self.env.np_random.binomial(1, self.zero_torque_rate):
                u = np.zeros_like(self.env.max_u)
                self.env.store_nn_data_path = self.zero_path
            else:
                u = self.get_random_u()
                self.env.store_nn_data_path = self.random_path

            self.env.step(u)
            self.env.target_torque = u

            if self.env.robot.get_contact_info():
                print('--COLLISON!--')
                self.restart()
            elif self.current_inner_step >= self.max_inner_step:
                print('--TIMEOUT!--')
                self.restart()
            else:
                self.env.save_nn_data()

    def data_collect_actificial(self):
        q_max = self.env.robot.q_max
        q_min = self.env.robot.q_min
        dq_max = self.env.robot.velocity_limit
        dq_min = -self.env.robot.velocity_limit
        u_max = self.env.robot.torque_limit[:self.env.actuated_dof]
        u_min = -self.env.robot.torque_limit[:self.env.actuated_dof]

        np.random.seed(int(time.time()))

        for i in range(self.total_steps):
            print(i)    
            q = np.random.uniform(low=q_min, high=q_max)
            dq = np.random.uniform(low=dq_min, high=dq_max)
            u = np.random.uniform(low=u_min, high=u_max)
            f, g = self.env.robot.dynamics_chain.compute_true_f_and_g(q, dq)
            ddq = (f + g @ u)[self.env.robot.dof:]
            self.save_nn_data(q, dq, ddq, u)

    def save_nn_data(self, q, dq, ddq, u):
        raw_data = {
            'x': np.concatenate((q, dq)),
            'dot_x': np.concatenate((dq, ddq)),
            'u': u,
        }
        file_name = self.store_nn_data_path + str(time.time()) + '.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(raw_data, file)


if __name__ == '__main__':
    data_collection = PandaRodDataCollectionRandom()
    data_collection.data_collect_actificial()

