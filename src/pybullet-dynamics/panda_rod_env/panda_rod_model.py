import pybullet as p
import pybullet_data
import os
import time
import numpy as np
import torch as th
from torch.autograd.functional import jacobian as pytorch_jacobian

try:
    from panda_rod_kinematics_chain import PandaRodKinematicsChain
    # from learned_dynamics.panda_rod_dynamics_chain import PandaRodDynamicsChain
    from panda_rod_utils import *
except:
    from panda_rod_env.panda_rod_kinematics_chain import PandaRodKinematicsChain
    # from panda_rod_env.learned_dynamics.panda_rod_dynamics_chain import PandaRodDynamicsChain
    from panda_rod_env.panda_rod_utils import *


import warnings

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

th.set_printoptions(
    precision=3,    
    threshold=1000,
    edgeitems=3,
    linewidth=150,  
    profile=None,
    sci_mode=False  
)

class PandaRodModel:
    def __init__(
        self, 
        physics_client_id, 
        robot_file_path='./src/pybullet-dynamics/panda_rod_env/urdf/panda_without_rod.urdf',
        dof=7,
        end_eff_id=7,
    ):
        self.physics_client_id = physics_client_id  
        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION
        robot_file_path = robot_file_path
        self.id = p.loadURDF(
            robot_file_path, 
            basePosition=[0.0, 0.0, 0.0], 
            useFixedBase=True, 
            flags=flags, 
            physicsClientId=self.physics_client_id
        )
        self.plane_id = p.loadURDF(
            os.path.join(pybullet_data.getDataPath(), 'plane.urdf'), 
            physicsClientId=self.physics_client_id
        )
        self.dof = dof
        self.joint_ids = list(range(self.dof))
        self.end_eff_id = end_eff_id
        self.jac_id = self.end_eff_id - 1   # facilitate the computation of real jacobian
        
        self.q_min = []
        self.q_max = []
        self.torque_limit = []
        self.velocity_limit = []
        self.torque_target = []
        for j in self.joint_ids:
            joint_info = p.getJointInfo(self.id, j)
            self.q_min.append(joint_info[8])
            self.q_max.append(joint_info[9])
            self.torque_limit.append(joint_info[10])
            self.velocity_limit.append(joint_info[11])
            self.torque_target.append(0.0)
        self.q_min = np.asarray(self.q_min)
        self.q_max = np.asarray(self.q_max)
        self.torque_limit = np.asarray(self.torque_limit)
        self.velocity_limit = np.asarray(self.velocity_limit)
        
        self.q_target = self.q_init = np.array([
            0, np.deg2rad(-60), 0, np.deg2rad(-150), 0, np.deg2rad(90), np.deg2rad(45), # Panda joints
            # -np.deg2rad(45), 0,  # rod joints
        ])
        self.q_target = self.q_target[:self.dof]
        self.q_init = self.q_init[:self.dof]

        self.torque_target = np.asarray(self.torque_target)
        
        self.Kp = 16 * np.eye(self.dof)
        # self.Kp[7, 7] = self.Kp[8, 8] = 10
        self.Kd = 8 * np.eye(self.dof)

        # prepare for getting real jacobian
        end_eff_link_info = p.getLinkState(self.id, self.jac_id, self.physics_client_id)
        # suppose end effector is geometircally symmetric
        self.end_eff_link_local_pos = list(np.asarray(end_eff_link_info[2]) * 1) 
        end_eff_link_local_ori = end_eff_link_info[3]
        self.end_eff_link_local_transform_matrix = self.get_transfrom_matirx(self.end_eff_link_local_pos, end_eff_link_local_ori)
        
        self.kinematics_chain = PandaRodKinematicsChain(robot_file_path=robot_file_path)
        # self.dynamics_chain = PandaRodDynamicsChain(self.physics_client_id, robot_id=self.id)

        self.reset()

    def get_transfrom_matirx(self, position, quaternion):
        transform_matrix = np.zeros((4, 4))
        transform_matrix[:3, 3] = position
        rotation_matrix = p.getMatrixFromQuaternion(quaternion, self.physics_client_id)
        transform_matrix[:3, :3] = np.array(list(rotation_matrix)).reshape(3, 3)
        transform_matrix[3, 3] = 1
        return transform_matrix

    def reset(self):
        for j in self.joint_ids:
            p.resetJointState(self.id, j, targetValue=self.q_init[j])
        self.reset_controller()
    
    def reset_controller(self):
        p.setJointMotorControlArray(
            bodyUniqueId=self.id,
            jointIndices=self.joint_ids,
            controlMode=p.VELOCITY_CONTROL,
            forces=[0.0 for _ in range(self.dof)],
            physicsClientId=self.physics_client_id
        )
    
    def set_target_torques(self, torque_target):
        self.torque_target = torque_target
        p.setJointMotorControlArray(
            bodyUniqueId=self.id,
            jointIndices=self.joint_ids,
            controlMode=p.TORQUE_CONTROL,
            forces=self.torque_target,
            physicsClientId=self.physics_client_id
        )
    
    def get_joint_states(self):
        joint_states = p.getJointStates(self.id, self.joint_ids, physicsClientId=self.physics_client_id)
        q = np.asarray([x[0] for x in joint_states])
        dq = np.asarray([x[1] for x in joint_states])
        return q, dq
    
    def set_joint_states(self, q_target, dq_target):
        for j in self.joint_ids:
            p.resetJointState(self.id, j, targetValue=q_target[j], targetVelocity=dq_target[j])

    def solve_inverse_dynamics(self, q, dq, ddq):
        inertia_matrix, gravity_vec, coriolois_vec = self.calculate_dynamic_matrices(q, dq)
        tau = inertia_matrix @ ddq + coriolois_vec + gravity_vec 
        return tau
        
    def calculate_dynamic_matrices(self, q, dq):
        q = q.tolist()
        dq = dq.tolist()
        inertia_matrix = np.asarray(
            p.calculateMassMatrix(self.id, q, physicsClientId=self.physics_client_id)
        )
        gravity_vec = np.asarray(
            p.calculateInverseDynamics(self.id, q, [0.0] * self.dof, [0.0] * self.dof, physicsClientId=self.physics_client_id)
        )
        coriolois_vec = np.asarray(
            p.calculateInverseDynamics(self.id, q, dq, [0.0] * self.dof, physicsClientId=self.physics_client_id)
        ) - gravity_vec
        return inertia_matrix, gravity_vec, coriolois_vec

    def get_end_eff_pose(self):
        end_eff_pos = p.getLinkState(self.id, self.end_eff_id, physicsClientId=self.physics_client_id)[4]
        return end_eff_pos

    def get_end_eff_orientation(self):
        end_eff_ori = p.getLinkState(self.id, self.end_eff_id, physicsClientId=self.physics_client_id)[5]
        return end_eff_ori
    
    def get_end_eff_velocity(self):
        end_eff_vel = p.getLinkState(self.id, self.end_eff_id, physicsClientId=self.physics_client_id, computeLinkVelocity=True)[6]
        return end_eff_vel

    def solve_inverse_kinematics(self, end_eff_target_pos, end_eff_target_ori=[1.0, 0.0, 0.0, 0.0]):
        q_target = p.calculateInverseKinematics(
            self.id,
            self.end_eff_id,
            end_eff_target_pos,
            end_eff_target_ori,
            maxNumIterations=50,
            residualThreshold=0.001,
            physicsClientId=self.physics_client_id
        )
        return np.asarray(q_target)

    def computed_torque_control(self, q_d, dq_d):
        q, dq = self.get_joint_states()
        fake_ddq = self.Kp @ (q_d - q) + self.Kd @ (dq_d - dq)
        tau = self.solve_inverse_dynamics(q, dq, fake_ddq)
        return tau

    def pd_control(self, q_d, dq_d):
        q, dq = self.get_joint_states()
        tau = self.Kp @ (q_d - q) + self.Kd @ (dq_d - dq) 
        return tau
    
    def operational_space_control(self, d_end_eff_pos, d_end_eff_ori):
        impedance_Kp = np.array([400.0] * 6)
        impedance_Kd = 2.0 * np.sqrt(impedance_Kp)
        end_eff_state = p.getLinkState(self.id, self.end_eff_id, computeLinkVelocity=1, physicsClientId=self.physics_client_id)
        x = end_eff_state[4]
        print(f'x: {x}')
        r = end_eff_state[5]  
        lin_vel = end_eff_state[6]
        ang_vel = end_eff_state[7]
        orientation_error = calculate_orientation_error(
            np.asanyarray(p.getMatrixFromQuaternion(r, physicsClientId=self.physics_client_id)).reshape((3, 3)),
            np.asanyarray(p.getMatrixFromQuaternion(d_end_eff_ori, physicsClientId=self.physics_client_id)).reshape((3, 3))
        )
        position_error = np.asarray(d_end_eff_pos) - np.asarray(x)
        desired_end_eff_force = impedance_Kp[0:3] * np.array(position_error) - impedance_Kd[0:3] * np.array(lin_vel)
        desired_end_eff_torque = impedance_Kp[3:6] * np.array(orientation_error) - impedance_Kd[3:6] * np.array(ang_vel)
        wrench = np.concatenate([desired_end_eff_force, desired_end_eff_torque])
        J = self.get_jacobian()
        q, dq = self.get_joint_states()
        _, g, C = self.calculate_dynamic_matrices(q, dq)
        tau_prior_task = J.T @ wrench + g + C

        second_task_Kp = 50.0
        second_task_Kd = 2.0 * np.sqrt(second_task_Kp)
        q_target = self.solve_inverse_kinematics(d_end_eff_pos, d_end_eff_ori)
        
        tau_second_task = second_task_Kp * (q_target - q) - second_task_Kd * dq

        # tau = tau_prior_task + tau_second_task
        tau = tau_prior_task
        return tau

    def get_jacobian(self):
        q, _ = self.get_joint_states()
        zero_vec = [0.0] * self.dof
        
        # jac_t, jac_r = p.calculateJacobian(
        #     self.id, self.jac_id, list(self.end_eff_link_local_pos), list(q), zero_vec, zero_vec,
        #     physicsClientId=self.physics_client_id
        # )
        # J_t = np.asarray(jac_t)
        # J_r = np.asarray(jac_r)
        # J = np.concatenate((J_t, J_r), axis=0)

        J_1_t, J_1_r = p.calculateJacobian(
            self.id, self.end_eff_id, [0.0, 0.0, 0.0], list(q), zero_vec, zero_vec,
            physicsClientId=self.physics_client_id
        )
        J_1_t = np.asarray(J_1_t)
        J_1_r = np.asarray(J_1_r)
        J_1 = np.concatenate((J_1_t, J_1_r), axis=0)
        # compare(J, J_1)

        return J_1

    def get_contact_info(self):
        point = p.getContactPoints(physicsClientId=self.physics_client_id)
        if len(point) > 0:
            return True
        else:
            return False

    def get_p_Mr_p_Xr(self, q, dq):
        '''
        Mr is of 6-dim (eff-X and eff-V)
        Xr is of (2*dof)-dim (q and dq)
        '''
        try:
            p_X_p_q = self.kinematics_chain.get_FK_grad(q)
            p_V_p_q, p_V_p_dq = self.kinematics_chain.get_VFK_grad(q, dq)
        except ValueError:
            p_V_p_dq = p_X_p_q
            p_V_p_q = np.zeros_like(p_X_p_q)
            warnings.warn('NAN in p_V_p_q!')
        p_Mr_p_Xr = np.zeros((6, 2 * self.dof))
        p_Mr_p_Xr[:3, :self.dof] = p_X_p_q
        p_Mr_p_Xr[3:, :self.dof] = p_V_p_q
        p_Mr_p_Xr[3:, self.dof:] = p_V_p_dq
        return p_Mr_p_Xr

    def get_rod_mass(self, link_id=None):
        if link_id is None:
            link_id = self.jac_id
        return p.getDynamicsInfo(self.id, link_id, physicsClientId=self.physics_client_id)[0]

    def change_rod_mass(self, link_id, mass):
        p.changeDynamics(self.id, link_id, mass=mass, physicsClientId=self.physics_client_id)

    
if __name__ == '__main__':
    physics_client_id = p.connect(p.DIRECT)
    # physics_client_id = p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    # p.loadURDF(os.path.join(pybullet_data.getDataPath(), 'plane.urdf'))
    
    robot = PandaRodModel(physics_client_id=physics_client_id)
    target_pos = [0.6, 0.3, 0.5]
    # p.stepSimulation()

    # print(robot.get_end_rod_mass())
    # robot.change_end_rod_mass(0.2)
    # print(robot.get_end_rod_mass())

    for i in range(480):
        q, dq = robot.get_joint_states()
        # print(dq)
        # print(robot.get_end_eff_pose())
        # print(robot.kinematics_chain.get_computed_end_eff_pose(q))

        real_pos = robot.get_end_eff_pose()
        calc_pos = robot.kinematics_chain.get_computed_end_eff_pose(q)
        compare(real_pos, calc_pos)
        real_vel = robot.get_end_eff_velocity()
        calc_vel = robot.kinematics_chain.get_computed_end_eff_velocity(q, dq)
        compare(real_vel, calc_vel)
        print(robot.kinematics_chain.get_computed_end_eff_acceleration(q, dq, np.zeros_like(q)))

        target_q = robot.solve_inverse_kinematics(target_pos, [1.0, 0.0, 0.0, 0.0])
        target_dq = [0.0] * robot.dof
        tau = robot.computed_torque_control(target_q, target_dq)
        # print(tau)
        robot.set_target_torques(tau)
        # print(robot.get_end_eff_pose())
        # print(robot.get_end_eff_orientation())
        
        # p_Mr_p_Xr = robot.get_p_Mr_p_Xr(q, dq)

        # robot.get_jacobian()
        # compare(robot.get_jacobian()[:3, :], robot.kinematics_chain.get_computed_end_eff_jacobian(q))

        # if compare(real_pos, target_pos) < 2e-2:
        #     print(i)
        #     break
        # print(robot.get_contact_info())
        # M, g, C = robot.calculate_dynamic_matrices(q, dq)

        # q = th.tensor(q).repeat(3, 1).float()
        # dq = th.tensor(dq).repeat(3, 1).float()
        # M_th, g_th, C_th = robot.dynamics_chain.calculate_dynamic_matrices(q, dq)
        # compare(M_th[0], M, ratio_flag=False)
        # compare(g_th[1], g, ratio_flag=False)
        # compare(C_th[2], C, ratio_flag=False)

        # f, g = robot.dynamics_chain.get_f_and_g(q, dq)

        p.stepSimulation()
        time.sleep(0.1)
    
    p.disconnect()

    
