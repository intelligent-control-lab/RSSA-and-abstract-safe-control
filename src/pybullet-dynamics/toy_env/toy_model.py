import pybullet as p
import pybullet_data
import os
import time
import numpy as np
import torch as th
from torch.autograd.functional import jacobian as pytorch_jacobian
import pytorch_kinematics as pk

from toy_kinematics_chain import ToyKinematicsChain
from toy_utils import *

class ToyModel:
    def __init__(self, physics_client_id):
        self.physics_client_id = physics_client_id
        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION
        robot_file_path = './src/pybullet-dynamics/toy_env/urdf/toy_model.urdf'
        self.id = p.loadURDF(
            robot_file_path, 
            basePosition=[0.0, 0.0, 0.0], 
            useFixedBase=True, 
            flags=flags, 
            physicsClientId=self.physics_client_id
        )
        self.dof = 3
        self.joint_ids = list(range(self.dof))
        self.end_eff_id = self.dof
        
        self.q_min = []
        self.q_max = []
        self.torque_limit = []
        self.velocity_limit = []
        self.q_target = []
        self.torque_target = []
        for j in self.joint_ids:
            joint_info = p.getJointInfo(self.id, j)
            self.q_min.append(joint_info[8])
            self.q_max.append(joint_info[9])
            self.torque_limit.append(joint_info[10])
            self.velocity_limit.append(joint_info[11])
            self.q_target.append((self.q_min[j] + self.q_max[j]) / 2)
            self.torque_target.append(0.0)
        self.q_min = np.asarray(self.q_min)
        self.q_max = np.asarray(self.q_max)
        self.torque_limit = np.asarray(self.torque_limit) * 0.1
        self.velocity_limit = np.asarray(self.velocity_limit)
        
        self.q_target = np.asarray(self.q_target)
        self.torque_target = np.asarray(self.torque_target)
        
        self.reset()

        self.kinematics_chain = ToyKinematicsChain()

    
    def reset(self):
        self.q_target = np.asarray([0.0] * self.dof)
        for j in self.joint_ids:
            p.resetJointState(self.id, j, targetValue=self.q_target[j])
        p.setJointMotorControlArray(
            bodyUniqueId=self.id,
            jointIndices=self.joint_ids,
            controlMode=p.VELOCITY_CONTROL,
            forces=[0.0 for _ in range(self.dof)],
            physicsClientId=self.physics_client_id
        )

    def set_joint_states(self, q_target, dq_target):
        for j in self.joint_ids:
            p.resetJointState(self.id, j, targetValue=q_target[j], targetVelocity=dq_target[j])
    
    def get_joint_states(self):
        joint_states = p.getJointStates(self.id, self.joint_ids, physicsClientId=self.physics_client_id)
        q = np.asarray([x[0] for x in joint_states])
        dq = np.asarray([x[1] for x in joint_states])
        return q, dq

    def set_target_torques(self, tau):
        p.setJointMotorControlArray(
            bodyUniqueId=self.id,
            jointIndices=self.joint_ids,
            controlMode=p.TORQUE_CONTROL,
            forces=tau,
            physicsClientId=self.physics_client_id
        )

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

    def solve_forward_dynamics(self, q, dq, tau):
        '''
        This function is used to calculate ddq when q, dq, tau is given.
        In the open chain case: 
        tau = M(q) @ ddq + C(q, dq) + g(q) ==> ddq = M(q)^(-1) @ (tau - C(q, dq) - g(q))
        '''
        M, g, C = self.calculate_dynamic_matrices(q, dq)
        M_inv = np.linalg.pinv(M)
        ddq = M_inv @ (tau - g - C)
        return ddq

    def get_end_eff_pose(self):
        info = p.getLinkState(self.id, self.end_eff_id, physicsClientId=self.physics_client_id)
        cartesian_pos = np.asarray(list(info[4]))[1:]   # just consider 2D situation: y-z
        return cartesian_pos

    def get_end_eff_velocity(self):
        end_eff_vel = p.getLinkState(self.id, self.end_eff_id, physicsClientId=self.physics_client_id, computeLinkVelocity=True)[6]
        end_eff_vel = np.asanyarray(end_eff_vel)[1:]    # just consider 2D situation: y-z
        return end_eff_vel

    def get_jacobian(self, q):
        zero_vec = [0.0] * self.dof
        jac_t, _ = p.calculateJacobian(
            self.id, self.end_eff_id, [0.0, 0.0, 0.0], list(q), zero_vec, zero_vec,
            physicsClientId=self.physics_client_id
        )
        J = np.asarray(jac_t)[1:, :]    # just consider 2D situation: y-z
        return J

    def calculate_end_eff_acceleration(self, q, dq, tau):
        ddq = self.solve_forward_dynamics(q, dq, tau)
        end_eff_acc = self.kinematics_chain.get_computed_end_eff_acceleration(q, dq, ddq)
        return end_eff_acc



if __name__ == '__main__':
    physics_client_id = p.connect(p.DIRECT)
    # physics_client_id = p.connect(p.GUI)
    p.setGravity(0, 0, 0)

    robot = ToyModel(physics_client_id=physics_client_id)
    tau = np.zeros(robot.dof)
    last_real_vel = np.zeros((2,))

    q, dq = robot.get_joint_states()

    for i in range(100):
        # q, dq = robot.get_joint_states()
        # print(q)
        # print(dq)
        
        real_pos = robot.get_end_eff_pose()
        calc_pos = robot.kinematics_chain.get_computed_end_eff_pose(q)
        # compare(real_pos, calc_pos)
        real_jac = robot.get_jacobian(q)
        calc_jac = robot.kinematics_chain.get_computed_end_eff_jacobian(q)
        compare(real_jac, calc_jac)
        real_vel = robot.get_end_eff_velocity()
        # print(real_vel)
        real_acc = (real_vel - last_real_vel) * 240
        last_real_vel = real_vel
        calc_acc = robot.calculate_end_eff_acceleration(q, dq, tau)
        # print(calc_acc)
        # print(real_acc)
        # compare(real_acc, calc_acc)

        M, g, C = robot.calculate_dynamic_matrices(q, dq)
        ddq = robot.solve_forward_dynamics(q, dq, tau)
        calc_tau = robot.solve_inverse_dynamics(q, dq, ddq)
        # compare(tau, cala_tau)

        q = q + 0.1
        dq = dq + 0.1
        # robot.set_joint_states(q, dq)
        tau += 0.01
        # robot.set_target_torques(tau)
        # p.stepSimulation()
        # time.sleep(1/240)
    





