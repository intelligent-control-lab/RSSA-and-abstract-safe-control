import numpy as np
import torch as th
from torch.autograd.functional import jacobian as pytorch_jacobian
import pytorch_kinematics as pk
import warnings

class ToyKinematicsChain:
    def __init__(self):
        robot_file_path = './src/pybullet-dynamics/toy_env/urdf/toy_model.urdf'
        self.kinematics_chain = pk.build_serial_chain_from_urdf(open(robot_file_path).read(), 'end_eff_link')

    def get_computed_end_eff_pose(self, q):
        q_tensor = th.tensor(q, dtype=th.float32)
        tg = self.kinematics_chain.forward_kinematics(q_tensor)
        tg = th.squeeze(tg.get_matrix())
        pos = tg[:3, 3]
        pos_2d = pos[1:]    # just consider 2D situation: y-z

        return pos_2d.numpy()

    def get_computed_end_eff_jacobian(self, q):
        '''
        2D situation: Jacobian is of the size (2 * self.dof)
        '''
        jk = self.get_FK_grad(q)
        return jk

    def get_computed_end_eff_velocity(self, q, dq):
        jk = self.get_computed_end_eff_jacobian(self, q)
        vec_2d = jk @ dq
        return vec_2d

    def get_computed_end_eff_acceleration(self, q, dq, ddq):
        '''
        A: end effector's acceleration
        V: end effector's velocity
        X: end effector's pose
        A = d_V_d_t
          = p_V_p_q @ p_q_p_t + p_V_p_dq @ p_dq_p_t
          = p_V_p_q @ dq + p_V_p_dq @ ddq
        '''
        p_V_p_q, p_V_p_dq = self.get_VFK_grad(q, dq)
        acc_2d = p_V_p_q @ dq + p_V_p_dq @ ddq
        return acc_2d

    def get_FK_grad(self, q):
        '''
        first get p_X_p_q, which is equal to jacobian (J)
        '''
        def func(q_tensor):
            tg = self.kinematics_chain.forward_kinematics(q_tensor)
            tg = th.squeeze(tg.get_matrix())
            pos = tg[:3, 3]
            pos_2d = pos[1:]    # just consider 2D situation: y-z
            return pos_2d
        
        q_tensor = th.tensor(q, requires_grad=True, dtype=th.float32)
        p_X_p_q = pytorch_jacobian(func, q_tensor)
        return p_X_p_q.numpy()
    
    def get_VFK_grad(self, q, dq):
        '''
        second get p_V_p_q and p_V_p_dq
        where p_V_p_dq = p_X_p_q = J and p_V_p_q = p_J_p_q @ dq 
        '''
        def func(q_tensor, dq_tensor):
            def func_inner(q_tensor):
                tg = self.kinematics_chain.forward_kinematics(q_tensor)
                tg = th.squeeze(tg.get_matrix())
                pos = tg[:3, 3]
                pos_2d = pos[1:]    # just consider 2D situation: y-z
                return pos_2d
            jacobian = pytorch_jacobian(func_inner, q_tensor, create_graph=True)
            vel_2d = jacobian @ dq_tensor
            return vel_2d

        q_tensor = th.tensor(q, requires_grad=True, dtype=th.float32)
        dq_tensor = th.tensor(dq, requires_grad=True, dtype=th.float32)

        p_V_p_q, p_V_p_dq = pytorch_jacobian(func, (q_tensor, dq_tensor))
        if not th.max(p_V_p_q).item() < 1e5:
            p_X_p_q = self.get_FK_grad(q)
            p_V_p_dq = p_X_p_q
            p_V_p_q = np.zeros_like(p_X_p_q)
            warnings.warn('NAN in p_V_p_q!')
            return p_V_p_q, p_V_p_dq

        return p_V_p_q.numpy(), p_V_p_dq.numpy()