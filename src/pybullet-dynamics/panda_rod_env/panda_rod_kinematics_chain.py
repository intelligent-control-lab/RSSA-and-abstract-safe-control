import warnings
import numpy as np
import torch as th
from torch.autograd.functional import jacobian as pytorch_jacobian

import pytorch_kinematics as pk

# from panda_rod_utils import compare

class PandaRodKinematicsChain:
    def __init__(self, robot_file_path):
        robot_file_path = robot_file_path
        self.kinematics_chain = pk.build_serial_chain_from_urdf(open(robot_file_path).read(), 'panda_eff_link')
        self.VFK_grad_random_num_list = np.array([
            0.2, 0.4, 0.6, 0.8, 1.0
        ])
        self.VFK_grad_random_num = 0.5
        self.device = 'cpu'
        self.kinematics_chain = self.kinematics_chain.to(device=self.device)

    def get_computed_end_eff_pose(self, q):
        q_tensor = th.tensor(q, requires_grad=True, dtype=th.float32)
        tg = self.kinematics_chain.forward_kinematics(q_tensor)
        tg = th.squeeze(tg.get_matrix())
        pos = tg[:3, 3]

        return pos.detach().numpy()

    def get_computed_end_eff_jacobian(self, q):
        jk = self.get_FK_grad(q)
        return jk
    
    def get_computed_end_eff_velocity(self, q, dq):
        jk = self.get_computed_end_eff_jacobian(q)
        vec = jk @ dq
        return vec

    def get_computed_end_eff_acceleration(self, q, dq, ddq):
        p_V_p_q, p_V_p_dq = self.get_VFK_grad(q, dq)
        acc = p_V_p_q @ dq + p_V_p_dq @ ddq
        return acc
    
    def get_FK_grad(self, q):
        '''
        first get p_X_p_q, which is equal to jacobian (J)
        '''
        def func(q_tensor):
            tg = self.kinematics_chain.forward_kinematics(q_tensor)
            tg = th.squeeze(tg.get_matrix())
            pos = tg[:3, 3]
            return pos
        
        q_tensor = th.tensor(q, requires_grad=True, dtype=th.float32, device=self.device)
        p_X_p_q = pytorch_jacobian(func, q_tensor)
        return p_X_p_q.cpu().detach().numpy()
    
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
                return pos
            jacobian = pytorch_jacobian(func_inner, q_tensor, create_graph=True)
            vel = jacobian @ dq_tensor
            return vel 

        q_tensor = th.tensor(q, requires_grad=True, dtype=th.float32, device=self.device)
        dq_tensor = th.tensor(dq, requires_grad=True, dtype=th.float32, device=self.device)

        try:
            p_V_p_q, p_V_p_dq = pytorch_jacobian(func, (q_tensor, dq_tensor))
            p_V_p_q = p_V_p_q.cpu().detach().numpy()
            p_V_p_dq = p_V_p_dq.cpu().detach().numpy()
            if not (np.max(p_V_p_q) < 1e5):
                raise Exception('NAN in p_V_p_q!')
        except:
            p_X_p_q = self.get_FK_grad(q)
            p_V_p_dq = p_X_p_q
            p_V_p_q = np.zeros_like(p_X_p_q)
            warnings.warn('NAN in p_V_p_q!')

        # th.cuda.empty_cache()
        return p_V_p_q, p_V_p_dq
    


            