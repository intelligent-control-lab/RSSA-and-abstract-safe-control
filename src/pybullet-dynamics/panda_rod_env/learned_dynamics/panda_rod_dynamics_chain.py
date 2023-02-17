from shutil import get_archive_formats
import pybullet as p
import os
import time
import numpy as np
import torch as th
from torch.autograd.functional import jacobian as pytorch_jacobian

try:
    from learned_dynamics.core import *
except Exception:
    from core import *

class PandaRodDynamicsChain:
    def __init__(
        self,
        physics_client_id,
        robot_id=None,
        end_eff_id=7, 
        q_init=None,
        dof=7,
        actuated_dof=7,
    ):
        self.physics_client_id = physics_client_id  

        if robot_id is None:
            flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION
            robot_file_path = './src/pybullet-dynamics/panda_rod_env/urdf/panda_with_rod.urdf'
            self.robot_id = p.loadURDF(
                robot_file_path, 
                basePosition=[0.0, 0.0, 0.0], 
                useFixedBase=True, 
                flags=flags, 
                physicsClientId=self.physics_client_id
            )
        else:
            self.robot_id = robot_id

        self.dof = self.end_eff_id = end_eff_id
        self.joint_ids = list(range(self.dof))

        if q_init is None:
            self.q_init = np.array([
                0, np.deg2rad(-60), 0, np.deg2rad(-150), 0, np.deg2rad(90), np.deg2rad(45), # Panda joints
                # -np.deg2rad(45), 0,  # rod joints
            ])
        else:
            self.q_init = np.asanyarray(q_init)

        for j in self.joint_ids:
            p.resetJointState(self.robot_id, j, targetValue=self.q_init[j])

        self.dof = dof
        self.actuated_dof = actuated_dof
        self.B = np.zeros((self.dof, self.actuated_dof))
        self.B[:self.actuated_dof, :self.actuated_dof] = np.eye(self.actuated_dof)
        
        self.compute_dynamics_param()
        # self.compute_dynamics_param_closed_form()


    def compute_dynamics_param(self):
        '''
        calculate the params for inverse dynamics
        get: Mlist, Glist, Slist, Alist
        '''
        # get Glist
        Glist = []
        com_Ad_T_list = [] # help to calculate G
        for i in self.joint_ids:
            dynamics_info = p.getDynamicsInfo(self.robot_id, i, physicsClientId=self.physics_client_id)
            mass = dynamics_info[0]
            local_inertia_diagonal = dynamics_info[2]
            local_inertia_pos = dynamics_info[3]
            local_inertia_orn = np.array(p.getMatrixFromQuaternion(dynamics_info[4])).reshape(3, 3)
            local_inertia_mat = np.diag(
                list(local_inertia_diagonal) + [mass] * 3
            )

            trans = rp_to_trans(
                local_inertia_orn,
                np.array(local_inertia_pos)
            )
            link_inertia_mat = adjoint(trans_inv(trans)).T @ local_inertia_mat @ adjoint(trans_inv(trans))
            Glist.append(link_inertia_mat)
            com_Ad_T_list.append(adjoint(trans_inv(trans)))
            # print(link_inertia_mat)
        
        self.Glist = np.asanyarray(Glist)
        self.com_Ad_T_list = np.asanyarray(com_Ad_T_list)

        # get Mlist
        Mlist_w2l = []
        Mlist = []
        Mlist_w2l.append(np.eye(4)) # for base link (id = -1)

        for i in (self.joint_ids + [self.end_eff_id]):
            link_state = p.getLinkState(self.robot_id, i, physicsClientId=self.physics_client_id)
            R_w2l = np.array(p.getMatrixFromQuaternion(link_state[5])).reshape(3, 3)
            P_w2l = np.array(link_state[4])
            T_w2l = rp_to_trans(R_w2l, P_w2l)
            Mlist_w2l.append(T_w2l)
        
        for i in range(1, len(Mlist_w2l)):
            Mlist.append(
                trans_inv(Mlist_w2l[i - 1]) @ Mlist_w2l[i]
            )
            # print(Mlist[-1])
        self.Mlist = np.asanyarray(Mlist)

        # Slist
        Slist = []
        for i in self.joint_ids:
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client_id)
            joint_axis = np.array(joint_info[-4])
            link_state = p.getLinkState(self.robot_id, i, physicsClientId=self.physics_client_id)
            R_w2l = np.array(p.getMatrixFromQuaternion(link_state[5])).reshape(3, 3)
            P_w2l = np.array(link_state[4])
            T_w2l = rp_to_trans(R_w2l, P_w2l)

            T_w2j = T_w2l
            R_w2j, P_w2j = trans_to_rp(T_w2j)
            omg = R_w2j @ joint_axis
            v = np.cross(P_w2j, omg)
            Slist.append(np.concatenate((omg, v), axis=0))
            # print(Slist[-1])
            # print('\n')
        
        self.Slist = np.asanyarray(Slist)

        # Alist
        self.Alist = np.zeros_like(self.Slist)
        M_tmp = np.eye(4)
        for i in self.joint_ids:
            M_tmp = M_tmp @ self.Mlist[i]
            self.Alist[i] = adjoint(trans_inv(M_tmp)) @ self.Slist[i]


    def inverse_dynamics(self, q, dq, ddq, g, Ftip=np.zeros(6)):
        jn = len(self.joint_ids)
        T_list = np.zeros((jn + 1, 4, 4))
        v_list = np.zeros((jn, 6))
        a_list = np.zeros((jn, 6))
        v_tmp = np.zeros(6)
        a_tmp = np.concatenate((np.zeros(3), -g))

        # forward iterations
        for i in range(jn):
            T_list[i] = matrix_exp6(-vec_to_se3(self.Alist[i]) * q[i]) @ trans_inv(self.Mlist[i])
            # print(T_list[i])
            v_list[i] = adjoint(T_list[i]) @ v_tmp + self.Alist[i] * dq[i]
            v_tmp = v_list[i]
            a_list[i] = adjoint(T_list[i]) @ a_tmp + ad(v_tmp) @ self.Alist[i] * dq[i] + self.Alist[i] * ddq[i]
            a_tmp = a_list[i]
        
        # backward iterations
        T_list[-1] = trans_inv(self.Mlist[-1])
        F_tmp = Ftip
        tau = np.zeros(jn)
        for i in reversed(range(jn)):
            F_tmp = adjoint(T_list[i + 1]).T @ F_tmp + self.Glist[i] @ a_list[i] - ad(v_list[i]).T @ (self.Glist[i] @ v_list[i])
            tau[i] = F_tmp.T @ self.Alist[i]

        return tau    

    def calculate_dynamic_matrices(self, q, dq):
        '''
        get inertia_matrix, gravity_vec, coriolois_vec
        IMPORTANT: q should minus q_init!
        '''
        q = q - self.q_init
        inertia_matrix = self.mass_matrix(q)
        coriolois_vec = self.vel_quadratic_forces(q, dq)
        gravity_vec = self.gravity_forces(q)
        return inertia_matrix, gravity_vec, coriolois_vec

    def mass_matrix(self, q):
        M = np.zeros((self.dof, self.dof))
        for i in range(self.dof):
            ddq = [0.0] * self.dof
            ddq[i] = 1.0
            M[:, i] = self.inverse_dynamics(
                q=q, dq=[0.0]*self.dof, ddq=ddq,
                g=np.zeros(3)
            )
        return M

    def vel_quadratic_forces(self, q, dq):
        C = self.inverse_dynamics(
            q=q, dq=dq, ddq=[0.0]*self.dof, 
            g=np.zeros(3)
        )
        return C

    def gravity_forces(self, q, g=np.array([0, 0, -9.8])):
        return self.inverse_dynamics(
            q=q, dq=[0.0]*self.dof, ddq=[0.0]*self.dof,
            g=g
        )

    def compute_true_f_and_g(self, q, dq):
        M, g, C = self.calculate_dynamic_matrices(q, dq)
        M_inv = np.linalg.pinv(M)

        f = np.zeros((2 * self.dof, ))
        f[:self.dof] = dq
        f[self.dof:] = -M_inv @ (g + C)

        g_full_dof = np.zeros((2 * self.dof, self.dof))
        g_full_dof[self.dof:, :] = M_inv
        g = g_full_dof @ self.B

        return f, g


    def compute_dynamics_param_closed_form(self):
        '''
        calculate the params for inverse dynamics in closed form,
        use: Mlist, Glist, Slist, Alist.
        get: A_mat, G_mat.
        '''
        n = self.dof

        self.A_mat = np.zeros((6*n, n))
        for i in range(n):
            self.A_mat[i*6: i*6+6, i] = self.Alist[i]
        
        self.G_mat = np.zeros((6*n, 6*n))
        for i in range(n):
            self.G_mat[i*6: i*6+6, i*6: i*6+6] = self.Glist[i]
        
    def get_adjoint_T_list(self, q):
        adjoint_T_list = []
        for i in range(len(q)):
            T_tmp = matrix_exp6(-vec_to_se3(self.Alist[i]) * q[i]) @ trans_inv(self.Mlist[i])
            adjoint_T_tmp = adjoint(T_tmp)
            adjoint_T_list.append(adjoint_T_tmp)
        return adjoint_T_list
    
    def get_L_and_W(self, adjoint_T_list):
        n = self.dof
        
        W = np.zeros((6*n, 6*n))
        for i in range(1, n):
            W[i*6 : i*6+6, i*6-6 : i*6] = adjoint_T_list[i]
        
        L = np.zeros((6*n, 6*n))
        for i in range(1, n):
            adjoint_T_tmp = adjoint_T_list[i]
            for j in range(i - 1, -1, -1):
                L[i*6 : i*6+6, j*6 : j*6+6] = adjoint_T_tmp
                adjoint_T_tmp = adjoint_T_tmp @ adjoint_T_list[j]
        for i in range(n):
            L[6*i : 6*i+6, 6*i : 6*i+6] = np.eye(6)
        
        return L, W
    
    def get_V_base_and_dot_V_base(self, adjoint_T_list, g=np.array([0, 0, -9.8])):
        n = self.dof
        V_base = np.zeros((6*n, 1))

        dot_V_0 = np.concatenate((np.zeros(3), -g))
        dot_V_base = np.zeros((6*n, 1))
        dot_V_base[:6, 0] = adjoint_T_list[0] @ dot_V_0

        return V_base, dot_V_base

    def get_V_map(self, V):
        n = self.dof
        assert V.shape[0] == 6 * n

        V_map = np.zeros((6*n, 6*n))
        for i in range(n):
            V_map[6*i : 6*i+6, 6*i : 6*i+6] = ad(V[6*i: 6*i+6, 0])
        return V_map

    def calculate_dynamic_matrices_closed_form(self, q, dq):
        '''
        get inertia_matrix, gravity_vec, coriolois_vec with pure matrix calculation.
        use: 
            G_mat, A_mat,
            L(q), W(q), V_base(q), dot_V_base(q)
        IMPORTANT: q should minus q_init!
        '''
        q = q - self.q_init

        adjoint_T_list = self.get_adjoint_T_list(q)
        L, W = self.get_L_and_W(adjoint_T_list)
        V_base, dot_V_base = self.get_V_base_and_dot_V_base(adjoint_T_list)

        q = q.reshape(-1, 1)
        dq = dq.reshape(-1, 1)
        V = L @ (self.A_mat @ q + V_base)
        V_map = self.get_V_map(V)
        A_dq_map = self.get_V_map(self.A_mat @ dq)

        inertia_matrix = self.A_mat.T @ L.T @ self.G_mat @ L @ self.A_mat
        tmp = self.G_mat @ L @ A_dq_map @ W - V_map.T @ self.G_mat
        coriolois_vec = self.A_mat.T @ L.T @ tmp @ L @ self.A_mat @ dq
        gravity_vec = self.A_mat.T @ L.T @ self.G_mat @ L @ dot_V_base

        return inertia_matrix, gravity_vec, coriolois_vec




if __name__ == '__main__':
    physics_client_id = p.connect(p.DIRECT)
    chain = PandaRodDynamicsChain(physics_client_id)
    print(chain.inverse_dynamics(None, None, None, None, None))