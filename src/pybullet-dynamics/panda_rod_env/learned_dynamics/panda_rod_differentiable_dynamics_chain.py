import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
import pybullet as p
import numpy as np

try:
    from learned_dynamics.panda_rod_dynamics_chain import PandaRodDynamicsChain
except Exception:
    from panda_rod_dynamics_chain import PandaRodDynamicsChain


device = th.device('cuda' if th.cuda.is_available() else 'cpu')
th.autograd.set_detect_anomaly(True)

class PandaRodDifferentiableDynamicsChain(nn.Module):
    def __init__(
        self, 
        physics_client_id,
        device,
        robot_id=None,
        end_eff_id=9, 
        q_init=None,
        dof=9,
        actuated_dof=7,
        learning_flag=True,
        Glist_learnable=[True, True],   # change G's component as a learnable parameter in a REVERSED manner
        Mlist_learnable=[False, False, False],    # change M's component as a learnable parameter in a REVERSED manner
        Alist_learnable=[False, False],   # change A's component as a learnable parameter in a REVERSED manner
    ):
        super().__init__()
        self.dynamics_chain = PandaRodDynamicsChain(
            physics_client_id=physics_client_id,
            robot_id=robot_id,
            end_eff_id=end_eff_id,
            q_init=q_init,
            dof=dof,
            actuated_dof=actuated_dof,
        )

        self.device = device

        def convert_data(data_list):
            converted_data_list = []
            for data in data_list:
                converted_data_list.append(th.unsqueeze(th.tensor(data), dim=0).float().to(self.device))
            return converted_data_list

        self.Glist = convert_data(self.dynamics_chain.Glist)
        self.Mlist = convert_data(self.dynamics_chain.Mlist)
        self.Slist = convert_data(self.dynamics_chain.Slist)
        self.Alist = convert_data(self.dynamics_chain.Alist)

        # auxiliary matrices
        self.com_Ad_T_list = convert_data(self.dynamics_chain.com_Ad_T_list)

        self.jn = self.dynamics_chain.dof
        self.q_init = th.tensor(self.dynamics_chain.q_init).float().to(self.device)

        self.dof = dof
        self.actuated_dof = actuated_dof
        self.B = th.zeros((1, self.dof, self.actuated_dof)).to(self.device)
        self.B[0, :self.actuated_dof, :self.actuated_dof] = th.eye(self.actuated_dof)

        self.Glist_learnable = Glist_learnable
        self.Mlist_learnable = Mlist_learnable
        self.Alist_learnable = Alist_learnable

        self.learning_flag = learning_flag
        if self.learning_flag:
            self.add_learnable_param()

    def get_learnable_param_indices(self, data_list, if_learnable_list):
        length = len(data_list)
        indices = []
        for i in range(len(if_learnable_list)):
            idx = length - 1 - i
            if if_learnable_list[i]:
                indices.append(idx)
        return indices       

    def add_learnable_param(self):
        '''
        Set some parameters in Glist, Mlist and Alist as learnable.
        IMPORTANT: parameters are set in a reversed order!
        '''
        self.parameter_names = []
        G_indices = self.get_learnable_param_indices(self.Glist, self.Glist_learnable)
        for i in G_indices:
            mass_name = 'Glist_mass_' + str(i)
            inertia_name = 'Glist_inertia_' + str(i)
            self.__setattr__(mass_name, nn.Parameter(th.tensor(0.0).to(self.device)))
            self.__setattr__(inertia_name, nn.Parameter(th.tensor([0.0, 0.0, 0.0]).to(self.device)))

    
    def inverse_dynamics(
        self, 
        q: th.Tensor, dq: th.Tensor, ddq: th.Tensor,
        g: th.Tensor,
        Ftip: th.Tensor,
    ):
        assert len(g.shape) == 1 # all data share the same g

        batch_size = q.shape[0]
        T_list = th.zeros((batch_size, self.jn + 1, 4, 4)).to(self.device)
        v_list = th.zeros((batch_size, self.jn + 1, 6)).to(self.device)
        a_list = th.zeros((batch_size, self.jn + 1, 6)).to(self.device)
        v_list[:, 0, :] = th.zeros((batch_size, 6)).to(self.device)
        a_list[:, 0, :] = th.cat((th.zeros(3).to(self.device), -g)).repeat(batch_size, 1).to(self.device)

        # forward iterations
        for i in range(1, self.jn + 1):
            T_list[:, i - 1, :, :] = matrix_exp6(-vec_to_se3(self.Alist[i - 1]) * q[:, i - 1].reshape(batch_size, 1, 1)) @ \
                            trans_inv(self.Mlist[i - 1])
            v_list[:, i, :] = th.squeeze(adjoint(T_list[:, i - 1, :, :].clone()) @ th.unsqueeze(v_list[:, i - 1, :].clone(), 2)) + \
                            self.Alist[i - 1] * dq[:, [i - 1]]
            a_list[:, i, :] = th.squeeze(adjoint(T_list[:, i - 1, :, :].clone()) @ th.unsqueeze(a_list[:, i - 1, :].clone(), 2)) + \
                            th.squeeze((ad(v_list[:, i, :].clone()) @ th.unsqueeze(self.Alist[i - 1], 2) * dq[:, i - 1].reshape(batch_size, 1, 1))) + \
                            self.Alist[i - 1] * ddq[:, [i - 1]]
        
        # backward iterations
        T_list[:, -1, :, :] = trans_inv(self.Mlist[-1]).repeat(batch_size, 1, 1)
        F_list = th.zeros((batch_size, self.jn + 1, 6)).to(self.device)
        F_list[:, -1, :] = Ftip.to(device)
        tau = th.zeros((batch_size, self.jn)).to(device)
        for i in reversed(range(self.jn)):
            F_list[:, i, :] = th.squeeze(th.transpose(adjoint(T_list[:, i + 1, :, :].clone()), 1, 2) @ th.unsqueeze(F_list[:, i + 1, :].clone(), 2)) + \
                    th.squeeze(self.Glist[i] @ th.unsqueeze(a_list[:, i + 1, :].clone(), 2)) - \
                    th.squeeze(th.transpose(ad(v_list[:, i + 1, :].clone()), 1, 2) @ self.Glist[i] @ th.unsqueeze(v_list[:, i + 1, :].clone(), 2))
            tau[:, i] = th.squeeze(th.unsqueeze(F_list[:, i, :].clone(), 1) @ th.unsqueeze(self.Alist[i], 2))
        
        return tau
    
    def mass_matrix(self, q: th.Tensor):
        batch_size = q.shape[0]
        M = th.zeros((batch_size, self.jn, self.jn)).to(self.device)
        dq = th.zeros_like(q).to(self.device)
        Ftip = th.zeros((batch_size, 6)).to(self.device)
        for i in range(self.jn):
            ddq = th.zeros_like(q)
            ddq[:, i] = 1.0
            M[:, :, i] = self.inverse_dynamics(
                q=q, dq=dq, ddq=ddq,
                g=th.zeros(3).to(self.device),
                Ftip=Ftip,
            )
        return M

    def vel_quadratic_forces(self, q: th.Tensor, dq: th.Tensor):
        ddq = th.zeros_like(q).to(self.device)
        Ftip = th.zeros((q.shape[0], 6)).to(self.device)
        return self.inverse_dynamics(
            q=q, dq=dq, ddq=ddq,
            g=th.zeros(3).to(self.device),
            Ftip=Ftip,
        )

    def gravity_forces(self, q: th.Tensor, g: th.Tensor = th.tensor([0, 0, -9.8])):
        dq = th.zeros_like(q).to(self.device)
        ddq = th.zeros_like(q).to(self.device)
        Ftip = th.zeros((q.shape[0], 6)).to(self.device)
        return self.inverse_dynamics(
            q=q, dq=dq, ddq=ddq,
            g=g.to(self.device), 
            Ftip=Ftip,
        )

    def calculate_dynamic_matrices(self, q: th.Tensor, dq: th.Tensor):
        '''
        get inertia_matrix, gravity_vec, coriolois_vec
        IMPORTANT: q should minus q_init!
        '''
        q = q - self.q_init
        inertia_matrix = self.mass_matrix(q)
        coriolois_vec = self.vel_quadratic_forces(q, dq)
        gravity_vec = self.gravity_forces(q)
        return inertia_matrix, gravity_vec, coriolois_vec

    def get_f_and_g_flat(self, q: th.Tensor, dq: th.Tensor):
        M, g, C = self.calculate_dynamic_matrices(q, dq)
        M_inv = th.linalg.pinv(M)

        batch_size = q.shape[0]
        f = th.zeros((batch_size, 2 * self.dof, 1)).to(self.device)
        f[:, :self.dof, 0] = dq
        f[:, self.dof:, :] = -M_inv @ th.unsqueeze(g + C, 2)

        g_full_dof = th.zeros((batch_size, 2 * self.dof, self.dof)).to(self.device)
        g_full_dof[:, self.dof:, :] = M_inv
        g = g_full_dof @ self.B

        f = f.cpu().detach().numpy().reshape(batch_size, -1)
        g_flat = g.cpu().detach().numpy().reshape(batch_size, -1)
        if batch_size == 1:
            f = np.squeeze(f)
            g_flat = np.squeeze(g_flat)

        return f, g_flat

    def forward(self, Xr: th.Tensor, dot_Xr: th.Tensor):
        q = Xr[:, :self.dof]
        dq = Xr[:, self.dof:]
        ddq = dot_Xr[:, self.dof:]
        
        # IMPORTANT: if you want to call inverse_dynamics directly, you should let q minus q_init
        q  = q - self.q_init
        Ftip = th.zeros((q.shape[0], 6)).to(self.device)
        g = th.tensor([0, 0, -9.8]).to(self.device)

        if self.learning_flag:
            G_indices = self.get_learnable_param_indices(self.Glist, self.Glist_learnable)
            for i in G_indices:
                mass_name = 'Glist_mass_' + str(i)
                inertia_name = 'Glist_inertia_' + str(i)
                m_loc = th.cat((self.__getattr__(inertia_name), self.__getattr__(mass_name).repeat(3)))
                G_loc = th.unsqueeze(th.diag(m_loc), 0)
                G = th.transpose(self.com_Ad_T_list[i], 1, 2) @ G_loc @ self.com_Ad_T_list[i]
                # print(G)
                # print(self.Glist[i])
                self.Glist[i] = G

        u = self.inverse_dynamics(q=q, dq=dq, ddq=ddq, g=g, Ftip=Ftip,)

        u = th.squeeze(u @ self.B)

        return u



def near_zero(z: th.Tensor):
    return (th.abs(z) < 1e-6).to(device)

def normalize(V: th.Tensor):
    assert len(V.shape) == 2
    norm_V = th.linalg.norm(V, dim=1, keepdim=True).to(device)
    return V / norm_V

def rot_inv(R: th.Tensor):
    assert len(R.shape) == 3 and R.shape[1] == 3 and R.shape[2] == 3
    return th.transpose(R, 1, 2).to(device)
    
def vec_to_so3(omg: th.Tensor):
    assert len(omg.shape) == 2 and omg.shape[1] == 3
    omg_so3 = th.zeros((omg.shape[0], 3, 3)).to(device)
    omg_so3[:, 0, 1] = -omg[:, 2]
    omg_so3[:, 1, 0] = omg[:, 2]
    omg_so3[:, 0, 2] = omg[:, 1]
    omg_so3[:, 2, 0] = -omg[:, 1]
    omg_so3[:, 1, 2] = -omg[:, 0]
    omg_so3[:, 2, 1] = omg[:, 0]
    return omg_so3

def so3_to_vec(so3mat: th.Tensor):
    assert len(so3mat.shape) == 3 and so3mat.shape[1] == 3 and so3mat.shape[2] == 3
    array = th.zeros((so3mat.shape[0], 3)).to(device)
    array[:, 0] = so3mat[:, 2, 1]
    array[:, 1] = so3mat[:, 0, 2]
    array[:, 2] = so3mat[:, 1, 0]
    return array

def axis_ang3(expc3: th.Tensor):
    assert len(expc3.shape) == 2 and expc3.shape[1] == 3
    return normalize(expc3), th.linalg.norm(expc3, dim=1, keepdim=True).to(device)

def matrix_exp3(so3mat: th.Tensor):
    assert len(so3mat.shape) == 3 and so3mat.shape[1] == 3 and so3mat.shape[2] == 3
    w = so3_to_vec(so3mat)
    w_norm = th.linalg.norm(w, dim=1, keepdim=True).to(device)
    w_near_zero = near_zero(w_norm).reshape(-1, 1, 1)
    # w_near_zero = near_zero(w_norm)
    # w_near_zero[:2, 0] = True
    # w_near_zero = w_near_zero.reshape(-1, 1, 1)

    I = th.eye(3).repeat(so3mat.shape[0], 1, 1).to(device)
    w_hat, theta = axis_ang3(w)
    w_hat_so3 = vec_to_so3(w_hat)
    exp3mat = I + \
            th.sin(theta).reshape(-1, 1, 1) * w_hat_so3 + \
            (1 - th.cos(theta)).reshape(-1, 1, 1) * w_hat_so3 @ w_hat_so3

    return th.where(w_near_zero, I, exp3mat)

def rp_to_trans(R: th.Tensor, p: th.Tensor):
    assert len(R.shape) == 3 and R.shape[1] == 3 and R.shape[2] == 3
    assert len(p.shape) == 2 and p.shape[1] == 3
    T = th.zeros((R.shape[0], 4, 4)).to(device)
    T[:, :3, :3] = R
    T[:, :3, 3] = p
    T[:, 3, 3] = 1.0
    return T

def trans_to_rp(T: th.Tensor):
    assert len(T.shape) == 3 and T.shape[1] == 4 and T.shape[2] == 4
    R = T[:, :3, :3]
    p = T[:, :3, 3]
    return R, p

def trans_inv(T: th.Tensor):
    assert len(T.shape) == 3 and T.shape[1] == 4 and T.shape[2] == 4
    R, p = trans_to_rp(T)
    R_inv = rot_inv(R)
    p_inv = th.squeeze(-rot_inv(R) @ p.reshape(p.shape[0], p.shape[1], -1), dim=2)
    return rp_to_trans(R_inv, p_inv)

def vec_to_se3(V: th.Tensor):
    assert len(V.shape) == 2 and V.shape[1] == 6
    w = vec_to_so3(V[:, :3])
    v = V[:, 3:]
    V_se3 = th.zeros((V.shape[0], 4, 4)).to(device)
    V_se3[:, :3, :3] = w
    V_se3[:, :3, 3] = v
    return V_se3

def adjoint(T: th.Tensor):
    assert len(T.shape) == 3 and T.shape[1] == 4 and T.shape[2] == 4
    R, p = trans_to_rp(T)
    Ad = th.zeros((T.shape[0], 6, 6)).to(device)
    Ad[:, :3, :3] = R
    Ad[:, 3:, :3] = vec_to_so3(p) @ R
    Ad[:, 3:, 3:] = R
    return Ad

def ad(V: th.Tensor):
    assert len(V.shape) == 2 and V.shape[1] == 6
    ad_V = th.zeros((V.shape[0], 6, 6)).to(device)
    ad_V[:, :3, :3] = vec_to_so3(V[:, :3])
    ad_V[:, 3:, 3:] = vec_to_so3(V[:, :3])
    ad_V[:, 3:, :3] = vec_to_so3(V[:, 3:])
    return ad_V

def matrix_exp6(se3mat: th.Tensor):
    assert len(se3mat.shape) == 3 and se3mat.shape[1] == 4 and se3mat.shape[2] == 4
    mat_exp = th.zeros((se3mat.shape[0], 4, 4)).to(device)
    w_so3 = se3mat[:, :3, :3]

    mat_exp[:, :3, :3] = matrix_exp3(w_so3)
    mat_exp[:, 3, 3] = 1.0

    w_hat, theta = axis_ang3(so3_to_vec(w_so3))
    theta_near_zero = near_zero(theta)
    # theta_near_zero[:2, 0] = True
    spare = se3mat[:, :3, 3]

    v = (spare / theta)
    w_hat_so3 = vec_to_so3(w_hat)
    G_theta = th.eye(3).repeat(se3mat.shape[0], 1, 1).to(device) * theta.reshape(se3mat.shape[0], 1, 1) + \
                (1 - th.cos(theta)).reshape(se3mat.shape[0], 1, 1) * w_hat_so3 + \
                (theta - th.sin(theta)).reshape(se3mat.shape[0], 1, 1) * w_hat_so3 @ w_hat_so3
    desired = th.squeeze(G_theta @ th.unsqueeze(v, dim=2))
    
    mat_exp[:, :3, 3] = th.where(theta_near_zero, spare, desired)
    return mat_exp




def convert_data(data, device):
    for i in range(len(data)):
        data[i] = data[i].to(device).float()



if __name__ == '__main__':
    physics_client_id = p.connect(p.DIRECT)
    

    # dataset = PandaRodDataset(data_path='./src/pybullet-dynamics/panda_rod_env/model/env_diff_2/')
    # print(len(dataset))
    # data_loader = DataLoader(dataset, 8192)
    # model = PandaRodDifferentiableDynamicsChain(physics_client_id=physics_client_id, device=device)
    # print(model.state_dict()['Glist_8'])
    # print(model.Glist[8])
    # model.load_state_dict(th.load('./src/pybullet-dynamics/panda_rod_env/model/env_diff_2/test_simple_best.pth'))
    # print(model.state_dict()['Glist_8'])
    # print(model.Glist[8])
    # # f_model.load_state_dict(th.load('./src/pybullet-dynamics/panda_rod_env/model/env_3/f/1_best.pth'))
    # # model.to(device)
    # # model.eval()
    # criterion = nn.MSELoss()
    # for data in data_loader:
    #     convert_data(data, device)
    #     x, dot_x, u = data
    #     u_nn = model(x, dot_x)
    #     loss = criterion(u, u_nn)
    #     print(loss.item())




