import numpy as np

from panda_rod_utils import to_np

class PandaRodBayesianRegression:
    def __init__(
        self,
        u_dim=7,
        Xr_dim=18,
        dT=1/240,
        max_batch=10,
        sigma_init=0.01,
        basis_func_info={'type': 'linear'},
    ):
        self.u_dim = u_dim
        self.Xr_dim = Xr_dim
        self.dT = dT
        self.max_batch = max_batch
        self.sigma_init=sigma_init
        
        self.basis_func_info = basis_func_info

        self.phi_last = None
        self.param_dim = None

        # store data for batch regression
        self.delta_phi_dot_list = []
        self.u_list = []
        self.Xr_list = []

    def get_phi_dot(self, phi):
        if self.phi_last is None:
            phi_dot = 0
        else:
            phi_dot = (phi - self.phi_last) / self.dT
        self.phi_last = phi
        return phi_dot

    def get_basis_func_batch(self, Xr_batch, u_batch):
        basis_func_batch = []

        for Xr, u in zip(Xr_batch, u_batch):
            # different type of basis function in terms of Xr
            if self.basis_func_info['type'] == 'linear':
                basis_func_Xr = self.get_linear_basis_func_Xr(Xr)

            basis_func = np.zeros(len(basis_func_Xr) * (self.u_dim + 1))
            basis_func[:len(basis_func_Xr)] = basis_func_Xr
            for i, u_one_component in enumerate(to_np(u), start=1):
                basis_func[len(basis_func_Xr)*i : len(basis_func_Xr)*(i+1)] = u_one_component * basis_func_Xr
            basis_func_batch.append(basis_func)
        
        return basis_func_batch
    
    def get_linear_basis_func_Xr(self, Xr):
        Xr = to_np(Xr)
        return np.concatenate(([1.0], Xr))
        
    def store_param(
        self, 
        phi, 
        LfP_nn, 
        LgP_nn, 
        Xr,
        u,
    ):
        self.phi_dot = self.get_phi_dot(phi)
        delta_phi_dot = self.phi_dot - (LfP_nn + LgP_nn @ u).item()
        self.delta_phi_dot_list.append(delta_phi_dot)
        self.Xr_list.append(Xr)
        self.u_list.append(u)

    def update_param(self):
        '''
        suppose param ~ N(mu, sigma_init^2 * self.Lambda^-1)
        '''
        if len(self.u_list) <= self.max_batch:
            u_batch = self.u_list
            Xr_batch = self.Xr_list
            delta_phi_dot_batch = self.delta_phi_dot_list
        else:
            u_batch = self.u_list[-self.max_batch:]
            Xr_batch = self.Xr_list[-self.max_batch:]
            delta_phi_dot_batch = self.delta_phi_dot_list[-self.max_batch:]

        basis_func_batch = self.get_basis_func_batch(Xr_batch, u_batch)

        if self.param_dim is None:
            self.param_dim = basis_func_batch[0].shape[0]
            self.Lambda = np.eye(self.param_dim)
            self.mu = np.zeros((self.param_dim, 1))
        
        basis_func_batch = to_np(basis_func_batch).T
        delta_phi_dot_batch = to_np(delta_phi_dot_batch).reshape(-1, 1)

        self.mu = np.linalg.inv(basis_func_batch @ basis_func_batch.T + self.Lambda) \
                    @ (basis_func_batch @ delta_phi_dot_batch + self.Lambda @ self.mu)
        self.Lambda = basis_func_batch @ basis_func_batch.T + self.Lambda

    def predict(self, Xr, u):
        Xr = [to_np(Xr)]
        u = [to_np(u)]
        basis_func = self.get_basis_func_batch(Xr, u)[0]
        delta_phi_dot_predict = np.dot(to_np(self.mu), to_np(basis_func))
        return delta_phi_dot_predict



