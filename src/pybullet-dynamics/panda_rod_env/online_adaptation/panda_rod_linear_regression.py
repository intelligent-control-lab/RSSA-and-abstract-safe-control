import numpy as np

from panda_rod_utils import to_np

class PandaRodLinearRegression:
    def __init__(
        self,
        x_dim=18,
        u_dim=7,
        max_batch_size=10,
        enable_adapt_size=5,
    ):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.max_batch_size = max_batch_size
        self.enable_adapt_size = enable_adapt_size # if batch_size < enable_adapt_size, do not do regression

        self.x_batch = []
        self.u_batch = []
        self.dot_x_residual_batch = []

    def update_data(self, x, u, dot_x_residual):
        if len(self.x_batch) >= self.max_batch_size:
            self.x_batch = self.x_batch[1:]
            self.u_batch = self.u_batch[1:]
            self.dot_x_residual_batch = self.dot_x_residual_batch[1:]
        self.x_batch.append(x)
        self.u_batch.append(u)
        self.dot_x_residual_batch.append(dot_x_residual)
    
    def adapt(self, x, u, dot_x_residual):
        self.update_data(x, u, dot_x_residual)
        if len(self.x_batch) >= self.enable_adapt_size:
            x_batch = to_np(self.x_batch)
            u_batch = to_np(self.u_batch)
            dot_x_residual_batch = to_np(self.dot_x_residual_batch)
            x_u_batch = np.hstack((x_batch, u_batch))
            self.W = np.linalg.inv(x_u_batch.T @ x_u_batch) @ x_u_batch.T @ dot_x_residual_batch
        else:
            self.W = np.zeros((self.u_dim + self.x_dim, self.x_dim))
    
    def pred(self, x):
        x = to_np(x).reshape(1, -1)
        f_residual = (x @ self.W[:self.x_dim, :]).T
        g_residual = self.W[self.x_dim:, :].T
        return f_residual, g_residual


