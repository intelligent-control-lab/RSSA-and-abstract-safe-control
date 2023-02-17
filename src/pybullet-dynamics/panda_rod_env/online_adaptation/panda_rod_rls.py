import numpy as np
from scipy.linalg import null_space

from panda_rod_utils import to_np

class PandaRodRLS:
    def __init__(
        self,
        lbd, 
        initial_W, 
        initial_lr,
        beta=0,
        x_dim=18,
        u_dim=7,
    ):
        self.lbd = lbd # forgetting factor
        self.W = initial_W
        self.initial_lr = initial_lr
        self.beta = beta # regularization factor

        self.x_dim = x_dim
        self.u_dim = u_dim
        self.u_bar_dim = self.u_dim + 1
        self.theta_dim = self.u_bar_dim * self.x_dim

        assert self.W.shape == (self.x_dim, self.u_bar_dim, self.x_dim)

        self.reset()

    def reset(self):
        self.H = np.zeros((self.x_dim, self.theta_dim, self.theta_dim))
        for i in range(self.x_dim):
            self.H[i] = 1.0 / self.initial_lr * np.eye(self.theta_dim)
    
    def d_reg(self, theta, x, u):
        ''' 
        Gradient of the regularization term
        '''
        return np.zeros_like(theta)

    def dd_reg(self, theta, x, u):
        '''
        Second derivative of the regularization term, used in H
        '''
        return np.zeros((self.theta_dim, self.theta_dim))

    def adapt(self, dot_x_residual, x, u):
        '''
        Update paramters:
        dot_x_residual[i] = u_bar.T @ W[i] @ x, where u_bar = [1, u]
        '''
        x = to_np(x).reshape(-1, 1)
        u = to_np(u).reshape(-1, 1)
        dot_x_residual = to_np(dot_x_residual)
        u_bar = np.vstack(([1.0], u))

        # For simplicity, we deal with one output dimension at a time.
        for i in range(self.x_dim):
            theta = self.W[i].reshape(-1, 1)
            G = (u_bar @ x.T).reshape(1, -1)
            e = dot_x_residual[i] - (u_bar.T @ theta.reshape(self.u_bar_dim, self.x_dim) @ x).item()
            print(e)
            H = self.H[i]
            F = np.linalg.inv(G.T @ G + self.lbd * H)
            theta = theta + F @ (G.T * e - self.beta * self.d_reg(theta, x, u))
            H = G.T @ G + self.beta * self.dd_reg(theta, x, u) + self.lbd * H
            self.W[i] = theta.reshape(self.u_bar_dim, self.x_dim)
            self.H[i] = H
        
    def pred(self, x):
        '''
        Predict residuals:
        [f_residual |  g_residual][i, :] = (W[i] @ x).T
        '''
        x = to_np(x).reshape(-1, 1)
        residual = np.zeros((self.x_dim, self.u_bar_dim))
        for i in range(self.x_dim):
            residual[[i], :] = (self.W[i] @ x).T
        
        f_residual = residual[:, [0]]
        g_residual = residual[:, 1:]
        return f_residual, g_residual


class PandaRodLinearRLS(PandaRodRLS):
    '''
    suppose dot_x_residual = A @ x + B @ u in local,
    then f_residual = A @ x, g_residual = B
    '''
    def __init__(
        self,
        lbd,
        initial_W, 
        initial_lr,
        beta=0,
        x_dim=18,
        u_dim=7,
    ):
        self.x_dim = x_dim
        self.u_dim = u_dim

        assert initial_W.shape == (self.x_dim, self.x_dim + self.u_dim)
        self.W = initial_W
        self.initial_lr = initial_lr
        self.theta_dim = self.x_dim + self.u_dim
        self.lbd = lbd
        self.beta = beta

        self.reset()

    def adapt(self, dot_x_residual, x, u):
        x = to_np(x).reshape(-1, 1)
        u = to_np(u).reshape(-1, 1)
        dot_x_residual = to_np(dot_x_residual)

        for i in range(self.x_dim):
            theta = self.W[[i], :].T
            G = np.vstack((x, u)).T
            e = dot_x_residual[i] - (theta.T @ np.vstack((x, u))).item()
            H = self.H[i]
            F = np.linalg.inv(G.T @ G + self.lbd * H)
            theta = theta + F @ (G.T * e - self.beta * self.d_reg(theta, x, u))
            H = G.T @ G + self.beta * self.dd_reg(theta, x, u) + self.lbd * H
            self.W[[i], :] = theta.T
            self.H[i] = H
    
    def pred(self, x):
        '''
        [A | B] = W
        '''
        x = to_np(x).reshape(-1, 1)
        A = self.W[:, :self.x_dim]
        B = self.W[:, self.x_dim:]

        f_residual = A @ x
        g_residual = B
        return f_residual, g_residual


class PandaRodRLSLinearNullL2(PandaRodLinearRLS):
    '''
    linear case with L2 regularization of sum theta*xj,
    where xj's is basis of the null space of x
    '''
    def d_reg(self, theta, x, u):
        '''
        reg = sum * (theta.T @ x)^2
        d_reg = sum (theta.T @ xj) * xj
        '''
        ret = np.zeros_like(theta)
        for j in range(len(self.x_u_null.T)):
            x_u_j = self.x_u_null[:, [j]]
            ret += x_u_j @ (theta.T @ x_u_j)
        return ret
    
    def dd_reg(self, theta, x, u):
        '''
        reg = sum * (theta.T @ x)^2
        dd_reg = sum xj.T @ xj
        '''
        ret = np.zeros_like(theta)
        for j in range(len(self.x_u_null.T)):
            x_u_j = self.x_u_null[:, [j]]
            ret += x_u_j.T @ x_u_j
        return ret
        
    def adapt(self, dot_x_residual, x, u):
        x = to_np(x).reshape(-1, 1)
        u = to_np(u).reshape(-1, 1)
        x_u = np.vstack((x, u))
        self.x_u_null = null_space(x_u.T)
        return super().adapt(dot_x_residual, x, u)



if __name__ == '__main__':
    rls = PandaRodRLS(0.5, np.ones((18, 8, 18)), 1)
    x = np.ones((18, 1))
    u = np.ones((7, 1))
    dot_x_residual = np.ones_like(x)
    rls.adapt(dot_x_residual, x, u)
    rls.pred(x)
