import torch
import scipy.stats as stats
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle


class DifferentiableChi2ppf(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, df):
        ctx.df = df
        y = stats.chi2.ppf(x, df)
        y = torch.tensor(y)
        ctx.save_for_backward(x, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_output / stats.chi2.pdf(y, ctx.df)
        return grad_x, None 

def draw_ellipsoid(ax, mu, sigma, confidence=None, k=None, color='blue', alpha=0.5):
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(sigma + np.eye(2)*0.00001)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Calculate angle (in degrees) to rotate the ellipse
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

    sqrt_cov = np.sqrt(eigenvalues)
    if confidence:
        width, height = stats.norm.ppf((confidence + 1)/2) * 2 * sqrt_cov
    if k:  # in case k is so large that confidence will be inf
        width, height = k * 2 * sqrt_cov

    # Plot ellipse
    ax.add_patch(Ellipse(mu, width=width, height=height, angle=angle,
                        edgecolor='none', facecolor=color, alpha=alpha))
    
def draw_rectangle(ax, mu, sigma, confidence=None, k=None, color='blue', alpha=0.5):
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(sigma + np.eye(2)*0.001)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Calculate angle (in degrees) to rotate the ellipse
    angle = np.arctan2(*eigenvectors[:, 0][::-1])

    sqrt_cov = np.sqrt(eigenvalues)
    if confidence:
        width, height = stats.norm.ppf((confidence + 1)/2) * 2 * sqrt_cov[0], 0.01
    if k:  # in case k is so large that confidence will be inf
        width, height = k * 2 * sqrt_cov[0], 0.01

    xy = -np.array([width, height])/2
    xy[0] = xy[0]*np.cos(angle) - xy[1]*np.sin(angle)
    xy[1] = xy[0]*np.sin(angle) + xy[1]*np.cos(angle)
    xy = mu.reshape(-1) + xy
    angle = np.degrees(angle)
    # Plot ellipse
    ax.add_patch(Rectangle(xy, width=width, height=height, angle=angle,
                        edgecolor='none', facecolor=color, alpha=alpha))


# Test
if __name__ == '__main__':
    p = 0.99
    x = torch.tensor(p, requires_grad=True) 
    y = DifferentiableChi2ppf.apply(x, 1) 
    y.backward()

    y_ = stats.chi2.ppf(p, 1)
    x_grad_ = (stats.chi2.ppf(p + 1e-7, 1) - stats.chi2.ppf(p, 1))/1e-7

    print(y.item(), y_)
    print(x.grad.item(), x_grad_)