import torch
import scipy.stats as stats
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.transforms import Affine2D


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

def draw_ellipsoid(ax, mu, sigma, confidence=None, k=None, edgecolor='none', color='blue', alpha=0.5):
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
    ellipse = Ellipse(mu, width=width, height=height, angle=angle,
                        edgecolor=edgecolor, facecolor=color, alpha=alpha)
    ax.add_patch(ellipse)
    return ellipse
    
def draw_rectangle(ax, mu, sigma, confidence=None, k=None, color='blue', alpha=0.5, height=0.02):
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(sigma + np.eye(2)*0.001)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    print(eigenvalues)
    # Calculate angle (in degrees) to rotate the ellipse
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

    sqrt_cov = np.sqrt(eigenvalues)
    if confidence:
        width, height = stats.norm.ppf((confidence + 1)/2) * 2 * sqrt_cov[0], height
    if k:  # in case k is so large that confidence will be inf
        width, height = k * 2 * sqrt_cov[0], height

    # Create the rectangle centered at the origin
    rectangle = Rectangle((mu[0]-width/2, mu[1]-height/2), width, height, angle=0, 
                          edgecolor='none', facecolor=color, alpha=alpha)

    # Transform the rectangle to have the desired center
    t = Affine2D().rotate_deg_around(mu[0], mu[1], angle)
    rectangle.set_transform(t + ax.transData)

    ax.add_patch(rectangle)

    return rectangle

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