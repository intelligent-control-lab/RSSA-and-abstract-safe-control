import torch
import scipy.stats as stats


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