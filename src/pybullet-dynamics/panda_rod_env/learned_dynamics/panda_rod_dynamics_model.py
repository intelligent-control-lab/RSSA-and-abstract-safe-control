import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader

# from panda_rod_dataset import PandaRodDataset

class PandaRodDynamicsModel(nn.Module):
    def __init__(self, num_layer, hidden_dim, dims, use_bn=True):
        super().__init__()
        self.num_layer = num_layer
        self.use_bn = use_bn

        self.nn_input_dim = dims['nn_input_dim']
        self.nn_output_dim = dims['nn_output_dim']
        self.hidden_dim = hidden_dim
        self.x_dim = self.f_dim = dims['f_dim']
        self.g_flat_dim = dims['g_flat_dim']
        self.u_dim = dims['u_dim']

        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(self.nn_input_dim, self.hidden_dim))
        if self.use_bn:
            self.fc.append(nn.BatchNorm1d(self.hidden_dim))
        self.fc.append(nn.ReLU())
        for _ in range(num_layer - 2):
            self.fc.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            if self.use_bn:
                self.fc.append(nn.BatchNorm1d(self.hidden_dim))
            self.fc.append(nn.ReLU())
        self.fc.append(nn.Linear(self.hidden_dim, self.nn_output_dim))
        self.fc = nn.Sequential(*self.fc)

    def forward(self, nn_input, u):
        nn_output = self.fc(nn_input)

        batch_size = nn_output.shape[0]
        f = nn_output[:, :self.f_dim]
        g = nn_output[:, self.f_dim:].reshape((batch_size, self.f_dim, self.u_dim))
        dot_x = f + (g @ u.view(batch_size, self.u_dim, 1)).squeeze_()
        return dot_x

    def get_f_and_g_flat(self, x):
        x = th.as_tensor(x)
        x = x.T.float()

        nn_output = self.fc(x)
        f = nn_output[:, :self.f_dim].numpy()
        g_flat = nn_output[:, self.f_dim:].numpy()

        return f, g_flat


class PandaRodDynamicsModelF(PandaRodDynamicsModel):
    '''
    train f(x) and g(x) seperatedly:
    when u = 0, \dot{x} = f(x)
    '''
    def __init__(self, num_layer, hidden_dim, dims, use_bn=True):
        dims['nn_output_dim'] = dims['f_dim']
        super().__init__(num_layer, hidden_dim, dims, use_bn)
    
    def forward(self, nn_input):
        '''
        in this case, u = 0, f(x) = \dot{x}
        '''
        f = self.fc(nn_input)
        return f

    def get_f(self, x):
        x = th.as_tensor(x)
        x = x.T.float()

        f = self.fc(x).numpy()
        return f


class PandaRodDynamicsModelG(PandaRodDynamicsModel):
    '''
    train f(x) and g(x) seperatedly:
    when f(x) is well trained, use \dot{x} - f(x) as the target
    '''
    def __init__(self, num_layer, hidden_dim, dims, use_bn=True):
        dims['nn_output_dim'] = dims['g_flat_dim']
        super().__init__(num_layer, hidden_dim, dims, use_bn)

    def forward(self, nn_input, u):
        '''
        in this case, u = 0, f(x) - \dot{x} = g(x) @ u
        '''
        g_flat = self.fc(nn_input)
        batch_size = g_flat.shape[0]
        g = g_flat.reshape((batch_size, self.f_dim, self.u_dim))

        dot_x_res = (g @ u.view(batch_size, self.u_dim, 1)).squeeze_()
        return dot_x_res

    def get_g_flat(self, x):
        x = th.as_tensor(x)
        x = x.T.float()

        g_flat = self.fc(x).numpy()
        return g_flat
    

def convert_data(data, device):
    for i in range(len(data)):
        data[i] = data[i].to(device).float()


# if __name__ == '__main__':
#     device = th.device('cuda' if th.cuda.is_available() else 'cpu')
#     dataset = PandaRodDataset(data_path='./src/pybullet-dynamics/panda_rod_env/model/env_3/f/')
#     print(len(dataset))
#     data_loader = DataLoader(dataset, 128)
#     f_model = PandaRodDynamicsModelF(3, 256, dataset.get_dim())
#     f_model.load_state_dict(th.load('./src/pybullet-dynamics/panda_rod_env/model/env_3/f/1_best.pth'))
#     f_model.to(device)
#     f_model.eval()
#     criterion = nn.MSELoss()
#     for data in data_loader:
#         convert_data(data, device)
#         x, dot_x, u = data
#         dot_x_nn = f_model(x)
#         loss = criterion(dot_x_nn, dot_x)
#         print(loss.item())





