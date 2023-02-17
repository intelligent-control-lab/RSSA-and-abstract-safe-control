import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
import pybullet as p
import numpy as np

from panda_rod_train_dynamics import PandaRodTrain
from panda_rod_differentiable_dynamics_chain import PandaRodDifferentiableDynamicsChain
from panda_rod_dataset import PandaRodDataset

class PandaRodDifferentiableDynamicsChainTrain(PandaRodTrain):
    def __init__(
        self, 
        physics_client_id,
        data_path, 
        best_model_save_path, 
        device, 
        batch_size=1024, lr=0.001, epochs=200, save_epoch_interval=10, 
        num_layer=3, hidden_dim=256, 
        if_finetune=False, finetune_epoch_interval=10,
        if_load_pretrained_model=False,
        pretrained_model_path=None,
    ):
        self.physics_client_id = physics_client_id
        self.device = device
        super().__init__(data_path, best_model_save_path, device, batch_size, lr, epochs, save_epoch_interval, 
                        num_layer, hidden_dim, if_finetune, finetune_epoch_interval, 
                        if_load_pretrained_model, pretrained_model_path)

        if self.if_load_pretrained_model:
            self.model.load_state_dict(th.load(self.pretrained_model_path))
        

    def get_model(self, dims):
        return PandaRodDifferentiableDynamicsChain(self.physics_client_id, self.device)
    
    def get_nn_output(self, nn_input, dot_x, u):
        return self.model(nn_input, dot_x)
    
    def get_target(self, nn_input, dot_x, u):
        return u

    def get_loss(self, nn_output, target, nn_input, dot_x, u):
        return self.criterion(nn_output, target)


def convert_data(data, device):
    for i in range(len(data)):
        data[i] = data[i].to(device).float()



if __name__ == '__main__':
    physics_client_id = p.connect(p.DIRECT)

    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    
    # train = PandaRodDifferentiableDynamicsChainTrain(
    #     physics_client_id=physics_client_id,
    #     data_path='./src/pybullet-dynamics/panda_rod_env/model/env_diff/',
    #     best_model_save_path='./src/pybullet-dynamics/panda_rod_env/model/env_diff/2_best.pth',
    #     device=device,
    #     batch_size=8192, lr=1e-1, epochs=1000,
    #     save_epoch_interval=10,
    #     if_load_pretrained_model=False, 
    #     pretrained_model_path='./src/pybullet-dynamics/panda_rod_env/model/env_diff_1/6_best.pth',
    # )
    # train()

    dataset = PandaRodDataset(data_path='./src/pybullet-dynamics/panda_rod_env/model/env_diff/')
    print(len(dataset))
    data_loader = DataLoader(dataset, 128)
    model = PandaRodDifferentiableDynamicsChain(physics_client_id=physics_client_id, device=device)
    # print(model.state_dict()['Glist_8'])
    # print(model.Glist[8])
    model.load_state_dict(th.load('./src/pybullet-dynamics/panda_rod_env/model/env_diff/0_best.pth'))
    # print(model.state_dict()['Glist_8'])
    # print(model.Glist[8])
    criterion = nn.MSELoss()
    for data in data_loader:
        convert_data(data, device)
        x, dot_x, u = data
        u_nn = model(x, dot_x)
        loss = criterion(u, u_nn)
        print(loss.item())

