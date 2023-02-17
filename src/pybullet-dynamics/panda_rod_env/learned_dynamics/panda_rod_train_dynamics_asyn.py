from panda_rod_dataset import PandaRodDataset
from panda_rod_dynamics_model import PandaRodDynamicsModelF, PandaRodDynamicsModelG
from panda_rod_train_dynamics import PandaRodTrain, convert_data

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import os
import sys

class PandaRodTrainF(PandaRodTrain):
    def __init__(
        self, 
        data_path, 
        best_model_save_path, 
        device, 
        batch_size=8192, lr=0.001, epochs=1000, 
        save_epoch_interval=10, 
        num_layer=3, hidden_dim=256, 
        if_finetune=False, finetune_epoch_interval=10
    ):
        super().__init__(data_path, best_model_save_path, device, batch_size, lr, epochs, 
            save_epoch_interval, num_layer, hidden_dim, if_finetune, finetune_epoch_interval)

    def get_model(self, dims):
        return PandaRodDynamicsModelF(self.num_layer, self.hidden_dim, dims)

    def get_nn_output(self, nn_input, dot_x, u):
        return self.model(nn_input)


class PandaRodTrainG(PandaRodTrain):
    def __init__(
        self, 
        data_path, 
        best_model_save_path,
        f_pretrained_model_path,
        device, 
        batch_size=8192, lr=0.001, epochs=1000, 
        save_epoch_interval=10, 
        num_layer=3, hidden_dim=256, 
        if_finetune=False, finetune_epoch_interval=10,
        
    ):
        super().__init__(data_path, best_model_save_path, device, batch_size, lr, epochs, 
            save_epoch_interval, num_layer, hidden_dim, if_finetune, finetune_epoch_interval)
    
        self.f_pretrained_model_path = f_pretrained_model_path
        dims = self.train_dataset.get_dim()
        self.f_model = PandaRodDynamicsModelF(num_layer=self.num_layer, hidden_dim=self.hidden_dim, dims=dims)
        self.f_model.load_state_dict(th.load(self.f_pretrained_model_path))
        self.f_model.to(self.device)
        self.f_model.eval()
    
    def get_model(self, dims):
        return PandaRodDynamicsModelG(num_layer=self.num_layer, hidden_dim=self.hidden_dim, dims=dims)
    
    def get_nn_output(self, nn_input, dot_x, u):
        return self.model(nn_input, u)
    
    def get_target(self, nn_input, dot_x, u):
        with th.no_grad():
            f = self.f_model(nn_input)
        return dot_x - f
    
        

if __name__ == '__main__':
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    data_path = './src/pybullet-dynamics/panda_rod_env/model/env_3/'
    best_model_name = '9_best.pth'
    
    f_data_path = data_path + 'f/'
    f_best_model_save_path = f_data_path + best_model_name
    train_f = PandaRodTrainF(
        data_path=f_data_path,
        device=device,
        best_model_save_path=f_best_model_save_path,
    )
    train_f()

    g_data_path = data_path + 'g/'
    g_best_model_save_path = g_data_path + best_model_name
    train_g = PandaRodTrainG(
        data_path=g_data_path,
        device=device,
        best_model_save_path=g_best_model_save_path,
        f_pretrained_model_path=f_best_model_save_path,
    )
    train_g()

