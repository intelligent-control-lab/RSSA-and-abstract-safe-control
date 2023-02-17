from panda_rod_dataset import PandaRodDataset
from panda_rod_dynamics_model import PandaRodDynamicsModel

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import os
import sys

class PandaRodTrain:
    '''
    Base class of training
    '''
    def __init__(
        self,
        data_path,
        best_model_save_path,
        device,
        batch_size=1024,
        lr=1e-3,
        epochs=200,
        save_epoch_interval=10,
        num_layer=3,
        hidden_dim=256,
        if_finetune=False,
        finetune_epoch_interval=10,
        if_load_pretrained_model=False,
        pretrained_model_path=None,
    ):
        self.data_path = data_path
        self.best_model_save_path = best_model_save_path
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.save_epoch_interval = save_epoch_interval
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.if_finetune = if_finetune
        self.finetune_epoch_interval = finetune_epoch_interval

        self.train_dataset = PandaRodDataset(self.data_path, mode='train')
        self.val_dataset = PandaRodDataset(self.data_path, mode='val')
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1024, drop_last=True)

        self.criterion = nn.MSELoss()

        self.if_load_pretrained_model = if_load_pretrained_model
        self.pretrained_model_path = pretrained_model_path

        dims = self.train_dataset.get_dim()
        self.model = self.get_model(dims=dims)
        self.model.to(device=device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.min_val_loss = float('inf')


    def get_model(self, dims):
        '''
        return a ordinary f and g_flat model
        '''
        return PandaRodDynamicsModel(num_layer=self.num_layer, hidden_dim=self.hidden_dim, dims=dims)


    def get_nn_output(self, nn_input, dot_x, u):
        '''
        get output of the model: dot_x_nn
        '''
        return self.model(nn_input, u)


    def get_target(self, nn_input, dot_x, u):
        '''
        get target: dot_x
        '''
        return dot_x


    def get_loss(self, nn_output, target, nn_input, dot_x, u):
        '''
        get loss: || dot_x - dot_x_nn ||^2
        '''
        return self.criterion(nn_output, target)


    def finetune(self):
        '''
        for parameter finetune
        '''
        pass


    def train_epoch(self):
        runnning_loss = 0.0
        train_set_size = 0
        for i, data in enumerate(self.train_loader):
            # print(i)
            convert_data(data, self.device)

            nn_input, dot_x, u = data
            nn_output = self.get_nn_output(nn_input, dot_x, u)
            target = self.get_target(nn_input, dot_x, u)

            self.optimizer.zero_grad()

            loss = self.get_loss(nn_output, target, nn_input, dot_x, u)
            # loss.backward(retain_graph=True)
            loss.backward()
            self.optimizer.step()

            train_set_size += 1
            runnning_loss += loss.item()
        runnning_loss /= train_set_size
        return runnning_loss


    def val(self):
        self.model.eval()

        running_loss = 0
        val_set_size = 0
        
        with th.no_grad():
            for data in self.val_loader:
                convert_data(data, self.device)
                nn_input, dot_x, u = data
                nn_output = self.get_nn_output(nn_input, dot_x, u)
                target = self.get_target(nn_input, dot_x, u)
                loss = self.get_loss(nn_output, target, nn_input, dot_x, u)
                
                val_set_size += 1
                running_loss += loss.item()
        
        self.model.train()

        running_loss /= val_set_size
        return running_loss

    
    def lr_schedule(self, epoch):
        if epoch % 100 == 99:
            for g in self.optimizer.param_groups:
                g['lr'] /= 2
            lr = self.optimizer.param_groups[0]['lr']
            print(f'change the learning rate to {lr}')


    def __call__(self):
        for epoch in range(self.epochs):
            if self.if_finetune and epoch % self.finetune_epoch_interval == 0:
                self.finetune()
            else:
                train_loss = self.train_epoch()
                print(f'epoch: {epoch}, train loss: {train_loss}')
            
            if epoch % self.save_epoch_interval == 0:
                val_loss = self.val()
                print(f'val loss: {val_loss}')
                if val_loss < self.min_val_loss:
                    self.min_val_loss = val_loss
                    print('---BEST MODEL!---')
                    th.save(self.model.state_dict(), self.best_model_save_path)
                    print(self.model.state_dict()['Glist_mass_7'], self.model.state_dict()['Glist_inertia_7'])
                    print(self.model.state_dict()['Glist_mass_8'], self.model.state_dict()['Glist_inertia_8'])

            self.lr_schedule(epoch)

        print('finish training')


    def train(self):
        self.__call__()
    

def convert_data(data, device):
    for i in range(len(data)):
        data[i] = data[i].to(device).float()


if __name__ == '__main__':
    data_path = './src/pybullet-dynamics/panda_rod_env/model/env_data_test_bn_2/'
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    best_model_save_path = './src/pybullet-dynamics/panda_rod_env/model/env_data_test_bn_2/9_best.pth'
    train = PandaRodTrain(
        data_path=data_path,
        best_model_save_path=best_model_save_path,
        device=device,
        epochs=1000,
    )

    train()