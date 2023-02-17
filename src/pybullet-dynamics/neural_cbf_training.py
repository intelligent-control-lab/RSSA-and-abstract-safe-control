import os
import shutil
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from loguru import logger
import wandb
import yaml
import warnings
from datetime import datetime

from neural_cbf_model import NeuralCBFModel
from latent_neural_cbf_model import LatentNeuralCBFModel
from fly_inv_pend_env.fly_inv_pend_SI_learning_env import FlyingInvertedPendulumLatentSILearningEnv

def turn_on_log(log_root_path, yaml_path: str = None):
    now = datetime.now()
    date_time = now.strftime('%Y-%m-%d__%H-%M-%S')
    log_path =log_root_path + date_time + '/'
    os.mkdir(log_path)
    logger.add(log_path + '/log.log')
    yaml_name = yaml_path.split('/')[-1]
    shutil.copy(src=yaml_path, dst=log_path + '/' + yaml_name)
    return log_path, date_time

class NeuralCBFTrain:
    def __init__(
        self,
        model: NeuralCBFModel,
        dataset_path,
        log_path,
        lr=1e-3,
        epochs=200,
        save_epoch_interval=10,
        lr_steps=20,
        indices_num=100,
        train_ratio=0.9,
    ):
        self.device = model.device
        self.seperate_dataset(indices_num, train_ratio)
        self.dataset_path = dataset_path
        self.epochs = epochs
        
        self.model = model
        logger.debug(self.model.phi_fmodel)
        
        self.criterion = nn.MSELoss()
        self.lr = lr
        self.optimizer = optim.Adam(self.model.phi_params, lr=lr)
        self.lr_steps = lr_steps
        self.min_val_loss = float('inf')
        
        self.best_model_save_path = log_path + 'best_model.pth'
        self.save_epoch_interval = save_epoch_interval
            
    def seperate_dataset(self, indices_num, train_ratio):
        indices_num = indices_num
        indices = th.arange(start=0, end=indices_num, step=1)
        train_num = int(train_ratio * indices_num)
        val_num = indices_num - train_num
        self.train_data_indices, self.val_data_indices = random_split(indices, [train_num, val_num])
        self.train_data_indices = self.train_data_indices.indices
        self.val_data_indices = self.val_data_indices.indices
        
    def get_data(self, data_idx):
        '''
        each batch of data should hold several components:
        - x
        - f_x and g_x
        - u_ref
        - safe_mask and unsafe_mask
        '''
        return th.load(self.dataset_path + str(data_idx) + '.pth')

    def get_nn_output(self, **kwargs):
        return self.model.forward(
            kwargs['x'], 
            kwargs['f_x'], kwargs['g_x'],
            kwargs['u_ref'],
            kwargs['safe_mask'], kwargs['unsafe_mask'],
        )

    def get_target(self, **kwargs):
        return None

    def get_loss(self, nn_output, target, **kwargs):
        return nn_output['loss']
    
    def wandb_train_log(self, nn_output, loss, **kwargs):
        for key, loss in nn_output.items():
            wandb.log({'train/' + key: loss.item()})
    
    def wandb_val_log(self, nn_output, loss, **kwargs):
        for key, loss in nn_output.items():
            wandb.log({'val/' + key: loss.item()})

    def train_epoch(self):
        runnning_loss = 0.0
        train_set_size = 1
        for data_idx in self.train_data_indices:
            kwargs = self.get_data(data_idx)
            try:
                nn_output = self.get_nn_output(**kwargs)
            except:
                warnings.warn('Solver scs returned status infeasible!')
                continue
            target = self.get_target(**kwargs)
            self.optimizer.zero_grad()
            loss = self.get_loss(nn_output, target, **kwargs)
            loss.backward()
            self.optimizer.step()
            train_set_size += 1
            runnning_loss += loss.item()
            self.wandb_train_log(nn_output, loss)
        runnning_loss /= train_set_size
        return runnning_loss

    def val(self):
        self.model.phi_fmodel.eval()
        running_loss = 0
        val_set_size = 0
        with th.no_grad():
            for data_idx in self.val_data_indices:
                kwargs = self.get_data(data_idx)
                try:
                    nn_output = self.get_nn_output(**kwargs)
                except:
                    warnings.warn('Solver scs returned status infeasible!')
                    continue
                target = self.get_target(**kwargs)
                self.optimizer.zero_grad()
                loss = self.get_loss(nn_output, target, **kwargs)
                val_set_size += 1
                running_loss += loss.item()
                self.wandb_val_log(nn_output, loss)
        self.model.phi_fmodel.train()
        running_loss /= val_set_size
        if val_set_size == 0:
            return np.inf
        return running_loss

    def lr_schedule(self, epoch):
        if epoch % self.lr_steps == (self.lr_steps - 1):
            for g in self.optimizer.param_groups:
                g['lr'] /= 2
            lr = self.optimizer.param_groups[0]['lr']
            logger.debug(f'change the learning rate to {lr}')
            
    def __call__(self):
        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            logger.debug(f'epoch: {epoch}, train loss: {train_loss}')
            if epoch % self.save_epoch_interval == 0:
                val_loss = self.val()
                if val_loss < 1e10:
                    logger.debug(f'val loss: {val_loss}')
                    if val_loss < self.min_val_loss:
                        self.min_val_loss = val_loss
                        logger.debug('---BEST MODEL!---')
                        th.save(self.model.phi_params, self.best_model_save_path)
            self.lr_schedule(epoch)
        logger.debug('finish training')


class LatentNeuralCBFTrain(NeuralCBFTrain):
    def __init__(
        self,
        model: LatentNeuralCBFModel,
        dataset_path,
        log_path,
        lr=1e-3,
        epochs=200,
        save_epoch_interval=10,
        lr_steps=20,
        indices_num=100,
        train_ratio=0.9,
    ):
        super().__init__(
            model, 
            dataset_path, log_path, 
            lr, epochs, save_epoch_interval, lr_steps,
            indices_num, train_ratio,
        )
    
    def get_data(self, data_idx):
        '''
        each batch of data should hold several components:
        - z
        - f_z and g_z
        - v_ref
        - M
        - safe_mask and unsafe_mask
        '''
        return th.load(self.dataset_path + str(data_idx) + '.pth')

    def get_nn_output(self, **kwargs):
        return self.model.forward(
            kwargs['z'], 
            kwargs['f_z'], kwargs['g_z'],
            kwargs['v_ref'],
            kwargs['M'],
            kwargs['safe_mask'], kwargs['unsafe_mask'],
        ) 
        

def finetune(
    best_model_path='./src/pybullet-dynamics/fly_inv_pend_env/latent_neural_cbf_log/2022-08-14__14-51-34/',
    infeasible_data_path='./src/pybullet-dynamics/fly_inv_pend_env/data/latent_neural_cbf_data/infeasible.pth',
    finetune_lr=1e-5,
    finetune_epoch=200,
):  
    log_root_path = '/'.join(best_model_path.split('/')[:-2]) + '/finetune/'
    log_path, wandb_exp_name = turn_on_log(log_root_path=log_root_path, yaml_path=best_model_path + 'fly_inv_pend_latent_neural_cbf.yaml')
    with open(best_model_path + 'fly_inv_pend_latent_neural_cbf.yaml', 'r') as file:
        cfg = yaml.load(file)
    cfg['train_kwargs']['lr'] = finetune_lr
    env = FlyingInvertedPendulumLatentSILearningEnv(**cfg['robot_kwargs'])
    model = LatentNeuralCBFModel(env=env, **cfg['model_kwargs'])
    model.phi_params = th.load(best_model_path + 'best_model.pth')
    train = LatentNeuralCBFTrain(model=model, log_path=log_path, **cfg['train_kwargs'])
    finetune_model_path = log_path + 'finetuned_model.pth'
    min_loss = th.inf
    kwargs = th.load(infeasible_data_path)
    wandb.init(project='flying_finetune_latent_neural_cbf', name=wandb_exp_name, config=cfg)
    for epoch in range(finetune_epoch):
        try:
            nn_output = train.get_nn_output(**kwargs)
        except:
            warnings.warn('Solver scs returned status infeasible!')
            continue
        target = train.get_target(**kwargs)
        train.optimizer.zero_grad()
        loss = train.get_loss(nn_output, target, **kwargs)
        loss.backward()
        train.optimizer.step()
        logger.debug(f'epoch: {epoch}, loss: {loss.item()}')
        train.wandb_train_log(nn_output, loss)
        if loss.item() < min_loss:
            th.save(train.model.phi_params, finetune_model_path)
            logger.debug('--BEST MODEL!--')
            min_loss = loss.item()
    wandb.finish()

class FakeEnv:
    def __init__(self, *args, **kwargs):
        self.z_dim = 2
        self.v_dim = 1
        self.dot_M_max = 100
       
if __name__ == '__main__':
    # finetune()
    # exit(0)
    
    
    yaml_path = './src/pybullet-dynamics/fly_inv_pend_env/fly_inv_pend_latent_neural_cbf.yaml'
    with open(yaml_path, 'r') as file:
        cfg = yaml.load(file)
    log_path, wandb_exp_name = turn_on_log(log_root_path=cfg['log_root_path'], yaml_path=yaml_path)
    
    env = FlyingInvertedPendulumLatentSILearningEnv(
        **cfg['robot_kwargs'],
    ) 
    # env = FakeEnv()
    model = LatentNeuralCBFModel(
        env=env,
        **cfg['model_kwargs'],
    )
    train = LatentNeuralCBFTrain(
        model=model,
        log_path=log_path,
        **cfg['train_kwargs'],
    )
    
    wandb.init(
        project='flying_train_latent_neural_cbf',
        name=wandb_exp_name,
        config=cfg,
    )
    train()
    wandb.finish()