import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from loguru import logger
import wandb
import yaml

from humanoid_nn_models import HumanoidVModel
from humanoid_utils import turn_on_log

class BaseTrain:
    def __init__(
        self,
        model: HumanoidVModel,
        log_path,
        device='cuda',
        lr=1e-3,
        epochs=200,
        save_epoch_interval=10,
        if_load_pretrained_model=False,
        pretrained_model_path=None,
        dataset_path='./data/v_data/',
        lr_steps=20,
    ):
        self.device = device
        self.seperate_dataset()
        self.dataset_path = dataset_path
        self.epochs = epochs
        
        self.model = model
        self.model.to(device=device)
        logger.debug(self.model)
        
        self.criterion = nn.MSELoss()
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.lr_steps = lr_steps
        self.min_val_loss = float('inf')
        
        self.best_model_save_path = log_path + '/best_model.pth'
        self.save_epoch_interval = save_epoch_interval
        
        self.if_load_pretrained_model = if_load_pretrained_model
        self.pretrained_model_path = pretrained_model_path
        if self.if_load_pretrained_model:
            self.model.load_state_dict(th.load(self.pretrained_model_path))
            
    def seperate_dataset(self):
        indices_num = 480
        indices = th.arange(start=0, end=indices_num, step=1)
        train_num = int(0.9 * indices_num)
        val_num = indices_num - train_num
        self.train_data_indices, self.val_data_indices = random_split(indices, [train_num, val_num])
        self.train_data_indices = self.train_data_indices.indices
        self.val_data_indices = self.val_data_indices.indices
        
    def get_data(self, data_idx):
        return th.load(self.dataset_path + str(data_idx) + '.pth')

    def get_nn_output(self, **kwargs):
        return self.model(kwargs['x'], kwargs['u'])

    def get_target(self, **kwargs):
        return kwargs['v']

    def get_loss(self, nn_output, target, **kwargs):
        return self.criterion(nn_output, target)

    def train_epoch(self):
        runnning_loss = 0.0
        train_set_size = 0
        for data_idx in self.train_data_indices:
            kwargs = self.get_data(data_idx)
            nn_output = self.get_nn_output(**kwargs)
            target = self.get_target(**kwargs)
            self.optimizer.zero_grad()
            loss = self.get_loss(nn_output, target, **kwargs)
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
            for data_idx in self.val_data_indices:
                kwargs = self.get_data(data_idx)
                nn_output = self.get_nn_output(**kwargs)
                target = self.get_target(**kwargs)
                self.optimizer.zero_grad()
                loss = self.get_loss(nn_output, target, **kwargs)
                
                val_set_size += 1
                running_loss += loss.item()
        self.model.train()
        running_loss /= val_set_size
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
            wandb.log({'train/loss': train_loss})
            logger.debug(f'epoch: {epoch}, train loss: {train_loss}')
            if epoch % self.save_epoch_interval == 0:
                val_loss = self.val()
                wandb.log({'val/loss': val_loss})
                logger.debug(f'val loss: {val_loss}')
                if val_loss < self.min_val_loss:
                    self.min_val_loss = val_loss
                    logger.debug('---BEST MODEL!---')
                    th.save(self.model.state_dict(), self.best_model_save_path)
            self.lr_schedule(epoch)
        logger.debug('finish training')


if __name__ == '__main__':
    yaml_path = './cfg/latent_train/v_train.yaml'
    with open(yaml_path, 'r') as file:
        cfg = yaml.load(file)
    log_path, wandb_exp_name = turn_on_log(log_root_path='./train_v_log/', yaml_path=yaml_path)
    wandb.init(
        project='humanoid_train_v',
        name=wandb_exp_name,
        config=cfg,
    )
    model = HumanoidVModel(**cfg['model'])
    train = BaseTrain(
        model=model,
        log_path=log_path, 
        **cfg['train'],
    )
    train()
    wandb.finish()