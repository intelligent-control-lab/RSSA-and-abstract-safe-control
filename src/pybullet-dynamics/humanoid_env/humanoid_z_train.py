import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from loguru import logger
import wandb
import yaml

from humanoid_nn_models import HumanoidZModel
from humanoid_v_train import BaseTrain
from humanoid_utils import turn_on_log

class ZTrain(BaseTrain):
    def __init__(
        self, 
        model: HumanoidZModel, 
        log_path,
        device='cuda',
        lr=1e-3,
        epochs=200,
        save_epoch_interval=10,
        if_load_pretrained_model=False,
        pretrained_model_path=None,
        dataset_path='./data/z_data/',
        lr_steps=20,
    ):
        super().__init__(
            model, log_path, 
            device, lr, epochs, save_epoch_interval, if_load_pretrained_model, pretrained_model_path, dataset_path, lr_steps
        )
    
    def get_nn_output(self, **kwargs):
        return self.model(kwargs['x'])

    def get_target(self, **kwargs):
        return kwargs['z']
    
    
if __name__ == '__main__':
    yaml_path = './cfg/latent_train/z_train.yaml'
    with open(yaml_path, 'r') as file:
        cfg = yaml.load(file)
    log_path, wandb_exp_name = turn_on_log(log_root_path='./train_z_log/', yaml_path=yaml_path)
    wandb.init(
        project='humanoid_train_z',
        name=wandb_exp_name,
        config=cfg,
    )
    model = HumanoidZModel(**cfg['model'])
    train = ZTrain(
        model=model,
        log_path=log_path, 
        **cfg['train'],
    )
    train()
    wandb.finish()