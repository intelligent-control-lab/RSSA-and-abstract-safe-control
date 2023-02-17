import os
import numpy as np
import pickle

from torch.utils.data import Dataset
import torch as th

def divide_data(
    raw_data_path,
    data_path,
    load_ratio=0.1,
    train_ratio=0.9,
):
    train_data_name_path = data_path + 'train_data_name.pkl'
    val_data_name_path = data_path + 'val_data_name.pkl'
    train_data_name_list = []
    val_data_name_list = []
    load_interval = int(1 / load_ratio)
    count = 0
    for file_name in os.listdir(raw_data_path):
        if count % load_interval == 0:
            if np.random.binomial(n=1, p=train_ratio) == 1:
                train_data_name_list.append(raw_data_path + file_name)
            else:
                val_data_name_list.append(raw_data_path + file_name)
        count += 1
    with open(train_data_name_path, 'wb') as file:
        pickle.dump(train_data_name_list, file)
    with open(val_data_name_path, 'wb') as file:
        pickle.dump(val_data_name_list, file)
    

class PandaRodDataset(Dataset):
    def __init__(
        self,
        data_path,
        mode='train',
    ):
        if mode == 'train':
            train_data_name_path = data_path + 'train_data_name.pkl'
            with open(train_data_name_path, 'rb') as file:
                self.data_name_list = pickle.load(file)
        elif mode == 'val':
            val_data_name_path = data_path + 'val_data_name.pkl'
            with open(val_data_name_path, 'rb') as file:
                self.data_name_list = pickle.load(file)
        else:
            raise Exception('dataset does not have such mode')
        
    def get_dim(self):
        nn_input, dot_x, u = self.__getitem__(0)
        u_dim = u.shape[0]
        nn_input_dim = nn_input.shape[0]
        f_dim = dot_x.shape[0]
        g_flat_dim = f_dim * u.shape[0]
        nn_output_dim = f_dim + g_flat_dim
        dims = {
            'nn_input_dim': nn_input_dim,
            'nn_output_dim': nn_output_dim,
            'f_dim': f_dim,
            'g_flat_dim': g_flat_dim,
            'u_dim': u_dim,
        }
        return dims
    
    def __getitem__(self, idx):
        file_name = self.data_name_list[idx]
        with open(file_name, 'rb') as file:
            data = pickle.load(file)
        x = np.squeeze(data['x'])
        dot_x = np.squeeze(data['dot_x'])
        u = np.squeeze(data['u'])

        return x, dot_x, u

    def __len__(self):
        return len(self.data_name_list)



if __name__ == '__main__':
    # raw_data_path = './src/pybullet-dynamics/panda_rod_env/data/nn_raw_data_env_3/zero/'
    # data_path = './src/pybullet-dynamics/panda_rod_env/model/env_3/f/'
    # divide_data(
    #     raw_data_path=raw_data_path, 
    #     data_path=data_path, 
    #     load_ratio=1
    # )
    raw_data_path = './src/pybullet-dynamics/panda_rod_env/data/raw_data_diff/'
    data_path = './src/pybullet-dynamics/panda_rod_env/model/env_diff/'
    divide_data(
        raw_data_path=raw_data_path, 
        data_path=data_path, 
        load_ratio=1
    )

    dataset = PandaRodDataset(data_path, mode='train')
    print(len(dataset))