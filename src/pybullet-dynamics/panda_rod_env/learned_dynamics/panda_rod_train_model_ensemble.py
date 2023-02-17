import numpy as np
import torch as th

from panda_rod_train_dynamics import train, val
from panda_rod_dataset import PandaRodDataset
from panda_rod_dynamics_model import PandaRodDynamicsModel

def train_model_ensemble(
    data_path, 
    n_ensemble=10,
    batch_size=8192,
    lr=1e-3,
    epochs=1000,
    save_epoch_interval=10,
    num_layer=3,
    hidden_dim=256,
):
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    model_save_prefix = data_path

    for i in range(n_ensemble):
        best_model_save_path = model_save_prefix + str(i) + '_best.pth'
        train(
            data_path=data_path,
            best_model_save_path=best_model_save_path,
            device=device,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            save_epoch_interval=save_epoch_interval,
            num_layer=num_layer,
            hidden_dim=hidden_dim,
        )

def test_single_model(
    data_path,
    idx=0, 
    num_layer=3,
    hidden_dim=256,
):
    dataset = PandaRodDataset(data_path=data_path, mode='val')
    model_path = data_path + str(idx) + '_best.pth'

    model = PandaRodDynamicsModel(num_layer, hidden_dim, dataset.get_dim())
    model.load_state_dict(th.load(model_path, map_location=th.device('cpu')))
    model.eval()

    for i in range(len(dataset)):
        nn_input, dot_x, u = dataset[i]
        nn_input = th.as_tensor(nn_input.reshape(1, -1)).float()
        u = th.as_tensor(u.reshape(1, -1)).float()
        nn_dot_x = np.squeeze(model(nn_input, u).detach().numpy())
        
        print(np.abs(nn_dot_x - dot_x))




if __name__ == '__main__':
    train_model_ensemble(
        data_path='./src/pybullet-dynamics/panda_rod_env/model/env_data_test_bn_2/',
    )

    # test_single_model(data_path='./src/pybullet-dynamics/panda_rod_env/model/env_data_test_bn_1/')
