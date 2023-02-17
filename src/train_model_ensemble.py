import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import sys, os
from dynamics_dataset import GymDynamicsDataset, UncertainUnicycleDataset
from dynamics_model import FC
sys.path.append('../')
import matplotlib.pyplot as plt
from utils import convert
from train_nn_dynamics import train_and_save_networks, compute_dot_x
from scipy.stats import norm
import matplotlib.pyplot as plt

def test_ensemble(models, test_loader, criterion):
    loss = 0
    tot = 0
    results = []
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = torch.zeros((len(models), *labels["dot_x"].shape))
            
            for i in range(len(models)):
                outputs[i] = compute_dot_x(models[i].forward(inputs), labels)
                # loss += criterion(outputs, labels)
            
            var, mean = torch.var_mean(outputs, axis=0)

            var = torch.mean(var, axis=1) # mean of dot_x dims.
            var = torch.sqrt(var)
            err = torch.norm(mean - labels["dot_x"], dim=1)
            # print(err.shape)
            # print(np.vstack((var, err)).shape)
            results.append(np.vstack((var, err)))
    results = np.array(results)
    results = np.hstack(results)
    print(np.shape(results))
    
    plt.figure()
    plt.scatter(results[0,:], results[1,:])
    plt.xlim([0,.2])
    plt.ylim([0,.2])
    plt.xlabel("std")
    plt.ylabel("error")
    plt.title("num ensemble: " +str(len(models)))
    # plt.show()
    # plt.savefig("[0].png")
    plt.savefig("[1].png")
    # plt.savefig("[2].png")
    # plt.savefig("[3].png")

def load_test_ensemble(env_name, n_ensemble, num_layer, hidden_dim, prefix):
    # dataset = GymDynamicsDataset(env_name)
    # dataset = UncertainUnicycleDataset(env_name, size=1000, region_idx=[0])
    dataset = UncertainUnicycleDataset(env_name, size=1000, region_idx=[1])
    # dataset = UncertainUnicycleDataset(env_name, size=1000, region_idx=[2])
    # dataset = UncertainUnicycleDataset(env_name, size=1000, region_idx=[3])
    data, label = dataset[0]
    u_dim = np.shape(label['u'])[0]
    x_dim = np.shape(label['dot_x'])[0]
    models = [FC(num_layer, x_dim+u_dim, hidden_dim, x_dim+u_dim*x_dim) for i in range(n_ensemble)]
    for i in range(n_ensemble):
        load_path = "../model/"+prefix+str(i)+"/epoch_1000.pth"
        models[i].load_state_dict(torch.load(load_path))
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=10000, shuffle=False)
    criterion = nn.MSELoss()
    test_ensemble(models, test_loader, criterion)

def train_model_ensemble():
    n_ensemble = 10
    num_layer = 4
    hidden_dim = 100
    data_size = 100000
    region_idx = [1,2,3]
    dataset = UncertainUnicycleDataset("Uncertain-Unicycle-v0", size=data_size, region_idx=region_idx)
    
    for i in range(n_ensemble):
        unicycle_args = {
            "dataset": dataset,
            "n_ensemble": n_ensemble,
            "num_layer": num_layer,
            "hidden_dim": hidden_dim,
            "epochs": 1005,
            "lr": 0.001,
            "prefix": 'uncertain-unicycle-ensemble-FC'+str(num_layer)+'-'+str(hidden_dim)+'-'+str(region_idx)+'-'+str(i),
            "load_path": None
        }

        train_and_save_networks(unicycle_args)
    
if __name__ == "__main__":
    train_model_ensemble()
    # load_test_ensemble("Uncertain-Unicycle-v0", 10, 3, 100, 'unicycle-ensemble-FC3-100-')
    # load_test_ensemble("Uncertain-Unicycle-v0", 10, 4, 100, 'uncertain-unicycle-ensemble-FC4-100-[1, 2, 3]-')
    # load_test_ensemble("Unicycle-v0", 9, 3, 100, 'unicycle-ensemble-FC3-100-')