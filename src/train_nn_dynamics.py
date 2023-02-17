from cProfile import label
from operator import mod
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

def compute_dot_x(outputs, labels):
    # outputs: batch * [fx; expanded_gx]
    # labels[0]: batch * [u;]
    # labels[1]: batch * [dot_x;]
    batch = labels['u'].shape[0]
    u_dim = labels['u'].shape[1]
    x_dim = labels['dot_x'].shape[1]
    fx = outputs[:,:x_dim]
    # print(outputs.shape)
    gx = outputs[:,x_dim:].reshape(batch, x_dim, u_dim)
    u = labels['u'].reshape(batch, u_dim, 1)
    dot_x_gt = labels['dot_x']
    # print(fx.shape)
    # print("gx.shape")
    # print(gx.shape)
    # print("u.shape")
    # print(u.shape)
    # print("dot_x_gt.shape")
    # print(dot_x_gt.shape)
    # print(u.type)
    dot_x = fx + torch.matmul(gx, u).reshape(batch, x_dim)
    return dot_x
    
def train(dataset, model, prefix, epochs, lr, criterion, test_only=False, load_file=None):

    if not os.path.exists('../model/'+prefix):
        os.makedirs('../model/'+prefix)
        os.makedirs('../nnet/'+prefix)

    # define train set and validation set
    
    indices = torch.randperm(len(dataset)).tolist()
    train_set_size = int(0.9 * len(dataset))
    train_set = torch.utils.data.Subset(dataset, indices[:train_set_size])
    test_set = torch.utils.data.Subset(dataset, indices[train_set_size:])
    
    # define data loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=10000, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=10000, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    loss_his = []

    if load_file:
        model.load_state_dict(torch.load(load_file))
        print("model loaded")
        avg_loss = test(model, test_loader, criterion)
        print("avg_loss: ", avg_loss)
    
    if test_only:
        avg_loss = test(model, test_loader, criterion)
        print('Average L2 loss: %f' % (avg_loss))
        
    save_idx = -1
    save_path_prefix = '../model/'+prefix+'/training_'

    # training

    # plt.figure()

    for epoch in range(epochs):  # loop over the dataset multiple times
        train_loss = train_epoch(model, train_loader, optimizer, criterion)

        PATH = '../model/'+prefix+'/epoch_'+str(epoch)+'.pth'
        if epoch % 100 == 0:
            torch.save(model.state_dict(), PATH)
        print('epoch %d' % (epoch + 1))
        print('train loss: %f' % (train_loss))
        loss_his.append(train_loss)

        if epoch % 100 == 0:
            test_loss = test(model, test_loader, criterion)
            print('test  loss: %f' % (test_loss))
            # plt.plot(loss_his, color='b')
            # plt.pause(0.01)
    print("minimum loss epoch:", np.argmin(loss_his))
    print("minimum loss: ", np.min(loss_his))
    print('Finished Training')
    # plt.show()

def train_epoch(model, train_loader, optimizer, criterion):
    #train for one epoch
    running_loss = 0.0
    train_set_size = 0
    saved_cnt = 0

    for i, data in enumerate(train_loader, 0):
        train_set_size += 1

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # print("input shape")
        # print(inputs.shape)
        # print(inputs)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # outputs = model.forward(inputs.float()).float()
        outputs = model.forward(inputs.float())
        dot_x = compute_dot_x(outputs, labels)
        
        loss = criterion(dot_x, labels["dot_x"])
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    running_loss /= train_set_size
    return running_loss

def test(model, test_loader, criterion):
    loss = 0
    tot = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model.forward(inputs)
            outputs = model.forward(inputs.float())
            dot_x = compute_dot_x(outputs, labels)
            loss = criterion(dot_x, labels["dot_x"])
            # loss += criterion(outputs, labels)
            tot += 1

    avg_loss = loss / tot

    return avg_loss


def train_and_save_networks(args):

    # dataset = GymDynamicsDataset(args["env_name"])
    dataset = args["dataset"]
    
    # print(dataset[0])
    data, label = dataset[0]
    print("data")
    print(type(data))
    print(type(data[0]))
    u_dim = np.shape(label['u'])[0]
    x_dim = np.shape(label['dot_x'])[0]
    print(x_dim)
    print(u_dim)
    model = FC(args["num_layer"], x_dim, args["hidden_dim"], x_dim + u_dim * x_dim)
    model.float()
    criterion = nn.MSELoss()
    # criterion = control_affine_criterion

    train(dataset, model, args["prefix"], args["epochs"], args["lr"], criterion, load_file=args["load_path"])
    
    # convert to nnet for NN verification.
    # convert(args["prefix"], "epoch_1000", args["num_layer"], len(data), args["hidden_dim"], len(label))

if __name__ == "__main__":

    # ant_args = {
    #     "env_name": "Free-Ant-v0",
    #     "num_layer": 2,
    #     "hidden_dim": 500,
    #     "epochs": 210,
    #     "lr": 0.01,
    #     "prefix": 'ant-FC2-100',
    # }

    # unicycle_args_3 = {
    #     "env_name": "Unicycle-v0",
    #     "num_layer": 3,
    #     "hidden_dim": 100,
    #     "epochs": 1005,
    #     "lr": 0.001,
    #     "prefix": 'unicycle-FC3-100-rk4-so',
    #     "load_path": "../model/unicycle-FC3-100-rk4/epoch_1000.pth"
    # }

    # unicycle_args_5 = {
    #     "env_name": "Unicycle-v0",
    #     "num_layer": 5,
    #     "hidden_dim": 100,
    #     "epochs": 1005,
    #     "lr": 0.001,
    #     "prefix": 'unicycle-FC5-100-rk4-so',
    #     "load_path": None
    # }

    # unicycle_args_4 = {
    #     "env_name": "Unicycle-v0",
    #     "num_layer": 4,
    #     "hidden_dim": 50,
    #     "epochs": 1005,
    #     "lr": 0.001,
    #     "prefix": 'unicycle-FC4-50-rk4-so',
    #     "load_path": None
    # }

    # unicycle_args_35 = {
    #     "env_name": "Unicycle-v0",
    #     "num_layer": 3,
    #     "hidden_dim": 50,
    #     "epochs": 1005,
    #     "lr": 0.001,
    #     "prefix": 'unicycle-FC3-50-rk4-so',
    #     "load_path": None
    # }

    # layers = [3,3,4,5]
    # hidden = [200, 300, 100, 100]
    # layers = [4, 5]
    # hidden = [200, 50]
    # layers = [2, 2, 2, 2, 2, 2]
    # hidden = [50, 100, 200, 300, 400, 500]
    layers = [3, 4]
    hidden = [100, 100]
    for i in range(len(layers)):
        unicycle_args = {
            "env_name": "Unicycle-v0",
            "num_layer": layers[i],
            "hidden_dim": hidden[i],
            "epochs": 1005,
            "lr": 0.001,
            "prefix": 'unicycle-FC'+str(layers[i])+'-'+str(hidden[i])+'-rk4-extra',
            "load_path": None
        }

        train_and_save_networks(unicycle_args)