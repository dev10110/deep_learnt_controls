from __future__ import print_function
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

import numpy as np
import pandas as pd

from rocket import Rocket

import random
import glob



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        d = 16
        self.fc1 = nn.Linear(5, d)
        self.fc2 = nn.Linear(d, d)
        self.fc3 = nn.Linear(d, d)
        self.fc4 = nn.Linear(d, 2)

        a = (12/(5+d))**0.5
        nn.init.uniform_(self.fc1.weight, a=-a, b=a)

        a = (12/(d+d))**0.5
        nn.init.uniform_(self.fc2.weight, a=-a, b=a)

        a = (12/(d+d))**0.5
        nn.init.uniform_(self.fc3.weight, a=-a, b=a)

        a = (6/(d+2))**0.5
        nn.init.uniform_(self.fc4.weight, a=-a, b=a)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))

        return x

class MyDataset(data.Dataset):

    def __init__(self, state_csv, control_csv, transform=None):

        df_s = pd.read_csv(state_csv)
        df_c = pd.read_csv(control_csv)

        #normalize by (df-mean)/std
        self.s_mean, self.s_std = df_s.mean(), df_s.std()
        #self.c_mean, self.c_std = df_c.mean(), df_c.std()

        n_df_s = (df_s - df_s.mean())/df_s.std()
        #n_df_c = (df_c - df_c.mean())/df_c.std()

        #normalize control range from -1 to 1
        #self.c_max = n_df_c.max()
        #self.c_min = n_df_c.min()

        #n2_df_c = 2*(n_df_c-n_df_c.min())/(n_df_c.max()-n_df_c.min()) - 1
        n_df_c = df_c
        n_df_c['0'] = 2*n_df_c['0']-1 #from [0,1] to [-1,1]
        n_df_c['1'] = 2*n_df_c['1']/np.pi #from [-pi/2, pi/2] to [-1,1]


        print(len(samp))
        print(len(n_df_s_2))


        self.df_s = torch.FloatTensor(n_df_s.values)
        self.df_c = torch.FloatTensor(n_df_c.values)

        self.length = len(self.df_s)

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        sample = {'s': self.df_s[index], 'c': self.df_c[index]}

        return sample

    def normalize_s(self, state):
        return ((state[i] - self.s_mean[i])/self.s_std[i] for i in range(5))

    def normalize_c(self, control):

        raise NotImplementedError

        #return ((control[i] - self.c_mean[i])/self.c_std[i] for i in range(2))

    def denormalize_c(self, nn_c):
        #undo the min max

        #n_c = [0.5*(nn_c[i]+1)*(self.c_max[i]-self.c_min[i]) + self.c_min[i] for i in range(2)]

        #undo the std and mean
        #c = (n_c[i]*self.c_std[i] + self.c_mean[i] for i in range(2))

        c1 = (nn_c[0] + 1)/2
        c2 = (nn_c[1]*np.pi/2)

        return c1, c2


def make_train_step(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop

    def train_step(x, y):
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        yhat = model(x)
        # Computes loss
        loss = loss_fn(y, yhat)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item()

    # Returns the function that will be called inside the train loop
    return train_step







def collate_data_from_folders():

    data_s = []
    first_row_s = []
    data_c = []
    first_row_c = []
    for infile in glob.glob("data_apollo/*_sim_s.csv"):
        df = pd.read_csv(infile)

        data_s.append(df)
        first_row_s.append(list(df.iloc[0]))

    for infile in glob.glob("data_apollo/*_sim_c.csv"):
        df = pd.read_csv(infile)

        data_c.append(df)
        first_row_c.append(list(df.iloc[0]))


        df_s_full = pd.concat(data_s)
        df_s_first = pd.DataFrame(first_row_s)

        df_c_full = pd.concat(data_c)
        df_c_first = pd.DataFrame(first_row_c)

    df_s_full.to_csv('data_apollo/df_s_full.csv', index=False)
    df_c_full.to_csv('data_apollo/df_c_full.csv', index=False)
