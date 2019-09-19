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

from training_apollo import Net, MyDataset, make_train_step, collate_data_from_folders

#collate data

collate_data_from_folders()

# create the net

net = Net()

# create the data sets (with partition)
path_s = 'data_apollo/df_s_full.csv'
path_c = 'data_apollo/df_c_full.csv'
full_data = MyDataset(path_s, path_c)

train_size = int(0.95 * len(full_data))
val_size = len(full_data) - train_size
train_dataset, validate_dataset = torch.utils.data.random_split(full_data, [train_size, val_size])

batch_size = 8
train_dl = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validate_dl = data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)

# initalize the training

net.zero_grad()


optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum=0.9)
criterion = nn.L1Loss(reduce=True)


train_step = make_train_step(model=net, loss_fn=criterion, optimizer=optimizer)
losses = []
val_losses = []

n_epochs = 50


# actual training


for epoch in range(n_epochs):

    for batch in train_dl:

        state = batch['s']
        control = batch['c']

        loss = train_step(state, control)

        losses.append(loss)

    with torch.no_grad():
        for batch in validate_dl:

            net.eval()

            state = batch['s']
            control = batch['c']

            c_hat = net(state)

            val_loss = criterion(control, c_hat)

            val_losses.append(val_loss.item())

    print(f'Epoch: {epoch}, Loss: {sum(losses)/len(losses):.6f}, Val_loss: {sum(val_losses)/len(val_losses):.6f}')


# save the net
torch.save(net.state_dict(), 'model.pt')
