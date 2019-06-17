# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 19:49:25 2019

@author: 李奇
"""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np

class AutoEncoder(nn.Module):
    def __init__(self,n_components):
        super(AutoEncoder, self).__init__()

        self.en_fc0 = nn.Sequential(
                nn.Linear(6, 24),
                nn.BatchNorm1d(24),
                nn.Tanh(),
                
                nn.Linear(24, 96),
                nn.BatchNorm1d(96),
                nn.Tanh(),
                
                nn.Linear(96, 384),
                nn.BatchNorm1d(384),
                nn.Tanh()
                )
        self.en_fc = nn.Linear(384, n_components)
        self.de_fc = nn.Linear(n_components, 384)
        self.de_fc2 = nn.Sequential(
                nn.Linear(384, 96),
                nn.BatchNorm1d(96),
                nn.Tanh(),
                
                nn.Linear(96, 24),
                nn.BatchNorm1d(24),
                nn.Tanh(),
                
                nn.Linear(24, 6),
                nn.BatchNorm1d(6),
                nn.Sigmoid()
        )

    def forward(self, x):
        en = self.en_fc0(x)
        code = self.en_fc(en)
        de = self.de_fc(code)
        decoded = self.de_fc2(de)
        return code, decoded

class AE():
    def __init__(self,n_components=2,epoches=100,lr=0.8):
        
        torch.set_default_tensor_type('torch.DoubleTensor')
        self.model=AutoEncoder(n_components)
        self.lr=lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,step_size=5,gamma = self.lr)
        self.loss_f = nn.MSELoss()
        self.epoches=epoches
       
    def fit_transform(self,data):
        data=torch.from_numpy(data)
        for epoch in range(self.epoches):
            self.optimizer.zero_grad()
            code, decoded = self.model(data)
            loss = self.loss_f(decoded, data)
            loss.backward()
            self.optimizer.step()
            if epoch%10==0:
                print('epoch: %d  \nloss: %.8f'%(epoch,loss.item()))
        print('=========训练完成=========')
        code, decoded = self.model(data)
        return code.detach().numpy()

            