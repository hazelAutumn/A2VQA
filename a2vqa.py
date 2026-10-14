from argparse import ArgumentParser
import os
import scipy.io
import torch
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


import warnings
warnings.filterwarnings("ignore")
from torch.nn import init
import torch.nn.functional as functions
import math

class A2VQA(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=128, hidden_dim2=128):
        super(A2VQA, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim


        self.affine_v = nn.Linear(2048, 54, bias=False) # W_v
        self.affine_a = nn.Linear(2048, 54, bias=False) # W_a
        self.affine_at = nn.Linear(54, 1) # 54 = 6*9


        self.down_a = nn.Linear(2048, hidden_dim2)
        self.down_v = nn.Linear(2048, hidden_dim2)

        self.lstm_a = nn.LSTM(hidden_dim2, hidden_dim//2, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_v = nn.LSTM(hidden_dim2, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(hidden_dim*3, hidden_dim)
        

        self.relu1 = nn.ReLU()
        self.dro2 = nn.Dropout()
        self.p7 = nn.Sequential(
            nn.Linear(30*hidden_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.linear2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def init_weights(self):
        " Initialize weights"
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(2, batch_size, self.hidden_dim, device=device)
        return h0
    
    def _get_initial_state_a(self, batch_size, device):
        h0 = torch.zeros(2, batch_size, self.hidden_dim//2, device=device)
        return h0
    
    def forward(self, v, a): # v:Bx60x2048x6x9, a: Tx60x1024
        " Forward pass of the network"
        # Audio-guided visual attention
        B, T, D, h, w = v.size()

        v_t = v.view(B*T, D, -1) #flatten the h and w to k 
        v_t= v_t.permute(0, 2, 1).contiguous() # B*T, k, D
        V = v_t
        
        a_t = a.view(B*T, -1)
        
        content_v = self.affine_v(v_t) + self.affine_a(a_t).unsqueeze(2) # B*T, k, k
        x_t = self.affine_at(torch.tanh(content_v)).squeeze(2)  # B*T, k
        att = F.softmax(x_t, dim=-1).view(x_t.size(0), -1, x_t.size(1))  # B, k, T
        c_t = torch.bmm(att, V) # B*T, 1, D
        v = c_t.view(B, T, -1) # B, T, D


        # For modeling temporal dependencies
        a_t = self.relu(self.down_a(a))
        v = self.relu(self.down_v(v))
        a, _ = self.lstm_a(a_t) 
        v, _ = self.lstm_v(v) 


        # Regression
        output = torch.cat((a.squeeze(1), v.squeeze(1)), -1)
        output = self.relu(output)
        out = self.linear1(output)
        out=self.dro2(out)
        out = torch.flatten(out, start_dim=1)
        out = self.relu(self.p7(out))
        out = self.linear2(out)
        score = self.fc3(out)

        return score