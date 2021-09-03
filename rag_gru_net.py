# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 11:22:52 2021

@author: Yan Wang
"""

import torch
from torch import nn
import torch.nn.functional as F

class GRUEncoderNetwork(nn.Module):
    """Encoder network for location finding example"""

    def __init__(self, design_dim, observation_dim, hidden_dim, encoding_dim):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.hidden_dim = hidden_dim
        self.design_dim = design_dim
        self.design_dim_flat = design_dim[0] * design_dim[1]  
        self.observation_dim = observation_dim
        input_dim = self.design_dim_flat + observation_dim
        # input_dim = self.design_dim_flat
        # input_dim = self.design_dim_flat * observation_dim
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                          num_layers=1, batch_first=True, dropout=0.1)        
        self.linear = nn.Linear(hidden_dim, encoding_dim)
        
        self.light_encoder = nn.Linear(observation_dim, input_dim)
        
    def forward(self, xi, y, h=None, **kwargs):
        x = xi.flatten(-2)
        # y = y.squeeze(-2)               
        # y = F.one_hot(y, num_classes=3)  
        # y = 2 * y - 1
        x = torch.cat([x, y], dim=-1) # When input_dim = self.design_dim_flat + observation_dim
        
        # siz = [1] * (len(xi.shape) - 1) + [self.observation_dim]
        # inputs = xi.repeat(*siz)
        # y_ = y.repeat_interleave(self.design_dim[1], -1)
        # inputs = inputs * y_
        
        # inputs = xi
        
        # y_ = self.light_encoder(y)
        # y_ = F.sigmoid(y_)
        # x = x + y_
        
        # if len(y.shape) >= 2:
        #     x = y[:, 0].unsqueeze(-1) * x_g + y[:, 1].unsqueeze(-1) * x_r + y[:, 2].unsqueeze(-1) * x_a
        # else:
        #     x = y[0] * x_g + y[1] * x_r + y[2] * x_a  
        
        # x = F.leaky_relu(x, 0.1)
        
        x = x.unsqueeze(1)
        x, h = self.gru(x, h)      
        x = x.squeeze(1)
        x = self.linear(x)
            
        return x, h

    
class GRUEmitterNetwork(nn.Module):
    """Emitter network for location finding example"""

    def __init__(self, encoding_dim, hidden_dim, design_dim):
        super().__init__()
        self.design_dim = design_dim
        self.design_dim_flat = design_dim[0] * design_dim[1]
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim,
                          num_layers=1, batch_first=True, dropout=0)
        self.hidden = nn.Linear(encoding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, self.design_dim_flat)
        self.layer_norm = nn.LayerNorm((hidden_dim,))

    def forward(self, emd, h=None):
        x = self.hidden(emd)        
        x = x.unsqueeze(1) # (N, L=1, H)        
        x = F.leaky_relu(x, negative_slope=0.1)
        x, h = self.gru(x, h)

        # x = self.layer_norm(x)
        # h = None
        # x = F.leaky_relu(x)
        x = x.squeeze(1)
        xi_flat = self.linear(x)
        return xi_flat.reshape(xi_flat.shape[:-1] + self.design_dim), h

