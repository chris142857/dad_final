# -*- coding: utf-8 -*-
"""
Encoder emitter networks for RNN
"""

import torch
from torch import nn
import torch.nn.functional as F

class GRUEncoderNetwork(nn.Module):
    """Encoder network for location finding example"""

    def __init__(self, design_dim, osbervation_dim, hidden_dim, encoding_dim):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.hidden_dim = hidden_dim
        self.design_dim_flat = design_dim[0] * design_dim[1]
        input_dim = self.design_dim_flat + osbervation_dim
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                          num_layers=1, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(hidden_dim, encoding_dim)
        
    def forward(self, xi, y, h=None, **kwargs):
        xi = xi.flatten(-2)
        inputs = torch.cat([xi, y], dim=-1)
        inputs = inputs.unsqueeze(1)
        x, h = self.gru(inputs, h)
        x = x.squeeze(1)
        # x = F.leaky_relu(x, 0.1)
        x = self.linear(x)
        return x, h

    
class GRUEmitterNetwork(nn.Module):
    """Emitter network for location finding example"""

    def __init__(self, encoding_dim, hidden_dim, design_dim):
        super().__init__()
        self.design_dim = design_dim
        self.design_dim_flat = design_dim[0] * design_dim[1]
        self.gru = nn.GRU(input_size=encoding_dim, hidden_size=hidden_dim,
                          num_layers=1, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(hidden_dim, self.design_dim_flat)

    def forward(self, emd, h=None):
        x = emd.unsqueeze(1) # (N, L=1, H)
        x = F.leaky_relu(x, negative_slope=0.1)
        x, h = self.gru(x, h)
        xi_flat = self.linear(x)
        xi_flat = xi_flat.squeeze(1)
        return xi_flat.reshape(xi_flat.shape[:-1] + self.design_dim), h

