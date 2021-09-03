# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 16:11:21 2021

@author: Yan Wang
"""

import math

import torch
import torch.nn.functional as F
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions import Categorical
from scipy.special import gamma

class GumbelSoftmax(TorchDistribution):
    
    def __init__(self, probs, tau=1.0):
        if type(probs) != torch.Tensor:
            raise TypeError("The input probs must be of tensor type")           
        batch_shape, event_shape = self.infer_shapes(probs)
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        self.tau = tau
        self.probs = probs
        self.k = probs.shape[-1]
        loc = torch.tensor(0., dtype=self.probs.dtype, device=self.probs.device)
        scale = torch.tensor(1., dtype=self.probs.dtype, device=self.probs.device)
        self.gumbel = torch.distributions.Gumbel(loc, scale)
        self.gamma_k = gamma(self.k) # k!
        self.has_rsample = True
        
    @staticmethod
    def infer_shapes(probs, tau=1.0):
        batch_shape, event_shape = probs.shape[:-1], probs.shape[-1:]
        return batch_shape, event_shape
    
    def log_prob(self, value):
        """
        Log of likelihood given the input value. For more information, refer to
        Jang, et al. "Categorical Reparameterization with Gumbel-Softmax", ICLR, 2017

        Parameters
        ----------
        value : TYPE
            Input value (observation).

        Returns
        -------
        result : TYPE
            Logarithm of likelihood.

        """
        # p(y_1,...,y_k) = gamma(k) \tau^(k-1) (Sigma_{1}{k} pi_i/y_i^\tau)^{-k} Prod_{1}{k} pi_i/y_i^{\tau+1}
        probs = self.probs.expand_as(value)
        result = math.log(self.gamma_k) + (self.k - 1) * math.log(self.tau) - \
            self.k * torch.log((probs / (value.pow(self.tau) + 1e-6)).sum(-1) + 1e-6) + \
                torch.log(probs + 1e-6).sum(-1) - (self.tau + 1) * torch.log(value + 1e-6).sum(-1)
                
        return result
    
    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        g = self.gumbel.sample(shape)
        logits = (torch.log(self.probs + 1e-10) + g) / self.tau
        y = F.softmax(logits, dim=-1)
        return y


if __name__ == "__main__":
    probs = torch.tensor([[0.1, 0.5, 0.4], [0.2, 0.4, 0.4]], requires_grad=True)
    
    cat = Categorical(probs=probs)
    s = cat.sample([10])
    
    gst = GumbelSoftmax(probs, tau=0.2)
    s = gst.rsample([10])
    log_p = gst.log_prob(s)