"""
This is a pytorch implementation of the sparse autoencoder technique used for learning LLM features in Anthropic's Towards Monosemanticity work
"""

import torch

class SpAE(torch.nn.Module):
    def __init__(self, 
        act_dim: int, 
        over_comp_factor: int 8
    ):
        super().__init__()
        self.hidden_d = over_comp_factor * act_dim
        self.enc = torch.nn.Linear(act_dim, self.hidden_d)
        self.dec = torch.nn.Linear(self.hidden_d, act_dim)

    def encode(self, x):
        return torch.nn.ReLU(self.enc(x - self.dec.bias))

    def decode(self, f):
        return self.dec(f)

    def forward(self, x):
        return self.decode(self.encode(x))
        
