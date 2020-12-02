import math

import torch.nn as nn


class SelfAttentionLayer(nn.Module):
    def __init__(self, input_dim, k_dim, v_dim):
        super().__init__()
        self.Wq = nn.Linear(input_dim, k_dim)  # dim of query must be the same as k_dim
        self.Wk = nn.Linear(input_dim, k_dim)
        self.Wv = nn.Linear(input_dim, v_dim)
        self.div_factor = math.sqrt(k_dim)
        
    
    def forward(self, x):
        """
        x is the embedding vectors of a sequence, of shape (N, T, D),
        N is the batch size, T is the length of the sequence and D is the
        dimension of the word embedding.
        """
        q = self.Wq(x)  # (N, T, k_dim)
        k = self.Wk(k)  # (N, T, k_dim)
        v = self.Wv(v)  # (N, T, v_dim)
        qk = torch.softmax(q @ k.permute(0, 2, 1) / self.div_factor, dim=2)  # (N, T, T)
        att = qk @ v  # (N, T, v_dim)
        return att
        