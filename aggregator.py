
import torch
import torch.nn as nn
import torch.nn.functional as F


class GNN(nn.Module):
    def __init__(self, hidden_size,  device):
        super(GNN, self).__init__()
        self.dim = hidden_size
        self.device = device
        self.W_x = nn.Linear(self.dim, self.dim)
        self.W_n = nn.Linear(self.dim, self.dim)
        self.W_w = nn.Linear(1, hidden_size)
        self.linear = nn.Linear(self.dim, 1)
        self.LeakRelu = nn.LeakyReLU(0.1)

    def forward(self, x,  x_nb , weight ):

        x_s = self.W_x(x) #  [n, 1,d]
        n_s = self.W_n(x_nb)  #  [n,m,d]
        w  = self.W_w(weight.unsqueeze(2))

        scores = self.linear(self.LeakRelu(x_s.unsqueeze(1) + n_s +  w))
        scores = scores.squeeze(-1)    ## [n, m]
        attn_scores = F.softmax(scores, dim=1)
        output = torch.matmul(attn_scores.unsqueeze(-2), x_nb)
        output = output.squeeze(-2)
        return output

