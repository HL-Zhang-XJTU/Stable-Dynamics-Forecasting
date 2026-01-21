import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from model_STLT import STLT
from model_NLF import NLF


class SGDM(nn.Module):
    """
    stability-guaranteed dynamics modeling method (SGDM)
    """
    def __init__(self, len_in, dim_in, len_out, dim_out, nlf_layersize, scale=16,
                 avgPool_dim=8, d_model=32, n_heads=16, e_layers=3, d_layers=2, d_ff=64,
                 dropout=0.1, d=1.0, eps=0.01, beta=0.99):
        super().__init__()
        self.len_in = len_in
        self.dim_in = dim_in
        self.len_out = len_out
        self.dim_out = dim_out

        self.beta = beta
        self.fhat = STLT(enc_s=dim_in,
                         enc_t=len_in,
                         out_s=dim_out,
                         out_t=len_out,
                         scale=scale,
                         avgPool_dim=avgPool_dim,
                         d_model=d_model, n_heads=n_heads,
                         e_layers=e_layers,
                         d_layers=d_layers,
                         d_ff=d_ff,
                         dropout=dropout)
        self.V = NLF(layer_sizes=nlf_layersize, d=d, eps=eps)

    def forward(self, enc_x, dec_x):
        t0 = enc_x[:, -1:, :self.dim_out]
        yhat = self.fhat(enc_x, dec_x).cuda()
        _yhat = torch.cat([t0, yhat], dim=1).cuda()
        target = self.beta * self.V(_yhat[:, :-1, :])
        current = self.V(_yhat[:, 1:, :])
        fx = yhat * ((target - F.relu(target - current)) / current)
        return fx


class Dataset(data.Dataset):
    def __init__(self, seqs, y):
        super(Dataset, self).__init__()
        assert len(seqs) == len(y)
        self.seqs = torch.from_numpy(seqs).float()
        self.y = torch.from_numpy(y).float()
        self.len = len(seqs)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.seqs[item].cuda(), self.y[item].cuda()


