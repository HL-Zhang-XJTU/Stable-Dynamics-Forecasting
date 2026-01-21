import torch
import torch.nn as nn
import torch.nn.functional as F


class ReHU(nn.Module):
    def __init__(self, d=1.0):
        super().__init__()
        self.a = 1/(2*d)
        self.b = d/2

    def forward(self, x):
        return torch.max(torch.clamp(torch.sign(x)*self.a*x**2, min=0, max=self.b), x-self.b)


class ICNN(nn.Module):
    def __init__(self, layer_sizes, d=1.0):
        super().__init__()

        self.W = nn.ParameterList([nn.Parameter(torch.Tensor(l, layer_sizes[0]))
                                   for l in layer_sizes[1:]]).cuda()
        self.U = nn.ParameterList([nn.Parameter(torch.Tensor(layer_sizes[i+1], layer_sizes[i]))
                                   for i in range(1, len(layer_sizes)-1)]).cuda()
        self.bias = nn.ParameterList([nn.Parameter(torch.Tensor(l)) for l in layer_sizes[1:]]).cuda()

        self.act = ReHU(d)
        self.reset_parameters()

    def reset_parameters(self):
        for W in self.W:
            nn.init.kaiming_uniform_(W, a=5**0.5)
        for U in self.U:
            nn.init.kaiming_uniform_(U, a=5**0.5)
        for i,b in enumerate(self.bias):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W[i])
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(b, -bound, bound)

    def forward(self, x):
        z = F.linear(x, self.W[0], self.bias[0])
        z = self.act(z)

        for W, b, U in zip(self.W[1:-1], self.bias[1:-1], self.U[:-1]):
            z = F.linear(x, W, b) + F.linear(z, F.softplus(U)) / U.shape[0]
            z = self.act(z)
        res = (F.linear(x, self.W[-1], self.bias[-1])
               + F.linear(z, F.softplus(self.U[-1])) / self.U[-1].shape[0])
        return res


class NLF(nn.Module):
    """ Neural Lyapunov Function"""
    def __init__(self, layer_sizes=[6,50,50,6], d=1.0, eps=0.01):
        super().__init__()
        self.f = ICNN(layer_sizes, d)
        self.eps = eps
        self.d = d
        self.rehu = ReHU(self.d)

    def forward(self, x):
        zero = self.f(torch.zeros((1, 1, x.shape[-1])).cuda())
        smoothed_output = self.rehu(self.f(x) - zero)
        quadratic_under = self.eps*(torch.norm(x, dim=-1, keepdim=True)**2)

        return smoothed_output + quadratic_under



