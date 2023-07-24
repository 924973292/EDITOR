import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.epls) + ')'


class LocalRefinementUnits(nn.Module):
    def __init__(self, dim, out_dim=768, kernel=1):
        super().__init__()
        self.channels = dim
        self.out_dim = out_dim
        self.dwconv = nn.Conv2d(self.channels, self.channels, kernel, 1, padding=0, groups=self.channels)
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.ptconv = nn.Conv2d(self.channels, self.out_dim, 1, 1)
        self.bn2 = nn.BatchNorm2d(self.out_dim)
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()

    def forward(self, x):
        x = self.act1(self.bn1(self.dwconv(x)))
        x = self.act2(self.bn2(self.ptconv(x)))
        return x


class ChannelUnite(nn.Module):
    def __init__(self, dim, outdim, kernel):
        super().__init__()
        self.local = LocalRefinementUnits(dim=dim, out_dim=outdim, kernel=kernel)
        self.avg_pool = GeM()

    def forward(self, x, weight, height):
        b, c, _, _ = x.size()
        x = self.local(x)
        y = self.avg_pool(x).view(b, c)
        return y
