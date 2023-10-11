import torch
import torch.nn as nn
from ..fusion_part.MultiScaleCE import ChannelUnite


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SE(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.GELU(),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            nn.GELU()
        )
        print('SE HERE!!!')

    def forward(self, x, weight, height):
        b, l, c = x.shape
        cls = x[:, 0, :]
        x = x[:, 1:, :]
        x = x.reshape(b, weight, height, c).permute(0, 3, 1, 2).contiguous()
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1, 1)
        out = (x * y).flatten(2).transpose(1, 2)
        out = torch.cat([cls.unsqueeze(1), out], dim=-2)
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, height=16, weight=8):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MSCE(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.local_1 = ChannelUnite(dim=channel, outdim=channel, kernel=1)
        self.local_3 = ChannelUnite(dim=channel, outdim=channel, kernel=3)
        self.local_5 = ChannelUnite(dim=channel, outdim=channel, kernel=5)
        self.local_7 = ChannelUnite(dim=channel, outdim=channel, kernel=7)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.GELU(),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            nn.GELU()
        )
        print('MSCE HERE!!!')

    def forward(self, x, weight, height):
        b, l, c = x.shape
        x = x.reshape(b, weight, height, c).permute(0, 3, 1, 2).contiguous()
        c_1 = self.local_1(x, weight, height)
        c_3 = self.local_3(x, weight, height)
        c_5 = self.local_5(x, weight, height)
        c_7 = self.local_7(x, weight, height)
        b, c, _, _ = x.size()
        y = c_1 + c_3 + c_5 + c_7
        y = self.fc(y).view(b, c, 1, 1)
        out = (x * y).flatten(2).transpose(1, 2)
        return out
