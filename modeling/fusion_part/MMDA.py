from itertools import repeat
import collections.abc as container_abcs
import torch
import torch.nn as nn
from .mv3 import _make_divisible
import torch.nn.functional as F
from ..backbones.vit_pytorch import Block


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class CA(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.normy = nn.LayerNorm(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, N, C = x.shape
        q = self.q_(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_(self.normy(y)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DCA(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.sample = nn.Linear(dim, 3)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def SelectSample(self, ma, mb, mc):
        gate = F.gumbel_softmax(self.sample(ma), hard=True)
        tokens = gate[:, :, 0].unsqueeze(2) * ma + gate[:, :, 1].unsqueeze(2) * mb + gate[:, :, 2].unsqueeze(2) * mc
        return tokens

    def forward(self, ma, mb, mc):
        tokens = self.SelectSample(ma, mb, mc)
        B, N, C = ma.shape
        q = self.q_(ma).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_(self.norm(tokens)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_(tokens).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.GELU(),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            nn.GELU()
        )
        print('New FFN here!!!')

    def forward(self, x, weight, height):
        b, l, c = x.shape
        x = x.reshape(b, weight, height, c).permute(0, 3, 1, 2)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = (x * y).flatten(2).transpose(1, 2)
        return out


class FuseUnit(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, mode=0):
        super().__init__()
        self.norma = norm_layer(dim)
        self.mode = mode
        if self.mode == 0:
            self.NI_attn = CA(dim=dim, num_heads=num_heads)
            self.TI_attn = CA(dim=dim, num_heads=num_heads)
            self.combine = nn.Linear(2 * dim, dim)
        elif self.mode == 1:
            self.DCA = DCA(dim=dim, num_heads=num_heads)
        elif self.mode == 2:
            self.NI_attn = CA(dim=dim, num_heads=num_heads)
            self.TI_attn = CA(dim=dim, num_heads=num_heads)
            self.combine = nn.Linear(2 * dim, dim)
            self.DCA = DCA(dim=dim, num_heads=num_heads)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(channel=dim)

    def forward(self, ma, mb, mc, weight=16, height=8):
        if self.mode == 0:
            mab = ma + self.drop_path(self.NI_attn(self.norma(ma), mb))
            mac = ma + self.drop_path(self.TI_attn(self.norma(ma), mc))
            ma = self.combine(torch.cat([mab, mac], dim=-1))
            ma = ma + self.drop_path(self.mlp(self.norm2(ma), weight=weight, height=height))
            return ma
        elif self.mode == 1:
            ma = ma + self.drop_path(self.DCA(self.norma(ma), mb, mc))
            ma = ma + self.drop_path(self.mlp(self.norm2(ma), weight=weight, height=height))
            return ma
        elif self.mode == 2:
            mab = ma + self.drop_path(self.NI_attn(self.norma(ma), mb))
            mac = ma + self.drop_path(self.TI_attn(self.norma(ma), mc))
            ma = self.combine(torch.cat([mab, mac], dim=-1))
            ma = ma + self.drop_path(self.mlp(self.norm2(ma), weight=weight, height=height))
            ma = ma + self.drop_path(self.DCA(self.norma(ma), mb, mc))
            ma = ma + self.drop_path(self.mlp(self.norm2(ma), weight=weight, height=height))
            return ma


class BlockFuse(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, depth=1, mode=0, mixdim=384):
        super().__init__()
        # self.reduction = nn.Linear(dim, mixdim)
        self.depth = depth
        self.blocks = nn.ModuleList()
        self.mode = mode
        for i in range(self.depth):
            self.blocks.append(
                FuseUnit(dim, num_heads, qkv_bias=False, qk_scale=None, drop=0.,
                         attn_drop=0.,
                         drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, mode=mode))

    def forward(self, ma, mb, mc, weight=16, height=8):
        for block in self.blocks:
            ma = block(ma, mb, mc, weight=weight, height=height)
        # ma = self.reduction(ma)
        return ma


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.normy = nn.LayerNorm(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, N, C = y.shape
        q = self.q_(x).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_(self.normy(y)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class BlockRotation(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, y):
        x = x + self.drop_path(self.attn(self.norm1(x), y))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class RotationAttention(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, mode=0):
        super().__init__()
        self.Rotation1 = BlockRotation(dim, num_heads, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop=0.,
                                       attn_drop=0.,
                                       drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        # self.Rotation2 = BlockRotation(dim, num_heads, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop=0.,
        #                                attn_drop=0.,
        #                                drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        # self.Rotation3 = BlockRotation(dim, num_heads, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop=0.,
        #                                attn_drop=0.,
        #                                drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)

        self.mode = mode
        if self.mode == 1:
            self.local_enchance = BlockFuse(dim, num_heads, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop=0.,
                                            attn_drop=0.,
                                            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)

    def forward(self, x, y, z):
        if self.mode == 0:
            x_cls = self.Rotation1(x[:, 0, :], y[:, 1:, :])
            y_cls = self.Rotation1(y[:, 0, :], z[:, 1:, :])
            z_cls = self.Rotation1(z[:, 0, :], x[:, 1:, :])
            x = torch.cat([x_cls.unsqueeze(1), x[:, 1:, :]], dim=-2)
            y = torch.cat([y_cls.unsqueeze(1), y[:, 1:, :]], dim=-2)
            z = torch.cat([z_cls.unsqueeze(1), z[:, 1:, :]], dim=-2)
            return x, y, z
        else:
            x_cls = self.Rotation1(x[:, 0, :], x[:, 1:, :])
            y_cls = self.Rotation1(y[:, 0, :], y[:, 1:, :])
            z_cls = self.Rotation1(z[:, 0, :], z[:, 1:, :])
            cls = torch.cat([x_cls, y_cls, z_cls], dim=-1)
            patch = torch.mean(self.local_enchance(x[:, 1:, :], y[:, 1:, :], z[:, 1:, :]),dim=-2)

            return cls, patch


class Rotation(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.Ro_start = RotationAttention(dim, num_heads)
        self.Ro_middle = RotationAttention(dim, num_heads)
        self.Ro_end = RotationAttention(dim, num_heads, mode=1)

    def forward(self, x, y, z):
        x, y, z = self.Ro_start(x=x, y=y, z=z)
        x, z, y = self.Ro_middle(x=x, y=z, z=y)
        cls, patch = self.Ro_end(x=x, y=y, z=z)
        return cls, patch
