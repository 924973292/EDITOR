import torch
import torch.nn as nn
from ..fusion_part.MLP import Mlp
from ..fusion_part.DropPath import DropPath


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
        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RotationAttention(nn.Module):

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


class BlockRotation(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, mode=0):
        super().__init__()
        self.Rotation = RotationAttention(dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                                          attn_drop=0.,
                                          drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.mode = mode

    def forward(self, x, y, z):
        if self.mode == 0:
            x_cls = self.Rotation(x[:, 0, :], y[:, 1:, :])
            y_cls = self.Rotation(y[:, 0, :], z[:, 1:, :])
            z_cls = self.Rotation(z[:, 0, :], x[:, 1:, :])
            x = torch.cat([x_cls.unsqueeze(1), x[:, 1:, :]], dim=-2)
            y = torch.cat([y_cls.unsqueeze(1), y[:, 1:, :]], dim=-2)
            z = torch.cat([z_cls.unsqueeze(1), z[:, 1:, :]], dim=-2)
            return x, y, z
        else:
            x_cls = self.Rotation(x[:, 0, :], x[:, 1:, :])
            y_cls = self.Rotation(y[:, 0, :], y[:, 1:, :])
            z_cls = self.Rotation(z[:, 0, :], z[:, 1:, :])
            cls = torch.cat([x_cls, y_cls, z_cls], dim=-1)
            return cls


class BlockRotationCross(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, mode=0):
        super().__init__()
        self.Rotation = RotationAttention(dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                                          attn_drop=0.,
                                          drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.mode = mode

    def forward(self, x, y):
        if self.mode == 0:
            x_cls = self.Rotation(x[:, 0, :], y[:, 1:, :])
            y_cls = self.Rotation(y[:, 0, :], x[:, 1:, :])
            x = torch.cat([x_cls.unsqueeze(1), x[:, 1:, :]], dim=-2)
            y = torch.cat([y_cls.unsqueeze(1), y[:, 1:, :]], dim=-2)
            return x, y
        else:
            x_cls = self.Rotation(x[:, 0, :], x[:, 1:, :])
            y_cls = self.Rotation(y[:, 0, :], y[:, 1:, :])
            cls = torch.cat([x_cls, y_cls], dim=-1)
            return cls


class Rotation(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, cross=False):
        super().__init__()
        self.cross = cross
        if self.cross:
            self.Ro_start = BlockRotationCross(dim, num_heads)
            self.Ro_end = BlockRotationCross(dim, num_heads, mode=1)
        else:
            self.Ro_start = BlockRotation(dim, num_heads)
            self.Ro_middle = BlockRotation(dim, num_heads)
            self.Ro_end = BlockRotation(dim, num_heads, mode=1)

    def forward(self, x, y, z):
        if self.cross:
            x, y = self.Ro_start(x=x, y=y)
            cls = self.Ro_end(x=x, y=y)
            return cls
        else:
            x, y, z = self.Ro_start(x=x, y=y, z=z)
            x, z, y = self.Ro_middle(x=x, y=z, z=y)
            cls = self.Ro_end(x=x, y=y, z=z)
            return cls

