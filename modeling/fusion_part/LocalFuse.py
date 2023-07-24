import torch
import torch.nn as nn
from MLP import Mlp, MSCE, SE
from DropPath import DropPath


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


class FuseUnit(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, mode=0):
        super().__init__()
        self.norma = norm_layer(dim)
        self.mode = mode
        if self.mode == 0:
            self.NI_attn = CrossAttention(dim=dim, num_heads=num_heads)
            self.TI_attn = CrossAttention(dim=dim, num_heads=num_heads)
            self.combine = nn.Linear(2 * dim, dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        # self.mlp = MSCE(channel=dim)
        self.mlp = SE(channel=dim)
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, ma, mb, mc, weight=16, height=8):
        if self.mode == 0:
            mab = ma + self.drop_path(self.NI_attn(self.norma(ma), mb))
            mac = ma + self.drop_path(self.TI_attn(self.norma(ma), mc))
            ma = self.combine(torch.cat([mab, mac], dim=-1))
            ma = ma + self.drop_path(self.mlp(self.norm2(ma), weight=weight, height=height))
            return ma


class BlockFuse(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, depth=1, mode=0, mixdim=768):
        super().__init__()
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
        return ma
