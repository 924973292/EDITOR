import torch
import torch.nn as nn
from MLP import Mlp
from LocalFuse import BlockFuse, CrossAttention
from DropPath import DropPath


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
        self.Rotation = RotationAttention(dim, num_heads, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop=0.,
                                          attn_drop=0.,
                                          drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)

        self.mode = mode
        if self.mode == 1:
            self.local_enchance = BlockFuse(dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                                            attn_drop=0.,
                                            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)

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
            patch = torch.mean(self.local_enchance(x[:, 1:, :], y[:, 1:, :], z[:, 1:, :]), dim=-2)

            return cls, patch


class Rotation(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.Ro_start = BlockRotation(dim, num_heads)
        self.Ro_middle = BlockRotation(dim, num_heads)
        self.Ro_end = BlockRotation(dim, num_heads, mode=1)

    def forward(self, x, y, z):
        x, y, z = self.Ro_start(x=x, y=y, z=z)
        x, z, y = self.Ro_middle(x=x, y=z, z=y)
        cls, patch = self.Ro_end(x=x, y=y, z=z)
        return cls, patch
