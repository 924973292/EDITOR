import torch
import torch.nn as nn
from ..fusion_part.MLP import Mlp, MSCE, SE
from ..fusion_part.DropPath import DropPath
from ..backbones.vit_pytorch import Attention


class ReUnit(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, weight=16, height=8):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x), weight=weight, height=height))
        return x


class ReBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, depth=1, mode=0, mixdim=768):
        super().__init__()
        self.depth = depth
        self.mode = mode
        self.reunite = ReUnit(dim, num_heads, qkv_bias=False, qk_scale=None, drop=0.,
                              attn_drop=0.,
                              drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)

    def forward(self, x, weight=16, height=8):
        x = self.reunite(x, weight=weight, height=height)
        return x


class Reconstruct(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, depth=1, mode=0, mixdim=768):
        super().__init__()
        self.re1 = ReBlock(dim, num_heads, depth=depth)
        self.re2 = ReBlock(dim, num_heads, depth=depth)

    def forward(self, x, weight=16, height=8):
        re1 = self.re1(x, weight=weight, height=height)
        re2 = self.re2(x, weight=weight, height=height)
        return re1, re2


class ReconstructAll(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, depth=1, mode=0, mixdim=768, miss='0'):
        super().__init__()
        self.RGBRE = Reconstruct(dim, num_heads, depth=depth)
        self.NIRE = Reconstruct(dim, num_heads, depth=depth)
        self.TIRE = Reconstruct(dim, num_heads, depth=depth)
        self.miss = miss

    def forward(self, ma, mb, mc, weight=16, height=8, cross_miss=None):
        if self.training:
            if cross_miss:
                if cross_miss == 'nt':
                    RGB_NI, RGB_TI = self.RGBRE(ma, weight=16, height=8)
                    return RGB_NI + ma, RGB_TI + ma
                elif cross_miss == 'rt':
                    NI_RGB, NI_TI = self.NIRE(mb, weight=16, height=8)
                    return NI_RGB + mb, NI_TI + mb
                elif cross_miss == 'rn':
                    TI_RGB, TI_NI = self.TIRE(mc, weight=16, height=8)
                    return TI_RGB + mc, TI_NI + mc
                elif cross_miss == 'r':
                    NI_RGB, NI_TI = self.NIRE(mb, weight=16, height=8)
                    TI_RGB, TI_NI = self.TIRE(mc, weight=16, height=8)
                    return (NI_RGB + mb + TI_RGB + mc) / 2
                elif cross_miss == "n":
                    RGB_NI, RGB_TI = self.RGBRE(ma, weight=16, height=8)
                    TI_RGB, TI_NI = self.TIRE(mc, weight=16, height=8)
                    return (RGB_NI + ma + TI_NI + mc) / 2
                elif cross_miss == 't':
                    RGB_NI, RGB_TI = self.RGBRE(ma, weight=16, height=8)
                    NI_RGB, NI_TI = self.NIRE(mb, weight=16, height=8)
                    return (RGB_TI + ma + NI_TI + mb) / 2
            else:
                RGB_NI, RGB_TI = self.RGBRE(ma, weight=16, height=8)
                NI_RGB, NI_TI = self.NIRE(mb, weight=16, height=8)
                TI_RGB, TI_NI = self.TIRE(mc, weight=16, height=8)
                loss_rgb = nn.MSELoss()(RGB_NI, mb - ma) + nn.MSELoss()(RGB_TI, mc - ma)
                loss_ni = nn.MSELoss()(NI_RGB, ma - mb) + nn.MSELoss()(NI_TI, mc - mb)
                loss_ti = nn.MSELoss()(TI_RGB, ma - mc) + nn.MSELoss()(TI_NI, mb - mc)
                loss = loss_rgb + loss_ni + loss_ti
                return loss
        else:
            if cross_miss:
                if cross_miss == 'nt':
                    RGB_NI, RGB_TI = self.RGBRE(ma, weight=16, height=8)
                    return RGB_NI + ma, RGB_TI + ma
                elif cross_miss == 'rt':
                    NI_RGB, NI_TI = self.NIRE(mb, weight=16, height=8)
                    return NI_RGB + mb, NI_TI + mb
                elif cross_miss == 'rn':
                    TI_RGB, TI_NI = self.TIRE(mc, weight=16, height=8)
                    return TI_RGB + mc, TI_NI + mc
                elif cross_miss == 'r':
                    NI_RGB, NI_TI = self.NIRE(mb, weight=16, height=8)
                    TI_RGB, TI_NI = self.TIRE(mc, weight=16, height=8)
                    return (NI_RGB + mb + TI_RGB + mc) / 2
                elif cross_miss == "n":
                    RGB_NI, RGB_TI = self.RGBRE(ma, weight=16, height=8)
                    TI_RGB, TI_NI = self.TIRE(mc, weight=16, height=8)
                    return (RGB_NI + ma + TI_NI + mc) / 2
                elif cross_miss == 't':
                    RGB_NI, RGB_TI = self.RGBRE(ma, weight=16, height=8)
                    NI_RGB, NI_TI = self.NIRE(mb, weight=16, height=8)
                    return (RGB_TI + ma + NI_TI + mb) / 2
            else:
                if self.miss == None:
                    pass
                elif self.miss == 'r':
                    NI_RGB, NI_TI = self.NIRE(mb, weight=16, height=8)
                    TI_RGB, TI_NI = self.TIRE(mc, weight=16, height=8)
                    return (NI_RGB + mb + TI_RGB + mc) / 2
                elif self.miss == "n":
                    RGB_NI, RGB_TI = self.RGBRE(ma, weight=16, height=8)
                    TI_RGB, TI_NI = self.TIRE(mc, weight=16, height=8)
                    return (RGB_NI + ma + TI_NI + mc) / 2
                elif self.miss == 't':
                    RGB_NI, RGB_TI = self.RGBRE(ma, weight=16, height=8)
                    NI_RGB, NI_TI = self.NIRE(mb, weight=16, height=8)
                    return (RGB_TI + ma + NI_TI + mb) / 2
                elif self.miss == 'rn':
                    TI_RGB, TI_NI = self.TIRE(mc, weight=16, height=8)
                    return TI_RGB + mc, TI_NI + mc
                elif self.miss == 'rt':
                    NI_RGB, NI_TI = self.NIRE(mb, weight=16, height=8)
                    return NI_RGB + mb, NI_TI + mb
                elif self.miss == 'nt':
                    RGB_NI, RGB_TI = self.RGBRE(ma, weight=16, height=8)
                    return RGB_NI + ma, RGB_TI + ma
