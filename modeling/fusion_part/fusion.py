import torch
import torch.nn as nn
from modeling.backbones.vit_pytorch import Block, DropPath, Mlp


class Shared_Encoding_Unit(nn.Module):
    def __init__(self, embed_dim=768,
                 depth=1,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., camera=0,
                 norm_layer=nn.LayerNorm, drop_path_rate=0.):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.norm = norm_layer(embed_dim)
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


class Attention_mix(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(dim)
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, cls, mid_fea_2):
        B, N, C = mid_fea_2.shape
        # cls = cls.reshape(B,1,C)
        q = self.q(cls).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(self.norm(mid_fea_2)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.v(mid_fea_2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = (attn @ v).transpose(1, 2).reshape(B, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block_mix(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_mix(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, cls, fea_kv):
        cls = cls + self.drop_path(self.attn(self.norm1(cls), fea_kv))
        cls = cls + self.drop_path(self.mlp(self.norm2(cls)))
        cls = cls.unsqueeze(1)
        return cls


class Mutual_Fsuion_Unit(nn.Module):

    def __init__(self, embed_dim=768,
                 depth=1,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., camera=0,
                 norm_layer=nn.LayerNorm, drop_path_rate=0.):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block_mix(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, cls, fea_kv):
        for blk in self.blocks:
            cls = blk(cls, fea_kv)
        return cls


class Reflection_item(nn.Module):

    def __init__(self, embed_dim=768,mode=0):
        super().__init__()
        self.mode = mode
        if self.mode == 0:
            self.sa = Shared_Encoding_Unit(embed_dim=embed_dim)
            self.res_q = Mutual_Fsuion_Unit(embed_dim=embed_dim)
            self.former_q = Mutual_Fsuion_Unit(embed_dim=embed_dim)
        elif self.mode ==1:
            self.sa = Shared_Encoding_Unit(embed_dim=embed_dim)
        elif self.mode ==2:
            self.res_q = Mutual_Fsuion_Unit(embed_dim=embed_dim)
            self.former_q = Mutual_Fsuion_Unit(embed_dim=embed_dim)
        elif self.mode == 3:
            self.sa = Shared_Encoding_Unit(embed_dim=embed_dim)
            self.res_q = Mutual_Fsuion_Unit(embed_dim=embed_dim)
            self.former_q = Mutual_Fsuion_Unit(embed_dim=embed_dim)

    def forward(self, res, former):
        if self.mode == 0:
            res = self.sa(res)
            former = self.sa(former)
            cls_q = self.res_q(res[:, 0, :], former[:, 1:, :])
            cls_f = self.former_q(former[:, 0, :], res[:, 1:, :])
        elif self.mode == 1:
            res = self.sa(res)
            former = self.sa(former)
            cls_q = res[:, 0, :].unsqueeze(1)
            cls_f = former[:, 0, :].unsqueeze(1)
        elif self.mode == 2:
            cls_q = self.res_q(res[:, 0, :], former[:, 1:, :])
            cls_f = self.former_q(former[:, 0, :], res[:, 1:, :])
        elif self.mode == 3:
            cls_q = self.res_q(res[:, 0, :], former[:, 1:, :])
            cls_f = self.former_q(former[:, 0, :], res[:, 1:, :])
            res = torch.cat([cls_q,res[:, 1:, :]],dim=-2)
            former = torch.cat([cls_f,former[:, 1:, :]],dim=-2)
            res = self.sa(res)
            former = self.sa(former)
            cls_q = res[:, 0, :].unsqueeze(1)
            cls_f = former[:, 0, :].unsqueeze(1)
        return cls_q, cls_f


class Self_Reflection_Module(nn.Module):

    def __init__(self, embed_dim=768, depth=6):
        super().__init__()
        self.mix_dim = embed_dim
        self.blocks = nn.ModuleList([Reflection_item(embed_dim=embed_dim) for i in range(depth)])

    def forward(self, res, former, fea_1, fea_2):
        for blk in self.blocks:
            fea_1, fea_2 = blk(torch.cat([fea_1, res], dim=1), torch.cat([fea_2, former], dim=1))
        return fea_1, fea_2
