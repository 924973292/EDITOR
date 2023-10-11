import torch
import torch.nn as nn
from modeling.fusion_part.MLP import Mlp


class MaskCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.normx = nn.LayerNorm(dim)
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

    def forward(self, x, y, mask=None):
        B, N, C = y.shape
        q = self.q_(self.normx(x)).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_(self.normy(y)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = None
        if mask is not None:
            mask_q = torch.ones(B, 1, C).cuda()
            mask_q = mask_q.reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            mask = mask.unsqueeze(2).expand(B, N, C).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                                                3)
            mask = ~((mask_q @ mask.transpose(-2, -1)).div(C // self.num_heads)).bool()
            attn = attn.masked_fill(mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        # has_nan = torch.isnan(attn).any()
        # if has_nan:
        #     print("attn向量中包含NaN值")
        # else:
        #     print("attn向量中没有NaN值")

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DynamicTransfer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.MLP_key = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=1, act_layer=act_layer,
                           drop=drop)

        self.mask_temp = torch.tensor([0, 0, 0, 1, 1, 0, 0, 0,
                                       0, 0, 0, 1, 1, 0, 0, 0,
                                       0, 0, 1, 1, 1, 1, 0, 0,
                                       0, 1, 1, 1, 1, 1, 1, 0,
                                       0, 1, 1, 1, 1, 1, 1, 0,
                                       0, 1, 1, 1, 1, 1, 1, 0,
                                       0, 1, 1, 1, 1, 1, 1, 0,
                                       0, 1, 1, 1, 1, 1, 1, 0,
                                       0, 1, 1, 1, 1, 1, 1, 0,
                                       0, 0, 1, 1, 1, 1, 1, 0,
                                       0, 0, 1, 0, 0, 1, 0, 0,
                                       0, 0, 1, 0, 0, 1, 0, 0,
                                       0, 0, 1, 0, 0, 1, 0, 0,
                                       0, 0, 1, 0, 0, 1, 0, 0,
                                       0, 1, 1, 0, 0, 1, 1, 0,
                                       0, 1, 1, 0, 0, 1, 1, 0
                                       ]).cuda()

    def forward(self, x, y, height, weight, miss=None):
        if x is None:
            b, n, d = y.shape
        else:
            b, n, d = x.shape
        if miss:
            if miss == 'n':
                x = self.MLP_key(x).squeeze()
                x = x.reshape(b, height, weight)
                x_act = torch.tanh(x)
                x_mask = (x_act > 0).float()
                batch_size, height, width = x_act.size()
                x_mask = x_mask.view(batch_size, -1)
                num_zeros = torch.sum(x_mask == 0, dim=1)
                x_mask[num_zeros == n] = self.mask_temp.type(x_mask.dtype)
                return x_mask
            elif miss == 'r':
                y = self.MLP_key(y).squeeze()
                y = y.reshape(b, height, weight)
                y_act = torch.tanh(y)
                y_mask = (y_act > 0).float()
                batch_size, height, width = y_act.size()
                y_mask = y_mask.view(batch_size, -1)
                num_zeros = torch.sum(y_mask == 0, dim=1)
                y_mask[num_zeros == n] = self.mask_temp.type(y_mask.dtype)
                return y_mask
        else:
            y = self.MLP_key(y).squeeze()
            y = y.reshape(b, height, weight)
            y_act = torch.tanh(y)
            y_mask = (y_act > 0).float()
            batch_size, height, width = y_act.size()
            y_mask = y_mask.view(batch_size, -1)
            num_zeros = torch.sum(y_mask == 0, dim=1)
            y_mask[num_zeros > (n // 2)] = self.mask_temp.type(y_mask.dtype)

            x = self.MLP_key(x).squeeze()
            x = x.reshape(b, height, weight)
            x_act = torch.tanh(x)
            x_mask = (x_act > 0).float()
            batch_size, height, width = x_act.size()
            x_mask = x_mask.view(batch_size, -1)
            num_zeros = torch.sum(x_mask == 0, dim=1)
            x_mask[num_zeros > (n // 2)] = self.mask_temp.type(x_mask.dtype)

        return x_mask, y_mask


class DynamicTransferALL(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.DT = DynamicTransfer(dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=False, qk_scale=None, drop=0.,
                                  attn_drop=0.,
                                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.xy_Cross = MaskCrossAttention(dim, num_heads)

    def forward(self, x, y, height=16, weight=8, cross_type='r2n', miss='r'):
        x_cls = x[:, 0, :]
        x_patch = x[:, 1:, :]
        y_cls = y[:, 0, :]
        y_patch = y[:, 1:, :]
        x_mask, y_mask = self.DT(x_patch, y_patch, height, weight)
        x_cls = self.xy_Cross(x_cls, y_patch, y_mask)
        y_cls = self.xy_Cross(y_cls, x_patch, x_mask)
        x = torch.cat([x_cls.unsqueeze(1), x_patch], dim=1)
        y = torch.cat([y_cls.unsqueeze(1), y_patch], dim=1)
        if self.training:
            return torch.cat([x, y], dim=0)
        else:
            if not cross_type:
                return torch.cat([x, y], dim=0)
            else:
                if miss == 'n':
                    x_cls = x[:, 0, :]
                    patch = x[:, 1:, :]
                    mask = self.DT(patch, None, height, weight, miss=miss)
                    x_cls = self.xy_Cross(x_cls, patch, mask)
                    x = torch.cat([x_cls.unsqueeze(1), x_patch], dim=1)
                    return x
                elif miss == 'r':
                    y_cls = y[:, 0, :]
                    patch = y[:, 1:, :]
                    mask = self.DT(None, patch, height, weight, miss=miss)
                    y_cls = self.xy_Cross(y_cls, patch, mask)
                    y = torch.cat([y_cls.unsqueeze(1), y_patch], dim=1)
                    return y
