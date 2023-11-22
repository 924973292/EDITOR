import torch.nn as nn
from modeling.backbones.vit_pytorch import PatchEmbed
from modeling.fusion_part.MLP import _make_divisible
import torch
from torch.nn.init import trunc_normal_

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, l, c = x.shape
        x = x.reshape(b, 16, 8, c).permute(0, 3, 1, 2).contiguous()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1).permute(0, 2, 1)
        return y


class ScaleChannel(nn.Module):
    def __init__(self, dim, init_value=1e-4):
        super().__init__()
        self.patch_s1 = PatchEmbed(img_size=(128, 64), patch_size=8, in_chans=3, embed_dim=768)
        self.patch_s2 = PatchEmbed(img_size=(64, 32), patch_size=4, in_chans=3, embed_dim=768)
        self.patch_s3 = PatchEmbed(img_size=(32, 16), patch_size=2, in_chans=3, embed_dim=768)
        self.patch_s4 = PatchEmbed(img_size=(16, 8), patch_size=1, in_chans=3, embed_dim=768)
        self.channel = SELayer(768)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self, x1, x2, x3, x4):
        x1 = self.patch_s1(x1)
        x2 = self.patch_s2(x2)
        x3 = self.patch_s3(x3)
        x4 = self.patch_s4(x4)
        x = (x1 + x2 + x3 + x4) / 4
        x = self.channel(x)
        return x

