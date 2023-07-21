import torch
import torch.nn as nn
from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224
import torch.nn.functional as F
from modeling.backbones.swin import swin_base_patch4_win8
from modeling.backbones.ResTV2 import restv2_tiny, restv2_small, restv2_base, restv2_large
from modeling.backbones.edgeViT import edgevit_s
from modeling.backbones.t2tvit import t2t_vit_t_24, t2t_vit_t_14
from modeling.fusion_part.MMDA import BlockFuse


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


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


class build_transformer(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH_T
        self.mode = cfg.MODEL.RES_USE
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        if 't2t' in cfg.MODEL.TRANSFORMER_TYPE:
            self.in_planes = 512
        if 'edge' in cfg.MODEL.TRANSFORMER_TYPE or cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224':
            self.in_planes = 384
        if '14' in cfg.MODEL.TRANSFORMER_TYPE:
            self.in_planes = 384
        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        num_classes=num_classes,
                                                        camera=camera_num, view=view_num,
                                                        stride_size=cfg.MODEL.STRIDE_SIZE,
                                                        drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate=cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label=0, view_label=None):
        cash_x = self.base(x, cam_label=cam_label, view_label=view_label)
        global_feat = cash_x[-1][:, 0]
        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
            if self.mode == 0:
                return cls_score, global_feat
            else:
                return cash_x, cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                if self.mode == 0:
                    return feat
                else:
                    return cash_x, feat
            else:
                if self.mode == 0:
                    return global_feat
                else:
                    return cash_x, global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class Branch_new(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(Branch_new, self).__init__()
        self.NI = build_transformer(num_classes, cfg, camera_num, view_num, factory)
        self.TI = build_transformer(num_classes, cfg, camera_num, view_num, factory)
        self.RGB = build_transformer(num_classes, cfg, camera_num, view_num, factory)

        self.num_classes = num_classes
        self.cfg = cfg
        self.camera = camera_num
        self.view = view_num
        self.dim_l = 768
        self.num_head = 12
        self.mix_depth = cfg.MODEL.DEPTH
        self.mix_dim = cfg.MODEL.MIX_DIM
        self.cash0_fusion = BlockFuse(dim=self.dim_l, num_heads=self.num_head, depth=self.mix_depth, mode=0)
        self.cash1_fusion = BlockFuse(dim=self.dim_l, num_heads=self.num_head, depth=self.mix_depth, mode=0)
        self.cash2_fusion = BlockFuse(dim=self.dim_l, num_heads=self.num_head, depth=self.mix_depth, mode=0)
        self.cash3_fusion = BlockFuse(dim=self.dim_l, num_heads=self.num_head, depth=self.mix_depth, mode=2)

        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        self.patch_num = (16, 8)

        self.classifier_FUSE0 = nn.Linear(self.mix_dim, self.num_classes, bias=False)
        self.classifier_FUSE0.apply(weights_init_classifier)
        self.bottleneck_FUSE0 = nn.BatchNorm1d(self.mix_dim)
        self.bottleneck_FUSE0.bias.requires_grad_(False)
        self.bottleneck_FUSE0.apply(weights_init_kaiming)

        self.classifier_FUSE1 = nn.Linear(self.mix_dim, self.num_classes, bias=False)
        self.classifier_FUSE1.apply(weights_init_classifier)
        self.bottleneck_FUSE1 = nn.BatchNorm1d(self.mix_dim)
        self.bottleneck_FUSE1.bias.requires_grad_(False)
        self.bottleneck_FUSE1.apply(weights_init_kaiming)

        self.classifier_FUSE2 = nn.Linear(self.mix_dim, self.num_classes, bias=False)
        self.classifier_FUSE2.apply(weights_init_classifier)
        self.bottleneck_FUSE2 = nn.BatchNorm1d(self.mix_dim)
        self.bottleneck_FUSE2.bias.requires_grad_(False)
        self.bottleneck_FUSE2.apply(weights_init_kaiming)

        self.classifier_FUSE3 = nn.Linear(self.mix_dim, self.num_classes, bias=False)
        self.classifier_FUSE3.apply(weights_init_classifier)
        self.bottleneck_FUSE3 = nn.BatchNorm1d(self.mix_dim)
        self.bottleneck_FUSE3.bias.requires_grad_(False)
        self.bottleneck_FUSE3.apply(weights_init_kaiming)

        self.test_feat = cfg.TEST.FEAT
        self.miss_type = cfg.TEST.MISS

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def forward(self, x, label=None, cam_label=5, view_label=None):
        if self.training:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']
            NI_cash, NI_score, NI_global = self.NI(NI, cam_label=cam_label, view_label=view_label)
            TI_cash, TI_score, TI_global = self.TI(TI, cam_label=cam_label, view_label=view_label)
            RGB_cash, RGB_score, RGB_global = self.RGB(RGB, cam_label=cam_label, view_label=view_label)
            CASH0 = self.cash0_fusion(RGB_cash[0], NI_cash[0], TI_cash[0], *self.patch_num)
            CASH1 = self.cash1_fusion(CASH0 + RGB_cash[1], NI_cash[1], TI_cash[1], *self.patch_num)
            CASH2 = self.cash2_fusion(CASH1 + RGB_cash[2], NI_cash[2], TI_cash[2], *self.patch_num)
            CASH3 = self.cash3_fusion(CASH2 + RGB_cash[3], NI_cash[3], TI_cash[3], *self.patch_num)
            FUSE0 = torch.mean(CASH0, dim=-2)
            FUSE1 = torch.mean(CASH1, dim=-2)
            FUSE2 = torch.mean(CASH2, dim=-2)
            FUSE3 = torch.mean(CASH3, dim=-2)
            FUSE0_global = self.bottleneck_FUSE0(FUSE0)
            FUSE1_global = self.bottleneck_FUSE1(FUSE1)
            FUSE2_global = self.bottleneck_FUSE2(FUSE2)
            FUSE3_global = self.bottleneck_FUSE3(FUSE3)
            FUSE0_score = self.classifier_FUSE0(FUSE0_global)
            FUSE1_score = self.classifier_FUSE1(FUSE1_global)
            FUSE2_score = self.classifier_FUSE2(FUSE2_global)
            FUSE3_score = self.classifier_FUSE3(FUSE3_global)
            return NI_score, NI_global, TI_score, TI_global, RGB_score, RGB_global, \
                FUSE0_global, FUSE0_score, FUSE1_global, FUSE1_score, FUSE2_global, FUSE2_score, FUSE3_global, FUSE3_score

        else:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']
            NI_cash, NI_global = self.NI(NI, cam_label=cam_label, view_label=view_label)
            TI_cash, TI_global = self.TI(TI, cam_label=cam_label, view_label=view_label)
            RGB_cash, RGB_global = self.RGB(RGB, cam_label=cam_label, view_label=view_label)
            CASH0 = self.cash0_fusion(RGB_cash[0], NI_cash[0], TI_cash[0], *self.patch_num)
            CASH1 = self.cash1_fusion(CASH0 + RGB_cash[1], NI_cash[1], TI_cash[1], *self.patch_num)
            CASH2 = self.cash2_fusion(CASH1 + RGB_cash[2], NI_cash[2], TI_cash[2], *self.patch_num)
            CASH3 = self.cash3_fusion(CASH2 + RGB_cash[3], NI_cash[3], TI_cash[3], *self.patch_num)
            FUSE0 = torch.mean(CASH0, dim=-2)
            FUSE1 = torch.mean(CASH1, dim=-2)
            FUSE2 = torch.mean(CASH2, dim=-2)
            FUSE3 = torch.mean(CASH3, dim=-2)
            FUSE0_global = self.bottleneck_FUSE0(FUSE0)
            FUSE1_global = self.bottleneck_FUSE1(FUSE1)
            FUSE2_global = self.bottleneck_FUSE2(FUSE2)
            FUSE3_global = self.bottleneck_FUSE3(FUSE3)

            if self.neck_feat == 'after':
                pass
            else:
                FUSE0_global = FUSE0
                FUSE1_global = FUSE1
                FUSE2_global = FUSE2
                FUSE3_global = FUSE3
            if self.test_feat == 0:
                return torch.cat([NI_global, TI_global, RGB_global,
                                  FUSE0_global, FUSE1_global, FUSE2_global, FUSE3_global], dim=-1)


__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
    'swin_base_patch4_win8': swin_base_patch4_win8,
    'restv2_tiny': restv2_tiny,
    'restv2_small': restv2_small,
    'restv2_base': restv2_base,
    'restv2_large': restv2_large,
    'edgevit_s': edgevit_s,
    't2t_vit_t_24': t2t_vit_t_24,
    't2t_vit_t_14': t2t_vit_t_14
}


def make_model(cfg, num_class, camera_num, view_num=0, ):
    model = Branch_new(num_class, cfg, camera_num, view_num, __factory_T_type)
    print('===========Building MMReID Only===========')
    return model
