import torch
import torch.nn as nn
from modeling.backbones.resnet import ResNet, Bottleneck
from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224, BlockRe, ReAttention
import torch.nn.functional as F
from modeling.backbones.swin import swin_base_patch4_win8
from modeling.backbones.ResTV2 import restv2_tiny, restv2_small, restv2_base, restv2_large
from modeling.backbones.edgeViT import edgevit_s
from modeling.backbones.t2tvit import t2t_vit_t_24, t2t_vit_t_14


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
            self.eps) + ')'


class LocalRefinementUnits(nn.Module):
    def __init__(self, dim, out_dim=768, kernel=1, choice=True):
        super().__init__()
        self.LRU = choice
        self.channels = dim
        self.out_dim = out_dim
        self.dwconv = nn.Conv2d(self.channels, self.channels, kernel, 1, padding=0, groups=self.channels)
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.ptconv = nn.Conv2d(self.channels, self.out_dim, 1, 1)
        self.bn2 = nn.BatchNorm2d(self.out_dim)
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        self.act = nn.ReLU()

    def forward(self, x):
        if self.LRU:
            x = self.act1(self.bn1(self.dwconv(x)))
            x = self.act2(self.bn2(self.ptconv(x)))
        else:
            x = self.act2(self.bn2(self.ptconv(x)))
        return x


class BuildTransformer(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(BuildTransformer, self).__init__()
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
        mid_fea = self.base(x, cam_label=cam_label, view_label=view_label)
        global_feat = mid_fea[:, 0]
        mid_fea_f = mid_fea[:, 1:, :]
        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
            if self.mode == 0:
                return cls_score, global_feat
            else:
                return mid_fea_f, cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                if self.mode == 0:
                    return feat
                else:
                    return mid_fea_f, feat
            else:
                if self.mode == 0:
                    return global_feat
                else:
                    return mid_fea_f, global_feat

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


class MMReID(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(MMReID, self).__init__()
        self.NI = BuildTransformer(num_classes, cfg, camera_num, view_num, factory)
        self.TI = BuildTransformer(num_classes, cfg, camera_num, view_num, factory)
        self.RGB = BuildTransformer(num_classes, cfg, camera_num, view_num, factory)

        self.num_classes = num_classes
        self.cfg = cfg
        self.camera = camera_num
        self.view = view_num
        self.dim_l = 768
        self.mix_dim = cfg.MODEL.MIX_DIM

        self.NI_LRU = LocalRefinementUnits(dim=self.dim_l, out_dim=self.mix_dim)
        self.TI_LRU = LocalRefinementUnits(dim=self.dim_l, out_dim=self.mix_dim)
        self.RGB_LRU = LocalRefinementUnits(dim=self.dim_l, out_dim=self.mix_dim)
        self.NI_GAP = GeM()
        self.TI_GAP = GeM()
        self.RGB_GAP = GeM()

        self.srm_layer = cfg.MODEL.SRM_LAYER
        if self.srm_layer:
            print("FUSED HERE!!!")
            self.RENI = ReAttention(dim=self.mix_dim)
            self.RETI = ReAttention(dim=self.mix_dim)
            # self.Cross_RGB = CYCLE(dim=self.mix_dim)
            # self.Cross_NI = CYCLE(dim=self.mix_dim)
            # self.Cross_TI = CYCLE(dim=self.mix_dim)
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        self.patch_num = (16, 8, 768)

        self.classifier_NIrefine = nn.Linear(self.mix_dim, self.num_classes, bias=False)
        self.classifier_NIrefine.apply(weights_init_classifier)
        self.bottleneck_NIrefine = nn.BatchNorm1d(self.mix_dim)
        self.bottleneck_NIrefine.bias.requires_grad_(False)
        self.bottleneck_NIrefine.apply(weights_init_kaiming)

        self.classifier_TIrefine = nn.Linear(self.mix_dim, self.num_classes, bias=False)
        self.classifier_TIrefine.apply(weights_init_classifier)
        self.bottleneck_TIrefine = nn.BatchNorm1d(self.mix_dim)
        self.bottleneck_TIrefine.bias.requires_grad_(False)
        self.bottleneck_TIrefine.apply(weights_init_kaiming)

        self.classifier_RGBrefine = nn.Linear(self.mix_dim, self.num_classes, bias=False)
        self.classifier_RGBrefine.apply(weights_init_classifier)
        self.bottleneck_RGBrefine = nn.BatchNorm1d(self.mix_dim)
        self.bottleneck_RGBrefine.bias.requires_grad_(False)
        self.bottleneck_RGBrefine.apply(weights_init_kaiming)

        self.classifier_RGB_V = nn.Linear(self.mix_dim, self.num_classes, bias=False)
        self.classifier_RGB_V.apply(weights_init_classifier)
        self.bottleneck_RGB_V = nn.BatchNorm1d(self.mix_dim)
        self.bottleneck_RGB_V.bias.requires_grad_(False)
        self.bottleneck_RGB_V.apply(weights_init_kaiming)

        self.classifier_NI_V = nn.Linear(self.mix_dim, self.num_classes, bias=False)
        self.classifier_NI_V.apply(weights_init_classifier)
        self.bottleneck_NI_V = nn.BatchNorm1d(self.mix_dim)
        self.bottleneck_NI_V.bias.requires_grad_(False)
        self.bottleneck_NI_V.apply(weights_init_kaiming)

        self.classifier_TI_V = nn.Linear(self.mix_dim, self.num_classes, bias=False)
        self.classifier_TI_V.apply(weights_init_classifier)
        self.bottleneck_TI_V = nn.BatchNorm1d(self.mix_dim)
        self.bottleneck_TI_V.bias.requires_grad_(False)
        self.bottleneck_TI_V.apply(weights_init_kaiming)
        self.decay = 1
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
            B = RGB.shape[0]
            NI_fea, NI_score, NI_global = self.NI(NI)
            TI_fea, TI_score, TI_global = self.TI(TI)
            RGB_fea, RGB_score, RGB_global = self.RGB(RGB, cam_label=cam_label, view_label=view_label)

            NI_refine = self.NI_LRU(NI_fea.reshape(B, *self.patch_num).permute(0, 3, 1, 2))
            TI_refine = self.TI_LRU(TI_fea.reshape(B, *self.patch_num).permute(0, 3, 1, 2))
            RGB_refine = self.RGB_LRU(RGB_fea.reshape(B, *self.patch_num).permute(0, 3, 1, 2))

            NI_refine_global = self.NI_GAP(NI_refine).view(B, -1)
            TI_refine_global = self.TI_GAP(TI_refine).view(B, -1)
            RGB_refine_global = self.RGB_GAP(RGB_refine).view(B, -1)

            NI_refine = NI_refine.flatten(-2).transpose(1, 2)
            TI_refine = TI_refine.flatten(-2).transpose(1, 2)
            RGB_refine = RGB_refine.flatten(-2).transpose(1, 2)

            RGB_NI, NI_RE = self.RENI(RGB_refine)
            RGB_TI, TI_RE = self.RETI(RGB_refine)

            feat_NIRefine = self.bottleneck_NIrefine(NI_refine_global)
            feat_TIRefine = self.bottleneck_TIrefine(TI_refine_global)
            feat_RGBRefine = self.bottleneck_RGBrefine(RGB_refine_global)
            feat_NI_V = self.bottleneck_NI_V(torch.mean(RGB_NI, dim=1))
            feat_TI_V = self.bottleneck_TI_V(torch.mean(RGB_TI, dim=1))

            score_NIRefine = self.classifier_NIrefine(feat_NIRefine)
            score_TIRefine = self.classifier_TIrefine(feat_TIRefine)
            score_RGBRefine = self.classifier_RGBrefine(feat_RGBRefine)
            score_NI_V = self.classifier_NI_V(feat_NI_V)
            score_TI_V = self.classifier_TI_V(feat_TI_V)

            loss_reni = nn.MSELoss()(NI_RE, NI_refine)
            loss_reti = nn.MSELoss()(TI_RE, TI_refine)

            return NI_score, NI_global, TI_score, TI_global, RGB_score, RGB_global, \
                score_NIRefine, feat_NIRefine, score_TIRefine, feat_TIRefine, score_RGBRefine, feat_RGBRefine, \
                score_NI_V, feat_NI_V, score_TI_V, feat_TI_V, loss_reni, loss_reti

        else:
            if self.miss_type == 'NIR':
                RGB = x['RGB']
                TI = x['TI']
                B = RGB.shape[0]
                TI_fea, TI_global = self.TI(TI, cam_label=cam_label, view_label=view_label)
                RGB_fea, RGB_global = self.RGB(RGB, cam_label=cam_label, view_label=view_label)

                TI_refine = self.TI_LRU(TI_fea.reshape(B, *self.patch_num).permute(0, 3, 1, 2))
                RGB_refine = self.RGB_LRU(RGB_fea.reshape(B, *self.patch_num).permute(0, 3, 1, 2))

                TI_refine_global = self.TI_GAP(TI_refine).view(B, -1)
                RGB_refine_global = self.RGB_GAP(RGB_refine).view(B, -1)

                RGB_refine = RGB_refine.flatten(-2).transpose(1, 2)
                TI_refine = TI_refine.flatten(-2).transpose(1, 2)
                RGB_NI, NI_RE = self.RENI(RGB_refine)
                # RGB_TI, TI_RE = self.RETI(RGB_refine)

                NI_RE_NI = NI_RE
                TI_RE_TI = TI_refine

                RGB_V = self.Cross_RGB(NI_RE_NI, TI_RE_TI, RGB_refine)
                NI_V = self.Cross_NI(RGB_refine, TI_RE_TI, NI_RE_NI)
                TI_V = self.Cross_TI(NI_RE_NI, RGB_refine, TI_RE_TI)

                feat_TIRefine = self.bottleneck_TIrefine(TI_refine_global)
                feat_RGBRefine = self.bottleneck_RGBrefine(RGB_refine_global)
                feat_RGB_V = self.bottleneck_RGB_V(torch.mean(RGB_V, dim=1))
                feat_NI_V = self.bottleneck_NI_V(torch.mean(NI_V, dim=1))
                feat_TI_V = self.bottleneck_TI_V(torch.mean(TI_V, dim=1))

                if self.neck_feat == 'after':
                    pass
                else:
                    feat_TIRefine = TI_refine_global
                    feat_RGBRefine = RGB_refine_global
                    feat_RGB_V = torch.mean(RGB_V, dim=1)
                    feat_NI_V = torch.mean(NI_V, dim=1)
                    feat_TI_V = torch.mean(TI_V, dim=1)
                if self.test_feat == 0:
                    return torch.cat([TI_global, RGB_global, \
                                      feat_TIRefine, feat_RGBRefine, \
                                      feat_RGB_V, feat_NI_V, feat_TI_V], dim=-1)
            elif self.miss_type == 'TIR':
                RGB = x['RGB']
                NI = x['NI']
                B = RGB.shape[0]
                NI_fea, NI_global = self.NI(NI, cam_label=cam_label, view_label=view_label)
                RGB_fea, RGB_global = self.RGB(RGB, cam_label=cam_label, view_label=view_label)

                NI_refine = self.NI_LRU(NI_fea.reshape(B, *self.patch_num).permute(0, 3, 1, 2))
                RGB_refine = self.RGB_LRU(RGB_fea.reshape(B, *self.patch_num).permute(0, 3, 1, 2))

                NI_refine_global = self.NI_GAP(NI_refine).view(B, -1)
                RGB_refine_global = self.RGB_GAP(RGB_refine).view(B, -1)

                RGB_refine = RGB_refine.flatten(-2).transpose(1, 2)
                NI_refine = NI_refine.flatten(-2).transpose(1, 2)
                # RGB_NI, NI_RE = self.RENI(RGB_refine)
                RGB_TI, TI_RE = self.RETI(RGB_refine)

                NI_RE_NI = NI_refine
                TI_RE_TI = TI_RE

                RGB_V = self.Cross_RGB(NI_RE_NI, TI_RE_TI, RGB_refine)
                NI_V = self.Cross_NI(RGB_refine, TI_RE_TI, NI_RE_NI)
                TI_V = self.Cross_TI(NI_RE_NI, RGB_refine, TI_RE_TI)

                feat_NIRefine = self.bottleneck_NIrefine(NI_refine_global)
                feat_RGBRefine = self.bottleneck_RGBrefine(RGB_refine_global)
                feat_RGB_V = self.bottleneck_RGB_V(torch.mean(RGB_V, dim=1))
                feat_NI_V = self.bottleneck_NI_V(torch.mean(NI_V, dim=1))
                feat_TI_V = self.bottleneck_TI_V(torch.mean(TI_V, dim=1))

                if self.neck_feat == 'after':
                    pass
                else:
                    feat_NIRefine = NI_refine_global
                    feat_RGBRefine = RGB_refine_global
                    feat_RGB_V = torch.mean(RGB_V, dim=1)
                    feat_NI_V = torch.mean(NI_V, dim=1)
                    feat_TI_V = torch.mean(TI_V, dim=1)
                if self.test_feat == 0:
                    return torch.cat([NI_global, RGB_global, \
                                      feat_NIRefine, feat_RGBRefine, \
                                      feat_RGB_V, feat_NI_V, feat_TI_V], dim=-1)
            elif self.miss_type == "NIR+TIR":
                RGB = x['RGB']
                B = RGB.shape[0]
                RGB_fea, RGB_global = self.RGB(RGB, cam_label=cam_label, view_label=view_label)
                RGB_refine = self.RGB_LRU(RGB_fea.reshape(B, *self.patch_num).permute(0, 3, 1, 2))

                RGB_refine_global = self.RGB_GAP(RGB_refine).view(B, -1)
                RGB_refine = RGB_refine.flatten(-2).transpose(1, 2)
                RGB_NI, NI_RE = self.RENI(RGB_refine)
                RGB_TI, TI_RE = self.RETI(RGB_refine)

                NI_RE_NI = NI_RE
                TI_RE_TI = TI_RE

                RGB_V = self.Cross_RGB(NI_RE_NI, TI_RE_TI, RGB_refine)
                NI_V = self.Cross_NI(RGB_refine, TI_RE_TI, NI_RE_NI)
                TI_V = self.Cross_TI(NI_RE_NI, RGB_refine, TI_RE_TI)

                feat_RGBRefine = self.bottleneck_RGBrefine(RGB_refine_global)
                feat_RGB_V = self.bottleneck_RGB_V(torch.mean(RGB_V, dim=1))
                feat_NI_V = self.bottleneck_NI_V(torch.mean(NI_V, dim=1))
                feat_TI_V = self.bottleneck_TI_V(torch.mean(TI_V, dim=1))
                if self.neck_feat == 'after':
                    pass
                else:
                    feat_RGBRefine = RGB_refine_global
                    feat_RGB_V = torch.mean(RGB_V, dim=1)
                    feat_NI_V = torch.mean(NI_V, dim=1)
                    feat_TI_V = torch.mean(TI_V, dim=1)
                if self.test_feat == 0:
                    return torch.cat([RGB_global, \
                                      feat_RGBRefine, \
                                      feat_RGB_V, feat_NI_V, feat_TI_V], dim=-1)
            else:
                RGB = x['RGB']
                NI = x['NI']
                TI = x['TI']
                B = RGB.shape[0]
                NI_fea, NI_global = self.NI(NI, cam_label=cam_label, view_label=view_label)
                TI_fea, TI_global = self.TI(TI, cam_label=cam_label, view_label=view_label)
                RGB_fea, RGB_global = self.RGB(RGB, cam_label=cam_label, view_label=view_label)

                NI_refine = self.NI_LRU(NI_fea.reshape(B, *self.patch_num).permute(0, 3, 1, 2))
                TI_refine = self.TI_LRU(TI_fea.reshape(B, *self.patch_num).permute(0, 3, 1, 2))
                RGB_refine = self.RGB_LRU(RGB_fea.reshape(B, *self.patch_num).permute(0, 3, 1, 2))

                NI_refine_global = self.NI_GAP(NI_refine).view(B, -1)
                TI_refine_global = self.TI_GAP(TI_refine).view(B, -1)
                RGB_refine_global = self.RGB_GAP(RGB_refine).view(B, -1)

                RGB_refine = RGB_refine.flatten(-2).transpose(1, 2)

                RGB_NI, NI_RE = self.RENI(RGB_refine)
                RGB_TI, TI_RE = self.RETI(RGB_refine)

                feat_NIRefine = self.bottleneck_NIrefine(NI_refine_global)
                feat_TIRefine = self.bottleneck_TIrefine(TI_refine_global)
                feat_RGBRefine = self.bottleneck_RGBrefine(RGB_refine_global)
                feat_NI_V = self.bottleneck_NI_V(torch.mean(RGB_NI, dim=1))
                feat_TI_V = self.bottleneck_TI_V(torch.mean(RGB_TI, dim=1))

                if self.neck_feat == 'after':
                    pass
                else:
                    feat_NIRefine = NI_refine_global
                    feat_TIRefine = TI_refine_global
                    feat_RGBRefine = RGB_refine_global
                    feat_NI_V = torch.mean(RGB_NI, dim=1)
                    feat_TI_V = torch.mean(RGB_TI, dim=1)
                if self.test_feat == 0:
                    return torch.cat([NI_global, TI_global, RGB_global, \
                                      feat_NIRefine, feat_TIRefine, feat_RGBRefine, \
                                      feat_NI_V, feat_TI_V], dim=-1)


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
    model = MMReID(num_class, cfg, camera_num, view_num, __factory_T_type)
    print('===========Building MMReID===========')
    return model
