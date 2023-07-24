import torch
import torch.nn as nn
from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224
import torch.nn.functional as F
from modeling.fusion_part.RotationLSE import BlockFuse, Rotation
from modeling.backbones.resnet import ResNet, Bottleneck


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


class build_resnet(nn.Module):
    def __init__(self, num_classes, cfg):
        super(build_resnet, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH_R
        self.mode = cfg.MODEL.TRANS_USE
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        self.in_planes = 2048
        self.pattern = 1
        if self.pattern == 1:
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif self.pattern == 2:
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 23, 3])
        elif self.pattern == 3:
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = GeM()
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, cam_label=None, view_label=None, label=None):  # label is unused if self.cos_layer == 'no'
        mid_fea = self.base(x)
        global_feat = self.gap(mid_fea)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:

            cls_score = self.classifier(feat)
            if self.mode == 0:
                return cls_score, global_feat
            else:
                return mid_fea, cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                if self.mode == 0:
                    return feat
                else:
                    return mid_fea, feat
            else:
                if self.mode == 0:
                    return global_feat
                else:
                    return mid_fea, global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


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
        # cash_x[-1] = cash_x[-1][:, 1:, :]
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
        self.mix_mode = cfg.MODEL.MIX_MODE
        # self.cash0_fusion = BlockFuse(dim=self.dim_l, num_heads=self.num_head, depth=self.mix_depth, mode=0,
        #                               mixdim=self.mix_dim)
        # self.cash1_fusion = BlockFuse(dim=self.dim_l, num_heads=self.num_head, depth=self.mix_depth, mode=0,
        #                               mixdim=self.mix_dim)
        # self.cash2_fusion = BlockFuse(dim=self.dim_l, num_heads=self.num_head, depth=self.mix_depth, mode=0,
        #                               mixdim=self.mix_dim)
        # self.cash3_fusion = BlockFuse(dim=self.dim_l, num_heads=self.num_head, depth=self.mix_depth, mode=self.mix_mode,
        #                               mixdim=self.mix_dim)
        # self.Rotation_1 = Rotation(dim=self.dim_l, num_heads=self.num_head)
        # self.Rotation_2 = Rotation(dim=self.dim_l, num_heads=self.num_head)
        # self.Rotation_3 = Rotation(dim=self.dim_l, num_heads=self.num_head)
        self.Rotation_4 = Rotation(dim=self.dim_l, num_heads=self.num_head)
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        self.patch_num = (16, 8)
        # self.NI_RE = ReAttention(dim=self.dim_l, num_heads=self.num_head)
        # self.TI_RE = ReAttention(dim=self.dim_l, num_heads=self.num_head)

        # self.classifier_FEA_RGBNI = nn.Linear(self.mix_dim, self.num_classes, bias=False)
        # self.classifier_FEA_RGBNI.apply(weights_init_classifier)
        # self.bottleneck_FEA_RGBNI = nn.BatchNorm1d(self.mix_dim)
        # self.bottleneck_FEA_RGBNI.bias.requires_grad_(False)
        # self.bottleneck_FEA_RGBNI.apply(weights_init_kaiming)
        #
        # self.classifier_FEA_RGBTI = nn.Linear(self.mix_dim, self.num_classes, bias=False)
        # self.classifier_FEA_RGBTI.apply(weights_init_classifier)
        # self.bottleneck_FEA_RGBTI = nn.BatchNorm1d(self.mix_dim)
        # self.bottleneck_FEA_RGBTI.bias.requires_grad_(False)
        # self.bottleneck_FEA_RGBTI.apply(weights_init_kaiming)
        # self.classifier_FUSE0 = nn.Linear(self.mix_dim, self.num_classes, bias=False)
        # self.classifier_FUSE0.apply(weights_init_classifier)
        # self.bottleneck_FUSE0 = nn.BatchNorm1d(self.mix_dim)
        # self.bottleneck_FUSE0.bias.requires_grad_(False)
        # self.bottleneck_FUSE0.apply(weights_init_kaiming)
        #
        # self.classifier_FUSE1 = nn.Linear(self.mix_dim, self.num_classes, bias=False)
        # self.classifier_FUSE1.apply(weights_init_classifier)
        # self.bottleneck_FUSE1 = nn.BatchNorm1d(self.mix_dim)
        # self.bottleneck_FUSE1.bias.requires_grad_(False)
        # self.bottleneck_FUSE1.apply(weights_init_kaiming)
        #
        # self.classifier_FUSE2 = nn.Linear(self.mix_dim, self.num_classes, bias=False)
        # self.classifier_FUSE2.apply(weights_init_classifier)
        # self.bottleneck_FUSE2 = nn.BatchNorm1d(self.mix_dim)
        # self.bottleneck_FUSE2.bias.requires_grad_(False)
        # self.bottleneck_FUSE2.apply(weights_init_kaiming)
        #
        # self.classifier_Rotation_1 = nn.Linear(3 * self.mix_dim, self.num_classes, bias=False)
        # self.classifier_Rotation_1.apply(weights_init_classifier)
        # self.bottleneck_Rotation_1 = nn.BatchNorm1d(3 * self.mix_dim)
        # self.bottleneck_Rotation_1.bias.requires_grad_(False)
        # self.bottleneck_Rotation_1.apply(weights_init_kaiming)
        #
        # self.classifier_Rotation_2 = nn.Linear(3 * self.mix_dim, self.num_classes, bias=False)
        # self.classifier_Rotation_2.apply(weights_init_classifier)
        # self.bottleneck_Rotation_2 = nn.BatchNorm1d(3 * self.mix_dim)
        # self.bottleneck_Rotation_2.bias.requires_grad_(False)
        # self.bottleneck_Rotation_2.apply(weights_init_kaiming)
        #
        # self.classifier_Rotation_3 = nn.Linear(3 * self.mix_dim, self.num_classes, bias=False)
        # self.classifier_Rotation_3.apply(weights_init_classifier)
        # self.bottleneck_Rotation_3 = nn.BatchNorm1d(3 * self.mix_dim)
        # self.bottleneck_Rotation_3.bias.requires_grad_(False)
        # self.bottleneck_Rotation_3.apply(weights_init_kaiming)

        self.classifier_Rotation_4 = nn.Linear(3 * self.mix_dim, self.num_classes, bias=False)
        self.classifier_Rotation_4.apply(weights_init_classifier)
        self.bottleneck_Rotation_4 = nn.BatchNorm1d(3 * self.mix_dim)
        self.bottleneck_Rotation_4.bias.requires_grad_(False)
        self.bottleneck_Rotation_4.apply(weights_init_kaiming)

        self.classifier_PATCH4 = nn.Linear(self.mix_dim, self.num_classes, bias=False)
        self.classifier_PATCH4.apply(weights_init_classifier)
        self.bottleneck_PATCH4 = nn.BatchNorm1d(self.mix_dim)
        self.bottleneck_PATCH4.bias.requires_grad_(False)
        self.bottleneck_PATCH4.apply(weights_init_kaiming)

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

            # RGB_NI, NI_RE = self.NI_RE(RGB_cash[-1])
            # RGB_TI, TI_RE = self.TI_RE(RGB_cash[-1])
            # loss_ni = nn.MSELoss()(NI_RE, NI_cash[-1])
            # loss_ti = nn.MSELoss()(TI_RE, TI_cash[-1])

            # RGBNI_FEA = self.bottleneck_FEA_RGBNI(torch.mean(RGB_NI, dim=-2))
            # RGBTI_FEA = self.bottleneck_FEA_RGBTI(torch.mean(RGB_TI, dim=-2))
            # RGBNI_SCORE = self.classifier_FEA_RGBNI(RGBNI_FEA)
            # RGBTI_SCORE = self.classifier_FEA_RGBTI(RGBTI_FEA)

            # CASH0 = self.cash0_fusion(RGB_cash[0], NI_cash[0], TI_cash[0], *self.patch_num)
            # CASH1 = self.cash1_fusion(RGB_cash[1], NI_cash[1], TI_cash[1], *self.patch_num)
            # CASH2 = self.cash2_fusion(RGB_cash[2], NI_cash[2], TI_cash[2], *self.patch_num)
            # rotation_1 = self.Rotation_1(RGB_cash[0], NI_cash[0], TI_cash[0])
            # rotation_2 = self.Rotation_2(RGB_cash[1], NI_cash[1], TI_cash[1])
            # rotation_3 = self.Rotation_3(RGB_cash[2], NI_cash[2], TI_cash[2])
            rotation_4, patch_4 = self.Rotation_4(RGB_cash[3], NI_cash[3], TI_cash[3])

            # CASH3 = self.cash3_fusion(RGB_cash[3][:, 1:, :], NI_cash[3][:, 1:, :], TI_cash[3][:, 1:, :],
            #                           *self.patch_num)
            # CASH3 = self.combine(torch.cat([CASH2, CASH3], dim=-1))
            # FUSE0 = torch.mean(CASH0, dim=-2)
            # FUSE1 = torch.mean(CASH1, dim=-2)
            # FUSE2 = torch.mean(CASH2, dim=-2)
            # FUSE3 = torch.mean(CASH3, dim=-2)
            # FUSE0_global = self.bottleneck_FUSE0(FUSE0)
            # FUSE1_global = self.bottleneck_FUSE1(FUSE1)
            # FUSE2_global = self.bottleneck_FUSE2(FUSE2)
            # FUSE3_global = self.bottleneck_FUSE3(FUSE3)
            # Rotation_global_1 = self.bottleneck_Rotation_1(rotation_1)
            # Rotation_global_2 = self.bottleneck_Rotation_2(rotation_2)
            # Rotation_global_3 = self.bottleneck_Rotation_3(rotation_3)
            Rotation_global_4 = self.bottleneck_Rotation_4(rotation_4)
            Rotation_patch_4 = self.bottleneck_PATCH4(patch_4)
            # FUSE0_score = self.classifier_FUSE0(FUSE0_global)
            # FUSE1_score = self.classifier_FUSE1(FUSE1_global)
            # FUSE2_score = self.classifier_FUSE2(FUSE2_global)
            # FUSE3_score = self.classifier_FUSE3(FUSE3_global)
            # Rotation_score_1 = self.classifier_Rotation_1(Rotation_global_1)
            # Rotation_score_2 = self.classifier_Rotation_2(Rotation_global_2)
            # Rotation_score_3 = self.classifier_Rotation_3(Rotation_global_3)
            Rotation_score_4 = self.classifier_Rotation_4(Rotation_global_4)
            Patch_score_4 = self.classifier_PATCH4(Rotation_patch_4)
            return RGB_score, RGB_global, NI_score, NI_global, TI_score, TI_global, \
                Rotation_score_4, Rotation_global_4, Patch_score_4, Rotation_patch_4

        else:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']
            NI_cash, NI_global = self.NI(NI, cam_label=cam_label, view_label=view_label)
            TI_cash, TI_global = self.TI(TI, cam_label=cam_label, view_label=view_label)
            RGB_cash, RGB_global = self.RGB(RGB, cam_label=cam_label, view_label=view_label)

            # RGB_NI, NI_RE = self.NI_RE(RGB_cash[-1])
            # RGB_TI, TI_RE = self.TI_RE(RGB_cash[-1])
            # loss_ni = nn.MSELoss()(NI_RE, NI_cash[-1])
            # loss_ti = nn.MSELoss()(TI_RE, TI_cash[-1])

            # RGBNI_FEA = self.bottleneck_FEA_RGBNI(torch.mean(RGB_NI, dim=-2))
            # RGBTI_FEA = self.bottleneck_FEA_RGBTI(torch.mean(RGB_TI, dim=-2))
            # RGBNI_SCORE = self.classifier_FEA_RGBNI(RGBNI_FEA)
            # RGBTI_SCORE = self.classifier_FEA_RGBTI(RGBTI_FEA)

            # CASH0 = self.cash0_fusion(RGB_cash[0], NI_cash[0], TI_cash[0], *self.patch_num)
            # CASH1 = self.cash1_fusion(RGB_cash[1], NI_cash[1], TI_cash[1], *self.patch_num)
            # CASH2 = self.cash2_fusion(RGB_cash[2], NI_cash[2], TI_cash[2], *self.patch_num)
            # rotation_1 = self.Rotation_1(RGB_cash[0], NI_cash[0], TI_cash[0])
            # rotation_2 = self.Rotation_2(RGB_cash[1], NI_cash[1], TI_cash[1])
            # rotation_3 = self.Rotation_3(RGB_cash[2], NI_cash[2], TI_cash[2])
            rotation_4, patch_4 = self.Rotation_4(RGB_cash[3], NI_cash[3], TI_cash[3])

            # CASH3 = self.cash3_fusion(RGB_cash[3][:, 1:, :], NI_cash[3][:, 1:, :], TI_cash[3][:, 1:, :],
            #                           *self.patch_num)
            # CASH3 = self.combine(torch.cat([CASH2, CASH3], dim=-1))
            # FUSE0 = torch.mean(CASH0, dim=-2)
            # FUSE1 = torch.mean(CASH1, dim=-2)
            # FUSE2 = torch.mean(CASH2, dim=-2)
            # FUSE3 = torch.mean(CASH3, dim=-2)
            # FUSE0_global = self.bottleneck_FUSE0(FUSE0)
            # FUSE1_global = self.bottleneck_FUSE1(FUSE1)
            # FUSE2_global = self.bottleneck_FUSE2(FUSE2)
            Rotation_global_4 = self.bottleneck_Rotation_4(rotation_4)
            Rotation_patch_4 = self.bottleneck_PATCH4(patch_4)
            # FUSE0_score = self.classifier_FUSE0(FUSE0_global)
            # FUSE1_score = self.classifier_FUSE1(FUSE1_global)
            # FUSE2_score = self.classifier_FUSE2(FUSE2_global)
            # FUSE3_score = self.classifier_FUSE3(FUSE3_global)

            if self.neck_feat == 'after':
                pass
            else:
                # RGBNI_FEA = torch.mean(RGB_NI, dim=-2)
                # RGBTI_FEA = torch.mean(RGB_TI, dim=-2)
                # FUSE2_global = FUSE2
                # FUSE3_global = FUSE3
                # Rotation_global_1 = rotation_1
                # Rotation_global_2 = rotation_2
                # Rotation_global_3 = rotation_3
                Rotation_global_4 = rotation_4
                Rotation_patch_4 = patch_4
            if self.test_feat == 0:
                return torch.cat([RGB_global, NI_global, TI_global,
                                  Rotation_global_4, Rotation_patch_4], dim=-1)


class BaselineRes(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(BaselineRes, self).__init__()
        self.NI = build_resnet(num_classes, cfg)
        self.TI = build_resnet(num_classes, cfg)
        self.RGB = build_resnet(num_classes, cfg)

        self.num_classes = num_classes
        self.cfg = cfg
        self.camera = camera_num
        self.view = view_num
        self.dim_l = 768
        self.num_head = 12
        self.mix_depth = cfg.MODEL.DEPTH
        self.mix_dim = cfg.MODEL.MIX_DIM
        self.mix_mode = cfg.MODEL.MIX_MODE

        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        self.patch_num = (16, 8)
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
            return RGB_score, RGB_global, NI_score, NI_global, TI_score, TI_global

        else:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']
            NI_cash, NI_global = self.NI(NI, cam_label=cam_label, view_label=view_label)
            TI_cash, TI_global = self.TI(TI, cam_label=cam_label, view_label=view_label)
            RGB_cash, RGB_global = self.RGB(RGB, cam_label=cam_label, view_label=view_label)

            if self.neck_feat == 'after':
                pass
            else:
                pass
            if self.test_feat == 0:
                return torch.cat([RGB_global, NI_global, TI_global], dim=-1)


class BaselineTrans(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(BaselineTrans, self).__init__()
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
        self.mix_mode = cfg.MODEL.MIX_MODE

        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        self.patch_num = (16, 8)
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
            return RGB_score, RGB_global, NI_score, NI_global, TI_score, TI_global

        else:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']
            NI_cash, NI_global = self.NI(NI, cam_label=cam_label, view_label=view_label)
            TI_cash, TI_global = self.TI(TI, cam_label=cam_label, view_label=view_label)
            RGB_cash, RGB_global = self.RGB(RGB, cam_label=cam_label, view_label=view_label)

            if self.neck_feat == 'after':
                pass
            else:
                pass
            if self.test_feat == 0:
                return torch.cat([RGB_global, NI_global, TI_global], dim=-1)


__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
}


def make_model(cfg, num_class, camera_num, view_num=0, ):
    if cfg.MODEL.BASE == 0:
        model = BaselineTrans(num_class, cfg, camera_num, view_num, __factory_T_type)
        print('===========Building BaselineTrans===========')
    elif cfg.MODEL.BASE == 1:
        model = BaselineRes(num_class, cfg, camera_num, view_num, __factory_T_type)
        print('===========Building BaselineRes===========')
    else:
        model = Branch_new(num_class, cfg, camera_num, view_num, __factory_T_type)
        print('===========Building DenseMMReID===========')
    return model
