import torch
import torch.nn as nn
from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224
from modeling.fusion_part.TPM import Rotation
from modeling.fusion_part.CRM import ReconstructAll, ReconstructAllCross
from modeling.backbones.t2t import t2t_vit_t_14, t2t_vit_t_24
from modeling.backbones.osnet import osnet_x1_0
from modeling.backbones.hacnn import HACNN
from modeling.backbones.mudeep import MuDeep
from modeling.backbones.pcb import pcb_p6
from modeling.backbones.mlfn import mlfn
from modeling.fusion_part.dynamic import dynamic_triplet
from modeling.fusion_part.MLP import Mlp


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


class build_transformer(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH_T
        self.mode = cfg.MODEL.RES_USE
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.trans_type = cfg.MODEL.TRANSFORMER_TYPE
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


class UniSReIDC(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(UniSReIDC, self).__init__()
        self.NI = build_transformer(num_classes, cfg, camera_num, view_num, factory)
        self.RGB = build_transformer(num_classes, cfg, camera_num, view_num, factory)

        self.num_classes = num_classes
        self.cfg = cfg
        self.camera = camera_num
        self.view = view_num
        self.num_head = 12
        self.mix_dim = 768

        self.Rotation_4 = Rotation(dim=self.mix_dim, num_heads=self.num_head, cross=True)
        self.re = cfg.MODEL.RE
        if self.re:
            self.reconstruct = ReconstructAllCross(dim=self.mix_dim, num_heads=self.num_head,
                                                   depth=cfg.MODEL.RE_LAYER)
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.patch_num = (16, 8)

        self.classifier_Rotation_4 = nn.Linear(2 * self.mix_dim, self.num_classes, bias=False)
        self.classifier_Rotation_4.apply(weights_init_classifier)
        self.bottleneck_Rotation_4 = nn.BatchNorm1d(2 * self.mix_dim)
        self.bottleneck_Rotation_4.bias.requires_grad_(False)
        self.bottleneck_Rotation_4.apply(weights_init_kaiming)

        self.classifier = nn.Linear(2 * self.mix_dim, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(2 * self.mix_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.test_feat = cfg.TEST.FEAT
        self.miss = cfg.TEST.MISS
        self.cross = 0
        self.layer = cfg.MODEL.LAYER
        self.cross_train = 0
        if self.cross_train:
            self.pro_CROSS = Mlp(in_features=2 * self.mix_dim)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def forward(self, x, cam_label=0, label=None, view_label=None, cross_type=None, mode=0):
        if self.training:
            RGB = x['RGB']
            NI = x['NI']
            RGB_cash, RGB_score, RGB_global = self.RGB(RGB, cam_label=cam_label, view_label=view_label)
            NI_cash, NI_score, NI_global = self.NI(NI, cam_label=cam_label, view_label=view_label)
            # ori = torch.cat([RGB_global, NI_global], dim=-1)
            # ori_global = self.bottleneck(ori)
            # ori_score = self.classifier(ori_global)
            rotation_4 = self.Rotation_4(RGB_cash[self.layer], NI_cash[self.layer], None)
            Rotation_global_4 = self.bottleneck_Rotation_4(rotation_4)
            Rotation_score_4 = self.classifier_Rotation_4(Rotation_global_4)
            if self.re:
                loss_re = self.reconstruct(RGB_cash[self.layer], NI_cash[self.layer])
                return Rotation_score_4, rotation_4, loss_re
            else:
                return Rotation_score_4, rotation_4,

        else:
            if mode == 1:
                self.cross = 0
            else:
                self.cross = 1
            if self.cross:
                RGB = x['RGB']
                NI = x['NI']
                RGB_cash, RGB_global = self.RGB(RGB, cam_label=cam_label, view_label=view_label)
                NI_cash, NI_global = self.NI(NI, cam_label=cam_label, view_label=view_label)

                RGB = self.reconstruct(ma=None, mb=NI_cash[self.layer], cross_miss='r')
                NI_FEA = self.Rotation_4(RGB, NI_cash[self.layer], None)

                NI = self.reconstruct(ma=RGB_cash[self.layer], mb=None, cross_miss='n')
                RGB_FEA = self.Rotation_4(RGB_cash[self.layer], NI, None)

                if cross_type == 'r2n':
                    return RGB_FEA, NI_FEA
                elif cross_type == 'n2r':
                    return NI_FEA, RGB_FEA
            else:
                RGB = x['RGB']
                NI = x['NI']
                RGB_cash, RGB_global = self.RGB(RGB, cam_label=cam_label, view_label=view_label)
                NI_cash, NI_global = self.NI(NI, cam_label=cam_label, view_label=view_label)
                # ori = torch.cat([RGB_global, NI_global], dim=-1)
                # ori_global = self.bottleneck(ori)
                # ori_score = self.classifier(ori_global)
                rotation_4 = self.Rotation_4(RGB_cash[self.layer], NI_cash[self.layer], None)
                return rotation_4


__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
    't2t_vit_t_14': t2t_vit_t_14,
    't2t_vit_t_24': t2t_vit_t_24,
    'osnet': osnet_x1_0,
    'hacnn': HACNN,
    'mudeep': MuDeep,
    'pcb': pcb_p6,
    'mlfn': mlfn
}


def make_model_cross(cfg, num_class, camera_num, view_num=0, ):
    if cfg.MODEL.BASE == 0:
        print('===========Building BASELINE===========')
    elif cfg.MODEL.BASE == 1:
        print('===========Building Single===========')
    else:
        model = UniSReIDC(num_class, cfg, camera_num, view_num, __factory_T_type)
        print('===========Building UniSReID===========')
    return model
# import torch
# import torch.nn as nn
# from torch.nn import init
# from modeling.backbones.resnet_cross import resnet50, resnet18

#
# class Normalize(nn.Module):
#     def __init__(self, power=2):
#         super(Normalize, self).__init__()
#         self.power = power
#
#     def forward(self, x):
#         norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
#         out = x.div(norm)
#         return out
#
#
# class Non_local(nn.Module):
#     def __init__(self, in_channels, reduc_ratio=2):
#         super(Non_local, self).__init__()
#
#         self.in_channels = in_channels
#         self.inter_channels = reduc_ratio // reduc_ratio
#
#         self.g = nn.Sequential(
#             nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
#                       padding=0),
#         )
#
#         self.W = nn.Sequential(
#             nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
#                       kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(self.in_channels),
#         )
#         nn.init.constant_(self.W[1].weight, 0.0)
#         nn.init.constant_(self.W[1].bias, 0.0)
#
#         self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
#                                kernel_size=1, stride=1, padding=0)
#
#         self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
#                              kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         '''
#                 :param x: (b, c, t, h, w)
#                 :return:
#                 '''
#
#         batch_size = x.size(0)
#         g_x = self.g(x).view(batch_size, self.inter_channels, -1)
#         g_x = g_x.permute(0, 2, 1)
#
#         theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
#         theta_x = theta_x.permute(0, 2, 1)
#         phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
#         f = torch.matmul(theta_x, phi_x)
#         N = f.size(-1)
#         # f_div_C = torch.nn.functional.softmax(f, dim=-1)
#         f_div_C = f / N
#
#         y = torch.matmul(f_div_C, g_x)
#         y = y.permute(0, 2, 1).contiguous()
#         y = y.view(batch_size, self.inter_channels, *x.size()[2:])
#         W_y = self.W(y)
#         z = W_y + x
#
#         return z
#
#
# # #####################################################################
# def weights_init_kaiming(m):
#     classname = m.__class__.__name__
#     # print(classname)
#     if classname.find('Conv') != -1:
#         init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#     elif classname.find('Linear') != -1:
#         init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
#         init.zeros_(m.bias.data)
#     elif classname.find('BatchNorm1d') != -1:
#         init.normal_(m.weight.data, 1.0, 0.01)
#         init.zeros_(m.bias.data)
#
#
# def weights_init_classifier(m):
#     classname = m.__class__.__name__
#     if classname.find('Linear') != -1:
#         init.normal_(m.weight.data, 0, 0.001)
#         if m.bias:
#             init.zeros_(m.bias.data)
#
#
# class visible_module(nn.Module):
#     def __init__(self, arch='resnet50'):
#         super(visible_module, self).__init__()
#
#         model_v = resnet50(pretrained=True,
#                            last_conv_stride=1, last_conv_dilation=1)
#         # avg pooling to global pooling
#         self.visible = model_v
#
#     def forward(self, x):
#         x = self.visible.conv1(x)
#         x = self.visible.bn1(x)
#         x = self.visible.relu(x)
#         x = self.visible.maxpool(x)
#         return x
#
#
# class thermal_module(nn.Module):
#     def __init__(self, arch='resnet50'):
#         super(thermal_module, self).__init__()
#
#         model_t = resnet50(pretrained=True,
#                            last_conv_stride=1, last_conv_dilation=1)
#         # avg pooling to global pooling
#         self.thermal = model_t
#
#     def forward(self, x):
#         x = self.thermal.conv1(x)
#         x = self.thermal.bn1(x)
#         x = self.thermal.relu(x)
#         x = self.thermal.maxpool(x)
#         return x
#
#
# class base_resnet(nn.Module):
#     def __init__(self, arch='resnet50'):
#         super(base_resnet, self).__init__()
#
#         model_base = resnet50(pretrained=True,
#                               last_conv_stride=1, last_conv_dilation=1)
#         # avg pooling to global pooling
#         model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.base = model_base
#
#     def forward(self, x):
#         x = self.base.layer1(x)
#         x = self.base.layer2(x)
#         x = self.base.layer3(x)
#         x = self.base.layer4(x)
#         return x
#
#
# class embed_net(nn.Module):
#     def __init__(self, class_num, no_local='on', gm_pool='on', arch='resnet50'):
#         super(embed_net, self).__init__()
#
#         self.thermal_module = thermal_module(arch=arch)
#         self.visible_module = visible_module(arch=arch)
#         self.base_resnet = base_resnet(arch=arch)
#         self.non_local = no_local
#         if self.non_local == 'on':
#             layers = [3, 4, 6, 3]
#             non_layers = [0, 2, 3, 0]
#             self.NL_1 = nn.ModuleList(
#                 [Non_local(256) for i in range(non_layers[0])])
#             self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
#             self.NL_2 = nn.ModuleList(
#                 [Non_local(512) for i in range(non_layers[1])])
#             self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
#             self.NL_3 = nn.ModuleList(
#                 [Non_local(1024) for i in range(non_layers[2])])
#             self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
#             self.NL_4 = nn.ModuleList(
#                 [Non_local(2048) for i in range(non_layers[3])])
#             self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])
#
#         pool_dim = 2048
#         self.l2norm = Normalize(2)
#         self.bottleneck = nn.BatchNorm1d(pool_dim)
#         self.bottleneck.bias.requires_grad_(False)  # no shift
#
#         self.classifier = nn.Linear(pool_dim, class_num, bias=False)
#
#         self.bottleneck.apply(weights_init_kaiming)
#         self.classifier.apply(weights_init_classifier)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.gm_pool = gm_pool
#
#     def forward_fea(self, x, cam_label=0, label=None, view_label=None, cross_type=None, modal=0):
#         if self.training:
#             x1 = x['RGB']
#             x2 = x['NI']
#             x1 = self.visible_module(x1)
#             x2 = self.thermal_module(x2)
#             x = torch.cat((x1, x2), 0)
#             # shared block
#             if self.non_local == 'on':
#                 NL1_counter = 0
#                 if len(self.NL_1_idx) == 0: self.NL_1_idx = [-1]
#                 for i in range(len(self.base_resnet.base.layer1)):
#                     x = self.base_resnet.base.layer1[i](x)
#                     if i == self.NL_1_idx[NL1_counter]:
#                         _, C, H, W = x.shape
#                         x = self.NL_1[NL1_counter](x)
#                         NL1_counter += 1
#                 # Layer 2
#                 NL2_counter = 0
#                 if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
#                 for i in range(len(self.base_resnet.base.layer2)):
#                     x = self.base_resnet.base.layer2[i](x)
#                     if i == self.NL_2_idx[NL2_counter]:
#                         _, C, H, W = x.shape
#                         x = self.NL_2[NL2_counter](x)
#                         NL2_counter += 1
#                 # Layer 3
#                 NL3_counter = 0
#                 if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
#                 for i in range(len(self.base_resnet.base.layer3)):
#                     x = self.base_resnet.base.layer3[i](x)
#                     if i == self.NL_3_idx[NL3_counter]:
#                         _, C, H, W = x.shape
#                         x = self.NL_3[NL3_counter](x)
#                         NL3_counter += 1
#                 # Layer 4
#                 NL4_counter = 0
#                 if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
#                 for i in range(len(self.base_resnet.base.layer4)):
#                     x = self.base_resnet.base.layer4[i](x)
#                     if i == self.NL_4_idx[NL4_counter]:
#                         _, C, H, W = x.shape
#                         x = self.NL_4[NL4_counter](x)
#                         NL4_counter += 1
#             else:
#                 x = self.base_resnet(x)
#             if self.gm_pool == 'on':
#                 b, c, h, w = x.shape
#                 x = x.view(b, c, -1)
#                 p = 3.0
#                 x_pool = (torch.mean(x ** p, dim=-1) + 1e-12) ** (1 / p)
#             else:
#                 x_pool = self.avgpool(x)
#                 x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))
#
#             feat = self.bottleneck(x_pool)
#
#             return x_pool, self.classifier(feat), 0
#         else:
#             if modal==1:
#                 x = x['NI']
#                 x = self.visible_module(x)
#             else:
#                 x = x['RGB']
#                 x = self.thermal_module(x)
#             if self.non_local == 'on':
#                 NL1_counter = 0
#                 if len(self.NL_1_idx) == 0: self.NL_1_idx = [-1]
#                 for i in range(len(self.base_resnet.base.layer1)):
#                     x = self.base_resnet.base.layer1[i](x)
#                     if i == self.NL_1_idx[NL1_counter]:
#                         _, C, H, W = x.shape
#                         x = self.NL_1[NL1_counter](x)
#                         NL1_counter += 1
#                 # Layer 2
#                 NL2_counter = 0
#                 if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
#                 for i in range(len(self.base_resnet.base.layer2)):
#                     x = self.base_resnet.base.layer2[i](x)
#                     if i == self.NL_2_idx[NL2_counter]:
#                         _, C, H, W = x.shape
#                         x = self.NL_2[NL2_counter](x)
#                         NL2_counter += 1
#                 # Layer 3
#                 NL3_counter = 0
#                 if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
#                 for i in range(len(self.base_resnet.base.layer3)):
#                     x = self.base_resnet.base.layer3[i](x)
#                     if i == self.NL_3_idx[NL3_counter]:
#                         _, C, H, W = x.shape
#                         x = self.NL_3[NL3_counter](x)
#                         NL3_counter += 1
#                 # Layer 4
#                 NL4_counter = 0
#                 if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
#                 for i in range(len(self.base_resnet.base.layer4)):
#                     x = self.base_resnet.base.layer4[i](x)
#                     if i == self.NL_4_idx[NL4_counter]:
#                         _, C, H, W = x.shape
#                         x = self.NL_4[NL4_counter](x)
#                         NL4_counter += 1
#             else:
#                 x = self.base_resnet(x)
#             if self.gm_pool == 'on':
#                 b, c, h, w = x.shape
#                 x = x.view(b, c, -1)
#                 p = 3.0
#                 x_pool = (torch.mean(x ** p, dim=-1) + 1e-12) ** (1 / p)
#             else:
#                 x_pool = self.avgpool(x)
#                 x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))
#             return self.l2norm(x_pool)
#
#     def forward(self, x, cam_label=0, label=None, view_label=None, cross_type=None, modal=0):
#         if self.training:
#             return self.forward_fea(x)
#         else:
#             RGB = x['RGB']
#             NI = x['NI']
#             x1 = {'RGB': RGB, 'NI': RGB}
#             x2 = {'RGB': NI, 'NI': NI}
#             RGB_FEA = self.forward_fea(x1,modal=1)
#             NI_FEA = self.forward_fea(x2,modal=2)
#             return RGB_FEA, NI_FEA
#
#
# def make_model_cross(cfg, num_class, camera_num, view_num=0, ):
#     model = embed_net(class_num=num_class)
#     return model
#
#
#
