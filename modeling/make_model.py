import torch
import torch.nn as nn
from modeling.backbones.resnet import ResNet, Bottleneck
from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224
from modeling.fusion_part.fusion import Heterogenous_Transmission_Module
import torch.nn.functional as F
from modeling.backbones.swin import swin_base_patch4_win8
from modeling.backbones.ResTV2 import restv2_tiny, restv2_small, restv2_base, restv2_large
from modeling.backbones.edgeViT import edgevit_s
from modeling.backbones.t2tvit import t2t_vit_t_24,t2t_vit_t_14
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
        self.pattern = cfg.MODEL.RES_MODE
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


class Branch_new(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(Branch_new, self).__init__()
        self.resnet = build_resnet(num_classes, cfg)
        self.transformer = build_transformer(num_classes, cfg, camera_num, view_num, factory)

        self.num_classes = num_classes
        self.cfg = cfg
        self.camera = camera_num
        self.view = view_num

        self.mix_dim = cfg.MODEL.MIX_DIM
        self.srm_layer = cfg.MODEL.SRM_LAYER
        self.res_LRU = LocalRefinementUnits(dim=2048, out_dim=self.mix_dim)
        if 'swin' in cfg.MODEL.TRANSFORMER_TYPE or 'large' in cfg.MODEL.TRANSFORMER_TYPE:
            self.dim_l = 1024
        elif '14' in cfg.MODEL.TRANSFORMER_TYPE:
            self.dim_l = 384
        else:
            self.dim_l = 512
        self.former_LRU = LocalRefinementUnits(dim=self.dim_l, out_dim=self.mix_dim)
        self.gap_f = GeM()
        self.gap_r = GeM()
        self.mix = Heterogenous_Transmission_Module(depth=self.srm_layer, embed_dim=self.mix_dim)
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if 'swin' in cfg.MODEL.TRANSFORMER_TYPE or 'large' in cfg.MODEL.TRANSFORMER_TYPE or 't2t' in cfg.MODEL.TRANSFORMER_TYPE:
            self.patch_num = (512,16,8)
        elif 'edge' in cfg.MODEL.TRANSFORMER_TYPE:
            self.patch_num = (384,8,8)
        else:
            self.patch_num = (768,21,10)
        if '14' in cfg.MODEL.TRANSFORMER_TYPE:
            self.patch_num = (384,16,8)
        self.classifier_1 = nn.Linear(self.mix_dim, self.num_classes, bias=False)
        self.classifier_1.apply(weights_init_classifier)

        self.bottleneck_1 = nn.BatchNorm1d(self.mix_dim)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)

        self.classifier_2 = nn.Linear(self.mix_dim, self.num_classes, bias=False)
        self.classifier_2.apply(weights_init_classifier)

        self.bottleneck_2 = nn.BatchNorm1d(self.mix_dim)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)

        self.classifier_3 = nn.Linear(self.mix_dim, self.num_classes, bias=False)
        self.classifier_3.apply(weights_init_classifier)

        self.bottleneck_3 = nn.BatchNorm1d(self.mix_dim)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)

        self.classifier_4 = nn.Linear(self.mix_dim, self.num_classes, bias=False)
        self.classifier_4.apply(weights_init_classifier)

        self.bottleneck_4 = nn.BatchNorm1d(self.mix_dim)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)
        self.test_feat = cfg.TEST.FEAT

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def get_attn(self, x, k, label=None, cam_label=5, view_label=None):
        if not self.training:
            B = x.shape[0]
            mid_fea_r, feat_r = self.resnet(x)
            mid_fea_f, feat_f = self.transformer(x, cam_label=cam_label, view_label=view_label)
            # resnet feature conv
            mid_fea_r = self.res_LRU(mid_fea_r)
            local_res = self.gap_r(mid_fea_r)
            mid_fea_r = mid_fea_r.reshape(B, self.mix_dim, -1).permute(0, 2, 1)
            # former feature conv
            C,H,W = self.patch_num
            mid_fea_f = mid_fea_f.permute(0, 2, 1).reshape(B, int(C), int(H), int(W))
            mid_fea_f = self.former_LRU(mid_fea_f)
            local_former = self.gap_f(mid_fea_f)
            mid_fea_f = mid_fea_f.reshape(B, self.mix_dim, -1).permute(0, 2, 1)

            attn = self.mix.get_attn(mid_fea_r, mid_fea_f, local_res.reshape(B, 1, self.mix_dim),
                                     local_former.reshape(B, 1, self.mix_dim), k=k)
            return attn

    def forward(self, x, label=None, cam_label=5, view_label=None):
        if self.training:
            B = x.shape[0]
            mid_fea_r, cls_score_r, global_feat_r = self.resnet(x)
            mid_fea_f, cls_score_f, global_feat_f = self.transformer(x, cam_label=cam_label, view_label=view_label)
            # # resnet feature conv
            mid_fea_r = self.res_LRU(mid_fea_r)
            local_res = self.gap_r(mid_fea_r)
            mid_fea_r = mid_fea_r.reshape(B, self.mix_dim, -1).permute(0, 2, 1)
            # former feature conv
            C, H, W = self.patch_num
            mid_fea_f = mid_fea_f.permute(0, 2, 1).reshape(B, int(C), int(H), int(W))
            mid_fea_f = self.former_LRU(mid_fea_f)
            local_former = self.gap_f(mid_fea_f)
            mid_fea_f = mid_fea_f.reshape(B, self.mix_dim, -1).permute(0, 2, 1)

            # mix
            mix_r_q, mix_f_q = self.mix(mid_fea_r, mid_fea_f, local_res.reshape(B, 1, self.mix_dim),
                                        local_former.reshape(B, 1, self.mix_dim))

            global_feat_1 = local_res.view(B, -1)
            global_feat_2 = local_former.view(B, -1)
            global_feat_3 = mix_r_q.squeeze()
            global_feat_4 = mix_f_q.squeeze()

            feat_1 = self.bottleneck_1(global_feat_1)
            feat_2 = self.bottleneck_2(global_feat_2)
            feat_3 = self.bottleneck_3(global_feat_3)
            feat_4 = self.bottleneck_4(global_feat_4)

            cls_score_1 = self.classifier_1(feat_1)
            cls_score_2 = self.classifier_2(feat_2)
            cls_score_3 = self.classifier_3(feat_3)
            cls_score_4 = self.classifier_4(feat_4)
            return cls_score_r, global_feat_r, cls_score_f, global_feat_f, cls_score_1, global_feat_1, \
                cls_score_2, global_feat_2, cls_score_3, global_feat_3, cls_score_4, global_feat_4
        else:
            B = x.shape[0]
            mid_fea_r, feat_r = self.resnet(x)
            mid_fea_f, feat_f = self.transformer(x, cam_label=cam_label, view_label=view_label)
            # resnet feature conv
            mid_fea_r = self.res_LRU(mid_fea_r)
            local_res = self.gap_r(mid_fea_r)
            mid_fea_r = mid_fea_r.reshape(B, self.mix_dim, -1).permute(0, 2, 1)
            # former feature conv
            C, H, W = self.patch_num
            mid_fea_f = mid_fea_f.permute(0, 2, 1).reshape(B, int(C), int(H), int(W))
            mid_fea_f = self.former_LRU(mid_fea_f)
            local_former = self.gap_f(mid_fea_f)
            mid_fea_f = mid_fea_f.reshape(B, self.mix_dim, -1).permute(0, 2, 1)

            # mix
            mix_r_q, mix_f_q = self.mix(mid_fea_r, mid_fea_f, local_res.reshape(B, 1, self.mix_dim),
                                        local_former.reshape(B, 1, self.mix_dim))

            global_feat_1 = local_res.view(B, -1)
            global_feat_2 = local_former.view(B, -1)
            global_feat_3 = mix_r_q.squeeze()
            global_feat_4 = mix_f_q.squeeze()

            feat_1 = self.bottleneck_1(global_feat_1)
            feat_2 = self.bottleneck_2(global_feat_2)
            feat_3 = self.bottleneck_3(global_feat_3)
            feat_4 = self.bottleneck_4(global_feat_4)
            # print(self.test_feat)
            if self.neck_feat == 'after':
                pass
            else:
                feat_1 = global_feat_1
                feat_2 = global_feat_2
                feat_3 = global_feat_3
                feat_4 = global_feat_4
            if self.test_feat == 0:
                return torch.cat([feat_r, feat_f, feat_1, feat_2, feat_3, feat_4], dim=-1)
            elif self.test_feat == 1:
                return feat_r
            elif self.test_feat == 2:
                return feat_f
            elif self.test_feat == 3:
                return feat_1
            elif self.test_feat == 4:
                return feat_2
            elif self.test_feat == 5:
                return feat_3
            elif self.test_feat == 6:
                return feat_4


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
    if cfg.MODEL.RES_USE and not cfg.MODEL.TRANS_USE:
        model = build_resnet(num_class, cfg)
        print('===========Building ResNet Only===========')
        return model
    elif cfg.MODEL.TRANS_USE and not cfg.MODEL.RES_USE:
        model = build_transformer(num_class, cfg, camera_num, view_num, __factory_T_type)
        print('===========Building Transformer Only===========')
        return model
    elif cfg.MODEL.TRANS_USE and cfg.MODEL.RES_USE:
        model = Branch_new(num_class, cfg, camera_num, view_num, __factory_T_type)
        print('===========Building FusionReID===========')
        return model
    else:
        print("===========Fail to build model,Please check cfg.MODEL.RES_USE and cfg.MODEL.TRANS_USE===========")
        return None
