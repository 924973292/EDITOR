import torch
import torch.nn as nn
from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224, trunc_normal_
from modeling.backbones.resnet import ResNet, Bottleneck
from modeling.backbones.t2t import t2t_vit_t_14, t2t_vit_t_24
from modeling.backbones.osnet import osnet_x1_0
from modeling.backbones.hacnn import HACNN
from modeling.backbones.mudeep import MuDeep
from modeling.backbones.pcb import pcb_p6
from modeling.backbones.mlfn import mlfn
from modeling.fusion_part.MLP import Mlp
from modeling.fusion_part.Person_Select import Person_Token_Select, Person_Token_SelectC
from modeling.fusion_part.Reconstruct import RAll, RAllC
from modeling.fusion_part.FUSE import FUSEAll, FUSEAllC
from layers.transfer_loss.mmd import MMDLoss


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
        self.token_dim = 768
        self.trans_type = cfg.MODEL.TRANSFORMER_TYPE
        if 't2t' in cfg.MODEL.TRANSFORMER_TYPE:
            self.token_dim = 512
        if 'edge' in cfg.MODEL.TRANSFORMER_TYPE or cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224':
            self.token_dim = 384
        if '14' in cfg.MODEL.TRANSFORMER_TYPE:
            self.token_dim = 384
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

        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

    def forward(self, x, cam_label, view_label=None, label=None):
        cash_x = self.base(x, cam_label=cam_label, view_label=view_label)
        return cash_x

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


def IOU(set1, set2):
    """
    Compute Jaccard similarity between two sets.

    Parameters:
    - set1 (torch.Tensor): Tensor representing set 1 (binary tensor).
    - set2 (torch.Tensor): Tensor representing set 2 (binary tensor).

    Returns:
    - float: Jaccard similarity score.
    """
    # Compute intersection and union
    intersection = torch.sum(set1 * set2)
    union = torch.sum(set1) + torch.sum(set2) - intersection

    # Jaccard similarity
    similarity = intersection / union
    return -similarity.item()


class UniSReID(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(UniSReID, self).__init__()

        self.BACKBONE = build_transformer(num_classes, cfg, camera_num, view_num, factory)
        # self.RGB_PROJECTOR = Mlp(in_features=self.BACKBONE.token_dim)
        # self.NIR_PROJECTOR = Mlp(in_features=self.BACKBONE.token_dim)
        # self.TIR_PROJECTOR = Mlp(in_features=self.BACKBONE.token_dim)

        self.RGB_GLOBAL_HEAD = nn.Linear(self.BACKBONE.token_dim, num_classes, bias=False)
        self.NIR_GLOBAL_HEAD = nn.Linear(self.BACKBONE.token_dim, num_classes, bias=False)
        self.TIR_GLOBAL_HEAD = nn.Linear(self.BACKBONE.token_dim, num_classes, bias=False)
        self.RGB_GLOBAL_BN = nn.BatchNorm1d(self.BACKBONE.token_dim)
        self.NIR_GLOBAL_BN = nn.BatchNorm1d(self.BACKBONE.token_dim)
        self.TIR_GLOBAL_BN = nn.BatchNorm1d(self.BACKBONE.token_dim)

        self.RGB_GLOBAL_HEAD.apply(weights_init_classifier)
        self.NIR_GLOBAL_HEAD.apply(weights_init_classifier)
        self.TIR_GLOBAL_HEAD.apply(weights_init_classifier)
        self.RGB_GLOBAL_BN.requires_grad_(False)
        self.RGB_GLOBAL_BN.apply(weights_init_kaiming)
        self.NIR_GLOBAL_BN.requires_grad_(False)
        self.NIR_GLOBAL_BN.apply(weights_init_kaiming)
        self.TIR_GLOBAL_BN.requires_grad_(False)
        self.TIR_GLOBAL_BN.apply(weights_init_kaiming)

        self.PERSON_TOKEN_SELECT = Person_Token_SelectC(dim=self.BACKBONE.token_dim, ratio=0.75)
        self.PATCH_RECONSTURCT = RAllC(dim=self.BACKBONE.token_dim, depth=4)

        self.RGB_LOCAL_HEAD = nn.Linear(self.BACKBONE.token_dim, num_classes, bias=False)
        self.NIR_LOCAL_HEAD = nn.Linear(self.BACKBONE.token_dim, num_classes, bias=False)
        self.TIR_LOCAL_HEAD = nn.Linear(self.BACKBONE.token_dim, num_classes, bias=False)
        self.RGB_LOCAL_BN = nn.BatchNorm1d(self.BACKBONE.token_dim)
        self.NIR_LOCAL_BN = nn.BatchNorm1d(self.BACKBONE.token_dim)
        self.TIR_LOCAL_BN = nn.BatchNorm1d(self.BACKBONE.token_dim)

        self.RGB_LOCAL_HEAD.apply(weights_init_classifier)
        self.NIR_LOCAL_HEAD.apply(weights_init_classifier)
        self.TIR_LOCAL_HEAD.apply(weights_init_classifier)
        self.RGB_LOCAL_BN.requires_grad_(False)
        self.RGB_LOCAL_BN.apply(weights_init_kaiming)
        self.NIR_LOCAL_BN.requires_grad_(False)
        self.NIR_LOCAL_BN.apply(weights_init_kaiming)
        self.TIR_LOCAL_BN.requires_grad_(False)
        self.TIR_LOCAL_BN.apply(weights_init_kaiming)

        self.FUSE = FUSEAllC(dim=self.BACKBONE.token_dim, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, )

        self.FUSE_HEAD = nn.Linear(self.BACKBONE.token_dim, num_classes, bias=False)
        self.FUSE_BN = nn.BatchNorm1d(self.BACKBONE.token_dim)
        self.FUSE_HEAD.apply(weights_init_classifier)
        self.FUSE_BN.requires_grad_(False)
        self.cross = 0

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def forward(self, x, cam_label=None, label=None, view_label=None, cross_type=None, img_path=None, mode=1):
        if self.training:
            RGB = x['RGB']
            NIR = x['NI']

            RGB_FEAT = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
            NIR_FEAT = self.BACKBONE(NIR, cam_label=cam_label, view_label=view_label)

            RGB_cls4tri = RGB_FEAT[:, 0, :]
            NIR_cls4tri = NIR_FEAT[:, 0, :]

            RGB_cls_score = self.RGB_GLOBAL_HEAD(self.RGB_GLOBAL_BN(RGB_cls4tri))
            NIR_cls_score = self.NIR_GLOBAL_HEAD(self.NIR_GLOBAL_BN(NIR_cls4tri))

            RGB_patch, RGB_index = self.PERSON_TOKEN_SELECT(RGB_FEAT, img_path, mode=1)
            NIR_patch, NIR_index = self.PERSON_TOKEN_SELECT(NIR_FEAT, img_path, mode=2)

            loss_cons = IOU(RGB_index, NIR_index)

            RGB_patch4tri, NIR_patch4tri, loss_cross = self.PATCH_RECONSTURCT(RGB_patch, NIR_patch)
            RGB_patch_score = self.RGB_LOCAL_HEAD(self.RGB_LOCAL_BN(RGB_patch4tri))
            NIR_patch_score = self.NIR_LOCAL_HEAD(self.NIR_LOCAL_BN(NIR_patch4tri))

            FUSE_cls4tri = self.FUSE(RGB_cls4tri, NIR_cls4tri, RGB_patch, NIR_patch)
            FUSE_score = self.FUSE_HEAD(self.FUSE_BN(FUSE_cls4tri))

            return RGB_cls_score, RGB_cls4tri, NIR_cls_score, NIR_cls4tri, RGB_patch_score, RGB_patch4tri, NIR_patch_score, NIR_patch4tri, FUSE_score, FUSE_cls4tri, loss_cross + loss_cons
        else:
            if mode == 1:
                self.cross = 0
            else:
                self.cross = 1

            RGB = x['RGB']
            NIR = x['NI']

            RGB_FEAT = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
            NIR_FEAT = self.BACKBONE(NIR, cam_label=cam_label, view_label=view_label)

            # RGB_cls4tri = RGB_FEAT[:, 0, :]
            # NIR_cls4tri = NIR_FEAT[:, 0, :]

            RGB_patch = self.PERSON_TOKEN_SELECT(RGB_FEAT, img_path, mode=1)
            NIR_patch = self.PERSON_TOKEN_SELECT(NIR_FEAT, img_path, mode=2)

            # FUSE_cls4tri = self.FUSE(RGB_cls4tri, NIR_cls4tri, RGB_patch, NIR_patch)

            if self.cross:
                RGB_cls, NIR_cls = self.PATCH_RECONSTURCT(RGB_patch, NIR_patch)
                return RGB_cls, NIR_cls, NIR_cls, RGB_cls
            else:
                return torch.tensor(1)


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


def make_model(cfg, num_class, camera_num, view_num=0):
    if cfg.MODEL.BASE == 0:
        model = UniSReID(num_class, cfg, camera_num, view_num, __factory_T_type)
        print('===========Building Baseline===========')
    return model
