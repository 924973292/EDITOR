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
from modeling.fusion_part.Person_Select import Person_Token_Select
from modeling.fusion_part.Reconstruct import RAll
from modeling.fusion_part.FUSE import FUSEAll
from layers.transfer_loss.mmd import MMDLoss
import torch.nn.functional as F


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

        self.token_dim = 2048
        self.pattern = cfg.MODEL.RES_MODE
        self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))


    def forward(self, x, cam_label=None, view_label=None, label=None):  # label is unused if self.cos_layer == 'no'
        mid_fea = self.base(x)
        return mid_fea

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
    Compute average Jaccard similarity between corresponding sets in a batch.

    Parameters:
    - set1 (torch.Tensor): Tensor representing set 1 (binary tensor).
    - set2 (torch.Tensor): Tensor representing set 2 (binary tensor).

    Returns:
    - torch.Tensor: Average Jaccard similarity score for the batch.
    """
    # Compute intersection and union for each pair of sets

    intersection = torch.sum(set1 & set2, dim=1)  # Bitwise AND and sum along N dimension
    union = torch.sum(set1 | set2, dim=1)  # Bitwise OR and sum along N dimension

    # Jaccard similarity for each pair
    similarity = intersection.float() / union.float()
    similarity.requires_grad = True
    # Average Jaccard similarity for the batch
    average_similarity = torch.mean(similarity)

    return -average_similarity  # Return negative to get similarity (optional)


def symmetric_kl_divergence(matrix1, matrix2):
    # 计算两个方向的 KL 散度
    kl1 = F.kl_div(matrix1.log(), matrix2, reduction='batchmean')
    kl2 = F.kl_div(matrix2.log(), matrix1, reduction='batchmean')

    # 对称化 KL 散度
    symmetric_kl = (kl1 + kl2)

    return symmetric_kl


def SelectionConsistency(matrix1, matrix2, matrix3=None):
    # 计算两两矩阵之间的 KL 散度损失
    kl_loss1 = symmetric_kl_divergence(matrix1, matrix2)
    if matrix3:
        kl_loss2 = symmetric_kl_divergence(matrix2, matrix3)
        kl_loss3 = symmetric_kl_divergence(matrix1, matrix3)

        # 求平均得到总的 KL 散度损失
        total_kl_loss = (kl_loss1 + kl_loss2 + kl_loss3)
    else:
        total_kl_loss = kl_loss1

    return total_kl_loss


class UniSReID(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(UniSReID, self).__init__()

        self.BACKBONE = build_transformer(num_classes, cfg, camera_num, view_num, factory)
        # self.BACKBONE = build_resnet(num_classes, cfg)
        self.RGB_GLOBAL_HEAD = nn.Linear(self.BACKBONE.token_dim, num_classes, bias=False)
        self.NIR_GLOBAL_HEAD = nn.Linear(self.BACKBONE.token_dim, num_classes, bias=False)
        self.TIR_GLOBAL_HEAD = nn.Linear(self.BACKBONE.token_dim, num_classes, bias=False)
        self.RGB_GLOBAL_BN = nn.BatchNorm1d(self.BACKBONE.token_dim)
        self.NIR_GLOBAL_BN = nn.BatchNorm1d(self.BACKBONE.token_dim)
        self.TIR_GLOBAL_BN = nn.BatchNorm1d(self.BACKBONE.token_dim)

        self.RGB_GLOBAL_HEAD.apply(weights_init_classifier)
        self.NIR_GLOBAL_HEAD.apply(weights_init_classifier)
        self.TIR_GLOBAL_HEAD.apply(weights_init_classifier)
        self.RGB_GLOBAL_BN.apply(weights_init_kaiming)
        self.NIR_GLOBAL_BN.apply(weights_init_kaiming)
        self.TIR_GLOBAL_BN.apply(weights_init_kaiming)
        self.ratio = 0.75
        self.PERSON_TOKEN_SELECT_R = Person_Token_Select(dim=self.BACKBONE.token_dim, ratio=self.ratio)
        self.PERSON_TOKEN_SELECT_N = Person_Token_Select(dim=self.BACKBONE.token_dim, ratio=self.ratio)
        self.PERSON_TOKEN_SELECT_T = Person_Token_Select(dim=self.BACKBONE.token_dim, ratio=self.ratio)
        self.PATCH_RECONSTURCT = RAll(dim=self.BACKBONE.token_dim, depth=4)

        self.RGB_LOCAL_HEAD = nn.Linear(self.BACKBONE.token_dim, num_classes, bias=False)
        self.NIR_LOCAL_HEAD = nn.Linear(self.BACKBONE.token_dim, num_classes, bias=False)
        self.TIR_LOCAL_HEAD = nn.Linear(self.BACKBONE.token_dim, num_classes, bias=False)
        self.RGB_LOCAL_BN = nn.BatchNorm1d(self.BACKBONE.token_dim)
        self.NIR_LOCAL_BN = nn.BatchNorm1d(self.BACKBONE.token_dim)
        self.TIR_LOCAL_BN = nn.BatchNorm1d(self.BACKBONE.token_dim)

        self.RGB_LOCAL_HEAD.apply(weights_init_classifier)
        self.NIR_LOCAL_HEAD.apply(weights_init_classifier)
        self.TIR_LOCAL_HEAD.apply(weights_init_classifier)
        self.RGB_LOCAL_BN.apply(weights_init_kaiming)
        self.NIR_LOCAL_BN.apply(weights_init_kaiming)
        self.TIR_LOCAL_BN.apply(weights_init_kaiming)

        self.FUSE = FUSEAll(dim=self.BACKBONE.token_dim, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, )

        self.FUSE_HEAD = nn.Linear(self.BACKBONE.token_dim, num_classes, bias=False)
        self.FUSE_BN = nn.BatchNorm1d(self.BACKBONE.token_dim)
        self.FUSE_HEAD.apply(weights_init_classifier)

        self.COMBINE_HEAD = nn.Linear(self.BACKBONE.token_dim, num_classes, bias=False)
        self.COMBINE_BN = nn.BatchNorm1d(self.BACKBONE.token_dim)
        self.COMBINE_HEAD.apply(weights_init_classifier)
        self.cross = 0
        self.mmd = MMDLoss()
        self.gap = GeM()

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def forward(self, x, cam_label=None, label=None, view_label=None, cross_type=None, img_path=None, mode=1,
                writer=None, epoch=None):
        if self.training:
            RGB = x['RGB']
            NIR = x['NI']
            TIR = x['TI']
            # 判断NIR和TIR的第一张图像是否不一样
            if not torch.equal(NIR[0], TIR[0]):
                RGB_FEAT = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
                NIR_FEAT = self.BACKBONE(NIR, cam_label=cam_label, view_label=view_label)
                TIR_FEAT = self.BACKBONE(TIR, cam_label=cam_label, view_label=view_label)

                RGB_cls4tri = RGB_FEAT[:, 0, :]
                NIR_cls4tri = NIR_FEAT[:, 0, :]
                TIR_cls4tri = TIR_FEAT[:, 0, :]
                RGB_cls_score = self.RGB_GLOBAL_HEAD(self.RGB_GLOBAL_BN(RGB_cls4tri))
                NIR_cls_score = self.NIR_GLOBAL_HEAD(self.NIR_GLOBAL_BN(NIR_cls4tri))
                TIR_cls_score = self.TIR_GLOBAL_HEAD(self.TIR_GLOBAL_BN(TIR_cls4tri))

                RGB_patch, RGB_sim = self.PERSON_TOKEN_SELECT_R(RGB_FEAT, img_path, mode=1, writer=writer, epoch=epoch)
                NIR_patch, NIR_sim = self.PERSON_TOKEN_SELECT_N(NIR_FEAT, img_path, mode=2, writer=writer, epoch=epoch)
                TIR_patch, TIR_sim = self.PERSON_TOKEN_SELECT_T(TIR_FEAT, img_path, mode=3, writer=writer, epoch=epoch)
                loss_cons = -0.001 * SelectionConsistency(RGB_sim, NIR_sim, TIR_sim)
                writer.add_scalar('loss_cons', loss_cons, global_step=epoch)
                print(loss_cons)

                RGB_patch, NIR_patch, TIR_patch, Patch_combine4tri, loss_cross = self.PATCH_RECONSTURCT(RGB_patch,
                                                                                                        NIR_patch,
                                                                                                        TIR_patch)
                Patch_combine_score = self.COMBINE_HEAD(self.COMBINE_BN(Patch_combine4tri))

                RGB_patch4tri = torch.mean(RGB_patch, dim=1)
                NIR_patch4tri = torch.mean(NIR_patch, dim=1)
                TIR_patch4tri = torch.mean(TIR_patch, dim=1)
                RGB_patch_score = self.RGB_LOCAL_HEAD(self.RGB_LOCAL_BN(RGB_patch4tri))
                NIR_patch_score = self.NIR_LOCAL_HEAD(self.NIR_LOCAL_BN(NIR_patch4tri))
                TIR_patch_score = self.TIR_LOCAL_HEAD(self.TIR_LOCAL_BN(TIR_patch4tri))

                FUSE_cls4tri = self.FUSE(RGB_cls4tri=RGB_cls4tri, NIR_cls4tri=NIR_cls4tri, TIR_cls4tri=TIR_cls4tri,
                                         RGB_patch=RGB_patch, NIR_patch=NIR_patch, TIR_patch=TIR_patch)
                FUSE_score = self.FUSE_HEAD(self.FUSE_BN(FUSE_cls4tri))

                if self.ratio == 1.0:
                    loss_cons = 0
                return RGB_cls_score, RGB_cls4tri, NIR_cls_score, NIR_cls4tri, TIR_cls_score, TIR_cls4tri, FUSE_score, FUSE_cls4tri, \
                    RGB_patch_score, RGB_patch4tri, NIR_patch_score, NIR_patch4tri, TIR_patch_score, TIR_patch4tri, Patch_combine_score, Patch_combine4tri, \
                    loss_cross + loss_cons
            else:
                RGB_FEAT = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
                NIR_FEAT = self.BACKBONE(NIR, cam_label=cam_label, view_label=view_label)

                RGB_cls4tri = RGB_FEAT[:, 0, :]
                NIR_cls4tri = NIR_FEAT[:, 0, :]
                # RGB_cls4tri = self.gap(RGB_FEAT).squeeze()
                # NIR_cls4tri = self.gap(NIR_FEAT).squeeze()

                RGB_cls_score = self.RGB_GLOBAL_HEAD(self.RGB_GLOBAL_BN(RGB_cls4tri))
                NIR_cls_score = self.NIR_GLOBAL_HEAD(self.NIR_GLOBAL_BN(NIR_cls4tri))
                loss = self.mmd(RGB_cls4tri, NIR_cls4tri)
                # RGB_patch, RGB_sim = self.PERSON_TOKEN_SELECT_R(RGB_FEAT, img_path, mode=1, writer=writer, epoch=epoch)
                # NIR_patch, NIR_sim = self.PERSON_TOKEN_SELECT_N(NIR_FEAT, img_path, mode=2, writer=writer, epoch=epoch)
                #
                # loss_cons = -0.001 * SelectionConsistency(RGB_sim, NIR_sim)
                # writer.add_scalar('loss_cons', loss_cons, global_step=epoch)
                # print(loss_cons)
                # 
                # RGB_patch, NIR_patch, Patch_combine4tri, loss_cross = self.PATCH_RECONSTURCT(RGB_patch,
                #                                                                              NIR_patch)
                # Patch_combine_score = self.COMBINE_HEAD(self.COMBINE_BN(Patch_combine4tri))
                #
                # RGB_patch4tri = torch.mean(RGB_patch, dim=1)
                # NIR_patch4tri = torch.mean(NIR_patch, dim=1)
                #
                # RGB_patch_score = self.RGB_LOCAL_HEAD(self.RGB_LOCAL_BN(RGB_patch4tri))
                # NIR_patch_score = self.NIR_LOCAL_HEAD(self.NIR_LOCAL_BN(NIR_patch4tri))
                #
                # FUSE_cls4tri = self.FUSE(RGB_cls4tri=RGB_cls4tri, NIR_cls4tri=RGB_cls4tri, RGB_patch=RGB_patch,
                #                          NIR_patch=NIR_patch)
                # FUSE_score = self.FUSE_HEAD(self.FUSE_BN(FUSE_cls4tri))
                #
                # if self.ratio == 1.0:
                #     loss_cons = 0
                return RGB_cls_score, RGB_cls4tri, NIR_cls_score, NIR_cls4tri, loss
                # return RGB_cls_score, RGB_cls4tri, NIR_cls_score, NIR_cls4tri, FUSE_score, FUSE_cls4tri, \
                #     RGB_patch_score, RGB_patch4tri, NIR_patch_score, NIR_patch4tri, Patch_combine_score, Patch_combine4tri, \
                #     loss_cross + loss_cons
        else:
            if mode == 1:
                self.cross = 0
            else:
                self.cross = 1

            RGB = x['RGB']
            NIR = x['NI']
            TIR = x['TI']
            if not torch.equal(NIR[0], TIR[0]):
                RGB_FEAT = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
                NIR_FEAT = self.BACKBONE(NIR, cam_label=cam_label, view_label=view_label)
                TIR_FEAT = self.BACKBONE(TIR, cam_label=cam_label, view_label=view_label)

                RGB_cls4tri = RGB_FEAT[:, 0, :]
                NIR_cls4tri = NIR_FEAT[:, 0, :]
                TIR_cls4tri = TIR_FEAT[:, 0, :]

                RGB_patch = self.PERSON_TOKEN_SELECT_R(RGB_FEAT, img_path, mode=1)
                NIR_patch = self.PERSON_TOKEN_SELECT_N(NIR_FEAT, img_path, mode=2)
                TIR_patch = self.PERSON_TOKEN_SELECT_T(TIR_FEAT, img_path, mode=3)

                RGB_patch, NIR_patch, TIR_patch, RGB_cls, NIR_cls, TIR_cls = self.PATCH_RECONSTURCT(RGB_patch,
                                                                                                    NIR_patch,
                                                                                                    TIR_patch)

                FUSE_cls4tri = self.FUSE(RGB_cls4tri=RGB_cls4tri, NIR_cls4tri=NIR_cls4tri, TIR_cls4tri=TIR_cls4tri,
                                         RGB_patch=RGB_patch, NIR_patch=NIR_patch, TIR_patch=TIR_patch)
                if self.cross:
                    return RGB_cls, NIR_cls, RGB_cls, TIR_cls
                else:
                    return FUSE_cls4tri
            else:
                RGB_FEAT = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
                NIR_FEAT = self.BACKBONE(NIR, cam_label=cam_label, view_label=view_label)

                RGB_cls = RGB_FEAT[:, 0, :]
                NIR_cls = NIR_FEAT[:, 0, :]

                # RGB_cls = self.gap(RGB_FEAT)
                # NIR_cls = self.gap(NIR_FEAT)

                # RGB_patch = self.PERSON_TOKEN_SELECT_R(RGB_FEAT, img_path, mode=1)
                # NIR_patch = self.PERSON_TOKEN_SELECT_N(NIR_FEAT, img_path, mode=2)

                # RGB_patch, NIR_patch, RGB_cls, NIR_cls = self.PATCH_RECONSTURCT(RGB_patch,
                #                                                                 NIR_patch)

                if self.cross:
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
