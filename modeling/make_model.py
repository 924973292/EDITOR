import torch
import torch.nn as nn
from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224
from modeling.fusion_part.SFTS import SFTS
from modeling.backbones.vit_pytorch import BlockMask
from modeling.fusion_part.Frequency import Frequency_based_Token_Selection


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
    def __init__(self, num_classes, cfg, camera_num, factory):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH_T
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

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        num_classes=num_classes,
                                                        camera=camera_num, view=0,
                                                        stride_size=cfg.MODEL.STRIDE_SIZE,
                                                        drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate=cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

    def forward(self, x, cam_label, view_label=None):
        cash_x, attn = self.base(x, camera_id=cam_label, view_id=view_label)
        return cash_x, attn

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


class EDITOR(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, factory):
        super(EDITOR, self).__init__()
        # Three Modalities share the same backbone
        self.BACKBONE = build_transformer(num_classes, cfg, camera_num, factory)
        self.num_patches = int(cfg.INPUT.SIZE_TRAIN[0] // cfg.MODEL.STRIDE_SIZE[0]) * int(
            cfg.INPUT.SIZE_TRAIN[1] // cfg.MODEL.STRIDE_SIZE[1])
        # Ratio means the keep ratio of the patches in each head
        self.ratio = (1 / self.num_patches) * int(cfg.MODEL.HEAD_KEEP)
        self.SFTS = SFTS(ratio=self.ratio)
        self.FREQ_INDEX = Frequency_based_Token_Selection(keep=cfg.MODEL.FREQUENCY_KEEP,
                                                          stride=cfg.MODEL.STRIDE_SIZE[0])
        self.FUSE_block = BlockMask(num_class=num_classes, dim=self.BACKBONE.token_dim, num_heads=12, mlp_ratio=4.,
                                    qkv_bias=False, momentum=0.8)
        # For the feature reduction, you can try to reduce in modality or across modalities, here is the across type
        # self.CLS_REDUCE = nn.Linear(3 * self.BACKBONE.token_dim, self.BACKBONE.token_dim)
        # self.CLS_REDUCE.apply(weights_init_kaiming)
        # self.PATCH_REDUCE = nn.Linear(3 * self.BACKBONE.token_dim, self.BACKBONE.token_dim)
        # self.PATCH_REDUCE.apply(weights_init_kaiming)
        # However, we use the in modality type in the paper as below
        # Reduce the dimension of the cls token and the patch token to token_dim
        self.RGB_REDUCE = nn.Linear(2 * self.BACKBONE.token_dim, self.BACKBONE.token_dim)
        self.RGB_REDUCE.apply(weights_init_kaiming)
        self.NIR_REDUCE = nn.Linear(2 * self.BACKBONE.token_dim, self.BACKBONE.token_dim)
        self.NIR_REDUCE.apply(weights_init_kaiming)
        self.TIR_REDUCE = nn.Linear(2 * self.BACKBONE.token_dim, self.BACKBONE.token_dim)
        self.TIR_REDUCE.apply(weights_init_kaiming)

        # The output learning params of fused features
        self.FUSE_HEAD = nn.Linear(3 * self.BACKBONE.token_dim, num_classes, bias=False)
        self.FUSE_BN = nn.BatchNorm1d(3 * self.BACKBONE.token_dim)
        self.FUSE_HEAD.apply(weights_init_classifier)

        # The output learning params of RGB/NIR/TIR cls tokens
        self.BACKBONE_HEAD = nn.Linear(self.BACKBONE.token_dim, num_classes, bias=False)
        self.BACKBONE_BN = nn.BatchNorm1d(self.BACKBONE.token_dim)
        self.BACKBONE_HEAD.apply(weights_init_classifier)
        # Here, you can choose to use different head for different modalities
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # self.BACKBONE_HEAD_2 = nn.Linear(self.BACKBONE.token_dim, num_classes, bias=False)
        # self.BACKBONE_BN_2 = nn.BatchNorm1d(self.BACKBONE.token_dim)
        # self.BACKBONE_HEAD_2.apply(weights_init_classifier)
        # self.BACKBONE_HEAD_3 = nn.Linear(self.BACKBONE.token_dim, num_classes, bias=False)
        # self.BACKBONE_BN_3 = nn.BatchNorm1d(self.BACKBONE.token_dim)
        # self.BACKBONE_HEAD_3.apply(weights_init_classifier)
        # If you use above head, you need to change the forward function to return the scores of different modalities
        # RGB_cls_score = self.BACKBONE_HEAD(self.BACKBONE_BN(RGB_cls4tri))
        # NIR_cls_score = self.BACKBONE_HEAD_2(self.BACKBONE_BN_2(NIR_cls4tri))
        # TIR_cls_score = self.BACKBONE_HEAD_3(self.BACKBONE_BN_3(TIR_cls4tri))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # In fact, you can choose the AL setting like TOP-ReID, here is the head for AL setting.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.AL = cfg.MODEL.AL
        if self.AL:
            self.AL_HEAD = nn.Linear(3 * self.BACKBONE.token_dim, num_classes, bias=False)
            self.AL_BN = nn.BatchNorm1d(3 * self.BACKBONE.token_dim)
            self.AL_HEAD.apply(weights_init_classifier)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def forward(self, x, cam_label=None, label=None, view_label=None, img_path=None, mode=1,
                writer=None, epoch=None):
        if self.training:
            RGB = x['RGB']
            NIR = x['NI']
            TIR = x['TI']
            mask_fre = self.FREQ_INDEX(x=RGB, y=NIR, z=TIR, img_path=img_path, mode=mode, writer=writer,
                                       step=epoch)
            RGB_feat, RGB_attn = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
            NIR_feat, NIR_attn = self.BACKBONE(NIR, cam_label=cam_label, view_label=view_label)
            TIR_feat, TIR_attn = self.BACKBONE(TIR, cam_label=cam_label, view_label=view_label)

            RGB_cls4tri = RGB_feat[:, 0, :]
            NIR_cls4tri = NIR_feat[:, 0, :]
            TIR_cls4tri = TIR_feat[:, 0, :]
            if self.AL:
                ori = torch.cat([RGB_cls4tri, NIR_cls4tri, TIR_cls4tri], dim=-1)
                ori_score = self.AL_HEAD(self.AL_BN(ori))
            else:
                RGB_cls_score = self.BACKBONE_HEAD(self.BACKBONE_BN(RGB_cls4tri))
                NIR_cls_score = self.BACKBONE_HEAD(self.BACKBONE_BN(NIR_cls4tri))
                TIR_cls_score = self.BACKBONE_HEAD(self.BACKBONE_BN(TIR_cls4tri))

            RGB_feat_s, NIR_feat_s, TIR_feat_s, mask, loss_bcc = self.SFTS(RGB_feat=RGB_feat,
                                                                           RGB_attn=RGB_attn,
                                                                           NIR_feat=NIR_feat,
                                                                           NIR_attn=NIR_attn,
                                                                           TIR_feat=TIR_feat,
                                                                           TIR_attn=TIR_attn,
                                                                           img_path=img_path,
                                                                           epoch=epoch, writer=writer,
                                                                           mask_fre=mask_fre)

            feat_s, loss_ocfr = self.FUSE_block(RGB_feat_s, NIR_feat_s, TIR_feat_s, mask=mask, label=label,
                                                epoch=epoch)

            RGB_feat_s = feat_s[:, :RGB_feat_s.shape[1]]
            NIR_feat_s = feat_s[:, RGB_feat_s.shape[1]:RGB_feat_s.shape[1] + NIR_feat_s.shape[1]]
            TIR_feat_s = feat_s[:, RGB_feat_s.shape[1] + NIR_feat_s.shape[1]:]
            RGB_cls = RGB_feat_s[:, 0, :]
            NIR_cls = NIR_feat_s[:, 0, :]
            TIR_cls = TIR_feat_s[:, 0, :]

            RGB_patch = RGB_feat_s[:, 1:, :]
            NIR_patch = NIR_feat_s[:, 1:, :]
            TIR_patch = TIR_feat_s[:, 1:, :]

            row_sum = torch.sum(RGB_patch, dim=2)
            num = (row_sum != 0).sum(dim=1).unsqueeze(-1)
            num_count = torch.mean(num.float())
            writer.add_scalar('num_count', num_count, epoch)
            RGB_patch = torch.sum(RGB_patch, dim=1) / num
            NIR_patch = torch.sum(NIR_patch, dim=1) / num
            TIR_patch = torch.sum(TIR_patch, dim=1) / num

            rgb = self.RGB_REDUCE(torch.cat([RGB_cls, RGB_patch], dim=-1))
            nir = self.NIR_REDUCE(torch.cat([NIR_cls, NIR_patch], dim=-1))
            tir = self.TIR_REDUCE(torch.cat([TIR_cls, TIR_patch], dim=-1))
            cls4t = torch.cat([rgb, nir, tir], dim=-1)
            score = self.FUSE_HEAD(self.FUSE_BN(cls4t))
            if self.AL:
                return score, cls4t, ori_score, ori, loss_bcc + loss_ocfr
            else:
                return score, cls4t, RGB_cls_score, RGB_cls4tri, NIR_cls_score, NIR_cls4tri, TIR_cls_score, TIR_cls4tri, loss_bcc + loss_ocfr
        else:
            RGB = x['RGB']
            NIR = x['NI']
            TIR = x['TI']
            mask_fre = self.FREQ_INDEX(x=RGB, y=NIR, z=TIR, img_path=img_path, mode=mode, writer=writer,
                                       step=epoch)
            RGB_feat, RGB_attn = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
            NIR_feat, NIR_attn = self.BACKBONE(NIR, cam_label=cam_label, view_label=view_label)
            TIR_feat, TIR_attn = self.BACKBONE(TIR, cam_label=cam_label, view_label=view_label)

            RGB_feat_s, NIR_feat_s, TIR_feat_s, mask = self.SFTS(RGB_feat=RGB_feat,
                                                                 RGB_attn=RGB_attn,
                                                                 NIR_feat=NIR_feat,
                                                                 NIR_attn=NIR_attn,
                                                                 TIR_feat=TIR_feat,
                                                                 TIR_attn=TIR_attn,
                                                                 img_path=img_path,
                                                                 epoch=epoch, writer=writer,
                                                                 mask_fre=mask_fre)

            feat_s = self.FUSE_block(RGB_feat_s, NIR_feat_s, TIR_feat_s, mask=mask, label=label,
                                     epoch=epoch)

            RGB_feat_s = feat_s[:, :RGB_feat_s.shape[1]]
            NIR_feat_s = feat_s[:, RGB_feat_s.shape[1]:RGB_feat_s.shape[1] + NIR_feat_s.shape[1]]
            TIR_feat_s = feat_s[:, RGB_feat_s.shape[1] + NIR_feat_s.shape[1]:]
            RGB_cls = RGB_feat_s[:, 0, :]
            NIR_cls = NIR_feat_s[:, 0, :]
            TIR_cls = TIR_feat_s[:, 0, :]

            RGB_patch = RGB_feat_s[:, 1:, :]
            NIR_patch = NIR_feat_s[:, 1:, :]
            TIR_patch = TIR_feat_s[:, 1:, :]

            row_sum = torch.sum(RGB_patch, dim=2)
            num = (row_sum != 0).sum(dim=1).unsqueeze(-1)
            RGB_patch = torch.sum(RGB_patch, dim=1) / num
            NIR_patch = torch.sum(NIR_patch, dim=1) / num
            TIR_patch = torch.sum(TIR_patch, dim=1) / num

            rgb = self.RGB_REDUCE(torch.cat([RGB_cls, RGB_patch], dim=-1))
            nir = self.NIR_REDUCE(torch.cat([NIR_cls, NIR_patch], dim=-1))
            tir = self.TIR_REDUCE(torch.cat([TIR_cls, TIR_patch], dim=-1))
            cls4t = torch.cat([rgb, nir, tir], dim=-1)
            return cls4t

    def forward_two_modalities(self, x, cam_label=None, label=None, view_label=None, cross_type=None, img_path=None,
                               mode=1,
                               writer=None, epoch=None):
        # This forward function is used for the two modalities datasets like RGBN300
        if self.training:
            RGB = x['RGB']
            NIR = x['NI']
            mask_fre = self.FREQ_INDEX(x=RGB, y=NIR, z=None, img_path=img_path, mode=mode, writer=writer,
                                       step=epoch)
            RGB_feat, RGB_attn = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label, img_path=img_path,
                                               epoch=epoch, modes=1, writer=writer)
            NIR_feat, NIR_attn = self.BACKBONE(NIR, cam_label=cam_label, view_label=view_label, img_path=img_path,
                                               epoch=epoch, modes=2, writer=writer)

            RGB_cls4tri = RGB_feat[:, 0, :]
            NIR_cls4tri = NIR_feat[:, 0, :]
            # Here, you need to change the head for the AL setting to 2*token_dim
            if self.AL:
                ori = torch.cat([RGB_cls4tri, NIR_cls4tri], dim=-1)
                ori_score = self.AL_HEAD(self.AL_BN(ori))
            else:
                RGB_cls_score = self.BACKBONE_HEAD(self.BACKBONE_BN(RGB_cls4tri))
                NIR_cls_score = self.BACKBONE_HEAD(self.BACKBONE_BN(NIR_cls4tri))

            RGB_feat_s, NIR_feat_s, mask, loss_bcc = self.SFTS(RGB_feat=RGB_feat,
                                                               RGB_attn=RGB_attn,
                                                               NIR_feat=NIR_feat,
                                                               NIR_attn=NIR_attn,
                                                               TIR_feat=None,
                                                               TIR_attn=None,
                                                               img_path=img_path,
                                                               epoch=epoch, writer=writer,
                                                               mask_fre=mask_fre)

            feat_s, loss_ocfr = self.FUSE_block(RGB_feat_s, NIR_feat_s, TIR=None, mask=mask, label=label,
                                                epoch=epoch)

            RGB_feat_s = feat_s[:, :RGB_feat_s.shape[1]]
            NIR_feat_s = feat_s[:, RGB_feat_s.shape[1]:RGB_feat_s.shape[1] + NIR_feat_s.shape[1]]
            RGB_cls = RGB_feat_s[:, 0, :]
            NIR_cls = NIR_feat_s[:, 0, :]

            RGB_patch = RGB_feat_s[:, 1:, :]
            NIR_patch = NIR_feat_s[:, 1:, :]

            row_sum = torch.sum(RGB_patch, dim=2)
            # 创建掩码来标记包含全零向量的行
            num = (row_sum != 0).sum(dim=1).unsqueeze(-1)
            RGB_patch = torch.sum(RGB_patch, dim=1) / num
            NIR_patch = torch.sum(NIR_patch, dim=1) / num

            rgb = self.RGB_REDUCE(torch.cat([RGB_cls, RGB_patch], dim=-1))
            nir = self.NIR_REDUCE(torch.cat([NIR_cls, NIR_patch], dim=-1))
            cls4t = torch.cat([rgb, nir], dim=-1)
            score = self.FUSE_HEAD(self.FUSE_BN(cls4t))
            if self.AL:
                return score, cls4t, ori_score, ori, loss_bcc + loss_ocfr
            else:
                return score, cls4t, RGB_cls_score, RGB_cls4tri, NIR_cls_score, NIR_cls4tri, loss_bcc + loss_ocfr

        else:
            RGB = x['RGB']
            NIR = x['NI']
            mask_fre = self.FREQ_INDEX(x=RGB, y=NIR, z=None, img_path=img_path, mode=mode, writer=writer,
                                       step=epoch)
            RGB_feat, RGB_attn = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label, img_path=img_path,
                                               epoch=epoch, modes=1, writer=writer)
            NIR_feat, NIR_attn = self.BACKBONE(NIR, cam_label=cam_label, view_label=view_label, img_path=img_path,
                                               epoch=epoch, modes=2, writer=writer)

            RGB_feat_s, NIR_feat_s, mask = self.PERSON_TOKEN_SELECT(RGB_feat=RGB_feat,
                                                                    RGB_attn=RGB_attn,
                                                                    NIR_feat=NIR_feat,
                                                                    NIR_attn=NIR_attn,
                                                                    TIR_feat=None,
                                                                    TIR_attn=None,
                                                                    img_path=img_path,
                                                                    epoch=epoch, writer=writer,
                                                                    mask_fre=mask_fre)

            feat_s = self.FUSE_block(RGB_feat_s, NIR_feat_s, TIR=None, mask=mask, label=label,
                                     epoch=epoch)

            RGB_feat_s = feat_s[:, :RGB_feat_s.shape[1]]
            NIR_feat_s = feat_s[:, RGB_feat_s.shape[1]:RGB_feat_s.shape[1] + NIR_feat_s.shape[1]]
            RGB_cls = RGB_feat_s[:, 0, :]
            NIR_cls = NIR_feat_s[:, 0, :]

            RGB_patch = RGB_feat_s[:, 1:, :]
            NIR_patch = NIR_feat_s[:, 1:, :]

            row_sum = torch.sum(RGB_patch, dim=2)
            # 创建掩码来标记包含全零向量的行
            num = (row_sum != 0).sum(dim=1).unsqueeze(-1)
            RGB_patch = torch.sum(RGB_patch, dim=1) / num
            NIR_patch = torch.sum(NIR_patch, dim=1) / num

            rgb = self.RGB_REDUCE(torch.cat([RGB_cls, RGB_patch], dim=-1))
            nir = self.NIR_REDUCE(torch.cat([NIR_cls, NIR_patch], dim=-1))
            cls4t = torch.cat([rgb, nir], dim=-1)
            return cls4t


__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
}


def make_model(cfg, num_class, camera_num):
    model = EDITOR(num_class, cfg, camera_num, __factory_T_type)
    print('===========Building EDITOR===========')
    return model
