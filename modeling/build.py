from layers import objectives
from modeling.backbones.clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

class FM_cat(nn.Module):
    def __init__(self,in_channels):
        super(FM_cat, self).__init__()

        self.W = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_channels)
        )
        nn.init.normal_(self.W[1].weight.data, 1.0, 0.01)
        nn.init.zeros_(self.W[1].bias.data)


        # self.bottleneck = nn.BatchNorm1d(in_channels)
        # self.bottleneck.bias.requires_grad_(False)  # no shift

        # nn.init.normal_(self.bottleneck.weight.data, 1.0, 0.01)
        # nn.init.zeros_(self.bottleneck.bias.data)

    def forward(self,f):

        f = f.view(f.size(0),f.size(1),1,1)
        f = self.W(f)
        f = f.view(f.size(0),-1)
        # f = self.bottleneck(f+feat)

        return f

class CLIP2ReID(nn.Module):
    def __init__(self, cfg, num_classes=11003):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.test_setting = cfg.test_setting

        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(cfg.pretrain_choice, cfg.img_size, cfg.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / cfg.temperature))  # 0.07
        # self.logit_scale = torch.ones([]) * np.log(1 / cfg.temperature)  # 0.07

        if cfg.fusion_way == 'weight add':
            self.gate = nn.Parameter(torch.FloatTensor(2))
            nn.init.constant_(self.gate, 0.5)
        if cfg.fusion_way == 'concat':
            scale = 512**-0.5
            proj_std = scale * ((2 * 4)**-0.5)
            self.dim_conv = nn.Linear(512*2,512)
            nn.init.normal_(self.dim_conv.weight, std=proj_std)
            
        if cfg.fusion_way == 'global concat':
            self.dim_conv = nn.Linear(512*2,512)
            self.global_attn_s = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.global_attn_t = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
             # init cross attn
            scale = 512**-0.5
            proj_std = scale * ((2 * 4)**-0.5)
            attn_std = scale
            fc_std = (2 * 512)**-0.5
            nn.init.normal_(self.global_attn_s.in_proj_weight, std=attn_std)
            nn.init.normal_(self.global_attn_s.out_proj.weight, std=proj_std)

             # init cross attn
            nn.init.normal_(self.global_attn_t.in_proj_weight, std=attn_std)
            nn.init.normal_(self.global_attn_t.out_proj.weight, std=proj_std)
            
            nn.init.normal_(self.dim_conv.weight, std=proj_std)


        if 'concat' in cfg.fusion_way:
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=cfg.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)

        if 'cross' in cfg.fusion_way:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=cfg.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            # self.pos_embedding = nn.Parameter(scale * torch.randn(self.embed_dim))
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

        if 'id' in cfg.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mcm' in cfg.loss_names or 'mcq' in cfg.loss_names or 'mlm' in cfg.loss_names or 'msm' in cfg.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=cfg.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            # self.pos_embedding = nn.Parameter(scale * torch.randn(self.embed_dim))
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            # mcm
            if 'mcm' in cfg.loss_names:
                self.mcm_head = nn.Sequential(
                    OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                                ('gelu', QuickGELU()),
                                ('ln', LayerNorm(self.embed_dim)),
                                ('fc', nn.Linear(self.embed_dim, cfg.num_colors))]))
                # self.mcm_head = nn.Linear(self.embed_dim, cfg.num_colors)
                # init mcm head
                nn.init.normal_(self.mcm_head.dense.weight, std=fc_std)
                nn.init.normal_(self.mcm_head.fc.weight, std=proj_std)
            
            # mcq
            # if 'mcq' in cfg.loss_names:
            #     self.mcq_proj = nn.Parameter(scale * torch.randn(self.cross_modal_transformer.width, self.embed_dim))
            
            # TODO mlm
            if 'mlm' in cfg.loss_names:
                self.mlm_head = nn.Sequential(
                    OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                                ('gelu', QuickGELU()),
                                ('ln', LayerNorm(self.embed_dim)),
                                ('fc', nn.Linear(self.embed_dim, cfg.vocab_size))]))
                # self.mlm_head = nn.Linear(self.embed_dim, cfg.num_colors)
                # init mlm head
                nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
                nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def _set_task(self):
        loss_names = self.cfg.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        # x = q + x # residual connection (invalid for mcq and mcqmlm, valid for mlm)
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.cross_modal_transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x
    
    def global_former_s(self, q, k, v):
        x = self.global_attn_s(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = self.ln_post(x)
        return x

    def global_former_t(self, q, k, v):
        x = self.global_attn_t(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x #[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()
    
    def  fusion_layer(self, text, sketch, caption_ids, pa=0.1, way='add'):

        if way == 'weight add':
            f_feats = self.gate[0] * text[torch.arange(text.shape[0]), caption_ids.argmax(dim=-1)] + self.gate[1] * sketch[:, 0, :]
        elif way == 'cross attention':
            f_feats = (self.cross_former(text[torch.arange(text.shape[0]), caption_ids.argmax(dim=-1)].unsqueeze(1),sketch,sketch) + self.cross_former(sketch[:,0,:].unsqueeze(1),text,text))
            f_feats = f_feats.squeeze(1).contiguous()
        elif way == 'cross attention text':
            # f_feats = (self.cross_former(text,sketch,sketch)[:, 0, :] + self.cross_former(sketch,text,text)[torch.arange(text.shape[0]), caption_ids.argmax(dim=-1)])
            f_feats = self.cross_former(sketch[:,0,:].unsqueeze(1),text,text).squeeze(1).contiguous()
        elif way == 'cross attention sketch':
            # f_feats = (self.cross_former(text,sketch,sketch)[:, 0, :] + self.cross_former(sketch,text,text)[torch.arange(text.shape[0]), caption_ids.argmax(dim=-1)])
            f_feats = self.cross_former(text[torch.arange(text.shape[0]), caption_ids.argmax(dim=-1)].unsqueeze(1),sketch,sketch).squeeze(1).contiguous()
        elif way == 'parameter add':
            f_feats = (1-pa)*text[torch.arange(text.shape[0]), caption_ids.argmax(dim=-1)] + pa*sketch[:, 0, :]
        elif way == 'concat':
            f_feats = self.dim_conv(torch.cat((text[torch.arange(text.shape[0]), caption_ids.argmax(dim=-1)], sketch[:, 0, :]),dim=1))
        elif way == 'global concat':
            s_global = self.global_former_s(sketch[:,0,:,],sketch[:,1:,:],sketch[:,1:,:])
            eos_indices = caption_ids.argmax(dim=-1)
            t_globel = text[torch.arange(text.shape[0]), eos_indices]
            text[torch.arange(text.shape[0]), eos_indices] = 0
            t_local = text
            t_global = self.global_former_t(t_globel,t_local, t_local)
            # f_feats = self.dim_conv(torch.cat((, caption_ids.argmax(dim=-1)], sketch[:, 0, :]),dim=1))
        elif way == 'concat transformer':
            l_t = text.size(1)
            f_feats = self.cross_modal_transformer(torch.cat((text,sketch),dim=1))
            f_feats = f_feats[:,l_t:,:][:, 0, :] + f_feats[:,:l_t,:][torch.arange(text.shape[0]), caption_ids.argmax(dim=-1)]
        elif way == 'concat transformer-s':
            l_t = text.size(1)
            f_feats = self.cross_modal_transformer(torch.cat((text,sketch),dim=1))
            f_feats = f_feats[:,l_t:,:][:, 0, :]
        elif way == 'concat transformer-t':
            l_t = text.size(1)
            f_feats = self.cross_modal_transformer(torch.cat((text,sketch),dim=1))
            f_feats = f_feats[:,:l_t,:][torch.arange(text.shape[0]), caption_ids.argmax(dim=-1)]
        else:
            
            f_feats = text[torch.arange(text.shape[0]), caption_ids.argmax(dim=-1)] + sketch[:, 0, :]
            
        return f_feats.float()

    def forward(self, batch):
        ret = dict()
        images = batch['images']
        caption_ids = batch['caption_ids']
        simages =  batch['simages']
        label = batch['pids']

        image_feats, text_feats = self.base_model(torch.cat((images,simages),dim=0), caption_ids)
        b = image_feats.size(0)
        simage_feats = image_feats[int(b/2):,:,:] # [64, 193, 512] text:[64, 77, 512]
        image_feats = image_feats[:int(b/2),:,:]

        logit_scale = self.logit_scale.exp()
        ret.update({'temperature': 1 / logit_scale})

        if self.cfg.only_sketch:
            i_feats = image_feats[:, 0, :].float()
            si_feats = simage_feats[:, 0, :].float()   
            ret.update({'itc_loss':(objectives.compute_itc(i_feats, si_feats, logit_scale))*self.cfg.cmm_loss_weight})
        elif self.cfg.only_text:
            i_feats = image_feats[:, 0, :].float()
            t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float() #[64, 512]
            ret.update({'itc_loss':(objectives.compute_itc(i_feats, t_feats, logit_scale))*self.cfg.cmm_loss_weight})
        # elif self.cfg.only_fusion:
        #     i_feats = image_feats[:, 0, :].float()
        #     si_feats = simage_feats[:, 0, :].float()
        #     t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float() #[64, 512]
        #     f_feats = t_feats + si_feats
        #     ret.update({'itc_loss':(objectives.compute_itc(i_feats, f_feats, logit_scale))*self.cfg.cmm_loss_weight})
        else: 
            if self.cfg.fusion_way in ['add', 'weight add', 'cross attention', 'parameter add', 'concat', 'global concat', 'cross attention text', 'cross attention sketch', 'concat transformer']:
                f_feats = self.fusion_layer(text_feats, simage_feats, caption_ids, pa=self.cfg.pa, way=self.cfg.fusion_way)
                i_feats = image_feats[:, 0, :].float()
                si_feats = simage_feats[:, 0, :].float()
                t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float() #[64, 512]
                if self.cfg.only_fusion_loss:
                    ret.update({'itc_loss':(objectives.compute_itc(i_feats, f_feats, logit_scale))*self.cfg.cmm_loss_weight})
                elif self.cfg.four_fusion_loss:
                    ret.update({'itc_loss':(objectives.compute_itc(i_feats, t_feats, logit_scale) + objectives.compute_itc(i_feats, si_feats, logit_scale) + objectives.compute_itc(i_feats, f_feats, logit_scale)+objectives.compute_itc(si_feats, t_feats, logit_scale))*self.cfg.cmm_loss_weight})
                elif self.cfg.focal_three_fusion_loss3:
                    ret.update({'itc_loss':(objectives.compute_itc_focal3(i_feats, t_feats, si_feats, f_feats, logit_scale, self.cfg.al, self.cfg.ga, self.cfg.klp))*self.cfg.cmm_loss_weight})
                else:
                    ret.update({'itc_loss':(objectives.compute_itc(i_feats, t_feats, logit_scale) + objectives.compute_itc(i_feats, si_feats, logit_scale) + objectives.compute_itc(i_feats, f_feats, logit_scale))*self.cfg.cmm_loss_weight})
            else:
                i_feats = image_feats[:, 0, :].float()
                si_feats = simage_feats[:, 0, :].float()
                t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float() #[64, 512]
                ret.update({'itc_loss':(objectives.compute_itc(i_feats, t_feats, logit_scale)+objectives.compute_itc(i_feats, si_feats, logit_scale))*self.cfg.cmm_loss_weight})

        return ret

        # if 'tcmpm' in self.current_task:
        #     if self.cfg.use_imageid:
        #         ret.update({'tcmpm_loss':objectives.compute_tcmpm(i_feats, t_feats, batch['pids'], logit_scale, image_id=batch['image_ids'])*self.cfg.cmm_loss_weight})
        #     else:
        #         ret.update({'tcmpm_loss':objectives.compute_tcmpm(i_feats, t_feats, batch['pids'], logit_scale)*self.cfg.cmm_loss_weight})
        
        # if 'itc' in self.current_task:
        #     ret.update({'itc_loss':(objectives.compute_itc(i_feats, t_feats, logit_scale)+objectives.compute_itc(i_feats, si_feats, logit_scale)+objectives.compute_itc(i_feats, f_feats, logit_scale))*self.cfg.cmm_loss_weight})
        
        # if 'sdm' in self.current_task:
        #     ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)*self.cfg.cmm_loss_weight})

        # if 'cmpm' in self.current_task:
        #     ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])*self.cfg.cmm_loss_weight})
        
        # if 'supcon' in self.current_task:
        #     bs, d = i_feats.size()
        #     i_feats = i_feats.view(-1, self.cfg.num_instance, d)
        #     si_feats = si_feats.view(-1, self.cfg.num_instance, d)
        #     t_feats = t_feats.view(-1, self.cfg.num_instance, d)
        #     f_feats = f_feats.view(-1, self.cfg.num_instance, d)
        #     label = label.view(-1, self.cfg.num_instance)[:,0]

        #     ret.update({'supcon_loss':(objectives.SupConLoss(torch.cat((i_feats, t_feats),dim=1), label)+objectives.SupConLoss(torch.cat((i_feats, si_feats),dim=1), label)+objectives.SupConLoss(torch.cat((i_feats, f_feats),dim=1), label))*self.cfg.cmm_loss_weight})
        
        # if 'mcm' in self.current_task:
        #     masked_caption_ids = batch['masked_caption_ids']
        #     # with torch.no_grad():
        #     masked_caption_feats = self.base_model.encode_text(masked_caption_ids)

        #     x = self.cross_former(masked_caption_feats, image_feats, image_feats)

        #     x = self.mcm_head(x)  # [batch_size, text_len, num_colors]

        #     scores = x.float().reshape(-1, self.cfg.num_colors)
        #     mcm_labels = batch['mcm_labels'].reshape(-1)
        #     ret.update({'mcm_loss': objectives.compute_mcm_or_mlm(scores, mcm_labels)*self.cfg.mcm_loss_weight})

        #     pred = scores.max(1)[1]
        #     mcm_label_idx = torch.nonzero(mcm_labels)
        #     acc = (pred[mcm_label_idx] == mcm_labels[mcm_label_idx]).float().mean()
        #     ret.update({'acc': acc})
        
        # if 'mlm' in self.current_task:
        #     mlm_ids = batch['mlm_ids']

        #     mlm_feats = self.base_model.encode_text(mlm_ids)

        #     x = self.cross_former(mlm_feats, image_feats, image_feats)

        #     x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

        #     scores = x.float().reshape(-1, self.cfg.vocab_size)
        #     mlm_labels = batch['mlm_labels'].reshape(-1)
        #     ret.update({'mlm_loss': objectives.compute_mcm_or_mlm(scores, mlm_labels)*self.cfg.mlm_loss_weight})

        #     pred = scores.max(1)[1]
        #     mlm_label_idx = torch.nonzero(mlm_labels)
        #     acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
        #     ret.update({'acc': acc})
        
        # if 'mcq' in self.current_task or 'msm' in self.current_task:
        #     question_ids = batch['question_ids']
        #     answer_ids = batch['answer_ids']

        #     question_feats = self.base_model.encode_text(question_ids)
        #     answer_feats = self.encode_text(answer_ids)

        #     x = self.cross_former(question_feats, image_feats, image_feats)

        #     # x = x @ self.mcq_proj

        #     pred_answer_feats = x[torch.arange(x.shape[0]), question_ids.argmax(dim=-1)].float()
        #     ret.update({'mcq_loss': objectives.compute_mcq(pred_answer_feats, answer_feats)*self.cfg.mcq_loss_weight})

        # return ret


def build_model(cfg, num_classes=11003):
    model = CLIP2ReID(cfg, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model