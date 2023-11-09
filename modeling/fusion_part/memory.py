import torch
import torch.nn as nn
import math
import torch.nn.functional as F


# Cut & paste from PyTorch official master until it's in a few official releases - RW
# Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf

class ModalityMemory(nn.Module):
    def __init__(self, dim=768, num_class=171, momentum=0.8, alpha=1.0, beta=1.0, temp=0.05):
        super(ModalityMemory, self).__init__()
        self.dim = dim
        self.RGB_centers = nn.Parameter(torch.zeros(num_class, self.dim), requires_grad=False)
        self.NIR_centers = nn.Parameter(torch.zeros(num_class, self.dim), requires_grad=False)
        self.TIR_centers = nn.Parameter(torch.zeros(num_class, self.dim), requires_grad=False)
        self.momentum = torch.tensor(momentum, dtype=torch.float32)
        self.alpha = alpha
        self.beta = beta
        self.temp = temp

    def compute_center(self, features, labels, mode=0):
        # 求取labels中独特的ID
        centers = []
        for label in labels.unique():
            # 求取每个ID的特征
            centers.append(features[labels == label].mean(dim=0))
        centers = torch.stack(centers, dim=0)
        return centers.detach()

    def compute_inter_loss(self, centers, features, label_):
        centers_modality = []
        chunk = features.size(0) // centers.size(0)
        unique_label = label_.unique()
        # 创建一个新的tensor，每隔16取原始label的数据
        label = label_[::chunk]
        for i in range(label.size(0)):
            index = unique_label == label[i]
            centers_modality.append(centers[index].repeat(features.size(0) // centers.size(0), 1))
        centers_modality = torch.stack(centers_modality, dim=0).reshape(-1, self.dim)
        loss = nn.MSELoss()(centers_modality, features)
        return loss

    def compute_nce(self, centers, features, label_):
        outputs = features.mm(centers.t())
        outputs /= self.temp
        loss = F.cross_entropy(outputs, label_)
        return loss

    def compute_between_loss(self, centers1, centers2, centers3):
        # 在模态内进行同ID特征的对齐
        loss = nn.MSELoss()(centers1, centers2) + nn.MSELoss()(centers1, centers3) + nn.MSELoss()(centers2, centers3)
        return loss

    def forward(self, RGB_feat, NIR_feat, TIR_feat, label_, epoch):
        # 为输入的特征进行归一化
        RGB_feat = F.normalize(RGB_feat, dim=1)
        NIR_feat = F.normalize(NIR_feat, dim=1)
        TIR_feat = F.normalize(TIR_feat, dim=1)
        # 更新中心
        self.update1(RGB_feat, NIR_feat, TIR_feat, label_)
        # 更新计算独特的ID
        label = label_.unique()
        # 求取三个模态内部的对齐损失
        intra_loss = (self.compute_inter_loss(self.RGB_centers[label], RGB_feat, label_) +
                      self.compute_inter_loss(self.NIR_centers[label], NIR_feat, label_) +
                      self.compute_inter_loss(self.TIR_centers[label], TIR_feat, label_))
        self.update2(label_)
        # mutual_loss = (self.compute_inter_loss(self.RGB_centers[label], RGB_feat, label_) +
        #               self.compute_inter_loss(self.NIR_centers[label], NIR_feat, label_) +
        #               self.compute_inter_loss(self.TIR_centers[label], TIR_feat, label_))
        total_intra_loss = self.alpha * intra_loss
        return total_intra_loss

    def update1(self, RGB_feat, NIR_feat, TIR_feat, label_):
        # 按照ID求取得ID的中心特征，然后进行更新
        RGB_centers = self.compute_center(RGB_feat, label_, mode=0)
        NIR_centers = self.compute_center(NIR_feat, label_, mode=0)
        TIR_centers = self.compute_center(TIR_feat, label_, mode=0)
        # 更新计算独特的ID
        label = label_.unique()
        # 提取当前中心
        self.RGB_centers[label] = self.momentum * RGB_centers + (1 - self.momentum) * self.RGB_centers[label]
        self.NIR_centers[label] = self.momentum * NIR_centers + (1 - self.momentum) * self.NIR_centers[label]
        self.TIR_centers[label] = self.momentum * TIR_centers + (1 - self.momentum) * self.TIR_centers[label]
        return None
    def update2(self, label_):
        # 更新计算独特的ID
        label = label_.unique()
        # 中心平均
        centers = (self.RGB_centers[label] + self.NIR_centers[label] + self.TIR_centers[label])
        centers /= 3
        self.RGB_centers[label] = centers
        self.NIR_centers[label] = centers
        self.TIR_centers[label] = centers
        return None
# import collections
# import numpy as np
# from abc import ABC
# import torch
# import torch.nn.functional as F
# from torch import nn, autograd
#
#
# class CM(autograd.Function):
#
#     @staticmethod
#     def forward(ctx, inputs, targets, features, momentum):
#         ctx.features = features
#         ctx.momentum = momentum
#         ctx.save_for_backward(inputs, targets)
#         outputs = inputs.mm(ctx.features.t())
#
#         return outputs
#
#     @staticmethod
#     def backward(ctx, grad_outputs):
#         inputs, targets = ctx.saved_tensors
#         grad_inputs = None
#         if ctx.needs_input_grad[0]:
#             grad_inputs = grad_outputs.mm(ctx.features)
#
#         # momentum update
#         for x, y in zip(inputs, targets):
#             ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
#             ctx.features[y] /= ctx.features[y].norm()
#
#         return grad_inputs, None, None, None
#
#
# def cm(inputs, indexes, features, momentum=0.5):
#     return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))
#
#
# class CM_Hard(autograd.Function):
#
#     @staticmethod
#     def forward(ctx, inputs, targets, features, momentum):
#         ctx.features = features.half()
#         inputs = inputs.half()
#         ctx.momentum = momentum
#         ctx.save_for_backward(inputs, targets)
#         outputs = inputs.mm(ctx.features.t())
#
#         return outputs
#
#     @staticmethod
#     def backward(ctx, grad_outputs):
#         inputs, targets = ctx.saved_tensors
#         grad_inputs = None
#         if ctx.needs_input_grad[0]:
#             grad_inputs = grad_outputs.mm(ctx.features)
#
#         batch_centers = collections.defaultdict(list)
#         for instance_feature, index in zip(inputs, targets.tolist()):
#             batch_centers[index].append(instance_feature)
#
#         for index, features in batch_centers.items():
#             distances = []
#             for feature in features:
#                 distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
#                 distances.append(distance.cpu().numpy())
#
#             median = np.argmin(np.array(distances))
#             ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
#             ctx.features[index] /= ctx.features[index].norm()
#             print(ctx.features[index])
#
#         return grad_inputs, None, None, None
#
#
# def cm_hard(inputs, indexes, features, momentum=0.5):
#     return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))
#
#
# class ClusterMemory(nn.Module, ABC):
#     def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
#         super(ClusterMemory, self).__init__()
#         self.num_features = num_features
#         self.num_samples = num_samples
#
#         self.momentum = momentum
#         self.temp = temp
#         self.use_hard = use_hard
#
#         self.register_buffer('features', torch.zeros(num_samples, num_features).half())
#
#     def forward(self, inputs, targets):
#
#         inputs = F.normalize(inputs, dim=1).cuda()
#         if self.use_hard:
#             outputs = cm_hard(inputs, targets, self.features, self.momentum)
#         else:
#             outputs = cm(inputs, targets, self.features, self.momentum)
#
#         outputs /= self.temp
#         loss = F.cross_entropy(outputs, targets)
#         return loss
#
# model = ClusterMemory(768, 171, use_hard=True)
# model = model.cuda()
# inputs = torch.randn(16, 768).cuda()
# targets = torch.randint(0, 171, (16,)).cuda()
# loss = model(inputs, targets)
