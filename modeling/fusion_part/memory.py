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

    def compute_center_wo(self, features, labels, mode=0):
        # 求取labels中独特的ID
        centers = []
        for label in labels.unique():
            # 求取每个ID的特征
            centers.append(features[labels == label].mean(dim=0))
        centers = torch.stack(centers, dim=0)
        return centers

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
        if TIR_feat is not None:
            TIR_feat = F.normalize(TIR_feat, dim=1)
            # 更新中心
            self.update1(RGB_feat, NIR_feat, TIR_feat, label_=label_)
            # 更新计算独特的ID
            label = label_.unique()
            # 求取三个模态内部的对齐损失
            intra_loss = (self.compute_inter_loss(self.RGB_centers[label], RGB_feat, label_) +
                          self.compute_inter_loss(self.NIR_centers[label], NIR_feat, label_) +
                          self.compute_inter_loss(self.TIR_centers[label], TIR_feat, label_))
            total_intra_loss = self.alpha * intra_loss
            return total_intra_loss
        else:
            # 更新中心
            self.update1(RGB_feat, NIR_feat, TIR_feat=None, label_=label_)
            # 更新计算独特的ID
            label = label_.unique()
            # 求取两个模态内部的对齐损失
            intra_loss = (self.compute_inter_loss(self.RGB_centers[label], RGB_feat, label_) +
                          self.compute_inter_loss(self.NIR_centers[label], NIR_feat, label_))
            total_intra_loss = self.alpha * intra_loss
            return total_intra_loss

    def update1(self, RGB_feat, NIR_feat, TIR_feat, label_):
        # 按照ID求取得ID的中心特征，然后进行更新
        RGB_centers = self.compute_center(RGB_feat, label_, mode=0)
        NIR_centers = self.compute_center(NIR_feat, label_, mode=0)
        if TIR_feat is not None:
            TIR_centers = self.compute_center(TIR_feat, label_, mode=0)
        # 更新计算独特的ID
        label = label_.unique()
        # 提取当前中心
        self.RGB_centers[label] = self.momentum * RGB_centers + (1 - self.momentum) * self.RGB_centers[label]
        self.NIR_centers[label] = self.momentum * NIR_centers + (1 - self.momentum) * self.NIR_centers[label]
        if TIR_feat is not None:
            self.TIR_centers[label] = self.momentum * TIR_centers + (1 - self.momentum) * self.TIR_centers[label]
        return None
