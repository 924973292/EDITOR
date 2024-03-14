import torch
import torch.nn as nn
import math
import torch.nn.functional as F


# Cut & paste from PyTorch official master until it's in a few official releases - RW
# Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf

class OCFR(nn.Module):
    def __init__(self, dim=768, num_class=171, momentum=0.8, alpha=1.0, beta=1.0, temp=0.05):
        super(OCFR, self).__init__()
        self.dim = dim
        self.RGB_centers = nn.Parameter(torch.zeros(num_class, self.dim), requires_grad=False)
        self.NIR_centers = nn.Parameter(torch.zeros(num_class, self.dim), requires_grad=False)
        self.TIR_centers = nn.Parameter(torch.zeros(num_class, self.dim), requires_grad=False)
        self.momentum = torch.tensor(momentum, dtype=torch.float32)
        self.alpha = alpha
        self.beta = beta
        self.temp = temp

    def compute_center(self, features, labels):
        # 求取labels中独特的ID
        centers = []
        for label in labels.unique():
            # 求取每个ID的特征
            centers.append(features[labels == label].mean(dim=0))
        centers = torch.stack(centers, dim=0)
        return centers.detach()

    def compute_intra_loss(self, centers, features, label_):
        centers_modality = []
        chunk = features.size(0) // centers.size(0)
        unique_label = label_.unique()
        # Create a new tensor that takes data from the original label every 16(number of instances per ID)
        label = label_[::chunk]
        for i in range(label.size(0)):
            index = unique_label == label[i]
            centers_modality.append(centers[index].repeat(features.size(0) // centers.size(0), 1))
        centers_modality = torch.stack(centers_modality, dim=0).reshape(-1, self.dim)
        loss = nn.MSELoss()(centers_modality, features)
        return loss

    def forward(self, RGB_feat, NIR_feat, TIR_feat, label_, epoch):
        # Normalize the features for the input
        RGB_feat = F.normalize(RGB_feat, dim=1)
        NIR_feat = F.normalize(NIR_feat, dim=1)
        if TIR_feat is not None:
            TIR_feat = F.normalize(TIR_feat, dim=1)
            # Update the center
            self.update(RGB_feat, NIR_feat, TIR_feat, label_=label_)
            # Update the unique ID
            label = label_.unique()
            # Compute the alignment loss of the same ID feature in the modality
            intra_loss = (self.compute_intra_loss(self.RGB_centers[label], RGB_feat, label_) +
                          self.compute_intra_loss(self.NIR_centers[label], NIR_feat, label_) +
                          self.compute_intra_loss(self.TIR_centers[label], TIR_feat, label_))
            total_intra_loss = self.alpha * intra_loss
            return total_intra_loss
        else:
            # Update the center
            self.update(RGB_feat, NIR_feat, TIR_feat=None, label_=label_)
            # Update the unique ID
            label = label_.unique()
            # Compute the alignment loss of the same ID feature in the modality
            intra_loss = (self.compute_intra_loss(self.RGB_centers[label], RGB_feat, label_) +
                          self.compute_intra_loss(self.NIR_centers[label], NIR_feat, label_))
            total_intra_loss = self.alpha * intra_loss
            return total_intra_loss

    def update(self, RGB_feat, NIR_feat, TIR_feat, label_):
        # Obtain the center feature of the ID by ID seeking and then update it
        RGB_centers = self.compute_center(RGB_feat, label_)
        NIR_centers = self.compute_center(NIR_feat, label_)
        if TIR_feat is not None:
            TIR_centers = self.compute_center(TIR_feat, label_)
        # Update the current center
        label = label_.unique()
        # Update the center of the current ID
        self.RGB_centers[label] = self.momentum * RGB_centers + (1 - self.momentum) * self.RGB_centers[label]
        self.NIR_centers[label] = self.momentum * NIR_centers + (1 - self.momentum) * self.NIR_centers[label]
        if TIR_feat is not None:
            self.TIR_centers[label] = self.momentum * TIR_centers + (1 - self.momentum) * self.TIR_centers[label]
        return None
