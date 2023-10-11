import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_tcmpm(image_fetures, text_fetures, pid,  logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Cross Modal Projection Matching 
    t2i_proj = ||t|| * cos(theta)
    i2j_proj = ||v|| * cos(theta)
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    image_proj_text = logit_scale * torch.matmul(image_fetures, text_norm.t())
    text_proj_image = logit_scale * torch.matmul(text_fetures, image_norm.t())


    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1) # original paper use sum, and use norm will lead minus loss
    # labels_distribute = F.softmax((labels * logit_scale), dim=1)
    # labels_distribute = F.softmax(labels, dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    # i2t2t2i_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - F.log_softmax(text_proj_image, dim=1))

    # loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1)) + torch.mean(torch.sum(i2t2t2i_loss, dim=1))
    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss


def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    # text_proj_image = logit_scale * (text_norm_value * t2i_cosine_theta)
    # image_proj_text = logit_scale * (image_norm_value * i2t_cosine_theta)

    # mean_norm_value = (text_norm_value + image_norm_value) / 2
    # text_proj_image = logit_scale * (mean_norm_value * t2i_cosine_theta)
    # image_proj_text = logit_scale * (mean_norm_value * i2t_cosine_theta)

    # k_value = 8
    # text_proj_image = logit_scale * (k_value * t2i_cosine_theta)
    # image_proj_text = logit_scale * (k_value * i2t_cosine_theta)

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1) # original paper use sum, and use norm will lead minus loss
    # labels_distribute = F.softmax((labels * logit_scale), dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss

def compute_mcm_or_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


def compute_itc(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    
    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss

class CMFL(nn.Module):
    """
    Cross Modal Focal Loss
    """

    def __init__(self, alpha=1, gamma=2, binary=False, multiplier=2, sg=False):
        super(CMFL, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.binary = binary
        self.multiplier = multiplier
        self.sg = sg

    def forward(self, inputs_a, inputs_b, targets):

        # bce_loss_a = F.binary_cross_entropy(inputs_a, targets, reduce=False)
        # bce_loss_b = F.binary_cross_entropy(inputs_b, targets, reduce=False)

        bce_loss_a = F.cross_entropy(inputs_a, targets, reduce=False)
        bce_loss_b = F.cross_entropy(inputs_b, targets, reduce=False)

        pt_a = torch.exp(-bce_loss_a)
        pt_b = torch.exp(-bce_loss_b)

        eps = 0.000000001

        if self.sg:
            d_pt_a = pt_a.detach()
            d_pt_b = pt_b.detach()
            wt_a = ((d_pt_b + eps) * (self.multiplier * pt_a * d_pt_b)) / (pt_a + d_pt_b + eps)
            wt_b = ((d_pt_a + eps) * (self.multiplier * d_pt_a * pt_b)) / (d_pt_a + pt_b + eps)
        else:
            wt_a = ((pt_b + eps) * (self.multiplier * pt_a * pt_b)) / (pt_a + pt_b + eps)
            wt_b = ((pt_a + eps) * (self.multiplier * pt_a * pt_b)) / (pt_a + pt_b + eps)

        if self.binary:
            wt_a = wt_a * (1 - targets)
            wt_b = wt_b * (1 - targets)

        f_loss_a = self.alpha * (1 - wt_a) ** self.gamma * bce_loss_a
        f_loss_b = self.alpha * (1 - wt_b) ** self.gamma * bce_loss_b

        loss = 0.5 * torch.mean(f_loss_a) + 0.5 * torch.mean(f_loss_b)

        return loss

def focal_loss_two(inputs_a, inputs_b, alpha, gamma):

    pt_a = torch.exp(-inputs_a)
    pt_b = torch.exp(-inputs_b)

    eps = 0.000000001


    wt_a = ((pt_b + eps) * (2 * pt_a * pt_b)) / (pt_a + pt_b + eps)
    wt_b = ((pt_a + eps) * (2 * pt_a * pt_b)) / (pt_a + pt_b + eps)


    f_loss_a = alpha * (1 + wt_a) ** gamma * inputs_a
    f_loss_b = alpha * (1 + wt_b) ** gamma * inputs_b

    loss = torch.mean(f_loss_a) + torch.mean(f_loss_b)

    return loss


def compute_itc_focal3(image_features, text_features, simage_features, fusion_features, logit_scale, alpha, gamma, klp):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    
    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    simage_norm = simage_features / simage_features.norm(dim=-1, keepdim=True)
    fusion_norm = fusion_features / fusion_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image0 = logit_scale * image_norm @ text_norm.t()
    logits_per_text0 = logits_per_image0.t()

    loss_i = F.cross_entropy(logits_per_image0, labels, reduce=False)
    loss_t =F.cross_entropy(logits_per_text0, labels, reduce=False)
    loss_it = (loss_i +  loss_t)/2

    # cosine similarity as logits
    logits_per_image1 = logit_scale * image_norm @ simage_norm.t()
    logits_per_text1 = logits_per_image1.t()

    loss_i = F.cross_entropy(logits_per_image1, labels, reduce=False)
    loss_t =F.cross_entropy(logits_per_text1, labels, reduce=False)
    loss_is = (loss_i +  loss_t)/2

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ fusion_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss_if = (loss_i +  loss_t)/2

    # focal loss
    # kl = F.kl_div(logits_per_text1.softmax(dim=-1).log(), logits_per_text0.detach().softmax(dim=-1), reduction='sum') + F.kl_div(logits_per_text0.softmax(dim=-1).log(), logits_per_text1.detach().softmax(dim=-1), reduction='sum')

    loss = focal_loss_two(loss_it, loss_is, alpha, gamma) + loss_if + klp*(CoRefineLoss(logits_per_text1, logits_per_text0.detach()))
 
    return loss



def CoRefineLoss(output1, output2):

    # Target is ignored at training time. Loss is defined as KL divergence
    # between the model output and the refined labels.
    if output2.requires_grad:
        raise ValueError("Refined labels should not require gradients.")

    output1_log_prob = F.log_softmax(output1, dim=1)
    output2_prob = F.softmax(output2, dim=1)

    _, pred_label = output2_prob.max(1)

    # Loss is normal cross entropy loss
    # base_loss = F.cross_entropy(output1, pred_label)

    # Loss is -dot(model_output_log_prob, refined_labels). Prepare tensors
    # for batch matrix multiplicatio

    model_output1_log_prob = output1_log_prob.unsqueeze(2)
    model_output2_prob = output2_prob.unsqueeze(1)

    # Compute the loss, and average/sum for the batch.
    kl_loss = -torch.bmm(model_output2_prob, model_output1_log_prob)

    return kl_loss.mean()
        
def compute_id(classifier, image_embeddings, text_embeddings, labels, verbose=False):
    image_logits = classifier(image_embeddings)
    text_logits = classifier(text_embeddings)

    criterion = nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(image_logits, labels) + criterion(text_logits, labels)

    # classification accuracy for observation
    if verbose:
        image_pred = torch.argmax(image_logits, dim=1)
        text_pred = torch.argmax(text_logits, dim=1)

        image_precision = torch.mean((image_pred == labels).float())
        text_precision = torch.mean((text_pred == labels).float())

        return loss, image_precision, text_precision
    
    return loss


def compute_cmpm(image_embeddings, text_embeddings, labels, epsilon=1e-8):
    """
    Cross-Modal Projection Matching Loss(CMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: cmpm loss for image projected to text
        t2i_loss: cmpm loss for text projected to image
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """

    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_proj_text = torch.matmul(image_embeddings, text_norm.t())
    text_proj_image = torch.matmul(text_embeddings, image_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return cmpm_loss


def compute_mcq(a, b, temperature=0.05, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    x = torch.mm(a_norm, b_norm.transpose(0, 1))

    i_logsm = F.log_softmax(x/temperature, dim=1)
    j_logsm = F.log_softmax(x.t()/temperature, dim=1)

    # sum over positives
    idiag = torch.diag(i_logsm)
    loss_i = idiag.sum() / len(idiag)

    jdiag = torch.diag(j_logsm)
    loss_j = jdiag.sum() / len(jdiag)

    return - loss_i - loss_j


def CrossModalSupConLoss(image_fetures, text_fetures, labels, temperature):
    """
    Args:
        features: hidden vector of shape [bsz, n_views, ...].
        labels: ground truth of shape [bsz].
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """
    device = (torch.device('cuda') if image_fetures.is_cuda else torch.device('cpu'))


    batch_size = image_fetures.shape[0]

    labels = labels.contiguous().view(-1, 1)
    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')
    mask = torch.eq(labels, labels.T).float().to(device)


    contrast_count = 2
    contrast_feature = torch.cat([image_fetures, text_fetures], dim=0)

    anchor_feature = contrast_feature
    anchor_count = contrast_count

    # compute logits
    anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T) * temperature
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    # logits_mask = torch.scatter(
    #     torch.ones_like(mask),
    #     1,
    #     torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
    #     0
    # )
    # mask = mask * logits_mask

    # compute log_prob
    # exp_logits = torch.exp(logits) * logits_mask
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = -mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()

    return loss 


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(
            1, targets.unsqueeze(1).data.cpu(), 1
        )
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss



# class SupConLoss(nn.Module):
#     """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
#     It also supports the unsupervised contrastive loss in SimCLR"""
#     def __init__(self, temperature=0.07, contrast_mode='all',
#                  base_temperature=0.07):
#         super(SupConLoss, self).__init__()
#         self.temperature = temperature
#         self.contrast_mode = contrast_mode
#         self.base_temperature = base_temperature

def SupConLoss(features, labels=None, mask=None, temperature=2.0, contrast_mode='all',
                 base_temperature=0.07):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (temperature / base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


