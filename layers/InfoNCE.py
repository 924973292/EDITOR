import torch
import torch.nn.functional as F


def ContrastiveLoss(a, b, logit_scale):
    batch_size = a.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(a.device)

    # normalized features
    a_norm = a / a.norm(dim=-1, keepdim=True)
    b_norm = b / b.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_a = logit_scale * a_norm @ b_norm.t()
    logits_per_b = logits_per_a.t()

    loss_i = F.cross_entropy(logits_per_a, labels)
    loss_t = F.cross_entropy(logits_per_b, labels)
    loss = (loss_i + loss_t) / 2

    return loss
