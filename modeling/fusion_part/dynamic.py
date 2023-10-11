import torch


# 动态调整边距的单个三元组损失函数
class DynamicMarginTripletLoss(torch.nn.Module):
    def __init__(self, margin=0.1):
        super(DynamicMarginTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = torch.norm(anchor - positive, p=2)
        distance_negative = torch.norm(anchor - negative, p=2)
        margin = self.margin * (1 - torch.exp(-distance_positive))
        loss = torch.relu(distance_positive - distance_negative + margin)

        return loss


def select_triplet(similarity, data):
    r = data[0]
    n = data[1]
    t = data[2]
    if similarity['r2n'] >= similarity['r2t'] and similarity['r2n'] >= similarity['n2t']:
        pos = t
        if similarity['r2t'] >= similarity['n2t']:
            anchor = n
            neg = r
        else:
            anchor = r
            neg = n
    elif similarity['r2t'] >= similarity['r2n'] and similarity['r2t'] >= similarity['n2t']:
        pos = n
        if similarity['r2n'] >= similarity['n2t']:
            anchor = t
            neg = r
        else:
            anchor = r
            neg = t

    elif similarity['n2t'] >= similarity['r2n'] and similarity['n2t'] >= similarity['r2t']:
        pos = r
        if similarity['r2n'] >= similarity['r2t']:
            anchor = t
            neg = n
        else:
            anchor = n
            neg = t

    return anchor, pos, neg


def dynamic_triplet(data):
    triplet = DynamicMarginTripletLoss(margin=0.1)
    dynamic_loss = 0
    batch_size = data.shape[0]
    weight = 1 / batch_size
    similarity_matrix = torch.matmul(data, data.permute(0, 2, 1))
    for i in range(batch_size):
        upper_triangle = similarity_matrix[i].triu(diagonal=1)  # triu函数返回主对角线以上的元素
        similarity = {'r2n': upper_triangle[0, 1], 'r2t': upper_triangle[0, 2], 'n2t': upper_triangle[1, 2]}
        anchor, pos, neg = select_triplet(similarity, data[i])
        dynamic_loss = dynamic_loss + weight * triplet(anchor, pos, neg)
    return dynamic_loss
