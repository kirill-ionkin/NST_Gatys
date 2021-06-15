import torch.nn as nn
import torch.nn.functional as F


from utils import gram_matrix


class ContentLoss(nn.Module):
    def __init__(self, content_target):
        super(ContentLoss, self).__init__()
        self.target = content_target.detach()
        self.mode = "create_model"
        self.loss = None

    def forward(self, x):
        if self.mode == "loss":
            self.loss = F.mse_loss(x, self.target)
        return x


class StyleLoss(nn.Module):
    def __init__(self, style_target):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(style_target.detach())
        self.mode = "create_model"
        self.loss = None

    def forward(self, x):
        if self.mode == "loss":
            self.loss = F.mse_loss(gram_matrix(x), self.target)
        return x


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        # normalize img before put it in vgg19 model layers
        return (img - self.mean) / self.std
