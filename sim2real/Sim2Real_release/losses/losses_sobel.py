import torch
import torch.nn as nn
from tools.registery import LOSS_REGISTRY
from torch.nn import functional as F

@LOSS_REGISTRY.register()
class SobelLoss():
    def __init__(self, loss_dict):
        self.loss_dict = loss_dict
        self.as_loss = loss_dict.as_loss
        self.weight = loss_dict.weight
        self.mse_loss = nn.MSELoss()
        
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.sobel_x = self.sobel_x.cuda()
            self.sobel_y = self.sobel_y.cuda()

    def forward(self, x, y):
        if not self.as_loss:
            pass
        else:
            mse_loss = self.mse_loss(x, y)
            edge_x_pred = F.conv2d(x, self.sobel_x, padding=1, groups=3)
            edge_y_pred = F.conv2d(x, self.sobel_y, padding=1, groups=3)
            edge_x_true = F.conv2d(y, self.sobel_x, padding=1, groups=3)
            edge_y_true = F.conv2d(y, self.sobel_y, padding=1, groups=3)
            sobel_loss = self.mse_loss(edge_x_pred, edge_x_true) + self.mse_loss(edge_y_pred, edge_y_true)
            return sobel_loss