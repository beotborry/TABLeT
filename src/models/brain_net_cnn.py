import torch
import torch.nn as nn
import torch.nn.functional as F

class E2EBlock(nn.Module):
    def __init__(
        self,
        in_planes,
        planes,
        roi_num,
        bias=True,
    ):
        super().__init__()
        self.d = roi_num
        self.cnn1 = nn.Conv2d(in_planes, planes, kernel_size=(1, self.d), bias=bias)
        self.cnn2 = nn.Conv2d(in_planes, planes, kernel_size=(self.d, 1), bias=bias)
    
    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a] * self.d, 3) + torch.cat([b] * self.d, 2)
    
class BrainNetCNN(nn.Module):
    def __init__(self, num_roi, num_classes):
        super().__init__()
        self.in_planes = 1
        self.d = num_roi

        self.e2econv1 = E2EBlock(1, 32, self.d, bias=True)
        self.e2econv2 = E2EBlock(32, 64, self.d, bias=True)
        self.E2N = nn.Conv2d(64, 1, (1, self.d))
        self.N2G = nn.Conv2d(1, 256, (self.d, 1))
        self.dense1 = nn.Linear(256, 128)
        self.dense2 = nn.Linear(128, 30)
        self.dense3 = nn.Linear(30, num_classes)

    def forward(self, node_feature, **kwargs):
        node_feature = node_feature.unsqueeze(1).float()
        out = F.leaky_relu(self.e2econv1(node_feature), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.leaky_relu(self.N2G(out), negative_slope=0.33)
        out = F.dropout(out, p=0.5)
        out = out.view(out.size(0), -1)
        out = F.leaky_relu(self.dense1(out), negative_slope=0.33)
        out = F.dropout(out, p=0.5)
        out = F.leaky_relu(self.dense2(out), negative_slope=0.33)
        out = F.dropout(out, p=0.5)
        out = F.leaky_relu(self.dense3(out), negative_slope=0.33)
        return out