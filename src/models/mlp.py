import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, feature_size, hidden_dim, num_classes, num_layer=3, with_dropout=False, relu_output=False, dropout_rate=0.5):
        super(MLP, self).__init__()
        in_features = feature_size

        self.relu_output = relu_output
        self._make_layer(in_features, hidden_dim, num_classes, num_layer, with_dropout)


    def forward(self, feature):
        feature = feature.view(feature.size(0), -1)
        h = self.features(feature)
        out = self.head(h)
        
        if self.relu_output:
            out = F.relu(out)
            
        out = out.squeeze()
        return out

    def _make_layer(self, in_dim, h_dim, num_classes, num_layer, with_dropout=False, dropout_rate=0.5):

        num_outputs = 1 if num_classes <= 2 else num_classes
        if num_layer == 1:
            self.features = nn.Identity()
            h_dim = in_dim
        else:
            features = []
            for i in range(num_layer - 1):
                features.append(nn.Linear(in_dim, h_dim) if i == 0 else nn.Linear(h_dim, h_dim))
                features.append(nn.ReLU())
                if with_dropout:
                    features.append(nn.Dropout(dropout_rate))
            self.features = nn.Sequential(*features)

        self.head = nn.Linear(h_dim, num_outputs)