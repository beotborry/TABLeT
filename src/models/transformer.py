import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, input_feature_size, num_tokens, num_classes, num_layers=6, num_heads=8, d_model=512, d_ff=2048, dropout=0.1):
        super(Transformer, self).__init__()
        self.input_feature_size = input_feature_size
        self.num_tokens = num_tokens
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout

        # Define the transformer encoder layer
        encoder_layers = nn.TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Define the final linear layer for classification
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()
        self.position_embedding = nn.Parameter(torch.randn(1, num_tokens, d_model))
        self.feature_projection = nn.Linear(input_feature_size, d_model)
        self.feature_norm = nn.LayerNorm(d_model)
        self.feature_dropout = nn.Dropout(dropout)
        self.feature_activation = nn.ReLU()

        # add global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.register_parameter("position_embedding", self.position_embedding)

    def forward(self, x):
        # x shape: (batch_size, num_frames, C, H , W)
        # change x shape to (batch_size, num_frames*H*W,C)
        B, N, C, H, W = x.shape
        x = x.permute(0, 1, 3, 4, 2)
        x = x.contiguous().view(B, N*H*W, C)

        # project into (B, N*H*W, d_model)
        x = self.feature_projection(x)
        x = x + self.position_embedding

        # x shape: (batch_size, num_frames, d_model)
        x = x.permute(1, 0, 2)
        # x shape: (num_frames, batch_size, d_model)
        x = self.transformer_encoder(x)

        # apply global average pooling
        # change x shape to (batch_size, d_model, num_frames)
        x = x.permute(1, 2, 0)
        x = self.global_avg_pool(x)
        # x shape: (batch_size, d_model, 1)
        x = x.squeeze(-1)
        # x shape: (batch_size, d_model)

        x = self.fc(x)
        x = x.squeeze()
        return x
