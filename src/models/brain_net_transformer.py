import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

class ClusterAssignment(nn.Module):
    def __init__(
        self,
        num_clusters,
        embedding_dim,
        alpha=1.0,
        cluster_centers=None,
        orthogonal=True,
        freeze_center=True,
        project_assignment=True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.project_assignment = project_assignment

        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(self.num_clusters, self.embedding_dim, dtype=torch.float32)
            nn.init.xavier_uniform_(initial_cluster_centers)
        
        else:
            initial_cluster_centers = cluster_centers

        if orthogonal:
            orthogonal_cluster_centers = torch.zeros(self.num_clusters, self.embedding_dim, dtype=torch.float32)
            orthogonal_cluster_centers[0] = initial_cluster_centers[0]
            for i in range(1, num_clusters):
                project = 0
                for j in range(i):
                    project += self.project(initial_cluster_centers[j], initial_cluster_centers[i])
                initial_cluster_centers[i] -= project
                orthogonal_cluster_centers[i] = initial_cluster_centers[i] / torch.norm(initial_cluster_centers[i], p=2)
            initial_cluster_centers = orthogonal_cluster_centers
        self.cluster_centers = nn.Parameter(initial_cluster_centers, requires_grad=not freeze_center)
    
    @staticmethod
    def project(u, v):
        return (u.dot(v) / u.dot(u)) * u

    def forward(self, batch):
        if self.project_assignment:
            assignment = batch @ self.cluster_centers.T
            assignment = assignment.pow(2)

            norm = torch.norm(self.cluster_centers, p=2, dim=-1)
            soft_assign = assignment / norm
            return F.softmax(soft_assign, dim=-1)

        else:
            norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
            numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
            power = float(self.alpha + 1) / 2
            numerator = numerator ** power
            return numerator / torch.sum(numerator, dim=1, keepdim=True)
    
    def get_cluster_centers(self):
        return self.cluster_centers


class DEC(nn.Module):
    def __init__(
        self,
        num_clusters,
        hidden_dim,
        encoder,
        alpha=1.0,
        orthogonal=True,
        freeze_center=True,
        project_assignment=True,
    ):
        super().__init__()

        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.assignment = ClusterAssignment(
            num_clusters=num_clusters,
            embedding_dim=hidden_dim,
            alpha=alpha,
            orthogonal=orthogonal,
            freeze_center=freeze_center,
            project_assignment=project_assignment,
        )
        self.loss_fn = nn.KLDivLoss(size_average=False)
    
    def forward(self, batch):
        batch_size = batch.size(0)
        num_nodes = batch.size(1)

        flattened_batch = batch.view(batch_size, -1)
        encoded = self.encoder(flattened_batch)
        encoded = encoded.view(batch_size * num_nodes, -1)
        assignment = self.assignment(encoded)
        assignment = assignment.view(batch_size, num_nodes, -1)
        encoded = encoded.view(batch_size, num_nodes, -1)
        node_repr = torch.bmm(assignment.transpose(1, 2), encoded)
        return node_repr, assignment

    def target_distribution(self, batch):
        weight = batch ** 2 / batch.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def loss(self, assignment):
        flattened_assignment = assignment.view(-1, assignment.size(-1))
        target = self.target_distribution(flattened_assignment).detach()
        return self.loss_fn(flattened_assignment.log(), target) / flattened_assignment.size(0)

    def get_cluster_centers(self):
        return self.assignment.get_cluster_centers()


class InterpretableTransformerEncoder(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=False,
        device=None,
        dtype=None
    ):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, device, dtype)

        self.attention_weights = None

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal):
        x, weights = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=True)
        self.attention_weights = weights
        return self.dropout1(x)

    def get_attention_weights(self):
        return self.attention_weights


class TransPoolingEncoder(nn.Module):
    def __init__(self, input_feature_size, num_heads, input_node_num, hidden_size, output_node_num, pooling=True, orthogonal=True, freeze_center=False, project_assignment=True):
        super().__init__()

        self.transformer = InterpretableTransformerEncoder(d_model=input_feature_size, nhead=num_heads, dim_feedforward=hidden_size, batch_first=True)
        self.pooling = pooling
        if pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size * input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, input_feature_size * input_node_num),
            )
            self.dec = DEC(num_clusters=output_node_num, hidden_dim=input_feature_size, encoder=self.encoder,
                           orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)

    def is_pooling_enabled(self):
        return self.pooling

    def forward(self, x):
        x = self.transformer(x)
        if self.pooling:
            x, assignment = self.dec(x)
            return x, assignment
        return x, None

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)


class BrainNetworkTransformer(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.attention_list = nn.ModuleList()
        forward_dim = config.dataset.node_sz
        self.pos_encoding = config.model.pos_encoding
        if self.pos_encoding == 'identity':
            self.node_identity = nn.Parameter(torch.zeros(config.dataset.node_sz, config.model.pos_embed_dim), requires_grad=True)
            forward_dim = config.dataset.node_sz + config.model.pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)
        num_heads = config.model.num_heads

        sizes = config.model.sizes
        sizes[0] = config.dataset.node_sz
        in_sizes = [config.dataset.node_sz] + sizes[:-1]
        num_classes = config.dataset.num_classes
        do_pooling = config.model.pooling
        self.do_pooling = do_pooling
        for index, size in enumerate(sizes):
            self.attention_list.append(
                TransPoolingEncoder(input_feature_size=forward_dim,
                                    num_heads=num_heads,
                                    input_node_num=in_sizes[index],
                                    hidden_size=1024,
                                    output_node_num=size,
                                    pooling=do_pooling[index],
                                    orthogonal=config.model.orthogonal,
                                    freeze_center=config.model.freeze_center,
                                    project_assignment=config.model.project_assignment))

        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 8),
            nn.LeakyReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(8 * sizes[-1], 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, node_feature, **kwargs):
        node_feature = node_feature.float()

        bz, _, _, = node_feature.shape

        if self.pos_encoding == 'identity':
            pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        assignments = []

        for atten in self.attention_list:
            node_feature, assignment = atten(node_feature)
            assignments.append(assignment)

        node_feature = self.dim_reduction(node_feature)

        node_feature = node_feature.reshape((bz, -1))

        out = self.fc(node_feature)
        out = out.squeeze(1)  # (B,)

        return out

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def get_cluster_centers(self) -> torch.Tensor:
        return self.dec.get_cluster_centers()

    def loss(self, assignments):
        decs = list(filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None

        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all