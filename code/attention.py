import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math


class GATAttention(nn.Module):

    def __init__(self, embedding_dims, num_heads=4, dropout=0.1, alpha=0.2):
        super(GATAttention, self).__init__()
        self.embedding_dims = embedding_dims
        self.num_heads = num_heads
        self.head_dim = embedding_dims // num_heads
        self.alpha = alpha
        self.dropout = dropout

        assert embedding_dims % num_heads == 0, "embedding_dims must be divisible by num_heads"

        self.W_q = nn.Linear(embedding_dims, embedding_dims, bias=False)
        self.W_k = nn.Linear(embedding_dims, embedding_dims, bias=False)
        self.W_v = nn.Linear(embedding_dims, embedding_dims, bias=False)

        self.a = nn.Parameter(torch.empty(size=(2 * self.head_dim, 1)))

        self.W_o = nn.Linear(embedding_dims, embedding_dims)

        self.dropout_layer = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(alpha)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W_q.weight, gain=gain)
        nn.init.xavier_uniform_(self.W_k.weight, gain=gain)
        nn.init.xavier_uniform_(self.W_v.weight, gain=gain)
        nn.init.xavier_uniform_(self.W_o.weight, gain=gain)
        nn.init.xavier_uniform_(self.a, gain=gain)

    def forward(self, center_node, neighbor_nodes):
        N = neighbor_nodes.size(0)

        if N == 1:
            return torch.ones(1, 1).to(neighbor_nodes.device)

        center_q = self.W_q(center_node)
        neighbor_k = self.W_k(neighbor_nodes)

        center_q = center_q.view(1, self.num_heads, self.head_dim)
        neighbor_k = neighbor_k.view(N, self.num_heads, self.head_dim)

        center_q_expanded = center_q.expand(N, -1, -1)
        attention_input = torch.cat([center_q_expanded, neighbor_k], dim=2)
        e = (attention_input * self.a[:, 0].view(1, 1, -1)).sum(dim=2)
        attention_weights = self.leaky_relu(e).mean(dim=1, keepdim=True)

        attention_weights = F.softmax(attention_weights, dim=0)

        attention_weights = self.dropout_layer(attention_weights)

        return attention_weights


class EnhancedGATAttention(nn.Module):

    def __init__(self, embedding_dims, num_heads=4, dropout=0.1, alpha=0.2,
                 use_residual=True, use_layer_norm=True):
        super(EnhancedGATAttention, self).__init__()
        self.embedding_dims = embedding_dims
        self.num_heads = num_heads
        self.head_dim = embedding_dims // num_heads
        self.alpha = alpha
        self.dropout = dropout
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm

        assert embedding_dims % num_heads == 0, "embedding_dims must be divisible by num_heads"

        self.W_q = nn.Linear(embedding_dims, embedding_dims, bias=False)
        self.W_k = nn.Linear(embedding_dims, embedding_dims, bias=False)
        self.W_v = nn.Linear(embedding_dims, embedding_dims, bias=False)

        self.attention_weight = nn.Parameter(torch.empty(num_heads, 2 * self.head_dim))

        self.W_o = nn.Linear(embedding_dims, embedding_dims)

        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(embedding_dims)

        self.temporal_weight = nn.Parameter(torch.tensor(1.0))

        self.dropout_layer = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(alpha)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W_q.weight, gain=gain)
        nn.init.xavier_uniform_(self.W_k.weight, gain=gain)
        nn.init.xavier_uniform_(self.W_v.weight, gain=gain)
        nn.init.xavier_uniform_(self.W_o.weight, gain=gain)

        nn.init.xavier_uniform_(self.attention_weight, gain=gain)

    def forward(self, center_node, neighbor_nodes, temporal_weights=None):
        N = neighbor_nodes.size(0)

        if N == 1:
            return torch.ones(1, 1).to(neighbor_nodes.device)

        center_q = self.W_q(center_node).view(1, self.num_heads, self.head_dim)
        neighbor_k = self.W_k(neighbor_nodes).view(N, self.num_heads, self.head_dim)

        center_q_expanded = center_q.expand(N, -1, -1)
        attention_input = torch.cat([center_q_expanded, neighbor_k], dim=2)
        multi_head_attention = (attention_input * self.attention_weight.view(1, self.num_heads, -1)).sum(dim=2)
        multi_head_attention = self.leaky_relu(multi_head_attention).mean(dim=1, keepdim=True)

        if temporal_weights is not None:
            multi_head_attention = multi_head_attention * (1 + self.temporal_weight * temporal_weights)

        attention_weights = F.softmax(multi_head_attention, dim=0)

        attention_weights = self.dropout_layer(attention_weights)

        return attention_weights


class Attention(EnhancedGATAttention):

    def __init__(self, embedding_dims, num_heads=4, dropout=0.1):
        super(Attention, self).__init__(
            embedding_dims=embedding_dims,
            num_heads=num_heads,
            dropout=dropout,
            alpha=0.2,
            use_residual=True,
            use_layer_norm=True
        )

    def forward(self, node1, node2):
        if node1.dim() == 2 and node1.size(0) > 1:
            center_node = node1[0:1, :]
        else:
            center_node = node1

        return super().forward(center_node, node2)
