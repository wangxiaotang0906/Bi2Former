import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import HeteroConv, GCNConv, global_mean_pool, TransformerConv, MessagePassing
import matplotlib.pyplot as plt
import seaborn as sns
import os

def binary(x: torch.Tensor, k: float) -> torch.Tensor:
    return 0 if x < k else 1

class NodeEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_prob=0.5):
        super(NodeEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.bn2(self.fc2(x))
        return x


class MultiHeadCrossAttentionConv(MessagePassing):
    def __init__(self, rna_in_channels, atac_in_channels, hidden_channels, num_heads=1, dropout=0.5):
        super(MultiHeadCrossAttentionConv, self).__init__(aggr='mean') 
        assert hidden_channels % num_heads == 0, "hidden_channels must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        self.hidden_channels = hidden_channels
        print(f'num_heads:{self.num_heads},head_dims:{self.head_dim}')

        # LayerNorm
        # self.input_rna_norm = nn.LayerNorm(rna_in_channels)
        # self.input_atac_norm = nn.LayerNorm(atac_in_channels)
        # # self.layernorm_rna = nn.LayerNorm(hidden_channels)
        # # self.layernorm_atac = nn.LayerNorm(hidden_channels)

        # QKV
        self.rna_query = nn.Linear(rna_in_channels, hidden_channels)
        # self.atac_query = nn.Linear(atac_in_channels, hidden_channels)
        # self.rna_key = nn.Linear(rna_in_channels, hidden_channels)
        self.atac_key = nn.Linear(atac_in_channels, hidden_channels)
        self.rna_value = nn.Linear(rna_in_channels, hidden_channels)
        self.atac_value = nn.Linear(atac_in_channels, hidden_channels)

        if self.num_heads>1:
            self.rna_out_proj = nn.Linear(hidden_channels, hidden_channels)
            self.atac_out_proj = nn.Linear(hidden_channels, hidden_channels)

        self.rna_self_attention = nn.Linear(rna_in_channels, hidden_channels)
        self.atac_self_attention = nn.Linear(atac_in_channels, hidden_channels)

        self.dim_reduction_rna = nn.Linear(2 * hidden_channels, hidden_channels)
        self.dim_reduction_atac = nn.Linear(2 * hidden_channels, hidden_channels)

        # self.ffn_rna = nn.Sequential(
        #     nn.Linear(hidden_channels, 2 * hidden_channels),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(2 * hidden_channels, hidden_channels),
        #     nn.Dropout(dropout)
        # )
        # self.ffn_atac = nn.Sequential(
        #     nn.Linear(hidden_channels, 2 * hidden_channels),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(2 * hidden_channels, hidden_channels),
        #     nn.Dropout(dropout)
        # )

        # Dropout
        self.dropout = nn.Dropout(dropout)


    def forward(self, x_rna, x_atac, chrom_mask):
        """
        x_rna: [num_rna_nodes, rna_in_channels]
        x_atac: [num_atac_nodes, atac_in_channels]
        """
        # RNA->ATAC attention
        rna_query = self.rna_query(x_rna).view(-1, self.num_heads, self.head_dim)  # [num_rna_nodes, num_heads, head_dim]
        atac_key = self.atac_key(x_atac).view(-1, self.num_heads, self.head_dim)  # [num_atac_nodes, num_heads, head_dim]
        atac_value = self.atac_value(x_atac).view(-1, self.num_heads, self.head_dim)  # [num_atac_nodes, num_heads, head_dim]
        rna_atac_attention = torch.bmm(
            rna_query.permute(1, 0, 2).reshape(self.num_heads, -1, self.head_dim),
            atac_key.permute(1, 2, 0).reshape(self.num_heads, self.head_dim, -1)
        )  # [num_heads, num_rna_nodes, num_atac_nodes]
        chrom_mask = chrom_mask.squeeze(-1)
        rna_atac_attention = rna_atac_attention * chrom_mask
        # rna_atac_attention = self.dropout(rna_atac_attention)

        # softmax
        # rna_atac_attention = F.softmax(rna_atac_attention, dim=-1)

        # binary
        # rna_atac_attention = torch.sigmoid(rna_atac_attention)
        # rna_atac_attention = (rna_atac_attention > 0.8).float()

        # Biological Pruning
        rna_atac_attention = torch.sigmoid(rna_atac_attention)
        threshold_mask = (rna_atac_attention > 0.5).float()  # [num_heads, num_rna_nodes, 1]
        topk_values, topk_indices = torch.topk(rna_atac_attention, k=10, dim=-1)  # [num_heads, num_rna_nodes, top_k]
        topk_values = F.softmax(topk_values, dim=-1)
        rna_atac_attention = torch.zeros_like(rna_atac_attention)  # [num_heads, num_rna_nodes, num_atac_nodes]
        rna_atac_attention.scatter_(-1, topk_indices, topk_values)  
        rna_atac_attention = rna_atac_attention * threshold_mask

        # crossmodal information
        rna_to_atac_agg = torch.bmm(rna_atac_attention, atac_value.permute(1, 0, 2).reshape(self.num_heads, -1, self.head_dim))  # [num_heads, num_rna_nodes, head_dim]
        rna_to_atac_agg = rna_to_atac_agg.permute(1, 0, 2).reshape(-1, self.hidden_channels)  # [num_rna_nodes, hidden_channels]

        atac_rna_attention = rna_atac_attention.permute(0, 2, 1) 
        # crossmodal information
        atac_to_rna_agg = torch.bmm(atac_rna_attention, rna_value.permute(1, 0, 2).reshape(self.num_heads, -1, self.head_dim))  # [num_heads, num_atac_nodes, head_dim]
        atac_to_rna_agg = atac_to_rna_agg.permute(1, 0, 2).reshape(-1, self.hidden_channels)  # [num_atac_nodes, hidden_channels]

        if self.num_heads > 1:
            rna_to_atac_agg = self.rna_out_proj(rna_to_atac_agg)  # [num_rna_nodes, hidden_channels]
            atac_to_rna_agg = self.atac_out_proj(atac_to_rna_agg)  # [num_atac_nodes, hidden_channels]

        # crossmodal message passing
        rna_to_atac_agg = torch.cat([rna_to_atac_agg, self.rna_self_attention(x_rna)], dim=1)
        atac_to_rna_agg = torch.cat([atac_to_rna_agg, self.atac_self_attention(x_atac)], dim=1)
        rna_to_atac_agg = self.dim_reduction_rna(rna_to_atac_agg)
        atac_to_rna_agg = self.dim_reduction_atac(atac_to_rna_agg)

        # Layernorm
        # rna_to_atac_agg = self.layernorm_rna(rna_to_atac_agg)  # [num_rna_nodes, hidden_channels]
        # atac_to_rna_agg = self.layernorm_atac(atac_to_rna_agg)  # [num_atac_nodes, hidden_channels]

        # FFN
        # rna_to_atac_agg = self.layernorm_rna(rna_to_atac_agg + self.ffn_rna(rna_to_atac_agg))  # FFN
        # atac_to_rna_agg = self.layernorm_atac(atac_to_rna_agg + self.ffn_atac(atac_to_rna_agg))  # FFN

        return rna_to_atac_agg, atac_to_rna_agg

class RNA_ATAC_Pairing(nn.Module):
    def __init__(self, in_channels_dict, id_embedding_dims, hidden_channels, out_channels, layer_num=1, multihead=False, num_heads=1, dropout=0.5):
        super(RNA_ATAC_Pairing, self).__init__()
        self.rna_idembedding = nn.Embedding(num_embeddings=in_channels_dict['rna_id'], embedding_dim=id_embedding_dims)
        self.atac_idembedding = nn.Embedding(num_embeddings=in_channels_dict['atac_id'], embedding_dim=id_embedding_dims)

        # self.rna_feature_embedding = nn.Linear(in_channels_dict['rna_feature'], in_channels_dict['rna_feature_hidden'])
        # self.atac_feature_embedding = nn.Linear(in_channels_dict['atac_feature'], in_channels_dict['atac_feature_hidden'])

        if multihead==True:
            self.cross_attention_layers = nn.ModuleList([
                MultiHeadCrossAttentionConv(
                    id_embedding_dims + in_channels_dict['rna_feature'] if i == 0 else hidden_channels,
                    id_embedding_dims + in_channels_dict['atac_feature'] if i == 0 else hidden_channels,
                    hidden_channels, num_heads=num_heads, dropout=dropout
                )
                for i in range(layer_num)
            ])
        else:
            self.cross_attention_layers = nn.ModuleList([
                CrossAttentionConv(
                    id_embedding_dims + in_channels_dict['rna_feature'] if i == 0 else hidden_channels,
                    id_embedding_dims + in_channels_dict['atac_feature'] if i == 0 else hidden_channels,
                    hidden_channels, dropout=dropout
                )
                for i in range(layer_num)
            ])

        # Classifier
        self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels)  
        self.fc2 = nn.Linear(hidden_channels, int(hidden_channels/2))
        self.fc3 = nn.Linear(int(hidden_channels/2), out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data, device):
        x_dict = data.x_dict
        x_dict = {key: x.float() for key, x in x_dict.items()} 

        #print("rna row id:",x_dict['rna'][:,0].long())

        x_rna_id = self.rna_idembedding(x_dict['rna'][:,0].long())
        x_atac_id = self.atac_idembedding(x_dict['atac'][:,0].long())
        #print("rna id:",x_rna_id.shape,x_rna_id)

        x_rna_feature = x_dict['rna'][:,1:]
        x_atac_feature = x_dict['atac'][:,1:]
        
        # x_rna_feature = self.rna_feature_embedding(x_rna_feature)
        # x_atac_feature = self.atac_feature_embedding(x_atac_feature)
        # print("rna feature:",x_rna_feature.shape,x_rna_feature)

        # full
        x_rna = torch.cat((x_rna_id, x_rna_feature), dim=1)
        x_atac = torch.cat((x_atac_id, x_atac_feature), dim=1)

        # ablation wo feature
        # x_rna = x_rna_id
        # x_atac = x_atac_id

        # ablation wo ID
        # x_rna = x_rna_feature
        # x_atac = x_atac_feature

        chrom_mask = data.chrom_mask


        # CrossmodalAttentionConv
        for i, cross_attention_layer in enumerate(self.cross_attention_layers):
            rna_to_atac_agg, atac_to_rna_agg = cross_attention_layer(x_rna, x_atac, chrom_mask)
            if i > 0:
                x_rna = x_rna + rna_to_atac_agg 
                x_atac = x_atac + atac_to_rna_agg
            else:
                x_rna, x_atac = rna_to_atac_agg, atac_to_rna_agg
        #print("rna_to_atac_agg:",rna_to_atac_agg.shape,rna_to_atac_agg)

        # pooling
        x_rna_agg = global_mean_pool(x_rna, data['rna'].batch)
        x_atac_agg = global_mean_pool(x_atac, data['atac'].batch)
        # print("x_rna_agg:",x_rna.shape,x_rna)

        x = torch.cat([x_rna_agg, x_atac_agg], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x
