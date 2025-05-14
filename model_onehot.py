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


class CrossAttentionConv(MessagePassing):
    def __init__(self, rna_in_channels, atac_in_channels, hidden_channels, dropout=0.5):
        super(CrossAttentionConv, self).__init__(aggr='mean')  # 使用加法聚合

        # 定义用于计算RNA和ATAC节点之间注意力的参数
        self.rna_query = nn.Linear(rna_in_channels, hidden_channels)
        self.atac_query = nn.Linear(atac_in_channels, hidden_channels)
        self.rna_key = nn.Linear(rna_in_channels, hidden_channels)
        self.atac_key = nn.Linear(atac_in_channels, hidden_channels)
        self.rna_value = nn.Linear(rna_in_channels, hidden_channels)
        self.atac_value = nn.Linear(atac_in_channels, hidden_channels)

        # 自注意力机制（Self-Attention）参数
        self.rna_self_attention = nn.Linear(rna_in_channels, hidden_channels)
        self.atac_self_attention = nn.Linear(atac_in_channels, hidden_channels)

        #正则化 & dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_rna, x_atac):
        """
        x_rna: [num_rna_nodes, rna_in_channels]
        x_atac: [num_atac_nodes, atac_in_channels]
        edge_index_rna_to_atac: 从rna节点到atac节点的边索引
        edge_index_atac_to_rna: 从atac节点到rna节点的边索引
        """
        # 为rna节点与atac节点之间的交互计算注意力分数
        rna_query = self.rna_query(x_rna)  # [num_rna_nodes, hidden_channels]
        atac_key = self.atac_key(x_atac)  # [num_atac_nodes, hidden_channels]
        rna_atac_attention = torch.matmul(rna_query, atac_key.transpose(0, 1))  # [num_rna_nodes, num_atac_nodes]
        rna_atac_attention = F.softmax(rna_atac_attention, dim=-1)  # softmax归一化
        rna_atac_attention = self.dropout(rna_atac_attention)  # dropout

        # 为atac节点与rna节点之间的交互计算注意力分数
        atac_query = self.atac_query(x_atac)  # [num_atac_nodes, hidden_channels]
        rna_key = self.rna_key(x_rna)  # [num_rna_nodes, hidden_channels]
        atac_rna_attention = torch.matmul(atac_query, rna_key.transpose(0, 1))  # [num_atac_nodes, num_rna_nodes]
        atac_rna_attention = F.softmax(atac_rna_attention, dim=-1)  # softmax归一化
        atac_rna_attention = self.dropout(atac_rna_attention)  # dropout

        # 使用注意力权重对值进行加权聚合
        rna_value = self.rna_value(x_rna)  # [num_rna_nodes, hidden_channels]
        atac_value = self.atac_value(x_atac)  # [num_atac_nodes, hidden_channels]

        # 使用自注意力计算节点与自身的关系
        rna_self_attention = self.rna_self_attention(x_rna)  # [num_rna_nodes, hidden_channels]
        #rna_self_attention = self.dropout(rna_self_attention)
        atac_self_attention = self.atac_self_attention(x_atac)  # [num_atac_nodes, hidden_channels]
        #atac_self_attention = self.dropout(atac_self_attention)

        rna_to_atac_agg = torch.matmul(rna_atac_attention, atac_value) + rna_self_attention  # [num_rna_nodes, hidden_channels]
        atac_to_rna_agg = torch.matmul(atac_rna_attention, rna_value) + atac_self_attention  # [num_atac_nodes, hidden_channels]

        # rna_to_atac_agg = self.dropout(rna_to_atac_agg) #dropout
        # atac_to_rna_agg = self.dropout(atac_to_rna_agg) #dropout
        # 返回聚合后的节点表示
        return rna_to_atac_agg, atac_to_rna_agg


class MultiHeadCrossAttentionConv(MessagePassing):
    def __init__(self, rna_in_channels, atac_in_channels, hidden_channels, num_heads=1, dropout=0.5):
        super(MultiHeadCrossAttentionConv, self).__init__(aggr='mean')  # 使用加法聚合
        assert hidden_channels % num_heads == 0, "hidden_channels 必须能被 num_heads 整除"

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

        # 输出线性映射
        if self.num_heads>1:
            self.rna_out_proj = nn.Linear(hidden_channels, hidden_channels)
            self.atac_out_proj = nn.Linear(hidden_channels, hidden_channels)

        # 自注意力机制
        self.rna_self_attention = nn.Linear(rna_in_channels, hidden_channels)
        self.atac_self_attention = nn.Linear(atac_in_channels, hidden_channels)

        # 拼接形式聚合
        self.dim_reduction_rna = nn.Linear(2 * hidden_channels, hidden_channels)
        self.dim_reduction_atac = nn.Linear(2 * hidden_channels, hidden_channels)

        # 前馈网络 (FFN)
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

        # 复现性实验专用
        self.sample_counter = 0  # **自动编号的计数器**

    def forward(self, x_rna, x_atac, chrom_mask):
        """
        x_rna: [num_rna_nodes, rna_in_channels]
        x_atac: [num_atac_nodes, atac_in_channels]
        """
        # RNA->ATAC 注意力计算
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

        # softmax正则化
        # rna_atac_attention = F.softmax(rna_atac_attention, dim=-1)

        # binary正则化
        rna_atac_attention = torch.sigmoid(rna_atac_attention)
        rna_atac_attention = (rna_atac_attention > 0.8).float()

        # Top-K + binary正则化
        # rna_atac_attention = torch.sigmoid(rna_atac_attention)
        # threshold_mask = (rna_atac_attention > 0.5).float()  # [num_heads, num_rna_nodes, 1]
        # topk_values, topk_indices = torch.topk(rna_atac_attention, k=10, dim=-1)  # [num_heads, num_rna_nodes, top_k]
        # topk_values = F.softmax(topk_values, dim=-1)
        # rna_atac_attention = torch.zeros_like(rna_atac_attention)  # [num_heads, num_rna_nodes, num_atac_nodes]
        # rna_atac_attention.scatter_(-1, topk_indices, topk_values)  # 仅保留 Top-K 连接
        # rna_atac_attention = rna_atac_attention * threshold_mask

        # 复现性/可解释性实验专用 保存注意力矩阵
        save_dir = "cd4_test_rna_atac_attention"
        os.makedirs(save_dir, exist_ok=True)
        sample_id = self.sample_counter
        self.sample_counter += 1  # **编号自增**
        save_path = os.path.join(save_dir, f"sample_{sample_id}.pt")
        torch.save(rna_atac_attention.cpu(), save_path)
        print(f"Saved attention matrix for sample {sample_id} at {save_path}")

        # 表达量降序观察激活
        # if not self.training:
        #     attention_martix = rna_atac_attention.squeeze(0).detach().cpu().numpy()  # [2226, N]
        #     x_rna_expression = x_rna[:, 128].detach().cpu().numpy()  # [2226,]
        #     sorted_indices = np.argsort(x_rna_expression)[::-1]  # 从高到低排序
        #     sorted_expression = x_rna_expression[sorted_indices]
        #     sorted_attention = attention_martix[sorted_indices]
        #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        #
        #     # 子图 1：绘制注意力权重矩阵热力图
        #     sns.heatmap(
        #         sorted_attention[:, :],
        #         cmap="viridis",
        #         yticklabels=False,
        #         xticklabels=False,
        #         cbar_kws={'label': 'Attention Weight'},
        #         ax=ax1
        #     )
        #     ax1.set_title("Attention Weights")
        #     ax1.set_xlabel("ATAC Nodes")
        #     ax1.set_ylabel("RNA Nodes (Sorted by Expression)")
        #
        #     # 子图 2：绘制 RNA 表达量的条形图
        #     ax2.barh(
        #         range(len(sorted_expression)),
        #         sorted_expression,
        #         color='darkred',
        #         height=1.0
        #     )
        #     ax2.set_title("RNA Expression Level")
        #     ax2.set_xlabel("Expression Value")
        #     ax2.set_yticks([])  # 隐藏y轴刻度
        #
        #     plt.tight_layout()
        #     plt.show()

        # 加权聚合
        rna_to_atac_agg = torch.bmm(rna_atac_attention, atac_value.permute(1, 0, 2).reshape(self.num_heads, -1, self.head_dim))  # [num_heads, num_rna_nodes, head_dim]
        rna_to_atac_agg = rna_to_atac_agg.permute(1, 0, 2).reshape(-1, self.hidden_channels)  # [num_rna_nodes, hidden_channels]

        # # ATAC->RNA 注意力计算
        # atac_query = self.atac_query(x_atac).view(-1, self.num_heads, self.head_dim)  # [num_atac_nodes, num_heads, head_dim]
        # rna_key = self.rna_key(x_rna).view(-1, self.num_heads, self.head_dim)  # [num_rna_nodes, num_heads, head_dim]
        rna_value = self.rna_value(x_rna).view(-1, self.num_heads, self.head_dim)  # [num_rna_nodes, num_heads, head_dim]
        # atac_rna_attention = torch.bmm(
        #     atac_query.permute(1, 0, 2).reshape(self.num_heads, -1, self.head_dim),
        #     rna_key.permute(1, 2, 0).reshape(self.num_heads, self.head_dim, -1)
        # )  # [num_heads, num_atac_nodes, num_rna_nodes]
        # atac_rna_attention = atac_rna_attention * chrom_mask.T
        # atac_rna_attention = F.softmax(atac_rna_attention, dim=-1)
        # atac_rna_attention = torch.sigmoid(atac_rna_attention)
        # atac_rna_attention = (atac_rna_attention > 0.5).float()
        # atac_rna_attention = self.dropout(atac_rna_attention)

        atac_rna_attention = rna_atac_attention.permute(0, 2, 1) # 对称注意力

        # 加权聚合
        atac_to_rna_agg = torch.bmm(atac_rna_attention, rna_value.permute(1, 0, 2).reshape(self.num_heads, -1, self.head_dim))  # [num_heads, num_atac_nodes, head_dim]
        atac_to_rna_agg = atac_to_rna_agg.permute(1, 0, 2).reshape(-1, self.hidden_channels)  # [num_atac_nodes, hidden_channels]

        # case_study 绘图代码
        # if not self.training:
        #     # 提取注意力矩阵
        #     attention_matrix1 = rna_atac_attention[0].detach().cpu().numpy()
        #     # attention_matrix2 = atac_rna_attention[0].detach().cpu().numpy()
        #
        #     # 创建绘图区域
        #     plt.figure(figsize=(16, 8))  # 设置总图大小
        #
        #     # 绘制第一个热力图
        #     # plt.subplot(1, 2, 1)  # 1 行 2 列的第 1 个子图
        #     sns.heatmap(
        #         attention_matrix1,
        #         annot=False,
        #         cmap='coolwarm',  # 使用配色方案
        #         cbar=True,  # 显示颜色条
        #         xticklabels=False,  # 关闭X轴标签（可根据需要自定义）
        #         yticklabels=False,  # 关闭Y轴标签（可根据需要自定义）
        #     )
        #     plt.title('RNA-ATAC Attention Heatmap')
        #     plt.xlabel('ATAC Nodes')
        #     plt.ylabel('RNA Nodes')
        #
        #     # # 绘制第二个热力图
        #     # plt.subplot(1, 2, 2)  # 1 行 2 列的第 2 个子图
        #     # sns.heatmap(
        #     #     attention_matrix2,  # 转置矩阵以便显示
        #     #     annot=False,
        #     #     cmap='coolwarm',  # 使用配色方案
        #     #     cbar=True,  # 显示颜色条
        #     #     xticklabels=False,  # 关闭X轴标签（可根据需要自定义）
        #     #     yticklabels=False,  # 关闭Y轴标签（可根据需要自定义）
        #     # )
        #     # plt.title('ATAC-RNA Attention Heatmap')
        #     # plt.xlabel('RNA Nodes')
        #     # plt.ylabel('ATAC Nodes')
        #
        #     # 调整布局并显示
        #     plt.tight_layout()
        #     plt.show()

        if self.num_heads > 1:
            rna_to_atac_agg = self.rna_out_proj(rna_to_atac_agg)  # [num_rna_nodes, hidden_channels]
            atac_to_rna_agg = self.atac_out_proj(atac_to_rna_agg)  # [num_atac_nodes, hidden_channels]

        # 自注意力聚合（GraphSAGE形式，原本信息和聚合信息相加）
        # rna_to_atac_agg = rna_to_atac_agg + self.rna_self_attention(x_rna)  # [num_rna_nodes, hidden_channels]
        # atac_to_rna_agg = atac_to_rna_agg + self.atac_self_attention(x_atac)  # [num_atac_nodes, hidden_channels]

        # 自注意力聚合（拼接形式，原本信息和聚合信息拼接）
        rna_to_atac_agg = torch.cat([rna_to_atac_agg, self.rna_self_attention(x_rna)], dim=1)
        atac_to_rna_agg = torch.cat([atac_to_rna_agg, self.atac_self_attention(x_atac)], dim=1)
        rna_to_atac_agg = self.dim_reduction_rna(rna_to_atac_agg)
        atac_to_rna_agg = self.dim_reduction_atac(atac_to_rna_agg)

        # 归一化
        # rna_to_atac_agg = self.layernorm_rna(rna_to_atac_agg)  # [num_rna_nodes, hidden_channels]
        # atac_to_rna_agg = self.layernorm_atac(atac_to_rna_agg)  # [num_atac_nodes, hidden_channels]

        # 残差连接和前馈网络 (FFN)
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

        # 定义CrossAttentionConv用于RNA和ATAC节点的交互
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

        # 最后的分类层
        self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels)  # RNA和ATAC的聚合输出
        self.fc2 = nn.Linear(hidden_channels, int(hidden_channels/2))
        self.fc3 = nn.Linear(int(hidden_channels/2), out_channels)
        self.dropout = nn.Dropout(dropout)

        # 可解释性实验专用
        self.sample_counter = 0  # **自动编号的计数器**


    def forward(self, data, device):
        # 输入：data是包含异构图的HeteroData对象
        x_dict = data.x_dict
        x_dict = {key: x.float() for key, x in x_dict.items()}  # 转换为float

        #print("rna row id:",x_dict['rna'][:,0].long())

        # 将 RNA 和 ATAC 节点id映射到相同维度
        x_rna_id = self.rna_idembedding(x_dict['rna'][:,0].long())
        x_atac_id = self.atac_idembedding(x_dict['atac'][:,0].long())
        #print("rna id:",x_rna_id.shape,x_rna_id)

        x_rna_feature = x_dict['rna'][:,1:]
        x_atac_feature = x_dict['atac'][:,1:]
        # x_rna_feature = self.rna_feature_embedding(x_rna_feature)
        # x_atac_feature = self.atac_feature_embedding(x_atac_feature)
        #print("rna feature:",x_rna_feature.shape,x_rna_feature)

        #正常
        x_rna = torch.cat((x_rna_id, x_rna_feature), dim=1)
        x_atac = torch.cat((x_atac_id, x_atac_feature), dim=1)

        # ablation wo feature
        # x_rna = x_rna_id
        # x_atac = x_atac_id

        # # ablation wo ID
        # x_rna = x_rna_feature
        # x_atac = x_atac_feature

        chrom_mask = data.chrom_mask

        # 可解释性实验专用 保存注意力矩阵对应的非0索引
        save_dir_atac = "./cd4_test_none_zero_atac_index"
        save_dir_rna = "./cd4_test_none_zero_rna_index"
        os.makedirs(save_dir_atac, exist_ok=True)
        os.makedirs(save_dir_rna, exist_ok=True)
        sample_id = self.sample_counter
        self.sample_counter += 1  # **编号自增**

        atac_non_zero_indices = data['atac'].non_zero_indices
        save_path = os.path.join(save_dir_atac, f"sample_{sample_id}.pt")
        torch.save(atac_non_zero_indices, save_path)
        rna_non_zero_indices = data['rna'].non_zero_indices
        save_path = os.path.join(save_dir_rna, f"sample_{sample_id}.pt")
        torch.save(rna_non_zero_indices, save_path)

        print(f"Saved attention matrix for sample {sample_id} at {save_path}")

        # 通过多层 CrossAttentionConv
        for i, cross_attention_layer in enumerate(self.cross_attention_layers):
            rna_to_atac_agg, atac_to_rna_agg = cross_attention_layer(x_rna, x_atac, chrom_mask)
            # 使用跳跃连接
            if i > 0:
                x_rna = x_rna + rna_to_atac_agg  # 加跳跃连接
                x_atac = x_atac + atac_to_rna_agg
            else:
                x_rna, x_atac = rna_to_atac_agg, atac_to_rna_agg
        #print("rna_to_atac_agg:",rna_to_atac_agg.shape,rna_to_atac_agg)

        # 对聚合后的特征进行池化操作
        x_rna_agg = global_mean_pool(x_rna, data['rna'].batch)
        x_atac_agg = global_mean_pool(x_atac, data['atac'].batch)
        #print("x_rna_agg:",x_rna.shape,x_rna)

        # 将聚合后的特征进行拼接并传入全连接层
        x = torch.cat([x_rna_agg, x_atac_agg], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x
