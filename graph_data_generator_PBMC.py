import torch
import torch_geometric
import numpy as np
import pandas as pd
import scanpy as sc
import itertools
from torch_geometric.data import HeteroData
from torch_geometric.data import InMemoryDataset
import os
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import scipy.sparse

select_cell_type = True #选择部分细胞
if_easy_neg = 0
neg_easy_ratio = 1
selected_cell_type = ["CD14 Mono"]
dataset_store_path = './bio_graph_datasets/10X_PBMC_CD14'

rna_anndata = sc.read_h5ad("./bio_datasets/10xMultiome-PBMC/10x-Multiome-Pbmc10k-RNA.h5ad")
print('rna_anndata:',rna_anndata) #9631个细胞,29095个基因
atac_anndata = sc.read_h5ad("./bio_datasets/10xMultiome-PBMC/10x-Multiome-Pbmc10k-ATAC.h5ad")
print('atac_anndata:',atac_anndata) #9631个细胞,107194个peak
#print('atac_anndata_var:',atac_anndata.var) #9631个细胞,107194个peak

# 检测是否为稀疏矩阵，并转换为 dense 格式
if scipy.sparse.issparse(rna_anndata.X):
    rna_anndata.X = rna_anndata.X.toarray()  # 转换为 NumPy 数组
    atac_anndata.X = atac_anndata.X.toarray()
else:
    rna_anndata.X = np.array(rna_anndata.X)  # 确保 X 是 NumPy
    atac_anndata.X = np.array(atac_anndata.X)  # 确保 X 是 NumPy 数组

if select_cell_type==True:
    rna_anndata = rna_anndata[rna_anndata.obs['cell_type'].isin(selected_cell_type)].copy()
    atac_anndata = atac_anndata[atac_anndata.obs['cell_type'].isin(selected_cell_type)].copy()
    #print(rna_anndata.obs)
    #print(atac_anndata.obs)

# 复现性实验专用：划分 RNA 和 ATAC 数据
# rna_anndata = rna_anndata[350:700]
# atac_anndata = atac_anndata[350:700]
# print(rna_anndata)

total_cells = len(rna_anndata)
if total_cells!=len(atac_anndata):
    print("ERROR: different size of rna and atac sequence!")
    exit()


rna_var_df = rna_anndata.var.copy()[['means','variances','variances_norm','chrom','strand','highly_variable','highly_variable_rank']]
rna_var_df['original_index'] = rna_var_df.index
atac_var_df = atac_anndata.var.copy()[['chrom']]
atac_var_df['original_index'] = atac_var_df.index

#处理index
rna_var_df = rna_var_df.reset_index(drop=True)
print(rna_var_df)
atac_var_df = atac_var_df.reset_index(drop=True)
print(atac_var_df)

#处理chrom，共享编码
all_categories = pd.concat([rna_var_df['chrom'], atac_var_df['chrom']]).unique()
rna_var_df['chrom'] = pd.Categorical(rna_var_df['chrom'], categories=all_categories).codes
atac_var_df['chrom'] = pd.Categorical(atac_var_df['chrom'], categories=all_categories).codes

#处理rna中非数值特征
mapping = {'-': 0, '+': 1}
rna_var_df['strand'] =  rna_var_df['strand'].map(mapping)
rna_var_df['highly_variable'] = rna_var_df['highly_variable'].astype(int)
encoder = OneHotEncoder(categories=[range(len(all_categories))], sparse_output=False)
print("Unique chrom categories:", all_categories)

#df转numpy，索引作为节点唯一标识
rna_var_rowindex = rna_var_df.index.values
rna_var_rowindex = rna_var_rowindex.astype(int)
rna_var_chrom_np=rna_var_df[['chrom']].to_numpy().reshape(-1, 1)
rna_var_chrom_np_onehot = encoder.fit_transform(rna_var_chrom_np)
rna_var_np=rna_var_df[['means','variances_norm','strand','highly_variable']].to_numpy()
rna_var_np = np.hstack((rna_var_rowindex.reshape(-1,1),rna_var_chrom_np_onehot,rna_var_np))
print("rna row features shape:",rna_var_np.shape)
print("rna row features:",rna_var_np)


##为ATAC节点加入DNA信息特征（169148~169179为nan）
# # 定义转换函数
# def dna_sequence_to_numeric(sequence):
#     """
#     将DNA序列中的碱基字符映射到数值。
#     输入: 字符串 (如 "ATGCN")
#     输出: 数值列表 (如 [0.25, 0.5, 0.75, 1.0, 0.0])
#     """
#     base_mapping = {
#         'A': 0.25,
#         'T': 0.5,
#         'G': 0.75,
#         'C': 1.0,
#         'N': 0.0  # 其他非标准碱基
#     }
#     return [base_mapping.get(base, 0.0) for base in sequence]
#
# descriptions = []
# sequences = []
# max_len = 256  # 最大长度
#
# with open("./bio_datasets/ISSAAC-seq/ISSAAC_peak_output.fa", 'r') as f:
#     seq = ""
#     for line in f:
#         line = line.strip()  # 去除每行开头和结尾的空白
#         if line.startswith(">"):
#             # 如果已在收集的序列，存储到列表并清空序列缓存
#             if seq:
#                 # 截断或填充到指定长度
#                 seq = seq[:max_len] + 'N' * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len]
#                 sequences.append(seq)
#                 seq = ""  # 清空缓存
#             descriptions.append(line[1:])  # 存储描述（去除>符号）
#         else:
#             seq += line  # 累积序列行
#     # 处理最后一条序列
#     if seq:
#         seq = seq[:max_len] + 'N' * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len]
#         sequences.append(seq)
#
# atac_dna_df = pd.DataFrame({
#         "Description": descriptions,
#         "Sequence": sequences
#     })
# atac_var_dna_df = pd.concat([atac_var_df.reset_index(drop=True), atac_dna_df[['Sequence']].reset_index(drop=True)], axis = 1)
# atac_var_dna_df['Sequence'] = atac_var_dna_df['Sequence'].fillna('N' * 256)
# #print(atac_var_dna_df[['Sequence']])
# atac_var_dna_np= atac_var_dna_df[['Sequence']].to_numpy().reshape(-1, 1)
#
# # 将整个 atac_var_dna_np 数据进行转换
# for i in range(atac_var_dna_np.shape[0]):
#     sequence = atac_var_dna_np[i, 0]  # 读取当前行的 DNA 序列字符串
#     numeric_sequence = dna_sequence_to_numeric(sequence)  # 转换为数值列表
#     atac_var_dna_np[i, 0] = numeric_sequence  # 原地替换为数值列表
#
# # 转换为 numpy 数组，确保所有行长度一致
# atac_var_dna_np = np.array(atac_var_dna_np.tolist(), dtype=np.float32)
# atac_var_dna_np = np.squeeze(atac_var_dna_np, axis=1)

# 打印结果的形状和示例
# print("Encoded DNA shape:", atac_var_dna_np.shape)
# print("Example encoded sequence:", atac_var_dna_np[0])

atac_var_rowindex = atac_var_df.index.values
atac_var_rowindex = atac_var_rowindex.astype(int)
atac_var_chrom_np=atac_var_df[['chrom']].to_numpy().reshape(-1, 1)
atac_var_chrom_np_onehot = encoder.fit_transform(atac_var_chrom_np)
#atac_var_np=atac_var_df[['n_counts']].to_numpy()
#atac_var_np = np.hstack((atac_var_rowindex.reshape(-1,1),atac_var_chrom_np_onehot,atac_var_np))
atac_var_np = np.hstack((atac_var_rowindex.reshape(-1,1),atac_var_chrom_np_onehot))
#atac_var_np = np.hstack((atac_var_rowindex.reshape(-1,1),atac_var_chrom_np_onehot,atac_var_dna_np,atac_var_np))
print("atac row features shape:",atac_var_np.shape)
print("atac row features:",atac_var_np)

all_cell_df =  rna_anndata.obs.copy()[['cell_type']]
all_cell_df = all_cell_df.reset_index(drop=True)
#print("cells with index:",all_cell_df)
print("cell type distribution:",rna_anndata.obs['cell_type'].value_counts())
grouped = all_cell_df.groupby('cell_type')

# 生成easy负例
if if_easy_neg == 1:
    # 所有细胞和对应的 cell_type
    all_cells = all_cell_df.index.tolist()
    all_types = all_cell_df['cell_type'].tolist()
    # 构建 easy 负例的配对
    total_easy_negative_pairs = total_cells * neg_easy_ratio  # 假设生成的负例总数是细胞数的两倍
    easy_negative_pairs = []

    while len(easy_negative_pairs) < total_easy_negative_pairs:
        # 随机选择两个不同的细胞
        cell1, cell2 = np.random.choice(all_cells, size=2, replace=False)
        # 检查两者是否来自不同的 cell_type
        if all_cell_df.loc[cell1, 'cell_type'] != all_cell_df.loc[cell2, 'cell_type']:
            easy_negative_pairs.append((cell1, cell2))

    # 打印一些负例
    # print(easy_negative_pairs[:10])  # 打印前10个负例
    # print(f"Total easy negative pairs: {len(easy_negative_pairs)}")

# 生成hard负例
if if_easy_neg == 0:
    # 按照 cell_type 分组
    total_hard_negative_pairs = total_cells * neg_easy_ratio
    hard_negative_pairs = []
    # 计算每个 cell_type 的比例
    cell_type_proportions = grouped.size() / total_cells
    # 计算每个 cell_type 需要生成的负例对数量
    cell_type_negative_counts = (cell_type_proportions * total_hard_negative_pairs).astype(int)
    # 生成hard负例
    for cell_type, group in grouped:
        cells = group.index.tolist()
        negative_count = cell_type_negative_counts[cell_type]
        if len(cells) > 1 and negative_count > 0:
            pairs = list(itertools.combinations(cells, 2))
            if len(pairs) > negative_count:
                selected_indices = np.random.choice(len(pairs), size=negative_count, replace=False)
                selected_pairs = [pairs[i] for i in selected_indices]
            hard_negative_pairs.extend(selected_pairs)

    # 打印一些负例
    # print(hard_negative_pairs[:10])  # 打印前10个负例
    # print(f"Total hard negative pairs: {len(hard_negative_pairs)}")

# 生成数据集
def create_single_hetero_graph(cellnumber_0, cellnumber_1=None, if_pos=1):
    data = HeteroData()
    #rna_non_zero_indices = [i for i, x in enumerate(rna_anndata.X[cellnumber_0]) if x != 0]
    rna_non_zero_indices = rna_anndata.X[cellnumber_0].nonzero()[0]
    rna_chorm_type = rna_var_chrom_np[rna_non_zero_indices]
    rna_x_expression = rna_anndata.X[cellnumber_0][rna_non_zero_indices]
    rna_x = np.hstack((rna_var_np[rna_non_zero_indices,0].reshape(-1, 1), rna_x_expression.reshape(-1, 1), rna_var_np[rna_non_zero_indices,1:]))
    data['rna'].x = torch.tensor(rna_x)

    if if_pos == 1:
        cellnumber_1 = cellnumber_0

    #atac_non_zero_indices = [i for i, x in enumerate(atac_anndata.X[cellnumber_1]) if x != 0]
    atac_non_zero_indices = atac_anndata.X[cellnumber_1].nonzero()[0]
    atac_chorm_type = atac_var_chrom_np[atac_non_zero_indices]
    atac_x_expression = atac_anndata.X[cellnumber_1][atac_non_zero_indices]
    atac_x = np.hstack((atac_var_np[atac_non_zero_indices,0].reshape(-1, 1), atac_x_expression.reshape(-1, 1), atac_var_np[atac_non_zero_indices,1:]))
    data['atac'].x = torch.tensor(atac_x)

    # 存储非0索引，生物可解释性实验专用
    data['rna'].non_zero_indices = rna_non_zero_indices
    data['atac'].non_zero_indices = atac_non_zero_indices

    # 创建 chrom_mask
    chrom_mask = (rna_chorm_type[:, None] == atac_chorm_type[None, :])
    data.chrom_mask = torch.tensor(chrom_mask, dtype=torch.bool)

    return data

class HeteroGraphClassificationDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, if_easy_neg=1):
        self.if_easy_neg = if_easy_neg
        super(HeteroGraphClassificationDataset, self).__init__(root, transform, pre_transform)

        # 这里使用 processed_dir 而非 processed_folder
        self.processed_dir = os.path.join(root, 'processed')
        os.makedirs(self.processed_dir, exist_ok=True)

        # 生成数据集并保存
        if not os.path.exists(self.processed_paths[0]):
            print(f"Processed data not found. Generating and saving dataset...")
            self.process()

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        # 使用图的索引作为文件名（如：graph_0.pt, graph_1.pt...）
        return [f'graph_{i}.pt' for i in range(total_cells * 2)]

    def download(self):
        pass

    def process(self):
        # 创建数据集，按需保存每个图
        for i in tqdm(range(total_cells), desc="Processing Positive Examples"):
            data = create_single_hetero_graph(i)
            data.y = torch.tensor(1)
            torch.save(data, os.path.join(self.processed_dir, f'graph_{i}.pt'))

        print('Positive examples done.')

        # j = total_cells
        #
        # if self.if_easy_neg == 1:
        #     for m, n in tqdm(easy_negative_pairs, desc="Processing Easy Negative Pairs"):
        #         data = create_single_hetero_graph(m, n, 0)
        #         data.y = torch.tensor(0)
        #         torch.save(data, os.path.join(self.processed_dir, f'graph_{j}.pt'))
        #         j=j+1
        # else:
        #     for m, n in tqdm(hard_negative_pairs, desc="Processing Hard Negative Pairs"):
        #         data = create_single_hetero_graph(m, n, 0)
        #         data.y = torch.tensor(0)
        #         torch.save(data, os.path.join(self.processed_dir, f'graph_{j}.pt'))
        #         j=j+1
        #
        # print('Negative examples done.')

    def __getitem__(self, idx):
        # 加载图数据
        file_path = os.path.join(self.processed_dir, self.processed_file_names[idx])
        data = torch.load(file_path)

        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        # 数据集的大小
        return len(self.processed_file_names)

# class HeteroGraphClassificationDataset(InMemoryDataset):
#     def __init__(self, root, transform=None, pre_transform=None, if_easy_neg=1):
#         self.if_easy_neg = if_easy_neg
#         super(HeteroGraphClassificationDataset, self).__init__(root, transform, pre_transform)
#         # 在第一次加载时，处理数据
#         if os.path.exists(self.processed_paths[0]):
#             print(f"文件 {self.processed_paths[0]} 已经存在，正在加载数据集")
#             self.data, self.slices = torch.load(self.processed_paths[0])
#         else:
#             print(f"文件 {self.processed_paths[0]} 不存在，正在生成数据集")
#             self.process()
#
#     @property
#     def raw_file_names(self):
#         return []  # 不需要原始文件
#
#     @property
#     def processed_file_names(self):
#         return ['ISSAAC_graph.pt']
#
#     def download(self):
#         # 不需要下载数据
#         pass
#
#     def process(self):
#         # 创建多个异构图，每个图有一个对应标签
#         data_list = []
#
#         for i in tqdm(range(total_cells), desc="Positive Examples"):
#             data = create_single_hetero_graph(i)
#             data.y = torch.tensor(1)
#             data_list.append(data)
#
#         print('pos_done')
#
#         if self.if_easy_neg == 1:
#             for m, n in tqdm(easy_negative_pairs, desc="Easy Negative Pairs"):  # 负例
#                 data = create_single_hetero_graph(m, n, 0)
#                 data.y = torch.tensor(0)
#                 data_list.append(data)
#         else:
#             for m, n in tqdm(hard_negative_pairs, desc="Hard Negative Pairs"):  # 负例
#                 data = create_single_hetero_graph(m, n, 0)
#                 data.y = torch.tensor(0)
#                 data_list.append(data)
#
#         print('neg_done')
#
#         # 将数据存储在 processed 路径中
#         data, slices = self.collate(data_list)
#         torch.save((data, slices), self.processed_paths[0])

dataset = HeteroGraphClassificationDataset(root=dataset_store_path, if_easy_neg=if_easy_neg)
