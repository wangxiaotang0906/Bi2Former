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

select_cell_type = False # select partical cell types
if_easy_neg = 1
neg_pos_ = 1
selected_cell_type = []
dataset_store_path = './bio_graph_datasets/10X_PBMC'

rna_anndata = sc.read_h5ad("./bio_datasets/10xMultiome-PBMC/10x-Multiome-Pbmc10k-RNA.h5ad")
print('rna_anndata:',rna_anndata) #9631 cells,29095 genes
atac_anndata = sc.read_h5ad("./bio_datasets/10xMultiome-PBMC/10x-Multiome-Pbmc10k-ATAC.h5ad")
print('atac_anndata:',atac_anndata) #9631 cells,107194 peaks

if scipy.sparse.issparse(rna_anndata.X):
    rna_anndata.X = rna_anndata.X.toarray()  
    atac_anndata.X = atac_anndata.X.toarray()
else:
    rna_anndata.X = np.array(rna_anndata.X)  
    atac_anndata.X = np.array(atac_anndata.X)  

if select_cell_type==True:
    rna_anndata = rna_anndata[rna_anndata.obs['cell_type'].isin(selected_cell_type)].copy()
    atac_anndata = atac_anndata[atac_anndata.obs['cell_type'].isin(selected_cell_type)].copy()
    #print(rna_anndata.obs)
    #print(atac_anndata.obs)

total_cells = len(rna_anndata)
if total_cells!=len(atac_anndata):
    print("ERROR: different size of rna and atac sequence!")
    exit()

rna_var_df = rna_anndata.var.copy()[['means','variances','variances_norm','chrom','strand','highly_variable','highly_variable_rank']]
rna_var_df['original_index'] = rna_var_df.index
atac_var_df = atac_anndata.var.copy()[['chrom']]
atac_var_df['original_index'] = atac_var_df.index

# process index
rna_var_df = rna_var_df.reset_index(drop=True)
print(rna_var_df)
atac_var_df = atac_var_df.reset_index(drop=True)
print(atac_var_df)

# process chrom
all_categories = pd.concat([rna_var_df['chrom'], atac_var_df['chrom']]).unique()
rna_var_df['chrom'] = pd.Categorical(rna_var_df['chrom'], categories=all_categories).codes
atac_var_df['chrom'] = pd.Categorical(atac_var_df['chrom'], categories=all_categories).codes

# process attributes
mapping = {'-': 0, '+': 1}
rna_var_df['strand'] =  rna_var_df['strand'].map(mapping)
rna_var_df['highly_variable'] = rna_var_df['highly_variable'].astype(int)
encoder = OneHotEncoder(categories=[range(len(all_categories))], sparse_output=False)
print("Unique chrom categories:", all_categories)

# df to numpy
rna_var_rowindex = rna_var_df.index.values
rna_var_rowindex = rna_var_rowindex.astype(int)
rna_var_chrom_np=rna_var_df[['chrom']].to_numpy().reshape(-1, 1)
rna_var_chrom_np_onehot = encoder.fit_transform(rna_var_chrom_np)
rna_var_np=rna_var_df[['means','variances_norm','strand','highly_variable']].to_numpy()
rna_var_np = np.hstack((rna_var_rowindex.reshape(-1,1),rna_var_chrom_np_onehot,rna_var_np))
print("rna row features shape:",rna_var_np.shape)
print("rna row features:",rna_var_np)

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

# easy_neg
if if_easy_neg == 1:
    all_cells = all_cell_df.index.tolist()
    all_types = all_cell_df['cell_type'].tolist()
    total_easy_negative_pairs = total_cells * neg_pos_ratio  
    easy_negative_pairs = []

    while len(easy_negative_pairs) < total_easy_negative_pairs:
        cell1, cell2 = np.random.choice(all_cells, size=2, replace=False)
        if all_cell_df.loc[cell1, 'cell_type'] != all_cell_df.loc[cell2, 'cell_type']:
            easy_negative_pairs.append((cell1, cell2))
    # print(easy_negative_pairs[:10])  
    # print(f"Total easy negative pairs: {len(easy_negative_pairs)}")

# hard_neg
if if_easy_neg == 0:
    total_hard_negative_pairs = total_cells * neg_pos_ratio
    hard_negative_pairs = []
    cell_type_proportions = grouped.size() / total_cells
    cell_type_negative_counts = (cell_type_proportions * total_hard_negative_pairs).astype(int)
    for cell_type, group in grouped:
        cells = group.index.tolist()
        negative_count = cell_type_negative_counts[cell_type]
        if len(cells) > 1 and negative_count > 0:
            pairs = list(itertools.combinations(cells, 2))
            if len(pairs) > negative_count:
                selected_indices = np.random.choice(len(pairs), size=negative_count, replace=False)
                selected_pairs = [pairs[i] for i in selected_indices]
            hard_negative_pairs.extend(selected_pairs)
    # print(hard_negative_pairs[:10])  
    # print(f"Total hard negative pairs: {len(hard_negative_pairs)}")

# create dataset
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

    data['rna'].non_zero_indices = rna_non_zero_indices
    data['atac'].non_zero_indices = atac_non_zero_indices

    chrom_mask = (rna_chorm_type[:, None] == atac_chorm_type[None, :])
    data.chrom_mask = torch.tensor(chrom_mask, dtype=torch.bool)

    return data

class HeteroGraphClassificationDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, if_easy_neg=1):
        self.if_easy_neg = if_easy_neg
        super(HeteroGraphClassificationDataset, self).__init__(root, transform, pre_transform)

        self.processed_dir = os.path.join(root, 'processed')
        os.makedirs(self.processed_dir, exist_ok=True)

        if not os.path.exists(self.processed_paths[0]):
            print(f"Processed data not found. Generating and saving dataset...")
            self.process()

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'graph_{i}.pt' for i in range(total_cells * 2)]

    def download(self):
        pass

    def process(self):
        for i in tqdm(range(total_cells), desc="Processing Positive Examples"):
            data = create_single_hetero_graph(i)
            data.y = torch.tensor(1)
            torch.save(data, os.path.join(self.processed_dir, f'graph_{i}.pt'))

        print('Positive examples done.')

        j = total_cells
        
        if self.if_easy_neg == 1:
            for m, n in tqdm(easy_negative_pairs, desc="Processing Easy Negative Pairs"):
                data = create_single_hetero_graph(m, n, 0)
                data.y = torch.tensor(0)
                torch.save(data, os.path.join(self.processed_dir, f'graph_{j}.pt'))
                j=j+1
        else:
            for m, n in tqdm(hard_negative_pairs, desc="Processing Hard Negative Pairs"):
                data = create_single_hetero_graph(m, n, 0)
                data.y = torch.tensor(0)
                torch.save(data, os.path.join(self.processed_dir, f'graph_{j}.pt'))
                j=j+1
        
        print('Negative examples done.')

    def __getitem__(self, idx):
        file_path = os.path.join(self.processed_dir, self.processed_file_names[idx])
        data = torch.load(file_path)

        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.processed_file_names)


dataset = HeteroGraphClassificationDataset(root=dataset_store_path, if_easy_neg=if_easy_neg)
