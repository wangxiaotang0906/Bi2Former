import time
import random
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import DataLoader, HeteroData
from torch.utils.data import random_split
from model_onehot import RNA_ATAC_Pairing
import matplotlib.pyplot as plt
from torch_geometric.data import InMemoryDataset
import os

dataset_store_path = 'bio_graph_datasets/10X_PBMC'
best_model_path = "checkpoint/10XPBMC_check_point_binary_08.pth"

class HeteroGraphClassificationDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, if_easy_neg=1):
        self.if_easy_neg = if_easy_neg
        super(HeteroGraphClassificationDataset, self).__init__(root, transform, pre_transform)
        # 动态获取数据集长度
        self.dataset_length = len([f for f in os.listdir(self.processed_dir)
                                   if f.startswith('graph_') and f.endswith('.pt')])
        if self.dataset_length == 0:
            print("No processed 'graph_*.pt' files found. Please ensure the dataset is generated correctly.")
            raise FileNotFoundError("No processed data files found in the processed directory.")

        if not os.path.exists(self.processed_paths[0]):
            print("Processed data not found. Please ensure the dataset is generated correctly.")
            raise FileNotFoundError("Required processed data file is missing.")

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'graph_{i}.pt' for i in range(self.dataset_length)]

    def download(self):
        pass

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        file_path = self.processed_paths[idx]
        data = torch.load(file_path, weights_only=False)
        if self.transform:
            data = self.transform(data)
        return data

    def __len__(self):
        return self.dataset_length


def set_seed(seed=906):
    """Set seed for reproducibility."""
    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # Numpy random seed
    torch.manual_seed(seed)  # PyTorch seed for CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # PyTorch seed for GPU
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable cudnn's auto-tuner

def train(model, data, optimizer, device):
    model.train()
    data = data.to(device)
    optimizer.zero_grad()
    out = model(data, device)
    loss = F.cross_entropy(out, data.y) 
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate(model, loader, device):
    model.eval()
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for data in loader:
        data = data.to(device)
        out = model(data, device)
        pred = out.argmax(dim=1)
        # print(f"label:{data.y},pred:{pred}")
        TP += ((pred == 1) & (data.y == 1)).sum().item()
        FP += ((pred == 1) & (data.y == 0)).sum().item()
        TN += ((pred == 0) & (data.y == 0)).sum().item()
        FN += ((pred == 0) & (data.y == 1)).sum().item()

    Accuracy = (TP+TN) / (TP+TN+FP+FN)

    if (TP+FP) == 0:
        P_Precision = 'all_predict_N'
    else:
        P_Precision = TP / (TP+FP)

    if (TN+FN) == 0:
        N_Precision = 'all_predict_P'
    else:
        N_Precision = TN / (TN+FN)

    if (TP+FN) == 0:
        P_Recall = 0
    else:
        P_Recall= TP / (TP+FN)

    if (FP+TN) == 0:
        N_Recall = 0
    else:
        N_Recall= TN / (FP+TN)

    return Accuracy, P_Precision, N_Precision, P_Recall, N_Recall


def main():
    set_seed(906)
    # set_seed(2000)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # in_channels_dict = {'rna_id': 32208, 'atac_id': 169180, 'rna_feature': 34, 'atac_feature': 286, 'rna_feature_hidden': 128, 'atac_feature_hidden': 128} #ISSAAC
    in_channels_dict = {'rna_id': 29095, 'atac_id': 107194, 'rna_feature': 42, 'atac_feature': 38, 'rna_feature_hidden': 128, 'atac_feature_hidden': 128} #10XPBMC
    # in_channels_dict = {'rna_id': 21478, 'atac_id': 340341, 'rna_feature': 36, 'atac_feature': 32, 'rna_feature_hidden': 128, 'atac_feature_hidden': 128} #SHARE
    # in_channels_dict = {'rna_id': 28930, 'atac_id': 241757, 'rna_feature': 34, 'atac_feature': 30, 'rna_feature_hidden': 128, 'atac_feature_hidden': 128} #SNARE

    model = RNA_ATAC_Pairing(in_channels_dict, id_embedding_dims=256, hidden_channels=128, out_channels=2, layer_num=1, multihead=True, num_heads=1,
                             dropout=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-6)
    print("model:", model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total_params: {total_params}")
    print(f"trainable_params: {trainable_params}")

    # 正常训练 加载数据集
    dataset = HeteroGraphClassificationDataset(root=dataset_store_path)
    print("dataset:", dataset)
    print("num_samples:", len(dataset))
    print("dataset samples:", dataset[0])
    num_samples = len(dataset)
    indices = torch.randperm(num_samples)
    shuffled_dataset = dataset[indices]   

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                             [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)

    train_losses = []
    val_accuracies = []
    best_val_accuracy = 0

    # Continue training from break
    # model.load_state_dict(torch.load(best_model_path))
    # model.to(device)

    # Training
    for epoch in range(1, 30):
        epoch_loss = 0
        start_time = time.time()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', unit='batch', dynamic_ncols=True, leave=False)
        for data in pbar:
            data = data.to(device)  # 确保每个 batch 的数据都转移到设备上
            loss = train(model, data, optimizer, device)
            epoch_loss += loss
            # 更新进度条
            pbar.set_postfix(loss=loss)
        pbar.close()  # 手动关闭进度条，避免"0%"行
    
        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - start_time
    
        # Validation 
        val_accuracy, val_P_Precision, val_N_Precision, val_P_Recall, val_N_Recall = evaluate(model, val_loader, device)
    
        tqdm.write(f'Epoch {epoch}, Time: {epoch_time:.2f}s, Loss: {avg_loss:.8f}, '
                   f'Validation Accuracy: {val_accuracy:.4f}, '
                   f'Validation P Precision: {val_P_Precision:.4f}, '
                   f'Validation N Precision: {val_N_Precision:.4f}, '
                   f'Validation P Recall: {val_P_Recall:.4f}, '
                   f'Validation N Recall: {val_N_Recall:.4f}')
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model of Epoch {epoch} with validation accuracy: {val_accuracy:.4f}")
    
        train_losses.append(avg_loss)
        val_accuracies.append(val_accuracy)
    
        time.sleep(1)
    

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy per Epoch')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)

    # Test
    test_accuracy, test_P_Precision, test_N_Precision, test_P_Recall, test_N_Recall = evaluate(model, test_loader,
                                                                                               device)
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Test P Precision: {test_P_Precision:.4f}')
    print(f'Test N Precision: {test_N_Precision:.4f}')
    print(f'Test P Recall: {test_P_Recall:.4f}')
    print(f'Test N Recall: {test_N_Recall:.4f}')

if __name__ == '__main__':
    main()
