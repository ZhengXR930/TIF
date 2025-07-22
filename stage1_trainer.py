import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
from datetime import datetime
import os
from collections import Counter
from scipy import sparse
from loss_mpc import PALSoftWithInterMargin
from utils import BalancedEnvSampler
import pickle

class Stg1CustomDataset(Dataset):
    def __init__(self, X, y, env):
        # First check if X is sparse and handle appropriately
        if sparse.issparse(X):
            self.X = torch.FloatTensor(X.toarray())
        else:
            self.X = torch.FloatTensor(X)
            
        # Convert labels and environment IDs to tensors
        self.y = torch.LongTensor(y)
        self.envs = torch.LongTensor(env)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.envs[idx]


class St1ModelTrainer:
    def __init__(self, model, device='cuda', batch_size=64, learning_rate=0.001, con_loss_weight=1.0, save_dir='models'):
        self.device = device
        self.model = model.to(device)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_dir = save_dir
        self.best_f1 = 0
        self.con_loss_weight = con_loss_weight
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        self.criterion = nn.CrossEntropyLoss()
        self.custom_loss = PALSoftWithInterMargin(
                device=device,
                input_dim=model.emb_dim,
                embed_dim=128,
                num_classes=2,
                n_proxy=5,
                tau=0.2,
                margin=1.5,
                lambda_margin=0.05,
                lambda_div=0.01
            ).to(device)

        self.optimizer = torch.optim.Adam(
            list(model.parameters()) + list(self.custom_loss.parameters()), lr=self.learning_rate
        )
        
    def create_dataloaders(self, X_train, X_val, y_train, y_val,env_train,env_val):
        train_dataset = Stg1CustomDataset(X_train, y_train, env_train)
        val_dataset = Stg1CustomDataset(X_val, y_val, env_val)

        sampler = BalancedEnvSampler(train_dataset, self.batch_size)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        return train_loader, val_loader
    
    def evaluate(self, dataloader):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels,*_ in dataloader:
                inputs = inputs.to(self.device)
                outputs,_ = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return metrics
    
    def save_model(self, epoch, metrics,model):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'model_epoch{epoch}_lr{self.learning_rate}_bs{self.batch_size}.pt'
        path = os.path.join(self.save_dir, filename)
        
        torch.save({
            'epoch': epoch,
            # 'model_state_dict': self.model.state_dict(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }, path)
        
        return path
    
    def train(self, X_train, X_val, y_train, y_val, env_train, env_val, epochs=50):
        train_loader, val_loader = self.create_dataloaders(X_train, X_val, y_train, y_val, env_train, env_val)
        best_model_path = None

        num_envs = len(np.unique(env_train))

        model = self.model

        interval = 20
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            step = 0


            for inputs, labels, env in train_loader:
                inputs, labels, env = inputs.to(self.device), labels.to(self.device), env.to(self.device)

                self.optimizer.zero_grad()

                for i in range(num_envs):
                    mask = env == i
                    env_inputs = inputs[mask]
                    env_labels = labels[mask]
                    env_outputs, env_features = self.model(env_inputs)
                    cls_loss = self.criterion(env_outputs, env_labels)
                    con_loss = self.custom_loss(env_features, env_labels)
                    loss = cls_loss + self.con_loss_weight * con_loss
                    loss.backward()
                
                self.optimizer.step()
                    
                total_loss += loss.item()

                if step % interval == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Step {step}/{len(train_loader)},Classification Loss: {cls_loss.item():.4f}, Contrastive Loss: {con_loss.item():.4f}")
                step += 1
 
            # Evaluate after each epoch
            train_metrics = self.evaluate(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Training - Loss: {total_loss/len(train_loader):.4f}")
            print(f"Training Metrics: {train_metrics}")
            print(f"Validation Metrics: {val_metrics}")
            
            
            # Save best model
            if val_metrics['f1'] > self.best_f1:
                self.best_f1 = val_metrics['f1']
                model = self.model
            
        
                # if best_model_path:
                #     os.remove(best_model_path)
                # best_model_path = self.save_model(epoch, val_metrics)
        best_model_path = self.save_model(epoch, val_metrics,model)
                
        return best_model_path

    @staticmethod
    def load_model(model_path, model_class, input_size, device='cuda'):
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model = model_class(input_size=input_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        return model


# if __name__ == '__main__':

#     with open(os.path.join('/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/dataset/processed_features', f'train_data.pkl'), 'rb') as f:
#         train_data = pickle.load(f)
#     print(f"train data shape: {train_data['X'].shape}, {train_data['y'].shape}, {train_data['env'].shape}, {train_data['t'].shape}")
#     print(f"distribution of train data: {Counter(train_data['y']), Counter(train_data['env'])}")

#     dataset = Stg1CustomDataset(train_data['X'], train_data['y'], train_data['env'])  # 你的自定义数据集
#     sampler = BalancedEnvSampler(dataset, batch_size=256)  # 用你改好的均衡采样器
#     loader = DataLoader(dataset, batch_size=256, sampler=sampler)

#     # 统计前 10 个 batch 的 env 分布
#     for batch_idx, (x, y, envs) in enumerate(loader):
#         env_counts = Counter(envs.tolist())
#         print(f"Batch {batch_idx} env distribution: {dict(env_counts)}")
#         if batch_idx == 9:
#             break