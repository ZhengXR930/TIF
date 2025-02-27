import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
from datetime import datetime
import os
from loss import MPL
from torch.utils.data import Sampler
from collections import Counter
import torch.nn.functional as F
from scipy import sparse

class Stg2CustomDataset(Dataset):
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

class BalancedUniformEnvSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.env_labels = dataset.envs.numpy()  
        self.sample_labels = dataset.y.numpy()  
        
        self.env_indices = {env: np.where(self.env_labels == env)[0].tolist()
                            for env in np.unique(self.env_labels)}
        
        self.num_envs = len(self.env_indices)
        self.batch_size = batch_size
        
        self.env_batch_ratios = {env: len(indices) / sum(len(i) for i in self.env_indices.values())
                                 for env, indices in self.env_indices.items()}
        
        self.samples_per_env_per_batch = {env: max(1, int(self.env_batch_ratios[env] * self.batch_size))
                                          for env in self.env_indices}

        self._initialize_indices()

    def _initialize_indices(self):
        self.env_sampling_pools = {}
        for env, indices in self.env_indices.items():
            np.random.shuffle(indices)  # 重新打乱数据
            self.env_sampling_pools[env] = indices.copy()  # 确保数据不会被修改

    def _get_samples_from_env(self, env, num_samples):
        if len(self.env_sampling_pools[env]) < num_samples:
            np.random.shuffle(self.env_indices[env])  
            self.env_sampling_pools[env] = self.env_indices[env].copy()

        selected_samples = self.env_sampling_pools[env][:num_samples]
        self.env_sampling_pools[env] = self.env_sampling_pools[env][num_samples:]  
        return selected_samples
    
    def __iter__(self):
        self._initialize_indices()  
        total_indices = []
        num_batches = len(self)  
        
        for _ in range(num_batches):
            batch_indices = []
            for env in self.env_indices:
                num_samples = self.samples_per_env_per_batch[env]
                batch_indices.extend(self._get_samples_from_env(env, num_samples))
            
            np.random.shuffle(batch_indices)
            total_indices.extend(batch_indices)
        
        return iter(total_indices)
    
    def __len__(self):
        total_samples = sum(len(indices) for indices in self.env_indices.values())
        return total_samples // self.batch_size  

class St2ModelTrainer:
    def __init__(self, model, device='cuda', batch_size=64, learning_rate=0.001, save_dir='models'):
        self.device = device
        self.model = model.to(device)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_dir = save_dir
        self.best_f1 = 0
        self.penalty_weight = 10.0
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        self.criterion = nn.CrossEntropyLoss()
        self.custom_loss = MPL(device=device,feat_dim=200)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    def reset_optimizer(self, learning_rate):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        print(f"reinitialize optimizer with learning rate {learning_rate}")
        
    def create_dataloaders(self, X_train, X_val, y_train, y_val,env_train,env_val):
        train_dataset = Stg2CustomDataset(X_train, y_train,env_train)
        val_dataset = Stg2CustomDataset(X_val, y_val,env_val)

        sampler = BalancedUniformEnvSampler(train_dataset, self.batch_size)

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

        # print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return metrics
    
    def save_model(self, epoch, metrics):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'model_epoch{epoch}_lr{self.learning_rate}_bs{self.batch_size}_f1_{metrics["f1"]:.3f}_{timestamp}.pt'
        path = os.path.join(self.save_dir, filename)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }, path)
        
        return path

    def _compute_gradient_penalty(self, logits, label):
        # IRM alignment
        if label.dim() >= 2 and label.shape[1] > 1:
            label = label[:,1]
        scale = torch.ones_like(logits, requires_grad=True, device=self.device)
        loss = F.binary_cross_entropy_with_logits(logits * scale, label, reduction='none')

        grad = torch.autograd.grad(loss.sum(), [scale], create_graph=True)[0]
        # return sum of grad
        return (grad ** 2).sum()
        # return (grad ** 2).mean()
    
    def train(self, X_train, X_val, y_train, y_val, env_train, env_val, epochs=50,learning_rate=0.001):
        train_loader, val_loader = self.create_dataloaders(X_train, X_val, y_train, y_val, env_train, env_val)
        best_model_path = None

        num_envs = len(np.unique(env_train))

        interval = 20
        penalty_weight = self.penalty_weight
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            step = 0

            for inputs, labels, env in train_loader:
                inputs, labels, env = inputs.to(self.device), labels.to(self.device), env.to(self.device)

                self.optimizer.zero_grad()

                features = self.model.encoder_model(inputs)
                con_loss = self.custom_loss(features, labels)
                all_env_losses = []
                all_env_penalties = []

                for i in range(num_envs):
                    mask = (env == i)
                    if mask.any():
                        env_inputs = inputs[mask]
                        env_labels = labels[mask]

                        env_features = self.model.encoder_model(env_inputs)
                        env_outputs = self.model.mlp_model(env_features)
                        cls_loss = self.criterion(env_outputs, env_labels)
                        penalty = self._compute_gradient_penalty(torch.sigmoid(env_outputs)[:, 1], env_labels.float())

                        env_loss = cls_loss
                        all_env_losses.append(env_loss)
                        all_env_penalties.append(penalty)

                
                if all_env_losses:
                    total_env_loss = torch.stack(all_env_losses).mean()
                    total_penalty = torch.stack(all_env_penalties).mean()
                    total_loss = total_env_loss + penalty_weight * total_penalty + con_loss

                    total_loss.backward()
                    self.optimizer.step()

                    total_loss += total_env_loss.item()

                if step % interval == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Step {step}/{len(train_loader)}, "
                        f"Classification Loss: {cls_loss:.4f}, Contrastive Loss: {con_loss:.4f}, "
                        f"IRM Penalty: {penalty_weight * total_penalty:.4f}")

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
                if best_model_path:
                    os.remove(best_model_path)
                best_model_path = self.save_model(epoch, val_metrics)
                
        return best_model_path

    @staticmethod
    def load_model(model_path, model_class, input_size, device='cuda'):
        checkpoint = torch.load(model_path, map_location=device)
        model = model_class(input_size=input_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        return model
