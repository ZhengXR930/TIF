import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
from datetime import datetime
import os
from collections import Counter
from scipy import sparse
from loss_mpc import MPC
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
    def __init__(self, model, device='cuda', batch_size=64, learning_rate=0.001, 
                 con_loss_weight=1.0, save_dir='models', num_envs=None,
                 embed_dim=128, num_classes=2, n_proxy=3, tau=0.2, 
                 margin=1.5, lambda_margin=0.05, lambda_div=0.01,
                 weight_decay=0, proxy_lr_multiplier=1.0, use_scheduler=False,
                 early_stop_patience=0, use_multi_proxy=True):
        self.device = device
        self.model = model.to(device)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_dir = save_dir
        self.best_f1 = 0
        self.con_loss_weight = con_loss_weight
        self.num_envs = num_envs  
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.n_proxy = n_proxy
        self.tau = tau
        self.margin = margin
        self.lambda_margin = lambda_margin
        self.lambda_div = lambda_div
        self.weight_decay = weight_decay
        self.proxy_lr_multiplier = proxy_lr_multiplier
        self.use_scheduler = use_scheduler
        self.early_stop_patience = early_stop_patience
        self.patience_counter = 0
        self.use_multi_proxy = use_multi_proxy  # Whether to use multi-proxy mode
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        self.criterion = nn.CrossEntropyLoss()
        
        if use_multi_proxy:
            # Multi-proxy mode: will be initialized in train() when we know num_envs
            self.custom_losses = None  # Dictionary: {env_id: MPC_loss_instance}
            self.custom_loss = None
            self.scheduler = None
        else:
            # Single-proxy mode: backward compatible
            self.custom_loss = MPC(
                device=device,
                input_dim=model.emb_dim,
                embed_dim=embed_dim,
                num_classes=num_classes,
                n_proxy=n_proxy,
                tau=tau,
                margin=margin,
                lambda_margin=lambda_margin,
                lambda_div=lambda_div
            ).to(device)
            self.custom_losses = None
            self.scheduler = None
            self.optimizer = torch.optim.Adam(
                list(self.model.parameters()) + list(self.custom_loss.parameters()), 
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        
    def create_dataloaders(self, X_train, X_val, y_train, y_val,env_train,env_val):
        train_dataset = Stg1CustomDataset(X_train, y_train, env_train)
        val_dataset = Stg1CustomDataset(X_val, y_val, env_val)

        sampler = BalancedEnvSampler(train_dataset, self.batch_size)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        return train_loader, val_loader
    
    def _initialize_env_losses(self, num_envs):
        """Initialize separate MPC loss for each environment (multi-proxy mode only)."""
        if not self.use_multi_proxy:
            return  # Not in multi-proxy mode
        
        if self.custom_losses is not None and len(self.custom_losses) == num_envs:
            return  # Already initialized
        
        self.num_envs = num_envs
        self.custom_losses = nn.ModuleDict()
        
        for env_id in range(num_envs):
            env_loss = MPC(
                device=self.device,
                input_dim=self.model.emb_dim,
                embed_dim=self.embed_dim,
                num_classes=self.num_classes,
                n_proxy=self.n_proxy,
                tau=self.tau,
                margin=self.margin,
                lambda_margin=self.lambda_margin,
                lambda_div=self.lambda_div
            ).to(self.device)
            self.custom_losses[str(env_id)] = env_loss
        
        # Create parameter groups with different learning rates
        param_groups = [
            {'params': self.model.parameters(), 'lr': self.learning_rate}
        ]
        
        # Each environment's proxy parameters get slightly higher learning rate
        for env_loss in self.custom_losses.values():
            param_groups.append({
                'params': env_loss.parameters(), 
                'lr': self.learning_rate * self.proxy_lr_multiplier
            })
        
        self.optimizer = torch.optim.Adam(param_groups, lr=self.learning_rate, weight_decay=self.weight_decay)
        
        # Learning rate scheduler
        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
            )
    
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
    
    def save_model(self, epoch, metrics, model):
        if self.use_multi_proxy:
            filename = f'stage1_model_multi_proxy_epoch{epoch}_lr{self.learning_rate}_bs{self.batch_size}.pt'
        else:
            filename = f'stage1_model_epoch{epoch}_lr{self.learning_rate}_bs{self.batch_size}.pt'
        path = os.path.join(self.save_dir, filename)
        
        if self.use_multi_proxy:
            # Save state dicts for all environment-specific losses
            env_losses_state_dict = {}
            if self.custom_losses is not None:
                for env_id, env_loss in self.custom_losses.items():
                    env_losses_state_dict[env_id] = env_loss.state_dict()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                # Save all environment-specific MPC proxy parameters
                'env_losses_state_dict': env_losses_state_dict,
                'num_envs': self.num_envs,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': metrics,
                # Save hyperparameters for reconstruction
                'embed_dim': self.embed_dim,
                'num_classes': self.num_classes,
                'n_proxy': self.n_proxy,
                'tau': self.tau,
                'margin': self.margin,
                'lambda_margin': self.lambda_margin,
                'lambda_div': self.lambda_div,
            }, path)
        else:
            # Single-proxy mode: backward compatible
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                # Save MPC proxy parameters so they can be reused in stage 2
                'custom_loss_state_dict': self.custom_loss.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': metrics,
            }, path)
        
        return path
    
    def train(self, X_train, X_val, y_train, y_val, env_train, env_val, epochs=50):
        train_loader, val_loader = self.create_dataloaders(X_train, X_val, y_train, y_val, env_train, env_val)
        best_model_path = None

        num_envs = len(np.unique(env_train))
        
        # Initialize environment-specific losses if in multi-proxy mode
        if self.use_multi_proxy:
            self._initialize_env_losses(num_envs)

        model = self.model
        interval = 20
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            step = 0

            for inputs, labels, env in train_loader:
                inputs, labels, env = inputs.to(self.device), labels.to(self.device), env.to(self.device)

                self.optimizer.zero_grad()

                if self.use_multi_proxy:
                    for i in range(num_envs):
                        mask = env == i
                        if not mask.any():
                            continue  

                        env_inputs = inputs[mask]
                        env_labels = labels[mask]
                        env_outputs, env_features = self.model(env_inputs)
                        cls_loss = self.criterion(env_outputs, env_labels)
                        
                        
                        env_loss = self.custom_losses[str(i)]
                        con_loss = env_loss(env_features, env_labels)
                        
                        loss = cls_loss + self.con_loss_weight * con_loss
                        loss.backward()
                else:
                    for i in range(num_envs):
                        mask = env == i
                        env_inputs = inputs[mask]
                        env_labels = labels[mask]
                        if len(env_inputs) > 0:
                            env_outputs, env_features = self.model(env_inputs)
                            cls_loss = self.criterion(env_outputs, env_labels)
                            con_loss = self.custom_loss(env_features, env_labels)
                            loss = cls_loss + self.con_loss_weight * con_loss
                            loss.backward()
                
                self.optimizer.step()
                    
                # Track loss for logging
                if self.use_multi_proxy:
                    if mask.any():
                        total_loss += loss.item()
                else:
                    total_loss += loss.item()

                if step % interval == 0:
                    if self.use_multi_proxy:
                        if mask.any():
                            print(f"Epoch {epoch+1}/{epochs}, Step {step}/{len(train_loader)}, "
                                  f"Classification Loss: {cls_loss.item():.4f}, "
                                  f"Contrastive Loss: {con_loss.item():.4f}")
                    else:
                        print(f"Epoch {epoch+1}/{epochs}, Step {step}/{len(train_loader)}, "
                              f"Classification Loss: {cls_loss.item():.4f}, "
                              f"Contrastive Loss: {con_loss.item():.4f}")
                step += 1
 
            # Evaluate after each epoch
            train_metrics = self.evaluate(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Training - Loss: {total_loss/len(train_loader):.4f}")
            print(f"Training Metrics: {train_metrics}")
            print(f"Validation Metrics: {val_metrics}")
            
            # Learning rate scheduling (multi-proxy mode)
            if self.use_multi_proxy and self.scheduler is not None:
                self.scheduler.step(val_metrics['f1'])
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Current learning rate: {current_lr:.6f}")
            
            # Save best model and early stopping
            if val_metrics['f1'] > self.best_f1:
                self.best_f1 = val_metrics['f1']
                model = self.model
                self.patience_counter = 0
                # Save best model immediately when found
                best_model_path = self.save_model(epoch, val_metrics, model)
            else:
                self.patience_counter += 1
                if self.early_stop_patience > 0 and self.patience_counter >= self.early_stop_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs (patience: {self.early_stop_patience})")
                    break
            
        # Save final model if no best model was saved during training
        if best_model_path is None:
            best_model_path = self.save_model(epoch, val_metrics, model)
                
        return best_model_path

    @staticmethod
    def load_model(model_path, model_class, input_size, device='cuda'):
        """Load model. Returns (model, env_losses) for multi-proxy or (model, None) for single-proxy."""
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model = model_class(input_size=input_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        if 'env_losses_state_dict' in checkpoint:
            env_losses_state_dict = checkpoint['env_losses_state_dict']
            num_envs = checkpoint.get('num_envs', len(env_losses_state_dict))
            
            # Reconstruct hyperparameters from checkpoint
            embed_dim = checkpoint.get('embed_dim', 128)
            num_classes = checkpoint.get('num_classes', 2)
            n_proxy = checkpoint.get('n_proxy', 3)
            tau = checkpoint.get('tau', 0.1)
            margin = checkpoint.get('margin', 1.5)
            lambda_margin = checkpoint.get('lambda_margin', 0.05)
            lambda_div = checkpoint.get('lambda_div', 0.01)
            
            custom_losses = nn.ModuleDict()
            for env_id in range(num_envs):
                env_loss = MPC(
                    device=device,
                    input_dim=model.emb_dim,
                    embed_dim=embed_dim,
                    num_classes=num_classes,
                    n_proxy=n_proxy,
                    tau=tau,
                    margin=margin,
                    lambda_margin=lambda_margin,
                    lambda_div=lambda_div
                ).to(device)
                if str(env_id) in env_losses_state_dict:
                    env_loss.load_state_dict(env_losses_state_dict[str(env_id)])
                custom_losses[str(env_id)] = env_loss
            
            return model, custom_losses
        

        return model, None
