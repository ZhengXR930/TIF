import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
from datetime import datetime
import os
from loss_mpc import MPC
from torch.utils.data import Sampler
from collections import Counter
import torch.nn.functional as F
from scipy import sparse
from utils import BalancedEnvSampler
from loss_mpc import MPC

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
 

class St2ModelTrainer:
    def __init__(self, model, device='cuda', batch_size=64, learning_rate=0.001,
                 con_loss_weight=1.0, penalty_weight=1.0, save_dir='models',
                 penalty_warmup_epochs=10, custom_loss_state_dict=None, mpc_load_mode='full',
                 weight_decay=1e-4):
        self.device = device
        self.model = model.to(device)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_dir = save_dir
        self.best_f1 = 0
        self.penalty_weight = penalty_weight
        self.con_loss_weight = con_loss_weight
        self.penalty_warmup_epochs = penalty_warmup_epochs
        self.weight_decay = weight_decay

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        self.criterion = nn.CrossEntropyLoss()
        self.custom_loss = MPC(
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
        # If provided, reuse the MPC proxy parameters from a previous stage (e.g., stage 1)
        # Option: 'full' (load all), 'proj_only' (load projection layer only, reinit proxies), 'none' (reinit all)
        if custom_loss_state_dict is not None:
            self._load_mpc_state(custom_loss_state_dict, load_mode=mpc_load_mode)
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.custom_loss.parameters()), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
    
    def _load_mpc_state(self, state_dict, load_mode='full'):
        """
        Load MPC state with different strategies.
        
        Args:
            state_dict: State dict from checkpoint
            load_mode: 
                - 'full': Load all parameters (proj + proxies) - use when data distribution is similar
                - 'proj_only': Load only projection layer, reinitialize proxies - use when data distribution differs
                - 'none': Don't load, keep random initialization - use when starting fresh
        """
        if load_mode == 'full':
            self.custom_loss.load_state_dict(state_dict)
            print("Loaded MPC: full state (proj + proxies)")
        elif load_mode == 'proj_only':
            # Only load projection layer if it exists
            if self.custom_loss.proj is not None:
                proj_state = {k: v for k, v in state_dict.items() if k.startswith('proj')}
                if proj_state:
                    self.custom_loss.load_state_dict(proj_state, strict=False)
                    print("Loaded MPC: projection layer only, proxies reinitialized")
                else:
                    print("Loaded MPC: no projection layer found, proxies reinitialized")
            else:
                print("Loaded MPC: no projection layer, proxies reinitialized")
        elif load_mode == 'none':
            print("Loaded MPC: none (keeping random initialization)")
        else:
            raise ValueError(f"Unknown load_mode: {load_mode}. Use 'full', 'proj_only', or 'none'")
    
    def reset_optimizer(self, learning_rate):
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.custom_loss.parameters()), 
            lr=learning_rate,
            weight_decay=self.weight_decay
        )
        print(f"reinitialize optimizer with learning rate {learning_rate}, weight_decay {self.weight_decay}")
        
    def create_dataloaders(self, X_train, X_val, y_train, y_val,env_train,env_val):
        train_dataset = Stg2CustomDataset(X_train, y_train,env_train)
        val_dataset = Stg2CustomDataset(X_val, y_val,env_val)

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

        # print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return metrics
    
    def save_model(self, epoch, metrics):
        filename = f'stage2_model_epoch{epoch}_lr{self.learning_rate}_bs{self.batch_size}.pt'
        path = os.path.join(self.save_dir, filename)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            # Save MPC proxy parameters so they can be reused when loading stage 2 checkpoints
            'custom_loss_state_dict': self.custom_loss.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }, path)
        
        return path

    def _compute_irm_penalty(self, logits, labels, scale):
        if labels.dim() >= 2 and labels.shape[1] > 1:
            labels = labels[:, 1]
        scaled_logits = logits * scale
        loss = F.binary_cross_entropy_with_logits(scaled_logits, labels, reduction='sum') 
        grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        return (grad ** 2).mean()
    
    def train(self, X_train, X_val, y_train, y_val, env_train, env_val, epochs=50):
        train_loader, val_loader = self.create_dataloaders(X_train, X_val, y_train, y_val, env_train, env_val)
        best_model_path = None

        num_envs = len(np.unique(env_train))

        interval = 20

        env_scales = [torch.tensor(1.0, requires_grad=True, device=self.device) for _ in range(num_envs)]
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            step = 0

            # Linearly warm up the effective penalty weight to avoid collapse
            if self.penalty_warmup_epochs > 0:
                warmup_factor = min(1.0, float(epoch + 1) / float(self.penalty_warmup_epochs))
            else:
                warmup_factor = 1.0
            effective_penalty_weight = self.penalty_weight * warmup_factor

            for inputs, labels, env in train_loader:
                inputs, labels, env = inputs.to(self.device), labels.to(self.device), env.to(self.device)

                self.optimizer.zero_grad()

                features = self.model.encoder_model(inputs)
                con_loss = self.con_loss_weight * self.custom_loss(features, labels)

                all_env_losses = []
                all_env_penalties = []
                total_samples = 0

                for i in range(num_envs):
                    mask = (env == i)
                    if mask.any():
                        env_labels = labels[mask]
                        env_features = features[mask]
                        # Get logits (before softmax) for classification loss
                        env_logits = self.model.mlp_model(env_features)
                        env_outputs = self.model.pred(env_logits)  # Apply softmax for classification
                        cls_loss = self.criterion(env_outputs, env_labels)
                        scale = env_scales[i]
                        penalty = self._compute_irm_penalty(env_logits[:, 1], env_labels.float(), scale)

                        n = env_labels.size(0)
                        env_loss = cls_loss * n
                        all_env_losses.append(env_loss)
                        all_env_penalties.append(penalty)
                        total_samples += n

                
                if all_env_losses:
                    total_env_loss = torch.stack(all_env_losses).sum() / total_samples
                    # Normalize penalty across environments to stabilize its scale
                    total_penalty = torch.stack(all_env_penalties).mean()
                    total_loss = total_env_loss + effective_penalty_weight * total_penalty + con_loss

                    total_loss.backward()
                    self.optimizer.step()

                    epoch_loss += total_loss.item()

                    if step % interval == 0:
                        print(
                            f"Epoch {epoch+1}/{epochs}, Step {step}/{len(train_loader)}, "
                            f"Cls Loss: {cls_loss.item():.4f}, Con Loss: {con_loss.item():.4f}, "
                            f"IRM Penalty: weight {effective_penalty_weight:.2f}, penalty {total_penalty.item():.4f}, "
                            f"Total Loss: {total_loss.item():.4f}"
                        )

                step += 1
 
            # Evaluate after each epoch
            train_metrics = self.evaluate(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs}")
            if step > 0:
                print(f"Training - Loss: {epoch_loss/step:.4f}")
            print(f"Training Metrics: {train_metrics}")
            print(f"Validation Metrics: {val_metrics}")
            
            
            # Save best model based on validation F1 score
            if val_metrics['f1'] > self.best_f1:
                self.best_f1 = val_metrics['f1']
                if best_model_path:
                    os.remove(best_model_path)
                best_model_path = self.save_model(epoch, val_metrics)
                print(f"  -> Saved best model (val F1: {self.best_f1:.4f})")
            elif best_model_path is None:
                # Save first model if no model has been saved yet (safety measure)
                best_model_path = self.save_model(epoch, val_metrics)
                self.best_f1 = val_metrics['f1']
                print(f"  -> Saved initial model (val F1: {self.best_f1:.4f})")
                
        if best_model_path is None:
            # Fallback: save the last model if no model was saved (should not happen)
            print("Warning: No model was saved during training, saving last model as fallback")
            best_model_path = self.save_model(epochs - 1, val_metrics)
        
        print(f"\nTraining completed. Best validation F1: {self.best_f1:.4f}")
        return best_model_path

    @staticmethod
    def load_model(model_path, model_class, input_size, device='cuda'):
        # Load full checkpoint so we can reuse both model and proxy (MPC) parameters
        checkpoint = torch.load(model_path, map_location=device)
        model = model_class(input_size=input_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        custom_loss_state_dict = checkpoint.get('custom_loss_state_dict', None)
        return model, custom_loss_state_dict
