import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import os
from loss_mpc import MPC
import torch.nn.functional as F
from scipy import sparse
from utils import BalancedEnvSampler
from loss_mpc import MPC

class Stg2CustomDataset(Dataset):
    def __init__(self, X, y, env):
        if sparse.issparse(X):
            self.X = torch.FloatTensor(X.toarray())
        else:
            self.X = torch.FloatTensor(X)
            
        self.y = torch.LongTensor(y)
        self.envs = torch.LongTensor(env)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.envs[idx]
 

class St2ModelTrainer:
    def __init__(self, model, device='cuda', batch_size=256, learning_rate=0.0001,
                 con_loss_weight=1.0, penalty_weight=1.0, save_dir='models',
                 penalty_warmup_epochs=10, custom_loss_state_dict=None, mpc_load_mode='full',
                 weight_decay=1e-4, n_proxy=5, use_progressive_irm=True,
                 early_stop_patience=5):
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
        self.use_progressive_irm = use_progressive_irm  
        self.early_stop_patience = early_stop_patience  

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        self.criterion = nn.CrossEntropyLoss()
        self.custom_loss = MPC(
                device=device,
                input_dim=model.emb_dim,
                embed_dim=128,
                num_classes=2,
                n_proxy=n_proxy,
                tau=0.2,
                margin=1.5,
                lambda_margin=0.05,
                lambda_div=0.01
            ).to(device)

        if custom_loss_state_dict is not None:
            self._load_mpc_state(custom_loss_state_dict, load_mode=mpc_load_mode, n_proxy=n_proxy)

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.custom_loss.parameters()), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
    
    def _load_mpc_state(self, state_dict, load_mode='full', n_proxy=None):
        # Check if proxy count matches (if 'proxies' key exists in state_dict)
        proxy_count_mismatch = False
        if 'proxies' in state_dict and n_proxy is not None:
            loaded_proxy_shape = state_dict['proxies'].shape
            # proxies shape is [C * K, embed_dim]
            loaded_n_proxy = loaded_proxy_shape[0] // self.custom_loss.C  # Assuming same num_classes
            if loaded_n_proxy != n_proxy:
                proxy_count_mismatch = True
                print(f"Warning: Proxy count mismatch! Loaded: {loaded_n_proxy} proxies per class, "
                      f"Current: {n_proxy} proxies per class")
        
        
        if load_mode == 'full':
            if proxy_count_mismatch:
                print("Warning: Attempting 'full' load with mismatched proxy count. This will fail.")
                print("Falling back to 'proj_only' mode...")
                load_mode = 'proj_only'
            else:
                try:
                    self.custom_loss.load_state_dict(state_dict)
                    print("Loaded MPC: full state (proj + proxies)")
                    return  # Successfully loaded, exit early
                except RuntimeError as e:
                    print(f"Failed to load full state: {e}")
                    print("Falling back to 'proj_only' mode...")
                    load_mode = 'proj_only'
        
        if load_mode == 'proj_only':
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
            
            if 'proxies' in state_dict and n_proxy is not None:
                self._smart_init_proxies(state_dict['proxies'], n_proxy)
            return  # Finished loading in proj_only mode
        elif load_mode == 'none':
            print("Loaded MPC: none (keeping random initialization)")
            return  # Finished loading in none mode
        else:
            raise ValueError(f"Unknown load_mode: {load_mode}. Use 'full', 'proj_only', 'none', or 'auto'")
    
    def _smart_init_proxies(self, loaded_proxies, target_n_proxy):
        C = self.custom_loss.C  # num_classes
        embed_dim = loaded_proxies.shape[1]  # Get from loaded_proxies shape
        loaded_n_proxy = loaded_proxies.shape[0] // C
        
        if loaded_n_proxy == target_n_proxy:
            # Same count, just copy
            self.custom_loss.proxies.data = loaded_proxies.clone()
            print(f"Loaded proxies: {loaded_n_proxy} proxies per class (same count)")
            return
        
        # Reshape loaded proxies: [C, K1, embed_dim]
        loaded_proxies_reshaped = loaded_proxies.view(C, loaded_n_proxy, embed_dim)
        
        # Initialize target proxies: [C, K2, embed_dim]
        target_proxies = torch.zeros(C, target_n_proxy, embed_dim, device=self.device)
        
        for c in range(C):
            # Copy existing proxies
            target_proxies[c, :loaded_n_proxy] = loaded_proxies_reshaped[c]
            
            # For additional proxies, use interpolation or add small noise
            if target_n_proxy > loaded_n_proxy:
                # Method: Average of existing proxies + small random noise
                mean_proxy = loaded_proxies_reshaped[c].mean(dim=0, keepdim=True)
                noise_scale = 0.1
                for k in range(loaded_n_proxy, target_n_proxy):
                    # Interpolate between mean and one of the existing proxies
                    alpha = (k - loaded_n_proxy + 1) / (target_n_proxy - loaded_n_proxy + 1)
                    base_proxy = loaded_proxies_reshaped[c][k % loaded_n_proxy]
                    new_proxy = (1 - alpha) * base_proxy + alpha * mean_proxy.squeeze(0)
                    # Add small noise
                    noise = torch.randn_like(new_proxy) * noise_scale
                    target_proxies[c, k] = new_proxy + noise
        
        # Normalize and assign
        target_proxies = target_proxies.view(C * target_n_proxy, embed_dim)
        target_proxies = torch.nn.functional.normalize(target_proxies, dim=1)
        self.custom_loss.proxies.data = target_proxies
        
        print(f"Initialized proxies: {loaded_n_proxy} -> {target_n_proxy} per class")

    
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
        
        loss = F.binary_cross_entropy_with_logits(scaled_logits, labels, reduction='mean')
        
        grad = torch.autograd.grad(
            loss, 
            [scale], 
            create_graph=True, 
            retain_graph=True
        )[0]
        
        grad = torch.clamp(grad, -10.0, 10.0)
        penalty = (grad ** 2).mean()
        
        return penalty
    
    def train(self, X_train, X_val, y_train, y_val, env_train, env_val, epochs=50):
        train_loader, val_loader = self.create_dataloaders(X_train, X_val, y_train, y_val, env_train, env_val)
        best_model_path = None
        patience_counter = 0  

        num_envs = len(np.unique(env_train))

        interval = 20

        env_scales = [torch.tensor(1.0, requires_grad=True, device=self.device) for _ in range(num_envs)]
        
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + 
            list(self.custom_loss.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        print(f"Initialized IRM scales (fixed at 1.0, standard IRM): {[s.item() for s in env_scales]}")
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            step = 0

            if self.use_progressive_irm:
                if epoch < self.penalty_warmup_epochs:
                    effective_penalty_weight = 0.0
                elif epoch < self.penalty_warmup_epochs * 2:
                    progress = (epoch - self.penalty_warmup_epochs) / self.penalty_warmup_epochs
                    effective_penalty_weight = self.penalty_weight * progress
                else:
                    effective_penalty_weight = self.penalty_weight
            else:
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
                all_env_cls_losses = []  # For diagnostic: per-sample loss per environment
                total_samples = 0

                for i in range(num_envs):
                    mask = (env == i)
                    if mask.any():
                        env_labels = labels[mask]
                        env_features = features[mask]
                        env_logits = self.model.mlp_model(env_features)
                        env_outputs = self.model.pred(env_logits)  # Apply softmax for classification
                        cls_loss = self.criterion(env_outputs, env_labels)
                        scale = env_scales[i]
                        penalty = self._compute_irm_penalty(env_logits[:, 1], env_labels.float(), scale)

                        n = env_labels.size(0)
                        env_loss = cls_loss * n
                        all_env_losses.append(env_loss)
                        all_env_penalties.append(penalty)
                        all_env_cls_losses.append(cls_loss.detach())  # Per-sample loss for this environment
                        total_samples += n

                
                if all_env_losses:
                    total_env_loss = torch.stack(all_env_losses).sum() / total_samples
                    
                    # Normalize penalty across environments to stabilize its scale
                    total_penalty = torch.stack(all_env_penalties)
                    
                    # Additional stabilization: normalize penalty to similar scale as other losses
                    # This helps balance the optimization
                    if total_penalty.std() > 1e-8:
                        # Normalize if there's variation
                        penalty_mean = total_penalty.mean()
                        penalty_std = total_penalty.std()
                        total_penalty_normalized = (total_penalty - penalty_mean) / (penalty_std + 1e-8)
                        total_penalty = penalty_mean + 0.1 * total_penalty_normalized.mean()  # Soft normalization
                    else:
                        total_penalty = total_penalty.mean()
                    
                    total_loss = total_env_loss + effective_penalty_weight * total_penalty + con_loss

                    total_loss.backward()
                    self.optimizer.step()
                    
                    for scale in env_scales:
                        scale.data.fill_(1.0)

                    epoch_loss += total_loss.item()

                    if step % interval == 0:
                        print(
                            f"Epoch {epoch+1}/{epochs}, Step {step}/{len(train_loader)}, "
                            f"Cls Loss: {cls_loss.item():.4f}, Con Loss: {con_loss.item():.4f}, "
                            f"IRM Penalty: weight {effective_penalty_weight:.2f}, penalty {total_penalty.item():.6f}, "
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
                patience_counter = 0  # Reset patience counter
                if best_model_path:
                    os.remove(best_model_path)
                best_model_path = self.save_model(epoch, val_metrics)
                print(f"  -> Saved best model (val F1: {self.best_f1:.4f})")
            elif best_model_path is None:
                # Save first model if no model has been saved yet (safety measure)
                best_model_path = self.save_model(epoch, val_metrics)
                self.best_f1 = val_metrics['f1']
                print(f"  -> Saved initial model (val F1: {self.best_f1:.4f})")
            else:
                # No improvement
                patience_counter += 1
                if self.early_stop_patience > 0 and patience_counter >= self.early_stop_patience:
                    print(f"  -> Early stopping triggered (no improvement for {patience_counter} epochs)")
                    print(f"  -> Best validation F1: {self.best_f1:.4f}")
                    break
                
        if best_model_path is None:
            # Fallback: save the last model if no model was saved (should not happen)
            print("Warning: No model was saved during training, saving last model as fallback")
            best_model_path = self.save_model(epochs - 1, val_metrics)
        
        print(f"\nTraining completed. Best validation F1: {self.best_f1:.4f}")
        return best_model_path

    @staticmethod
    def load_model(model_path, model_class, input_size, device='cuda'):
        # Load full checkpoint so we can reuse both model and proxy (MPC) parameters
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model = model_class(input_size=input_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Check if this is a multi-proxy stage1 checkpoint
        if 'env_losses_state_dict' in checkpoint:
            env_losses_state_dict = checkpoint['env_losses_state_dict']
            if len(env_losses_state_dict) > 0:
                custom_loss_state_dict = St2ModelTrainer._fuse_multi_proxy_stage1(
                    env_losses_state_dict
                )
                n_proxy = checkpoint.get('n_proxy', None)
                return model, custom_loss_state_dict, n_proxy
        
        # Single-proxy stage1 or regular checkpoint
        custom_loss_state_dict = checkpoint.get('custom_loss_state_dict', None)
        n_proxy = None
        if custom_loss_state_dict is not None and 'proxies' in custom_loss_state_dict:
            # Extract n_proxy from proxies shape: [C*K, embed_dim]
            num_classes = 2  # Assuming binary classification
            proxy_shape = custom_loss_state_dict['proxies'].shape
            n_proxy = proxy_shape[0] // num_classes
        
        return model, custom_loss_state_dict, n_proxy
    
    @staticmethod
    def _fuse_multi_proxy_stage1(env_losses_state_dict):

        env_ids = sorted(env_losses_state_dict.keys())
        num_envs = len(env_ids)
                
        # Get the first environment's state dict as template
        first_env_state = env_losses_state_dict[env_ids[0]]
        fused_state = {}
        
        # Copy projection layer (same for all environments, represents learned feature transformation)
        for key, value in first_env_state.items():
            if key.startswith('proj'):
                fused_state[key] = value.clone()
                print(f"  Loaded {key} from environment {env_ids[0]}")
        
        # Extract invariant proxies from all environments' knowledge
        if 'proxies' in first_env_state:
            all_proxies = []
            for env_id in env_ids:
                env_proxies = env_losses_state_dict[env_id]['proxies']
                all_proxies.append(env_proxies)
                print(f"  Environment {env_id}: proxies shape {env_proxies.shape}")
            
            # Stack: [num_envs, C*K, embed_dim]
            all_proxies = torch.stack(all_proxies, dim=0)
            
            fused_proxies = all_proxies.mean(dim=0)
            print(f"  Extracted invariant proxies: mean (common center) of {num_envs} environments")
            
            # Normalize to unit sphere
            fused_proxies = torch.nn.functional.normalize(fused_proxies, dim=1)
            fused_state['proxies'] = fused_proxies
            print(f"  Final extracted invariant proxies shape: {fused_proxies.shape}")
        else:
            print("  Warning: No proxies found in environment state dicts")
        
        return fused_state
    