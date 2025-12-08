import torch
import torch.nn as nn
import torch.nn.functional as F

class MPC(nn.Module):
    def __init__(self,
                 device,
                 input_dim=1000,
                 embed_dim=128,
                 num_classes=2,
                 n_proxy=5,
                 tau=0.07,
                 margin=1.5,
                 lambda_margin=0.05,
                 lambda_div=0.01):
        super().__init__()
        
        if input_dim != embed_dim:
            self.proj = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            self.proj = None
        
        self.device = device
        self.C = num_classes
        self.K = n_proxy
        self.tau = tau
        self.margin = margin
        self.lambda_margin = lambda_margin
        self.lambda_div = lambda_div
        
        # Proxies 
        buf = torch.randn(self.C * self.K, embed_dim)
        self.proxies = nn.Parameter(F.normalize(buf, dim=1))
        
    def forward(self, x, labels):
        # (1) normalize
        if self.proj:
            z = self.proj(x)            # [B, E]
            z = F.normalize(z, dim=1)
        else:
            z = F.normalize(x, dim=1)
        
        # reshape proxies to [C, K, E]
        P = F.normalize(self.proxies, dim=1).view(self.C, self.K, -1)
        
        # (2) PAL Soft Loss
        pal, cnt = 0.0, 0
        for c in range(self.C):
            idx = (labels == c).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            
            Fc = z[idx]         # [B_c, E]
            Pc = P[c]           # [K, E]
            
            sim = Fc @ Pc.t() / self.tau     # [B_c, K]
            
            # soft assignment
            p_soft = F.softmax(sim, dim=1)  # [B_c, K]
            logp = F.log_softmax(sim, dim=1)
            
            per_sample_loss = -(p_soft * logp).sum(dim=1)   # [B_c]
            
            pal += per_sample_loss.mean()
            cnt += 1
        
        pal = pal / max(cnt, 1)
        
        # (3) Inter-class margin loss (only centers)
        centers = P.mean(dim=1)      # [C, E]
        inter_mgn, pairs = 0.0, 0
        for c1 in range(self.C):
            for c2 in range(self.C):
                if c1 == c2:
                    continue
                dist = F.pairwise_distance(
                    centers[c1].unsqueeze(0),
                    centers[c2].unsqueeze(0),
                    p=2
                )
                
                inter_mgn += F.relu(self.margin - dist)
                # print(f"margin = {self.margin}, dist = {dist.item():.6f}, penalty = {F.relu(self.margin - dist).item():.6f}")

                pairs += 1
        inter_mgn = inter_mgn / max(pairs, 1)
        
        # (4) Intra-class diversity loss
        diversity_loss, div_cnt = 0.0, 0
        for c in range(self.C):
            pc = P[c]        # [K, E]
            if self.K < 2:
                continue
            dists = torch.pdist(pc, p=2)   # pairwise distances
            diversity_loss += dists.mean()
            div_cnt += 1
        diversity_loss = diversity_loss / max(div_cnt, 1)
        
        # (5) total loss
        loss = pal + \
               self.lambda_margin * inter_mgn + \
               self.lambda_div * diversity_loss
        

        return loss