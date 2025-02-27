import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Multi-Proxy Contrastive Learning

class MPL(nn.Module):
    def __init__(self, device, num_classes=2, n_proxy=5, proxy_m=0.99, temp=0.05, lambda_pcon=1, k=5, feat_dim=200, epsilon=0.05):
        super(MPL, self).__init__()
        self.num_classes = num_classes
        self.temp = temp  
        self.lambda_pcon = lambda_pcon  
        self.device = device
        self.feat_dim = feat_dim
        self.epsilon = epsilon
        self.sinkhorn_iterations = 5  
        self.k = k  
        self.n_proxy = n_proxy
        self.proxy_m = proxy_m   

        total_proxy = self.num_classes * self.n_proxy
        self.register_buffer("proxy", torch.rand(total_proxy, feat_dim))
        self.proxy = F.normalize(self.proxy, dim=-1)  


    def sinkhorn(self, features, class_proxy):
        proxy = class_proxy.to(features.device)
        out = torch.matmul(features, proxy.T)  # (B, num_class_proxy)

        out = torch.clamp(out, min=-10.0, max=10.0)

        u = torch.ones(out.size(0)).to(features.device)
        v = torch.ones(out.size(1)).to(features.device)

        for _ in range(self.sinkhorn_iterations):
            u = 1.0 / torch.matmul(out, v).clamp(min=1e-10)  
            v = 1.0 / torch.matmul(out.T, u).clamp(min=1e-10)  

        Q = torch.matmul(torch.diag(u), torch.exp(out / self.epsilon)).matmul(torch.diag(v))
        Q = Q / Q.sum()  
        return Q  


    def mle_loss(self, features, targets):
        loss = 0
       
        for c in range(self.num_classes):
            class_idx = (targets == c).nonzero(as_tuple=True)[0]
            if len(class_idx) == 0:
                continue
            class_features = features[class_idx].to(self.device)  
            # print(f"class_features: {class_features.shape}")
            num_class_proxy = self.n_proxy 
            class_proxy = self.proxy[c * num_class_proxy:(c+1) * num_class_proxy].to(self.device) 
            # print(f"len(class_proxy): {len(class_proxy)}")
            
            # proability between the sample with multiple proxies
            W_c = self.sinkhorn(class_features, class_proxy) 
            # print(f"W_c: {W_c.shape}") 

            if self.k > 0:
                _, topk_idx = torch.topk(W_c, self.k, dim=1)
                topk_mask = torch.zeros_like(W_c).to(self.device)
                topk_mask.scatter_(1, topk_idx, 1)
                W_c = W_c * topk_mask
                # print(f"topk_mask: {topk_mask}, and W_c: {W_c.shape}")

            proxy_dis = torch.matmul(class_features, class_proxy.T) 
            proxy_dis = torch.clamp(proxy_dis, min=-10.0, max=10.0)  
            logits = torch.div(proxy_dis, self.temp)
            logits = F.log_softmax(logits, dim=1)

            pos = torch.sum(W_c * logits, dim=1)
            neg = torch.logsumexp(logits, dim=1) 
            log_prob = pos - neg
            loss += -torch.mean(log_prob) 
        
        return loss
    
    def proxy_contra(self):
        loss = 0
        num_class_proxy = self.n_proxy
        
        # Proxy-to-proxy contrast
        for c in range(self.num_classes):
            class_proxy = self.proxy[c * num_class_proxy:(c + 1) * num_class_proxy].to(self.device)

            # Intra-class proxy distance minimization
            proxy_sim_intra = torch.matmul(class_proxy, class_proxy.T)  
            proxy_sim_intra = proxy_sim_intra / self.temp  
            proxy_sim_intra = torch.clamp(proxy_sim_intra, min=-10.0, max=10.0)  
            logits_intra = F.log_softmax(proxy_sim_intra, dim=1)  
            
            pos_mask_intra = torch.eye(num_class_proxy).to(self.device)  
            pos_intra = torch.sum(pos_mask_intra * logits_intra, dim=1)
            neg_intra = torch.logsumexp(logits_intra, dim=1)  
            log_prob_intra = pos_intra - neg_intra
            loss += -torch.mean(log_prob_intra)  

            # Inter-class proxy distance maximization
            for c_other in range(self.num_classes):
                if c != c_other:
                    class_proxy_other = self.proxy[c_other * num_class_proxy:(c_other + 1) * num_class_proxy].to(self.device)
                    proxy_sim_inter = torch.matmul(class_proxy, class_proxy_other.T)
                    proxy_sim_inter = proxy_sim_inter / self.temp
                    proxy_sim_inter = torch.clamp(proxy_sim_inter, min=-10.0, max=10.0)
                    logits_inter = F.log_softmax(proxy_sim_inter, dim=1)
                    neg_inter = torch.logsumexp(logits_inter, dim=1)  
                    loss += torch.mean(neg_inter)  # Maximize the negative log-prob

        return loss
    
    def proxy_updating(self, features, targets):
        for c in range(self.num_classes):
            class_idx = (targets == c).nonzero(as_tuple=True)[0]
            if len(class_idx) == 0:
                continue
            class_features = features[class_idx].detach()
            num_class_proxy = self.n_proxy  
            class_proxy = self.proxy[c * num_class_proxy:(c+1) * num_class_proxy].to(self.device)

            W_c = self.sinkhorn(class_features, class_proxy)  

            if self.k > 0:
                _, topk_idx = torch.topk(W_c, self.k, dim=1)
                topk_mask = torch.zeros_like(W_c).to(self.device)
                topk_mask.scatter_(1, topk_idx, 1)
                W_c = W_c * topk_mask
            proxy_update = torch.matmul(W_c.T, class_features)  
            new_proxy = self.proxy_m * class_proxy + (1 - self.proxy_m) * proxy_update
            self.proxy[c * num_class_proxy:(c+1) * num_class_proxy] = F.normalize(new_proxy, dim=1)

    def forward(self, features, targets):
        features = F.normalize(features, dim=-1)

        mle_loss = self.mle_loss(features, targets)

        # total_loss = mle_loss
        
        proxy_contra_loss = self.proxy_contra()

        # print(f"mle_loss: {mle_loss}")
        # print(f"proxy_contra_loss: {proxy_contra_loss}")

        self.proxy_updating(features, targets)
        
        total_loss = mle_loss + self.lambda_pcon * proxy_contra_loss
        total_loss = total_loss / self.num_classes

        return total_loss


# test



def test_mpl_single_sample():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MPL(device=device).to(device)
    
    feat_dim = model.feat_dim  
    num_classes = model.num_classes  
    
    features = torch.randn(1, feat_dim).to(device)  
    targets = torch.tensor([0]).to(device)  
    
    loss = model(features, targets)
    print(f"Loss: {loss.item()}")

def test_mpl_batch_sample():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MPL(device=device).to(device)
    
    batch_size = 4
    feat_dim = model.feat_dim  
    num_classes = model.num_classes  
    
    features = torch.randn(batch_size, feat_dim).to(device)  
    targets = torch.randint(0, num_classes, (batch_size,)).to(device)  
    
    loss = model(features, targets)
    print(f"Loss (batch): {loss.item()}")




# test_mpl_single_sample()
# test_mpl_batch_sample()