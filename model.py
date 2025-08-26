import torch.nn.functional as F
from torch import nn

class DrebinMLP(nn.Module):
    """
    Network architecture used by Grosse et al. in the paper
    'Adversarial Examples for Malware Detection'

    Modifications (don't change the architecture, only the definition): 
    * Splitting of the layers into backbone and classifier
    * Variable output size; target to project to a large-dim output space and train.
    """
    def __init__(
            self, 
            input_size,
            output_size=2
    ):
        super(DrebinMLP, self).__init__()
        self.encoder_model = nn.Sequential(
            nn.Linear(input_size, 200),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.mlp_model = nn.Linear(200, output_size)
        self.pred = nn.Softmax(dim=1)
        self.emb_dim = 200

    def forward(self, x):
        x = self.encoder_model(x)
        feature = x
        x = self.mlp_model(x)
        x = self.pred(x)
        return x, feature

class DrebinMLP_IRM(nn.Module):
    def __init__(
            self, 
            input_size,
            output_size=2
    ):
        super(DrebinMLP_IRM, self).__init__()
        self.emb_dim = 200

        self.encoder_model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.ReLU(),
        )
        
        self.mlp_model = nn.Linear(self.emb_dim, output_size)
        self.pred = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.encoder_model(x)
        feature = x
        x = self.mlp_model(x)
        x = self.pred(x)
        return x, feature


