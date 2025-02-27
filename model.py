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

    def forward(self, x):
        x = self.encoder_model(x)
        feature = x
        x = self.mlp_model(x)
        x = self.pred(x)
        return x, feature

# class DrebinMLP(nn.Module):
#     """
#     Network architecture used by Grosse et al. in the paper
#     'Adversarial Examples for Malware Detection'

#     Modifications:
#     * Adjusted layer sizes to reduce overfitting
#     * Added Batch Normalization for stable training
#     * Changed Dropout from 0.5 to 0.3 to balance regularization
#     * Used LeakyReLU to prevent dying neurons
#     """
#     def __init__(self, input_size, output_size=2):
#         super(DrebinMLP, self).__init__()
#         self.encoder_model = nn.Sequential(
#             nn.Linear(input_size, 200),
#             nn.BatchNorm1d(200),  # Batch Normalization for stable training
#             nn.LeakyReLU(0.1),  # Prevent dead neurons
#             nn.Dropout(0.3),  # Reduce from 0.5 to 0.3

#             nn.Linear(200, 150),
#             nn.BatchNorm1d(150),
#             nn.LeakyReLU(0.1),
#             nn.Dropout(0.3),

#             nn.Linear(150, 100),
#             nn.BatchNorm1d(100),
#             nn.LeakyReLU(0.1),
#             nn.Dropout(0.3),

#             nn.Linear(100, 50),
#             nn.BatchNorm1d(50),
#             nn.LeakyReLU(0.1),
#             nn.Dropout(0.3),
#         )
#         self.mlp_model = nn.Linear(50, output_size)
#         self.pred = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.encoder_model(x)
#         feature = x  # Extracted feature representation
#         x = self.mlp_model(x)
#         x = self.pred(x)
#         return x, feature