import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNClassifier(nn.Module):
    def __init__(self, hidden_dim=64):
        super(GNNClassifier, self).__init__()
        self.conv1 = GCNConv(1, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        # TODO: consider adding dropout layers to prevent overfitting

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        # NOTE: global_mean_pool might not capture complex subgraph interactions
        x = global_mean_pool(x, batch)
        return torch.sigmoid(self.classifier(x))
        # FIXME: maybe add option to return logits instead of sigmoid for flexible loss functions

# REVIEW: test model with batch size >1 to ensure batch pooling works correctly
# TODO: add unit test for GNN forward pass with dummy graph data
