import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# Load configuration
with open("config/model_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

class GNNClassifier(nn.Module):
    def __init__(self, hidden_dim=cfg["hidden_dim"], dropout_rate=cfg["dropout_rate"]):
        super(GNNClassifier, self).__init__()
        self.conv1 = GCNConv(1, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, cfg["num_classes"])
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return torch.sigmoid(self.classifier(x))
