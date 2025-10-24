import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# ===============================
# Load consistent thresholds from config
# ===============================
with open("config/model_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

LOW_THRESHOLD = cfg.get("low_threshold", 0.45)
HIGH_THRESHOLD = cfg.get("high_threshold", 0.55)

# ===============================
# GNN Model Definition
# ===============================
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

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return torch.sigmoid(self.classifier(x))

# ===============================
# Risk Classification Function
# ===============================
def classify_risk(prob_tensor):
    """
    Convert model outputs into LOW / MEDIUM / HIGH risk levels
    based on consistent global thresholds.
    """
    categories = []
    probs = prob_tensor.detach().cpu()
    for p in probs:
        p_val = p.item() if isinstance(p, torch.Tensor) else float(p)
        if p_val < LOW_THRESHOLD:
            categories.append("LOW")
        elif p_val > HIGH_THRESHOLD:
            categories.append("HIGH")
        else:
            categories.append("MEDIUM")
    return categories

# ===============================
# Sanity Test
# ===============================
if __name__ == "__main__":
    dummy_output = torch.tensor([[0.2], [0.5], [0.7]])
    print(f"Thresholds → LOW: {LOW_THRESHOLD}, HIGH: {HIGH_THRESHOLD}")
    print("Risk categories:", classify_risk(dummy_output))
