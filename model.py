import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# ===============================
# Load configuration (Issue #1)
# ===============================
with open("config/model_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

LOW_THRESHOLD = cfg.get("low_threshold", 0.45)
HIGH_THRESHOLD = cfg.get("high_threshold", 0.55)
HIDDEN_DIM = cfg.get("hidden_dim", 64)
DROPOUT_RATE = cfg.get("dropout_rate", 0.2)
NUM_CLASSES = cfg.get("num_classes", 1)

# ===============================
# GNN Model Definition (Issue #3 / #5)
# ===============================
class GNNClassifier(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, dropout_rate=DROPOUT_RATE):
        super(GNNClassifier, self).__init__()
        self.conv1 = GCNConv(1, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, NUM_CLASSES)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return torch.sigmoid(self.classifier(x))

# ===============================
# Risk Classification Function (Issue #1 & #5)
# ===============================
def classify_risk(prob_tensor):
    """
    Convert sigmoid outputs into LOW / MEDIUM / HIGH risk
    using consistent thresholds.
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
# Vectorized Batch Inference (Issue #2)
# ===============================
def batch_classify(model, data_list, device='cpu'):
    """
    Perform vectorized inference over multiple drug graphs.

    Args:
        model: trained GNNClassifier
        data_list: list of PyG Data objects (drug pair graphs)
        device: 'cpu' or 'cuda'

    Returns:
        tuple: (torch.Tensor of probabilities, list of risk labels)
    """
    from torch_geometric.loader import DataLoader

    model.eval()
    loader = DataLoader(data_list, batch_size=len(data_list))  # all at once
    probs_all = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            probs_all.append(out)

    probs_tensor = torch.cat(probs_all, dim=0)  # shape: [num_graphs, 1]
    risk_labels = classify_risk(probs_tensor)

    return probs_tensor, risk_labels

# ===============================
# Sanity Test
# ===============================
if __name__ == "__main__":
    # Dummy test for classify_risk
    dummy_output = torch.tensor([[0.2], [0.5], [0.7]])
    print(f"Thresholds → LOW: {LOW_THRESHOLD}, HIGH: {HIGH_THRESHOLD}")
    print("Risk categories:", classify_risk(dummy_output))

    # Dummy test for batch_classify
    from torch_geometric.data import Data
    data_list = []
    for i in range(3):
        num_nodes = 4
        x = torch.rand((num_nodes, 1))
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        data_list.append(Data(x=x, edge_index=edge_index))
    model = GNNClassifier()
    probs, risks = batch_classify(model, data_list)
    print("Batch probabilities:", probs.squeeze().tolist())
    print("Batch risk labels:", risks)
