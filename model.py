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
        # Optional: consider adding dropout layers to prevent overfitting

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return torch.sigmoid(self.classifier(x))
        # Optionally return logits instead of sigmoid if needed

def classify_risk(prob_tensor, low_thresh=0.45, high_thresh=0.55):
    """
    Convert sigmoid output probabilities into LOW / MEDIUM / HIGH risk.

    Args:
        prob_tensor: torch.Tensor of shape (batch_size, 1)
        low_thresh: float, below which risk is LOW
        high_thresh: float, above which risk is HIGH

    Returns:
        List of strings: ["LOW", "MEDIUM", "HIGH"] for each example
    """
    categories = []
    probs = prob_tensor.detach().cpu()
    for p in probs:
        # If output is a tensor with shape (1,), get scalar value
        p_val = p.item() if isinstance(p, torch.Tensor) else float(p)
        if p_val < low_thresh:
            categories.append("LOW")
        elif p_val > high_thresh:
            categories.append("HIGH")
        else:
            categories.append("MEDIUM")
    return categories

# Example test (can remove in final version)
if __name__ == "__main__":
    # Dummy test
    dummy_output = torch.tensor([[0.2], [0.5], [0.7]])
    print("Risk categories:", classify_risk(dummy_output))
    # Expected: ['LOW', 'MEDIUM', 'HIGH']
