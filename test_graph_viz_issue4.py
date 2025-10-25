from model import GNNClassifier
from torch_geometric.data import Data
from utils.graph_viz import grad_cam_node_importance, plot_graph_with_importance
import torch

# Create a dummy molecular graph
x = torch.rand((4,1))  # 4 nodes, 1 feature each
edge_index = torch.tensor([[0,1,2],[1,2,3]], dtype=torch.long)
data = Data(x=x, edge_index=edge_index)

# Load model
model = GNNClassifier()

# Compute node importance
node_importance = grad_cam_node_importance(model, data)
print("Node importance scores:", node_importance)

# Visualize the graph with node importance
plot_graph_with_importance(data, node_importance)
