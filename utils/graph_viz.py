import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

def grad_cam_node_importance(model, data, device='cpu'):
    model.eval()
    data = data.to(device)
    data.x.requires_grad = True

    output = model(data)
    score = output[0,0]

    model.zero_grad()
    score.backward(retain_graph=True)

    node_grads = data.x.grad.abs().sum(dim=1)
    node_importance = node_grads / node_grads.max()

    return node_importance.detach().cpu()

def plot_graph_with_importance(data, node_importance):
    G = to_networkx(data, to_undirected=True)
    colors = node_importance.tolist()
    
    plt.figure(figsize=(6,6))
    nx.draw(G, with_labels=True, node_color=colors, cmap=plt.cm.Reds, node_size=500)
    plt.show()
