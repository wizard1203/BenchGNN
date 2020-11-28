from .mnist_graph_pred_GNN import Mnist_graph_pred_GNN
from .mnist_node_pred_GNN import Mnist_node_pred_GNN
from .mol_pred_DNN import Mol_pred_DNN
from .mol_pred_GNN import Mol_pred_GNN
from .mnist_CNN import Mnist_CNN

__all__ = [
    'Mnist_graph_pred_GNN',
    'Mnist_node_pred_GNN',
    'Mol_pred_DNN',
    'Mol_pred_GNN',
    'Mnist_CNN',
]

classes = __all__
