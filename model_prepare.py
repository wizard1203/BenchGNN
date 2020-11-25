from models import Mnist_graph_pred_GNN
from models import Mnist_node_pred_GNN
# from models import Mnist_CNN

def get_model(model_name):
    if model_name == 'mnist_graph_pred_GNN':
        in_feats = 1
        hidden_sizes = [2, 4, 2, 1]
        model = Mnist_graph_pred_GNN(in_feats, hidden_sizes)
    elif model_name == 'mnist_node_pred_GNN':
        in_feats = 784
        hidden_sizes = [300, 100, 30]
        model = Mnist_node_pred_GNN(in_feats, hidden_sizes)
    elif model_name == 'mnist_CNN':
        pass
    else:
        raise NotImplementedError("Not implemented!")

    return model










