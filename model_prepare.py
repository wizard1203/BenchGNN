from models import Mnist_graph_pred_GNN
from models import Mnist_node_pred_GNN
from models import Mol_pred_DNN
from models import Mol_pred_GNN
from models import Mnist_CNN

def get_model(model_name):
    if model_name == 'mnist_graph_pred_GNN':
        in_feats = 1
        hidden_sizes = [2, 4, 2, 1]
        # hidden_sizes = [4, 8, 16, 8, 4, 1]
        model = Mnist_graph_pred_GNN(in_feats, hidden_sizes)

    elif model_name == 'mnist_node_pred_GNN':
        in_feats = 784
        hidden_sizes = [300, 100, 30]
        # hidden_sizes = [300, 200, 100, 30]
        model = Mnist_node_pred_GNN(in_feats, hidden_sizes)

    elif model_name == 'mol_pred_DNN':
        in_feats = 502*3+222*9
        hidden_sizes = [1000, 300, 100, 30, 10]
        model = Mol_pred_DNN(in_feats, hidden_sizes)

    elif model_name == 'mol_pred_GNN':
        model = Mol_pred_GNN()

    elif model_name == 'mnist_CNN':
        model = Mnist_CNN()
    else:
        raise NotImplementedError("Not implemented!")
    return model










