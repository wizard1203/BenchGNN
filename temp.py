from data_prepare import dataset_prepare
import networkx as nx

# g = dgl.DGLGraph()
# g.add_nodes(10)
g = nx.Graph()
g.add_nodes_from([i for i in range(10)])

for i in range(1, 4):
    g.add_edge(i, 0)


dataset_name = 'mnist_graph_pred_GNN_prepare'
# dataset_name = 'mnist_node_pred_GNN_prepare'
# dataset_name = 'mnist_CNN_prepare'
# dataset_name = 'molhiv'

data_dir = '.'

train_loader, test_loader = dataset_prepare(dataset_name, data_dir)























