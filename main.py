# import dgl
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch

# import graph as graph
from data_prepare import dataset_prepare
from trainer import Trainer


# Not used
def dgl_test():
    # Since the actual graph is undirected, we convert it for visualization
    # purpose.
    G = graph.build_karate_club_graph()
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())

    nx_G = G.to_networkx().to_undirected()
    # Kamada-Kawaii layout usually looks pretty for arbitrary graphs
    pos = nx.kamada_kawai_layout(nx_G)

    inputs, labeled_nodes, labels = data_prepare()
    # trainer.train(G, inputs, labeled_nodes, labels)

    # nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
    # plt.show()

if __name__ == '__main__':

    # train_loader, test_loader = dataset_prepare('mnist_graph_pred_GNN_prepare', data_dir='.')
    # trainer = Trainer(model_name='mnist_graph_pred_GNN', optim_name=None, criterion=None,
    #         train_loader=train_loader, test_loader=test_loader)
    # trainer.train()

    # mnist_node_pred_GNN_dataset = dataset_prepare('mnist_node_pred_GNN_prepare', data_dir='.')
    # trainer = Trainer(model_name='mnist_node_pred_GNN', optim_name=None, criterion=None,
    #         train_loader=mnist_node_pred_GNN_dataset.data, test_loader=None)
    # trainer.train()

    # train_loader, test_loader = dataset_prepare('mol_pred_DNN_prepare', data_dir='.')
    # trainer = Trainer(model_name='mol_pred_DNN', optim_name=None, criterion=None,
    #         train_loader=train_loader, test_loader=test_loader)
    # trainer.train()

    train_loader, test_loader = dataset_prepare('mol_pred_GNN_prepare', data_dir='.')
    trainer = Trainer(model_name='mol_pred_GNN', optim_name=None, criterion=None,
            train_loader=train_loader, test_loader=test_loader)
    trainer.train()

    # train_loader, test_loader = dataset_prepare('mnist_CNN_prepare', data_dir='.')
    # trainer = Trainer(model_name='mnist_CNN', optim_name=None, criterion=None,
    #         train_loader=train_loader, test_loader=test_loader)
    # trainer.train()



 
























