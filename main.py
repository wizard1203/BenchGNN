import dgl
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import gnn_creater as gnn
import graph as graph
from data_prepare import data_prepare
from trainer import Trainer


if __name__ == '__main__':

    # Since the actual graph is undirected, we convert it for visualization
    # purpose.
    G = graph.build_karate_club_graph()
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())

    nx_G = G.to_networkx().to_undirected()
    # Kamada-Kawaii layout usually looks pretty for arbitrary graphs
    pos = nx.kamada_kawai_layout(nx_G)

    inputs, labeled_nodes, labels = data_prepare()

    trainer = Trainer(model_name=None, optim_name=None, criterion='nll_loss')
    trainer.train(G, inputs, labeled_nodes, labels)

    # nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
    # plt.show()












