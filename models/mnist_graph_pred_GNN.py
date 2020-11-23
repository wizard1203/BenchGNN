import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool

from torch_geometric.nn.conv import GCNConv



class mnist_graph_pred_GNN(torch.nn.Module):
    def __init__(self, in_feats, hidden_sizes: list, drop_ratio=0.5, gnn_type=None):
        super(mnist_graph_pred_GNN, self).__init__()

        self.in_feats = in_feats
        self.hidden_sizes = hidden_sizes
        self.conv_list = []
        self.conv_list.append(GCNConv(in_feats, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.conv_list.append(GCNConv(hidden_sizes[i-1], hidden_sizes[i]))

        # Maybe use fc is a little tricky
        # self.fc1 = nn.Linear(784, 128)
        # self.fc2 = nn.Linear(128, 10)
        self.classifier = self.get_classifier()


    def get_classifier(self):
        # 10 classes of numbers
        return nn.Linear(784, 10)


    # def forward(self, data):
    #     h = self.conv1(g, inputs)
    #     h = torch.relu(h)
    #     h = self.conv2(g, h)
    #     return h

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = \
            batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        for conv in self.conv_list:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

        x = self.classifier(x)

        print("x.shape:", x.shape)
        output = F.log_softmax(x, dim=1)
        return output



















