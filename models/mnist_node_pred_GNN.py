import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool

from torch_geometric.nn.conv import GCNConv



class mnist_node_pred_GNN(torch.nn.Module):
    def __init__(self, in_feats, hidden_sizes: list, drop_ratio=0.5, gnn_type=None):
        super(mnist_node_pred_GNN, self).__init__()

        self.in_feats = in_feats
        self.hidden_sizes = hidden_sizes
        self.conv_list = []
        self.conv_list.append(GCNConv(in_feats, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.conv_list.append(GCNConv(hidden_sizes[i-1], hidden_sizes[i]))

        self.classifier = self.get_classifier()


    def get_classifier(self):
        # 10 classes of numbers
        return GCNConv(self.hidden_sizes[-1], 10)


    # def forward(self, data):
    #     h = self.conv1(g, inputs)
    #     h = torch.relu(h)
    #     h = self.conv2(g, h)
    #     return h

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = \
            batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        print("x.shape:", x.shape)
        x = x.reshape([-1, 784])
        for conv in self.conv_list:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

        x = self.classifier(x, edge_index)

        print("x.shape:", x.shape)
        output = F.log_softmax(x, dim=1)
        return output












