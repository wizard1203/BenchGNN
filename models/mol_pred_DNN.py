import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder


class Mol_pred_DNN(torch.nn.Module):
    def __init__(self, in_feats=502*3+222*9, hidden_sizes: list=[], drop_ratio=0.25):
        super(Mol_pred_DNN, self).__init__()
        self.drop_ratio = drop_ratio
        self.in_feats = in_feats
        self.hidden_sizes = hidden_sizes

        # emb_dim = 300
        # self.atom_encoder = AtomEncoder(emb_dim)
        # self.bond_encoder = BondEncoder(emb_dim)

        self.linear_list = []
        self.batch_norms = []
        self.linear_list.append(nn.Linear(in_feats, hidden_sizes[0]).cuda())
        self.batch_norms.append(nn.BatchNorm1d(hidden_sizes[0]).cuda())

        for i in range(1, len(hidden_sizes)):
            self.linear_list.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]).cuda())
            self.batch_norms.append(nn.BatchNorm1d(hidden_sizes[i]).cuda())

        # Maybe use fc is a little tricky
        # self.fc1 = nn.Linear(784, 128)
        # self.fc2 = nn.Linear(128, 10)

        self.classifier = self.get_classifier()


    def get_classifier(self):
        # 10 classes of numbers
        return nn.Linear(self.hidden_sizes[-1], 1)


    # def forward(self, data):
    #     h = self.conv1(g, inputs)
    #     h = torch.relu(h)
    #     h = self.conv2(g, h)
    #     return h

    def forward(self, x):
    # def forward(self, batched_data):
        # x, edge_index, edge_attr, batch = \
        #     batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        # x = data[0], edge_attr = data[1]

        # x = self.atom_encoder(x)
        # edge_attr = self.bond_encoder(edge_attr)
        # x = torch.cat([x, edge_attr], dim=0)
            

        for i, linear in enumerate(self.linear_list):
            x = linear(x)
            if x.shape[0] != 1:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, self.drop_ratio, training=self.training)
        # print("x.shape:", x.shape)

        output = self.classifier(x)

        # print("x.shape:", x.shape)
        # output = F.log_softmax(x, dim=1)
        return output



























