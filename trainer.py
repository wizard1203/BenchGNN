import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F



import gnn_creater as gnn
from data_prepare import data_prepare


class Trainer(object):
    def __init__(self, model_name, optim_name, criterion):
        super(Trainer, self).__init__()
        self.all_logits = []
        self.model = gnn.gnn_create(model_name)
        self.optimizer = self._create_optimizer(optim_name)
        self.criterion = self._create_criterion(criterion)


    def _create_optimizer(self, optim_name):
        return torch.optim.Adam(
            self.model.parameters(),
            # itertools.chain(
            #     self.model.parameters(),
            #     self.embed.parameters()
            # ), 
            lr=0.01)

    def _create_criterion(self, criterion):
        return F.nll_loss


    def train_one_step(self, graph, inputs, labeled_nodes, labels):

        logits = self.model(graph, inputs)
        # we save the logits for visualization later
        self.all_logits.append(logits.detach())
        logp = F.log_softmax(logits, 1)
        # we only compute loss for labeled nodes
        # loss = F.nll_loss(logp[labeled_nodes], labels)
        loss = self.criterion(logp[labeled_nodes], labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, graph, inputs, labeled_nodes, labels):

        for epoch in range(50):
            loss = self.train_one_step(graph, inputs, labeled_nodes, labels)

            print('Epoch %d | Loss: %.4f' % (epoch, loss))







