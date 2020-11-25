import itertools

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F



from model_prepare import get_model



class Trainer(object):
    def __init__(self, model_name, optim_name, criterion):
        super(Trainer, self).__init__()
        self.all_logits = []
        self.model_name = model_name
        self.model = get_model(model_name)

        self.optimizer = self._create_optimizer(optim_name)
        self.criterion = self._create_criterion(criterion)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device=self.device)
        # self.model = self.model.cuda()


    def _create_optimizer(self, optim_name, lr=0.01, momentum=0.9):
        # return torch.optim.Adam(
        #     self.model.parameters(),
        #     # itertools.chain(
        #     #     self.model.parameters(),
        #     #     self.embed.parameters()
        #     # ), 
        #     lr=0.01)
        return torch.optim.SGD(
            self.model.parameters(), 
            lr=lr, 
            momentum=momentum)


    def _create_criterion(self, criterion):
        return F.nll_loss

    # used for DGL
    # def train_one_step(self, graph, inputs, labeled_nodes, labels):

    #     logits = self.model(graph, inputs)
    #     # we save the logits for visualization later
    #     self.all_logits.append(logits.detach())
    #     logp = F.log_softmax(logits, 1)
    #     # we only compute loss for labeled nodes
    #     # loss = F.nll_loss(logp[labeled_nodes], labels)
    #     loss = self.criterion(logp[labeled_nodes], labels)

    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    #     return loss.item()

    # used for PyG
    def train_one_step(self, data):
        if self.model_name == 'mnist_graph_pred_GNN':
            loss = self._train_one_step_Mnist_graph_pred_GNN(data)

        elif self.model_name == 'mnist_node_pred_GNN':
            loss = self._train_one_step_Mnist_node_pred_GNN(data)

        elif self.model_name == 'mnist_CNN':
            pass
        else:
            raise NotImplementedError("Not implemented!")

        return loss



    def _train_one_step_Mnist_graph_pred_GNN(self, graph, inputs, labeled_nodes, labels):
        pass


    def _train_one_step_Mnist_node_pred_GNN(self, data):
        x, edge_index = data.x, data.edge_index
        self.optimizer.zero_grad()
        out = self.model(x, edge_index)
        # we only compute loss for labeled nodes
        # loss = F.nll_loss(logp[labeled_nodes], labels)
        loss = self.criterion(out[data.train_mask], data.y[data.train_mask].long())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def train_with_one_sample(self, data):

        iteration = 10
        # data = data.to(device=self.device, dtype=torch.int64)
        data = data.to(device=self.device)
        # print(data.device)
        for epoch in range(50):
            self.model.train()
            for i in range(iteration):
                loss = self.train_one_step(data)

            self.model.eval()
            print('Epoch %d | Loss: %.4f' % (epoch, loss))


    def train_with_loader(self, dataloader):


        for epoch in range(50):
            self.model.train()
            for step, batch in enumerate(tqdm(loader, desc="Iteration")):
                batch = batch.to(self.device, dtype=torch.int64)

                loss = self.train_one_step(data)

            self.model.eval()
            print('Epoch %d | Loss: %.4f' % (epoch, loss))

        batch = batch.to(device)

        # if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
        #     pass
        # else:
        #     pred = model(batch)
        #     optimizer.zero_grad()
        #     ## ignore nan targets (unlabeled) when computing training loss.
        #     is_labeled = batch.y == batch.y
        #     if "classification" in task_type: 
        #         loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
        #     else:
        #         loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
        #     loss.backward()
        #     optimizer.step()


    def _test_Mnist_graph_pred_GNN(self, data):
        return None


    def _test_Mnist_node_pred_GNN(self, data):
        return None



    def test(self, data):
        if self.model_name == 'mnist_graph_pred_GNN':
            acc = self._test_Mnist_graph_pred_GNN(data)

        elif self.model_name == 'mnist_node_pred_GNN':
            acc = self._test_Mnist_node_pred_GNN(data)

        elif self.model_name == 'mnist_CNN':
            pass
        else:
            raise NotImplementedError("Not implemented!")

        return acc





