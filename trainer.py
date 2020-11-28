import itertools
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from model_prepare import get_model

mol_evaluator = Evaluator(name='ogbg-molhiv')
dataset = PygGraphPropPredDataset(name='ogbg-molhiv')

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
strhdlr = logging.StreamHandler()
logger.addHandler(strhdlr)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
strhdlr.setFormatter(formatter)

class Trainer(object):
    def __init__(self, model_name, optim_name, criterion, train_loader=None, test_loader=None):
        super(Trainer, self).__init__()

        self.lr_decay = 0.99
        self.all_logits = []
        self.model_name = model_name
        self.model = get_model(model_name)
        self.logfile_name = 'log_' + self.model_name + '.log'

        hdlr = logging.FileHandler(self.logfile_name)
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr) 

        if model_name == 'mnist_graph_pred_GNN':
            ''' 
            work: 
                0.0001
            work little:
                0.01 0.001
            not work:
                0.5 0.1, 0.2, 0.0005
            '''
            # self.lr = 0.001
            self.lr = 0.0001
            # self.model = get_model(model_name, deep=False)
            self.optimizer = self._create_optimizer(optim_name=optim_name, lr=self.lr, momentum=0.99)
            self._train_one_step = self._train_one_step_Mnist_graph_pred_GNN
            self.train = self.train_with_loader

        elif model_name == 'mnist_node_pred_GNN':
            self.lr = 0.01
            # self.model = get_model(model_name, deep=true)
            self.optimizer = self._create_optimizer(optim_name=optim_name, lr=self.lr, momentum=0.99)
            self._train_one_step = self._train_one_step_Mnist_node_pred_GNN
            self.train = self.train_with_one_sample

        elif model_name == 'mol_pred_DNN':
            ''' 
            work: 
                
            work little:
                0.05
            not work:
                
            '''
            self.lr = 0.0001
            # self.model = get_model(model_name, deep=true)
            self.optimizer = self._create_optimizer(optim_name=optim_name, lr=self.lr, momentum=0.99)
            self._train_one_step = self._train_one_step_Mol_pred_DNN
            self.train = self.train_with_loader

        elif model_name == 'mol_pred_GNN':
            self.lr = 0.0001
            # self.model = get_model(model_name, deep=true)
            self.optimizer = self._create_optimizer(optim_name=optim_name, lr=self.lr, momentum=0.99)
            self._train_one_step = self._train_one_step_Mol_pred_GNN
            self.train = self.train_with_loader

        elif model_name == 'mnist_CNN':
            self.lr = 0.01
            # self.model = get_model(model_name, deep=true)
            self.optimizer = self._create_optimizer(optim_name=optim_name, lr=self.lr, momentum=0.9)
            self._train_one_step = self._train_one_step_Mnist_CNN
            self.train = self.train_with_loader

        self.base_lr = self.lr
        self.criterion = self._create_criterion(criterion)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device=self.device)
        # self.model = self.model.cuda()
        self.train_loader = train_loader
        self.test_loader = test_loader

        logger.info('Base_lr: %.4f' % self.base_lr)

    def _create_optimizer(self, optim_name='SGD', lr=0.1, momentum=0.9):
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
        if self.model_name == 'mol_pred_GNN' or self.model_name == 'mol_pred_DNN':
            return nn.BCEWithLogitsLoss()
        return F.nll_loss


    def eval(self, train=False):
        self.model.eval()
        test_loss = 0
        correct = 0
        test_num = 0
        with torch.no_grad():
            if self.model_name == 'mnist_node_pred_GNN':
                data = self.train_loader.to(device=self.device)
                x, edge_index = data.x, data.edge_index
                out = self.model(x, edge_index)
                # we only compute loss for labeled nodes
                # loss = F.nll_loss(logp[labeled_nodes], labels)
                # print("out[data.train_mask].shape:", out[data.train_mask].shape)
                # print("out[data.train_mask][0:5]:", out[data.train_mask][0:5])
                # print("data.y[data.train_mask].long().shape:", data.y[data.train_mask].long().shape)
                # print("data.y[data.train_mask][0:5]:", data.y[data.train_mask][0:5])
                # print("data.y[data.train_mask].long()[0:5]:", data.y[data.train_mask].long()[0:5])
                test_loss = self.criterion(out[data.test_mask], data.y[data.test_mask].long()).item()

                _, pred = out.max(dim=1)
                correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
                acc = correct / len(data.test_mask)
                test_num = len(data.test_mask)

            elif self.model_name == 'mnist_graph_pred_GNN':
                for data in self.test_loader:
                    x, edge_index, graph_label = \
                        data.x.cuda(), data.edge_index.cuda(), data.graph_label.cuda()

                    out = self.model(x, edge_index)
                    test_loss += self.criterion(out, graph_label.long())
                    _, pred = out.max(dim=1)
                    correct += int(pred.eq(graph_label).sum().item())

                test_num = len(self.test_loader.dataset)
                test_loss /= len(self.test_loader.dataset)
                acc = correct / test_num
            elif self.model_name == 'mnist_CNN':
                for data, label in self.test_loader:
                    data = data.to(device=self.device, dtype=torch.float32)
                    label = label.to(device=self.device)

                    output = self.model(data)
                    test_loss += self.criterion(output, label).item()  # sum up batch loss
                    _, pred = output.max(dim=1)  # get the index of the max log-probability
                    correct += int(pred.eq(label).sum().item())

                test_num = len(self.test_loader.dataset)
                test_loss /= len(self.test_loader.dataset)
                acc = correct / test_num
            elif self.model_name == 'mol_pred_DNN':
                y_true = []
                y_pred = []
                if train:
                    data_loader = self.train_loader
                else:
                    data_loader = self.test_loader
                for i, data in enumerate(data_loader):
                    with torch.no_grad():
                        data, label = data
                        data = data.to(device=self.device, dtype=torch.float32)
                        # label = label.cuda().reshape(-1)
                        label = label.to(self.device)
                        pred = self.model(data)
                        # print("out.shape:", out.shape)
                        # print("label.shape:", label.shape)
                        test_loss += self.criterion(pred, label.to(torch.float32)).item()

                        # _, pred = out.max(dim=1)
                        # correct = int(pred.eq(label).sum().item())
                        # acc = correct / len(label)
                        # y_true.append(label.view(-1).detach().cpu())
                        # y_pred.append(pred.view(-1).detach().cpu())
                        y_true.append(label.view([-1, 1]).detach().cpu())
                        y_pred.append(pred.view([-1, 1]).detach().cpu())
                y_true = torch.cat(y_true, dim = 0).numpy()
                y_pred = torch.cat(y_pred, dim = 0).numpy()

                print('y_true[0:5]:', y_true[0:5])
                print('y_pred[0:5]:', y_pred[0:5])
                # print('y_true.shape:', y_true.shape)
                # print('y_pred.shape:', y_pred.shape)
                input_dict = {"y_true": y_true, "y_pred": y_pred}
                test_loss /= len(data_loader)
                return test_loss, mol_evaluator.eval(input_dict)

            elif self.model_name == 'mol_pred_GNN':
                y_true = []
                y_pred = []
                if train:
                    data_loader = self.train_loader
                else:
                    data_loader = self.test_loader
                for i, batch in enumerate(data_loader):
                    batch = batch.to(self.device)

                    if batch.x.shape[0] == 1:
                        pass
                    else:
                        with torch.no_grad():
                            pred = self.model(batch)
                            test_loss += self.criterion(pred.to(torch.float32), batch.y.to(torch.float32)).item()
                        y_true.append(batch.y.view(pred.shape).detach().cpu())
                        y_pred.append(pred.detach().cpu())

                y_true = torch.cat(y_true, dim = 0).numpy()
                y_pred = torch.cat(y_pred, dim = 0).numpy()
                print('y_true[0:5]:', y_true[0:5])
                print('y_pred[0:5]:', y_pred[0:5])
                # print('y_true.shape:', y_true.shape)
                # print('y_pred.shape:', y_pred.shape)
                input_dict = {"y_true": y_true, "y_pred": y_pred}
                test_loss /= len(data_loader)
                return test_loss, mol_evaluator.eval(input_dict)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, test_num,
            100. * correct / test_num))
        return test_loss, acc

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
    # def train_one_step(self, data):
    #     if self.model_name == 'mnist_graph_pred_GNN':
    #         loss = self._train_one_step_Mnist_graph_pred_GNN(data)

    #     elif self.model_name == 'mnist_node_pred_GNN':
    #         loss = self._train_one_step_Mnist_node_pred_GNN(data)

    #     elif self.model_name == 'mnist_CNN':
    #         pass
    #     else:
    #         raise NotImplementedError("Not implemented!")

    #     return loss



    def _train_one_step_Mnist_graph_pred_GNN(self, data):
        x, edge_index, graph_label = \
                data.x.cuda(), data.edge_index.cuda(), data.graph_label.cuda()

        # print("x.shape:", x.shape)
        # print("x[0:5]:", x[0:5])
        # print("graph_label.long().shape:", graph_label.long().shape)

        self.optimizer.zero_grad()
        out = self.model(x, edge_index)

        loss = self.criterion(out, graph_label.long())
    
        _, pred = out.max(dim=1)
        correct = int(pred.eq(graph_label).sum().item())
        acc = correct / len(graph_label)
        loss.backward()
        self.optimizer.step()

        return loss.item(), acc


    def _train_one_step_Mnist_node_pred_GNN(self, data):
        x, edge_index = data.x, data.edge_index
        self.optimizer.zero_grad()
        out = self.model(x, edge_index)
        # we only compute loss for labeled nodes
        # loss = F.nll_loss(logp[labeled_nodes], labels)
        # print("out[data.train_mask].shape:", out[data.train_mask].shape)
        # print("out[data.train_mask][0:5]:", out[data.train_mask][0:5])
        # print("data.y[data.train_mask].long().shape:", data.y[data.train_mask].long().shape)
        # print("data.y[data.train_mask][0:5]:", data.y[data.train_mask][0:5])
        # print("data.y[data.train_mask].long()[0:5]:", data.y[data.train_mask].long()[0:5])

        loss = self.criterion(out[data.train_mask], data.y[data.train_mask].long())
    
        _, pred = out.max(dim=1)
        correct = int(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
        acc = correct / len(data.train_mask)
        loss.backward()
        self.optimizer.step()

        return loss.item(), acc


    def _train_one_step_Mol_pred_DNN(self, data):
        data, label = data
        data = data.to(device=self.device, dtype=torch.float32)
        label = label.to(device=self.device)
        # label = label.cuda().reshape(-1)
        self.optimizer.zero_grad()
        pred = self.model(data)
        # print("out.shape:", out.shape)
        # print("label.shape:", label.shape)
        # print('type(out):', type(out))
        # print('type(label):', type(label))
        # print('out.dtype:', out.dtype)
        # print('label.dtype:', label.dtype)
        loss = self.criterion(pred, label.to(torch.float32))
        loss.backward()
        self.optimizer.step()
        test_loss = loss.item()

        y_true = label.view([-1, 1]).detach().cpu()
        y_pred = pred.view([-1, 1]).detach().cpu()

        # print('y_true[0:5]:', y_true[0:5])
        # print('y_pred[0:5]:', y_pred[0:5])


        return test_loss, y_true, y_pred


    def _train_one_step_Mol_pred_GNN(self, batch):
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            return None, None, None
        else:
            batch = batch.to(device=self.device)
            pred = self.model(batch)
            self.optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            # is_labeled = batch.y == batch.y
            # print("is_labeled:", is_labeled)
            # print("batch.y:", batch.y)
            # loss = self.criterion(pred.to(torch.float32)[is_labeled], 
            #     batch.y.to(torch.float32)[is_labeled])
            # print("pred before to device:", pred)
            loss = self.criterion(pred.to(torch.float32), 
                batch.y.to(torch.float32))

            loss.backward()
            self.optimizer.step()   
            test_loss = loss.item()     
            # _, pred = pred.max(dim=1)
            # correct = int(pred.to(torch.float32)[is_labeled]\
            #         .eq(batch.y.to(torch.float32)[is_labeled]).sum().item())
            # print("pred:", pred)
            # print("batch.y:", batch.y)
            # correct = int(pred.eq(batch.y).sum().item())
            # acc = correct / len(is_labeled)
            # acc = correct / len(batch.y)
        y_true = batch.y.view([-1, 1]).detach().cpu()
        y_pred = pred.view([-1, 1]).detach().cpu()

        # print('y_true[0:5]:', y_true[0:5])
        # print('y_pred[0:5]:', y_pred[0:5])

        return test_loss, y_true, y_pred


    def _train_one_step_Mnist_CNN(self, data):
        data, label = data
        data = data.to(device=self.device, dtype=torch.float32)
        label = label.to(device=self.device)
        # label = label.cuda().reshape(-1)
        self.optimizer.zero_grad()
        out = self.model(data)
        # print("out.shape:", out.shape)
        # print("label.shape:", label.shape)
        # print('type(out):', type(out))
        # print('type(label):', type(label))
        # print('out.dtype:', out.dtype)
        # print('label.dtype:', label.dtype)
        loss = self.criterion(out, label)

        _, pred = out.max(dim=1)
        correct = int(pred.eq(label).sum().item())
        acc = correct / len(label)
        loss.backward()
        self.optimizer.step()

        return loss.item(), acc


    def train_with_one_sample(self):

        iteration = 100
        # print("data.y[data.train_mask].long()[0:5]:", data.y[0:5])
        # data = data.to(device=self.device, dtype=torch.int64)
        data = self.train_loader.to(device=self.device)
        # print("data.y[data.train_mask][0:5]:", data.y[data.train_mask][0:5])
        # print(data.device)
        for epoch in range(50):
            self.model.train()
            for i in range(iteration):
                train_loss, train_acc = self._train_one_step(data)

                if i % 10 == 0:
                    print("Iteration %d...... train_loss: %.4f" % (i, train_loss))
            print('Train Epoch %d | Loss: %.4f | acc: %.4f' % (epoch, train_loss, train_acc))

            self.eval()

            print('Evaluating...')

            test_loss, test_acc = self.eval()
            print('Test Epoch %d | Loss: %.4f | acc: %.4f' % (epoch, test_loss, test_acc))
            logger.info('Train Epoch %d | Loss: %.4f | acc: %.4f' % (epoch, train_loss, train_acc))
            logger.info('Test Epoch %d | Loss: %.4f | acc: %.4f' % (epoch, test_loss, test_acc))


        print('Finished training!')


    def train_with_loader(self):
        valid_curve = []
        test_curve = []
        train_curve = []
        for epoch in range(50):
            self.model.train()
            train_losses = 0
            train_accs = 0
            y_true = []
            y_pred = []

            self.lr = self.base_lr * (self.lr_decay**epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

            for i, batch in enumerate(self.train_loader):
                # if i > 5:
                #     break   
                # print("type(batch):", type(batch))
                # print("batch:",batch)
                # batch = batch.to(self.device)
                
                if self.model_name == 'mol_pred_DNN' or self.model_name == 'mol_pred_GNN':
                    train_loss, y_true_i, y_pred_i = self._train_one_step(batch)
                    train_losses += train_loss if train_loss != None else 0
                    y_true.append(y_true_i) if  y_true_i != None else 0
                    y_pred.append(y_pred_i) if  y_pred_i != None else 0
                else:
                    train_loss, train_acc = self._train_one_step(batch)
                    train_losses += train_loss
                    train_accs += train_acc
                if i % 100 == 0:
                    print("Iteration %d...... train_loss: %.4f" % (i, train_loss))

            if self.model_name == 'mol_pred_DNN' or self.model_name == 'mol_pred_GNN':
                y_true = torch.cat(y_true, dim = 0).numpy()
                y_pred = torch.cat(y_pred, dim = 0).numpy()

                print('y_true[0:5]:', y_true[0:5])
                print('y_pred[0:5]:', y_pred[0:5])
                # print('y_true.shape:', y_true.shape)
                # print('y_pred.shape:', y_pred.shape)
                input_dict = {"y_true": y_true, "y_pred": y_pred}

                train_perf = mol_evaluator.eval(input_dict)
                train_loss = train_losses / len(self.train_loader.dataset)

                # train_loss, train_perf = self.eval(train=True)
                test_loss, test_perf = self.eval(train=False)

                # train_curve.append(train_perf[dataset.eval_metric])
                # test_curve.append(test_perf[dataset.eval_metric])
                # print('Train Epoch %d | Loss: %.4f | acc: %.4f' % 
                #     (epoch, train_loss if train_loss else 10, train_perf[dataset.eval_metric]))
                # print('Test Epoch %d | Loss: %.4f | acc: %.4f' % 
                #     (epoch, 10, test_perf[dataset.eval_metric]))
                logger.info('Train Epoch %d | Loss: %.4f | acc: %.4f' % 
                    (epoch, train_loss if train_loss else 10, train_perf[dataset.eval_metric]))
                logger.info('Test Epoch %d | Loss: %.4f | acc: %.4f' % 
                    (epoch, test_loss, test_perf[dataset.eval_metric]))
            else:
                train_loss = train_losses / len(self.train_loader.dataset)
                train_acc = train_accs / i
                test_loss, test_acc = self.eval()
                # print('Train Epoch %d | Loss: %.4f | acc: %.4f' % (epoch, train_loss, train_acc))
                # print('Test Epoch %d | Loss: %.4f | acc: %.4f' % (epoch, test_loss, test_acc))

                logger.info('Train Epoch %d | Loss: %.4f | acc: %.4f' % (epoch, train_loss, train_acc))
                logger.info('Test Epoch %d | Loss: %.4f | acc: %.4f' % (epoch, test_loss, test_acc))

            # self.eval()

            # print('Evaluating...')

            # test_loss, test_acc = self.eval()
            # print('Test Epoch %d | Loss: %.4f | acc: %.4f' % (epoch, test_loss, test_acc))

        # if self.model_name == 'mol_pred_DNN' or self.model_name == 'mol_pred_GNN':
        #     torch.save({'Test': test_curve, 'Train': train_curve}, 'aaaa.txt')



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


    # def _test_Mnist_graph_pred_GNN(self, data):
    #     return None


    # def _test_Mnist_node_pred_GNN(self, data):
    #     return None



    # def test(self, data):
    #     if self.model_name == 'mnist_graph_pred_GNN':
    #         acc = self._test_Mnist_graph_pred_GNN(data)

    #     elif self.model_name == 'mnist_node_pred_GNN':
    #         acc = self._test_Mnist_node_pred_GNN(data)

    #     elif self.model_name == 'mnist_CNN':
    #         pass
    #     else:
    #         raise NotImplementedError("Not implemented!")

    #     return acc





