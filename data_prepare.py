
import os
import os.path as osp
import copy

import networkx as nx
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np


from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import DataLoader
import torch_geometric.utils.convert as convert
# from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

batch_size = 10
# dataset = "ogbn-arxiv"
# n_node_feats, n_classes = 0, 0
# global n_node_feats, n_classes


# data = DglNodePropPredDataset(name=dataset)
# evaluator = Evaluator(name=dataset)

# splitted_idx = data.get_idx_split()
# train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
# graph, labels = data[0]

# n_node_feats = graph.ndata["feat"].shape[1]
# n_classes = (labels.max() + 1).item()



def data_prepare11111111():
    embed = nn.Embedding(34, 5)  # 34 nodes with embedding dim equal to 5
    inputs = embed.weight

    labeled_nodes = torch.tensor([0, 33])  # only the instructor and the president nodes are labeled
    labels = torch.tensor([0, 1])  # their labels are different

    return inputs, labeled_nodes, labels



def molhiv_prepare(batch_size=10):
    dataset_name = 'ogbg-molhiv'

    dataset = PygGraphPropPredDataset(name=dataset_name)
    evaluator = Evaluator(name=dataset_name)

    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader


def matrix2nodeid(i, j, dim):
    index_i = i*dim + j
    return 0

class Mnist_node_pred_GNN_dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        self.root = root
        self.data_dir = root
        self.processed_paths[0] = 'Mnist_node_pred.pt'
        self.transform = transform
        self.train_dataset = torchvision.datasets.MNIST(self.data_dir, train=True, download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ]))
        # self.data_file = 'train_Mnist_graph_pred.pt'

        self.test_dataset = torchvision.datasets.MNIST(self.data_dir, train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ]))
        super(Mnist_node_pred_GNN_dataset, self).__init__(self.data_dir, transform, pre_transform)
        path = self.processed_paths[0]
        self.data = torch.load(path)


    @property
    def processed_file_names(self):
        return ['Mnist_node_pred.pt']

    def download(self):
        return None

    def _download(self):
        pass

    @property
    def raw_file_names(self):
        return None

    def process(self):
        torch.save(self.process_set(), self.processed_paths[0])

    def process_set(self):
        self.split_train = np.random.choice(60000, 6000)
        self.split_test = np.random.choice(10000, 1000)
        random_list = list(range(0, 7000))
        np.random.shuffle(random_list)
        self.train_mask = random_list[0:6000]
        self.test_mask = random_list[6000:7000]
        # self.origin_dataset = self.train_dataset + self.test_dataset
        # self.test_file = 'test_Mnist_graph_pred.pt'
        # num_nodes = self.origin_dataset[0].shape[0] * self.origin_dataset[0].shape[0]
        num_nodes = 6000 + 1000
        n = num_nodes
        node_list = [i for i in range(num_nodes)]
        edge_list = []
        for i in range(num_nodes):
            neighbor_i = np.random.choice(7000, 10)
            edge_i = [(i, idx) for idx in neighbor_i]
            edge_list += (edge_i)

        g = nx.Graph()
        g.add_nodes_from(node_list)
        g.add_edges_from(edge_list)
        data = convert.from_networkx(g)

        imgs = torch.Tensor([])
        labels = torch.Tensor([])
        # for i in range(len(self.origin_dataset)):
        for i in range(7000):
            if i < 6000:
                (img, label) = self.train_dataset[self.split_train[i]]
            else:
                idx = i-6000
                (img, label) = self.test_dataset[self.split_test[idx]]
            print("Processing %d-th Image" % (i))
            img = img.reshape([1, 28, 28])
            imgs = torch.cat((imgs, img))
            labels = torch.cat((labels, torch.Tensor(label)))

        data.train_mask = self.train_mask
        data.test_mask = self.test_mask
        data.x = imgs
        data.y = labels

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.data = data
        return self.data


    def len(self):
        #return len(self.origin_dataset)
        return 1

    # def get(self, idx):
    #     return self.data



class Mnist_graph_pred_GNN_dataset(Dataset):
    def __init__(self, root, train, transform=None, pre_transform=None):
        # print(dir(super(Mnist_graph_pred_GNN_dataset, self)))
        # print(dir(Dataset))
        # super(Mnist_graph_pred_GNN_dataset, self)._download = self._download
        self.train = train
        self.data_dir = root
        self.root = root
        self.train = train
        self.transform = transform
        if self.train == True:
            self.origin_dataset = torchvision.datasets.MNIST(self.data_dir, train=True, download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ]))
            self.data_prefix = 'train_Mnist_graph_pred_'
        else:
            self.origin_dataset = torchvision.datasets.MNIST(self.data_dir, train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ]))
            self.data_prefix = 'test_train_Mnist_graph_pred_'
        super(Mnist_graph_pred_GNN_dataset, self).__init__(root, transform, pre_transform)
        # self.data = torch.load(osp.join(self.processed_dir, self.data_prefix+'data.pt'))
        # self.data = torch.load(osp.join(self.processed_dir, self.data_prefix+'data_{}.pt'.format(i)))

    @property
    def processed_file_names(self):
        n_datas = 60000 if self.train else 10000
        # data_prefix = 'train_Mnist_graph_pred_' if self.train else 'test_train_Mnist_graph_pred_'
        # return [osp.join(self.processed_dir, self.data_prefix+'data_{}.pt'.format(i)) \
        #     for i in range(n_datas)]
        return [self.data_prefix+'data_{}.pt'.format(i) for i in range(n_datas)]

    def download(self):
        return 'None'

    @property
    def raw_file_names(self):
        return 'None'

    def process(self):
        (img0, label0) = self.origin_dataset[0]
        print(img0)
        num_nodes = img0.shape[1] * img0.shape[1]
        print("img0.shape: " , img0.shape)
        print("num_nodes: %d" % num_nodes)
        n = num_nodes
        node_list = [i for i in range(num_nodes)]
        edge_list = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                edge_ij = [ (matrix2nodeid(i-1+n//n, j, n), matrix2nodeid(i, j, n)), 
                            (matrix2nodeid(i+1+n//n, j, n), matrix2nodeid(i, j, n)),
                            (matrix2nodeid(i, j-1+n//n, n), matrix2nodeid(i, j, n)),
                            (matrix2nodeid(i, j-1+n//n, n), matrix2nodeid(i, j, n)),
                            (matrix2nodeid(i-1+n//n, j-1+n//n, n), matrix2nodeid(i, j, n)),
                            (matrix2nodeid(i+1+n//n, j-1+n//n, n), matrix2nodeid(i, j, n)),
                            (matrix2nodeid(i-1+n//n, j+1+n//n, n), matrix2nodeid(i, j, n)),
                            (matrix2nodeid(i+1+n//n, j+1+n//n, n), matrix2nodeid(i, j, n))
                        ]
                edge_list += edge_ij

        data_list = []
        g0 = nx.Graph()
        g0.add_nodes_from(node_list)
        g0.add_edges_from(edge_list)
        for i in range(len(self.origin_dataset)):
            print("Processing %d-th Image" % (i))
            if i > 10:
                break
            # Constructing a new g with node_list and edge_list is very time-consuming
            # So here we can use deepcopy
            # g = nx.Graph()
            # g.add_nodes_from(node_list)
            # g.add_edges_from(edge_list)
            g = copy.deepcopy(g0)
            img, label = self.origin_dataset[i]
            data = convert.from_networkx(g)
            data.x = img.reshape([num_nodes, 1])
            data.graph_label = label

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            # data_list.append(data)
            # data, slices = self.collate(data_list)
            # torch.save((data, slices), osp.join(self.data_dir, self.data_prefix+'data_{}.pt'.format(i)))
            torch.save(data, osp.join(self.processed_dir, self.data_prefix+'data_{}.pt'.format(i)))

        # data, slices = self.collate(data_list)
        # torch.save((data, slices), osp.join(self.processed_dir, self.data_prefix+'data.pt'))

    def len(self):
        return len(self.origin_dataset)

    def get(self, idx):
        # data = torch.load(osp.join(self.data_dir, self.data_prefix+'data_{}.pt'.format(idx)))
        data = torch.load(osp.join(self.processed_dir, self.data_prefix+'data_{}.pt'.format(idx)))
        return data



def mnist_CNN_prepare(data_dir, batch_size):

    image_size = 28
    input_shape = (batch_size, 1, image_size, image_size)
    output_shape = (batch_size, 10)

    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])

    data_path=os.path.join(data_dir, 'mnist')

    trainset = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=transform)

    testset = torchvision.datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ]))

    shuffle = True


    trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=batch_size, shuffle=shuffle, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset,
            batch_size=batch_size, shuffle=False, num_workers=4)

    # return input_shape, output_shape, trainloader, testloader
    return trainloader, testloader



def mnist_node_pred_GNN_prepare(data_dir):

    mnist_node_pred_GNN_dataset \
            = Mnist_node_pred_GNN_dataset(data_dir)
    # mnist_node_pred_GNN_dataset_test \
    #         = Mnist_node_pred_GNN_dataset(data_dir, train=False)

    return mnist_node_pred_GNN_dataset


def mnist_graph_pred_GNN_prepare(data_dir, batch_size):

    mnist_graph_pred_GNN_dataset_train \
            = Mnist_graph_pred_GNN_dataset(data_dir, train=True)
    mnist_graph_pred_GNN_dataset_test \
            = Mnist_graph_pred_GNN_dataset(data_dir, train=False)

    train_loader = DataLoader(mnist_graph_pred_GNN_dataset_train, batch_size=batch_size)
    test_loader = DataLoader(mnist_graph_pred_GNN_dataset_test, batch_size=batch_size)
    return train_loader, test_loader


def dataset_prepare(dataset_name, data_dir):
    if dataset_name == 'mnist_graph_pred_GNN_prepare':
        train_loader, test_loader = \
            mnist_graph_pred_GNN_prepare(data_dir, batch_size)
    elif dataset_name == 'mnist_node_pred_GNN_prepare':
        mnist_node_pred_GNN_dataset = \
            mnist_node_pred_GNN_prepare(data_dir)
        return mnist_node_pred_GNN_dataset
    elif dataset_name == 'mnist_CNN_prepare':
        train_loader, test_loader = \
            mnist_CNN_prepare(data_dir, batch_size)

    # TODO
    elif dataset_name == 'molhiv_GNN_prepare':
        train_loader, valid_loader, test_loader = \
            molhiv_GNN_prepare(batch_size)
    elif dataset_name == 'molhiv_DNN_prepare':
        train_loader, valid_loader, test_loader = \
            molhiv_DNN_prepare(batch_size)
    else:
        raise NotImplementedError("Not implemented!")

    return train_loader, test_loader


if __name__ == '__main__':
    # dataset_name = 'mnist_graph_pred_GNN_prepare'
    dataset_name = 'mnist_node_pred_GNN_prepare'
    # dataset_name = 'mnist_CNN_prepare'
    # dataset_name = 'molhiv'

    data_dir = '.'

    # train_loader, test_loader = dataset_prepare(dataset_name, data_dir)
    mnist_node_pred_GNN_dataset = dataset_prepare(dataset_name, data_dir)















