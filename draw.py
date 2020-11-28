# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import matplotlib.pyplot as plt
import matplotlib
import copy
matplotlib.use("TkAgg")
import numpy as np
import datetime
import itertools
# import utils as u

#markers=['.','x','o','v','^','<','>','1','2','3','4','8','s','p','*']
markers=['.','x','o','v','^','<','>']
# markers=[None]
colors = ['b', 'g', 'r', 'm', 'y', 'k', 'orange', 'purple', 'olive']
#colors = colors[2:7]
#colors = colors[0:4]
colors = colors[0:6]
markeriter = itertools.cycle(markers)
coloriter = itertools.cycle(colors)

FONTSIZE=15

OUTPUTPATH = '.'

datasets = [
    'MNIST',
    'MNIST semi',
    'mol-hiv'
]

all_exps = [
    'mnist_graph_pred_GNN',
    'mnist_node_pred_GNN',
    'mol_pred_DNN',
    'mol_pred_GNN',
    'mnist_CNN'
]


EXPS_OF_DATASETS = {
    'MNIST': [  'mnist_graph_pred_GNN',
                'mnist_CNN'],
    'MNIST semi': ['mnist_node_pred_GNN'],
    'mol-hiv': ['mol_pred_DNN',
                'mol_pred_GNN']
}



STANDARD_TITLES = {
        'mnist_graph_pred_GNN': 'MNIST with GNN',
        'mnist_node_pred_GNN': 'semi-supervised MNIST with GNN',
        'mol_pred_DNN': 'mol-hiv with DNN',
        'mol_pred_GNN': 'mol-hiv with GNN',
        'mnist_CNN': 'MNIST with CNN'
}

LEGENDS = {        
        'mnist_graph_pred_GNN': 'MNIST GNN',
        'mnist_node_pred_GNN': 'semi-supervised GNN',
        'mol_pred_DNN': 'mol-hiv DNN',
        'mol_pred_GNN': 'mol-hiv GNN',
        'mnist_CNN': 'MNIST CNN'
}






def get_real_title(title):
    return STANDARD_TITLES.get(title, title)

def get_legend(title):
    return LEGENDS.get(title, title)

def get_exps(title):
    return EXPS_OF_DATASETS.get(title, title)





def get_data(file_name, train=True):
    epochs = []
    losses = []
    accs = []

    if train:
        filter_word = 'Train'
    else:
        filter_word = 'Test'

    with open(file_name, 'r') as f:
        for line in f.readlines():
            if filter_word in line:                    
                items = line.split()
                epoch = items[6].strip()
                loss = items[9].strip()
                acc = items[12].strip()

                epochs.append(int(epoch))
                losses.append(float(loss))
                accs.append(float(acc))

    return epochs, losses, accs


# def plot_line():
#     pass



def convergence(dataset, train=True, draw_loss=True):

    plt.figure()
    fig, ax = plt.subplots(1,1,figsize=(5,3.4))


    exps = get_exps(dataset)
    for exp in exps:
        file_name = 'log_' + exp + '.log'
        epochs, losses, accs = get_data(file_name, train=train)
        # if draw_loss:
        #     # plot_line(epochs, losses)
        #     ax.plot(epochs, losses, label=label, marker=marker, linewidth=1.5, markerfacecolor='none', color=color)
        # else:
        #     # plot_line(epochs, accs)
        #     ax.plot(epochs, accs, label=label, marker=marker, linewidth=1.5, markerfacecolor='none', color=color)
        label = get_legend(exp)
        if draw_loss:
            ax.plot(epochs, losses, label=label, marker=next(markeriter), linewidth=1.5, markerfacecolor='none', color=next(coloriter))
        else:
            ax.plot(epochs, accs, label=label, marker=next(markeriter), linewidth=1.5, markerfacecolor='none', color=next(coloriter))

    if draw_loss:
        ax.set_ylabel(('Train ' if train else 'Test ') + 'loss', fontsize=FONTSIZE)
        ax.legend(fontsize=FONTSIZE, loc='upper right')
    else:
        ax.set_ylabel(('Train ' if train else 'Test ') + \
            ('Accuracy [%]' if dataset != 'mol-hiv' else 'ROC-AUC'), fontsize=FONTSIZE)
        ax.legend(fontsize=FONTSIZE, loc='lower right')
    ax.set_xlabel('Epoch', fontsize=FONTSIZE)

    ax.set_xlim(xmin=-1)

    plt.subplots_adjust(bottom=0.16, left=0.15, right=0.96, top=0.95)
    plt.savefig('%s/%s_%s_%s_convergence.pdf' % (OUTPUTPATH, dataset, 'Training' if train else 'Test', 'loss' if draw_loss else 'acc'))
    # plt.show()


for train in [True, False]:
    for draw_loss in [True, False]:
        for dataset in datasets:
            convergence(dataset, train, draw_loss)



