# In DGL, you can add features for all nodes at once, using a feature tensor that
# batches node features along the first dimension. The code below adds the learnable
# embeddings for all nodes:

import torch
import torch.nn as nn


def data_prepare():
    embed = nn.Embedding(34, 5)  # 34 nodes with embedding dim equal to 5
    inputs = embed.weight

    labeled_nodes = torch.tensor([0, 33])  # only the instructor and the president nodes are labeled
    labels = torch.tensor([0, 1])  # their labels are different

    return inputs, labeled_nodes, labels











