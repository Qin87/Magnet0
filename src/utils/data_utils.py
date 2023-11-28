import random

import torch
import numpy as np
from dgl.data import CiteseerGraphDataset
from torch_scatter import scatter_add
import torch_geometric.transforms as T

import os, time, argparse, csv
from torch_geometric.datasets import WebKB, WikipediaNetwork, WikiCS
# from utilsMag.Citation import *
# from utilsMag.preprocess import geometric_dataset, load_syn

import torch.nn as nn
import torch.nn.functional as F

from src.utils.Citation import citation_datasets
from src.utils.preprocess import load_syn


class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, input, target, weight=None, reduction='mean'):
        return F.cross_entropy(input, target, weight=weight, reduction=reduction)



def load_directedData(args):
    load_func, subset = args.dataset.split('/')[0], args.dataset.split('/')[1]
    print("dataset is ", load_func)  # Ben WebKB
    if load_func == 'WebKB':
        load_func = WebKB
        dataset = load_func(root=args.data_path, name=subset)
    elif load_func == 'WikipediaNetwork':
        load_func = WikipediaNetwork
        dataset = load_func(root=args.data_path, name=subset)
    elif load_func == 'WikiCS':
        load_func = WikiCS
        dataset = load_func(root=args.data_path)
    elif load_func == 'cora_ml':
        dataset = citation_datasets(root='../dataset/data/tmp/cora_ml/cora_ml.npz')
    elif load_func == 'citeseer_npz':
        dataset = citation_datasets(root='../dataset/data/tmp/citeseer_npz/citeseer_npz.npz')
    elif load_func == 'dglCitation':    # Ben
        return CiteseerGraphDataset(reverse_edge=False)
    else:
        dataset = load_syn(args.data_path + args.dataset, None)

    return dataset


def get_dataset(name, path, split_type='public'):
    """

    :param name: dataset name
    :param path: dataset path
    :param split_type: it determines train_mask, val_mask
    :return: the
    """
    if name == "Cora" or name == "CiteSeer" or name == "PubMed":
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(path, name, transform=T.NormalizeFeatures(), split=split_type)
    elif name == 'Amazon-Computers':
        from torch_geometric.datasets import Amazon
        print("Load Amazon-Computers data")
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())
    elif name == 'Amazon-Photo':
        from torch_geometric.datasets import Amazon

        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())
    elif name == 'Coauthor-CS':
        from torch_geometric.datasets import Coauthor
        print("Load CoAuthor data")
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())
    elif name == 'WiKiCS':
        from torch_geometric.datasets import wikics
        print("Load WiKics data")
        return wikics(root=path, transform=T.NormalizeFeatures())
    elif name == 'DglCitation':
        print("Loading dgl citeseer dataset")
        return CiteseerGraphDataset(raw_dir=path, reverse_edge=False, transform=T.NormalizeFeatures())
    elif name.startswith('konect'):
        from konect import konect
        print("Loading konect data", name, path, name.split('_')[1])
        # name = name.split('_')[1]
        return konect(root=path, name = name.split('_')[1], transform=T.NormalizeFeatures())
    else:
        raise NotImplementedError("Not Implemented Dataset!")

    return dataset


def get_idx_info(label, n_cls, train_mask):
    """

    :param label: class
    :param n_cls:
    :param train_mask:
    :return:# all the sample for training that belongs to each class.
    """
    index_list = torch.arange(len(label))   # [0,1,2...]
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[((label == i) & train_mask)]
        # creating a boolean mask by comparing elements of label with the value i
        # and then performing element-wise logical AND operation with train_mask.
        # print("cls_indices is ", cls_indices)   # it's a tensor
        idx_info.append(cls_indices)
    return idx_info

def keep_all_data(edge_index, label, n_data, n_cls, ratio, train_mask):
    """
    just keep all training data
    :param edge_index:
    :param label:
    :param n_data:
    :param n_cls:
    :param ratio:
    :param train_mask:
    :return:
    """
    class_num_list = n_data
    data_train_mask = train_mask

    index_list = torch.arange(len(train_mask))
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[(label == i) & train_mask]
        idx_info.append(cls_indices)

    train_node_mask = train_mask

    row, col = edge_index[0], edge_index[1]
    row_mask = train_mask[row]
    col_mask = train_mask[col]
    edge_mask = row_mask & col_mask
    # print(torch.sum(train_mask), torch.sum(row_mask), torch.sum(col_mask), torch.sum(edge_mask))  # tensor(250) tensor(414) tensor(410) tensor(51)

    return class_num_list, data_train_mask, idx_info, train_node_mask, edge_mask

def make_longtailed_data_remove(edge_index, label, n_data, n_cls, ratio, train_mask):
    """

    :param edge_index: all edges in the graph
    :param label: classes of all nodes
    :param n_data:num of train in each class
    :param n_cls:
    :param ratio:
    :param train_mask:
    :return: list(class_num_list), train_mask, idx_info, node_mask, edge_mask
    """
    # Sort from major to minor
    n_data = torch.tensor(n_data)   # from list to tensor
    # print(n_data)
    sorted_n_data, indices = torch.sort(n_data, descending=True)
    # print(sorted_n_data, indices)   # tensor([341, 196, 196, 160, 138,  90,  87]) tensor([3, 2, 4, 0, 5, 1, 6])
    inv_indices = np.zeros(n_cls, dtype=np.int64)
    # print(inv_indices)      # [0 0 0 0 0 0 0]
    for i in range(n_cls):
        inv_indices[indices[i].item()] = i
    # print(inv_indices)      # [3 5 1 0 2 4 6]
    assert (torch.arange(len(n_data))[indices][torch.tensor(inv_indices)] - torch.arange(len(n_data))).sum().abs() < 1e-12

    # Compute the number of nodes for each class following LT rules
    mu = np.power(1/ratio, 1/(n_cls - 1))
    # print(mu, 1/ratio, 1/(n_cls-1))     # 0.4641588833612779 0.01 0.16666666666666666
    n_round = []
    class_num_list = []
    for i in range(n_cls):
        # assert int(sorted_n_data[0].item() * np.power(mu, i)) >= 1
        temp = int(sorted_n_data[0].item() * np.power(mu, i))
        if temp< 1:
            temp = 1
        class_num_list.append(int(min(temp, sorted_n_data[i])))
        """
        Note that we remove low degree nodes sequentially (10 steps)
        since degrees of remaining nodes are changed when some nodes are removed
        """
        if i < 1:  # We do not remove any nodes of the most frequent class
            n_round.append(1)
        else:
            n_round.append(10)
    class_num_list = np.array(class_num_list)   # from list to np.array
    class_num_list = class_num_list[inv_indices]    # sorted
    n_round = np.array(n_round)[inv_indices]        # sorted  #

    # Compute the number of nodes which would be removed for each class
    remove_class_num_list = [n_data[i].item()-class_num_list[i] for i in range(n_cls)]
    remove_idx_list = [[] for _ in range(n_cls)]
    # print(remove_idx_list)  # [[], [], [], [], [], [], []]
    cls_idx_list = []   # nodes belong to class i
    index_list = torch.arange(len(train_mask))
    original_mask = train_mask.clone()
    for i in range(n_cls):
        cls_idx_list.append(index_list[(label == i) & original_mask])

    for i in indices.numpy():
        for r in range(1, n_round[i]+1):
            # Find removed nodes
            node_mask = label.new_ones(label.size(), dtype=torch.bool)
            # new_ones is a PyTorch function used to create a new tensor of ones with the specified shape and data type.
            # print("Initialize all true: ", node_mask[:10])
            node_mask[sum(remove_idx_list, [])] = False
            # print("Setting some as false", node_mask[:10])

            # Remove connection with removed nodes
            row, col = edge_index[0], edge_index[1]
            # print("row is ", row.shape, row[:10])
            # # torch.Size([10556]) tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
            # print("col is ", row.shape, col[:10])
            # # torch.Size([10556]) tensor([ 633, 1862, 2582,    2,  652,  654,    1,  332, 1454, 1666])
            row_mask = node_mask[row]
            col_mask = node_mask[col]
            edge_mask = row_mask & col_mask  # elementwise "and"

            # Compute degree
            degree = scatter_add(torch.ones_like(col[edge_mask]), col[edge_mask], dim_size=label.size(0)).to(row.device)
            degree = degree[cls_idx_list[i]]
            # creating a new tensor that contains the elements from the original degree tensor
            # at the positions specified by the values in cls_idx_list[i]

            # Remove nodes with low degree first (number increases as round increases)
            # Accumulation does not be problem since
            # print("Degree is: ", degree, degree.shape, (r*remove_class_num_list[i])//n_round[i])
            # # tensor([0, 5, 0, 4, 0, 4, 7, 0, 0, 0]) torch.Size([87]) 75
            _, remove_idx = torch.topk(degree, (r*remove_class_num_list[i])//n_round[i], largest=False)
            remove_idx = cls_idx_list[i][remove_idx]
            # print("After index from class list: ", remove_idx)
            # # the actual nodes

            remove_idx_list[i] = list(remove_idx.numpy())
            # print("nodes to be removed: ", remove_idx_list[i])
            # remove_idx.numpy() converts the remove_idx tensor to a NumPy array

    # Find removed nodes
    node_mask = label.new_ones(label.size(), dtype=torch.bool)
    node_mask[sum(remove_idx_list, [])] = False
    # sum(remove_idx_list, []) is used to flatten this list into a single list of indices.
    # print("node_mask is ", node_mask)

    # Remove connection with removed nodes
    row, col = edge_index[0], edge_index[1]
    row_mask = node_mask[row]
    col_mask = node_mask[col]
    edge_mask = row_mask & col_mask

    train_mask = node_mask & train_mask
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[(label == i) & train_mask]
        idx_info.append(cls_indices)

    return list(class_num_list), train_mask, idx_info, node_mask, edge_mask


def get_step_split(imb_ratio, valid_each, labeling_ratio, all_idx, all_label, nclass):
    """
    get train, test, valid date split.
    :param imb_ratio: in training, the head class has more nodes than tail class
    :param valid_each: num o f valid node for each class
    :param labeling_ratio:
    :param all_idx:  all nodes
    :param all_label: all y
    :param nclass: num of classes
    :return:
    """
    # print("All nodes is ", len(all_idx))
    base_valid_each = valid_each
    # print("base_valid_each : ", base_valid_each)    # base_valid_each :  95

    head_list = [i for i in range(nclass//2)]   # the first half of classes

    all_class_list = [i for i in range(nclass)]
    tail_list = list(set(all_class_list) - set(head_list))  # the second half of the classes

    h_num = len(head_list)
    t_num = len(tail_list)

    base_train_each = int(len(all_idx) * labeling_ratio / (t_num + h_num * imb_ratio))
    # print("base_train_each : ", base_train_each)    # base_train_each :  1

    idx2train, idx2valid = {}, {}

    total_train_size = 0
    total_valid_size = 0

    for i_h in head_list:
        idx2train[i_h] = int(base_train_each * imb_ratio)
        idx2valid[i_h] = int(base_valid_each * 1)

        total_train_size += idx2train[i_h]
        total_valid_size += idx2valid[i_h]

    for i_t in tail_list:
        idx2train[i_t] = int(base_train_each * 1)
        idx2valid[i_t] = int(base_valid_each * 1)

        total_train_size += idx2train[i_t]
        total_valid_size += idx2valid[i_t]

    train_list = [0 for _ in range(nclass)]
    train_node = [[] for _ in range(nclass)]
    train_idx = []

    # print("total_train_size is", total_train_size)      # total_train_size is 404
    # print(idx2train)    # {0: 100, 1: 100, 2: 100, 3: 100, 4: 1, 5: 1, 6: 1, 7: 1}
    for iter1 in all_idx:
        iter_label = all_label[iter1]
        # print("iter1 is: ", iter1, "its label is : ", iter_label)
        if train_list[iter_label] < idx2train[iter_label]:
            # distribute train nodes for each class until idx2train is satisfied
            train_list[iter_label] += 1
            train_node[iter_label].append(iter1)
            train_idx.append(iter1)
            # print("hello, add to train of class", iter_label)

        # print("train_list is ", train_list)   # train_list is  [100, 100, 100, 100, 1, 1, 1, 1]
        if sum(train_list) == total_train_size:
            # print("It happened!")
            break

    assert sum(train_list) == total_train_size

    after_train_idx = list(set(all_idx)-set(train_idx))

    valid_list = [0 for _ in range(nclass)]
    valid_idx = []
    for iter2 in after_train_idx:
        iter_label = all_label[iter2]
        if valid_list[iter_label] < idx2valid[iter_label]:
            valid_list[iter_label] += 1
            valid_idx.append(iter2)
        if sum(valid_list) == total_valid_size:
            break

    test_idx = list(set(after_train_idx)-set(valid_idx))
    # print(len(train_idx), len(valid_idx), len(test_idx))
    # All nodes is 7650
    # 404 760 6486

    return train_idx, valid_idx, test_idx, train_node
    # train_node is divided in classes, while train_idx is all together


def generate_masks(data_y, minClassTrain, ratio_val2train):
    n_cls = data_y.max().item() + 1

    n_Alldata = []  # num of train in each class
    for i in range(n_cls):
        data_num = (data_y == i).sum()
        n_Alldata.append(int(data_num.item()))
    # print("In all nodes, class number:", n_Alldata)  # [264, 590, 668, 701, 596, 508]

    num_train_sample = []
    Tmin = min(n_Alldata)
    for i in range(len(n_Alldata)):
        Tnum_sample = int(minClassTrain * n_Alldata[i] / Tmin)
        num_train_sample.append(Tnum_sample)
    print(num_train_sample)  # [20, 44, 50, 53, 45, 38]

    train_mask = torch.zeros(len(data_y), dtype=torch.bool)
    val_mask = torch.zeros(len(data_y), dtype=torch.bool)
    test_mask = torch.zeros(len(data_y), dtype=torch.bool)

    for class_label, num_samples in zip(range(len(num_train_sample)), num_train_sample):
        class_indices = (data_y == class_label).nonzero().view(-1)
        shuffled_indices = torch.randperm(len(class_indices))

        # Divide the sampled indices into train, val, and test sets
        train_indices = class_indices[shuffled_indices[:num_samples]]
        val_indices = class_indices[shuffled_indices[num_samples:num_samples * (ratio_val2train+1)]]
        # test_indices = class_indices[shuffled_indices[num_samples * (ratio_val2train+1):num_samples * int(2*(ratio_val2train+1))]]
        test_indices = class_indices[shuffled_indices[num_samples * (ratio_val2train+1):]]

        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
    print(torch.sum(train_mask), torch.sum(val_mask), torch.sum(test_mask))
    return train_mask, val_mask, test_mask

def generate_masksRatio(data_y, TrainRatio, ValRatio):
    n_cls = data_y.max().item() + 1

    n_Alldata = []  # num of train in each class
    for i in range(n_cls):
        data_num = (data_y == i).sum()
        n_Alldata.append(int(data_num.item()))
    # print("In all nodes, class number:", n_Alldata)  # [264, 590, 668, 701, 596, 508]

    num_train_sample = []
    # Tmin = min(n_Alldata)
    for i in range(len(n_Alldata)):
        # Tnum_sample = int(minClassTrain * n_Alldata[i] / Tmin)
        num_train_sample.append(int(TrainRatio*n_Alldata[i]))
    print(num_train_sample)  # [20, 44, 50, 53, 45, 38]

    train_mask = torch.zeros(len(data_y), dtype=torch.bool)
    val_mask = torch.zeros(len(data_y), dtype=torch.bool)
    test_mask = torch.zeros(len(data_y), dtype=torch.bool)

    for class_label, num_samples in zip(range(len(num_train_sample)), num_train_sample):
        class_indices = (data_y == class_label).nonzero().view(-1)
        shuffled_indices = torch.randperm(len(class_indices))

        # Divide the sampled indices into train, val, and test sets
        train_indices = class_indices[shuffled_indices[:num_samples]]
        val_indices = class_indices[shuffled_indices[num_samples:num_samples * (ValRatio/TrainRatio+1)]]
        test_indices = class_indices[shuffled_indices[num_samples * (ValRatio/TrainRatio+1):]]

        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
    print("train, val, test sample number: ", torch.sum(train_mask), torch.sum(val_mask), torch.sum(test_mask))
    return train_mask, val_mask, test_mask
# Example usage:
# train_mask, val_mask, test_mask = generate_masks(data_y, num_train_sample, n_Alldata)
