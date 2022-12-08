import sys
sys.path.append('../')

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.loader import DataLoader

from torch.nn.utils import _stateless
from torch.autograd.functional import hessian

from model.model import GNN_graphpred
from module.common import MoleculeDataset, random_scaffold_split, scaffold_split

from sklearn.metrics import roc_auc_score

# torch.cuda.is_available()
# torch.cuda.get_device_name(0)
device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

num_layer = 2
embedding_dim = 5
num_worker = 0

DATASET_NAME = "hiv"
if DATASET_NAME == "tox21":
    NUM_TASK = 12
elif DATASET_NAME == "hiv":
    NUM_TASK = 1
elif DATASET_NAME == "pcba":
    NUM_TASK = 128
elif DATASET_NAME == "muv":
    NUM_TASK = 17
elif DATASET_NAME == "bace":
    NUM_TASK = 1
elif DATASET_NAME == "bbbp":
    NUM_TASK = 1
elif DATASET_NAME == "toxcast":
    NUM_TASK = 617
elif DATASET_NAME == "sider":
    NUM_TASK = 27
elif DATASET_NAME == "clintox":
    NUM_TASK = 2
elif DATASET_NAME == 'geno':
    NUM_TASK = 1

dataset = MoleculeDataset('../dataset/' + DATASET_NAME, dataset = DATASET_NAME)
smiles_list = pd.read_csv('../dataset/' + DATASET_NAME + '/processed/smiles.csv', header = None)[0].tolist()


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def train(model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float32)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred, (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()


@torch.no_grad()
def eval(model, device, loader):
    model.eval()
    
    y_true = []
    y_score = []
    loss = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float32)

        is_valid = y**2 > 0
        loss_mat = criterion(pred, (y+1)/2)
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

        _loss = torch.sum(loss_mat)/torch.sum(is_valid)

        y_true.append(batch.y.view(pred.shape))
        y_score.append(pred)
    
    loss.append(_loss)
    y_true = torch.cat(y_true, dim = 0).data.cpu().numpy()
    y_score = torch.cat(y_score, dim = 0).data.cpu().numpy()
    y_pred = np.where(sigmoid(y_score) > 0.5, 1.0, 0.0)
    
    roc_list = []
    
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_score[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(loss)/len(loss), sum(roc_list)/len(roc_list), roc_list, y_pred 


if __name__ == '__main__':
    test_roc_list = []

    for seed in range(10):
        # seed = 0
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed=seed)

        train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = False, num_workers = num_worker)
        val_loader = DataLoader(valid_dataset, batch_size = 64, shuffle = False, num_workers = num_worker)
        test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False, num_workers = num_worker)
        
        model = GNN_graphpred(num_layer, embedding_dim, num_tasks=NUM_TASK, JK='last', drop_ratio=0, graph_pooling='mean').to(device)
        
        criterion = nn.BCEWithLogitsLoss(reduction = "none")

        model_param_group = []
        model_param_group.append({"params": model.gnn.parameters()})

        model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":0.001*1})
        optimizer = optim.Adam(model_param_group, lr=0.001, weight_decay=0)
        print(optimizer)
        
        eval_train = 1
        test_roc_per_epochs = []
        
        for epoch in range(1, 30+1):
            # print("====epoch " + str(epoch))
            
            # train(model, device, train_loader, optimizer)
            train(model, device, train_loader, optimizer)

            # print("====Evaluation")
            if eval_train:
                train_loss, train_roc, train_roc_list, train_pred = eval(model, device, train_loader)
            else:
                # print("omit the training accuracy computation")
                train_roc = 0
            val_loss, val_roc, val_roc_list, val_pred = eval(model, device, val_loader)
            test_loss, test_roc, test_roc_list_, test_pred = eval(model, device, test_loader)
            
            test_roc_per_epochs.append(test_roc)

            print("epoch: %f " %int(epoch),
                "\ntrain_loss: %f val_loss: %f test_loss: %f" %(train_loss, val_loss, test_loss),
                "\ntrain_auc: %f val_auc: %f test_auc: %f" %(train_roc, val_roc, test_roc))

        print("Seed:", seed)
        
        test_y = []
        for d, s in enumerate(test_dataset):
            y_tmp = [0 if i == -1 else i for i in s.y.numpy()]
            test_y.append(y_tmp[0])
            
        pred = [int(i[0]) for i in test_pred]
        
        test_roc_list.append(max(test_roc_per_epochs))

    result = pd.DataFrame({'auc': test_roc_list})
    print(result.auc.mean(), result.auc.sem())
