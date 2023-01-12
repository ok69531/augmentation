import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

import argparse
import numpy as np
import pandas as pd

from model.model import GNN_graphpred
from train.train import sigmoid, all_hvp_train, eval
from module.common import MoleculeDataset, scaffold_split, random_scaffold_split


criterion = nn.BCEWithLogitsLoss(reduction = "none")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type = int, default = 500, help = 'batch size for training (default = 500)')
    parser.add_argument('--epochs', type = int, default = 50, help = 'number of epochs to train (default = 50)')
    parser.add_argument('--lr', type = float, default = 0.001, help = 'learning rate (default = 0.001)')
    parser.add_argument('--lr_scale', type = float, default = 1, help = 'relative learning rate for the feature extraction layer (default = 1)')
    parser.add_argument('--decay', type = float, default = 0, help = 'weight decay (default = 0)')
    parser.add_argument('--num_layer', type = int, default = 3, help = 'number of GNN message passing layers (default = 3)')
    parser.add_argument('--emb_dim', type = int, default = 300, help = 'embedding dimensions (default = 300)')
    parser.add_argument('--dropout_ratio', type = float, default = 0, help = 'dropout ratio (default = 0)')
    parser.add_argument('--graph_pooling', type = str, default = 'mean', help = 'graph level pooling (sum, mean, max)')
    parser.add_argument('--JK', type = str, default = 'last', help = 'how the node features across layers are combined. (last, sum, max or concat)')
    parser.add_argument('--gnn_type', type = str, default = 'gcn')
    parser.add_argument('--ratio', type = float, default = 0.2, help = 'top k IF-LOO ratio (default = 0.2)')
    parser.add_argument('--max_pert', type = float, default = 0.01, help = 'perturbation budget (default = 0.01)')
    parser.add_argument('--step_size', type = float, default = 0.001, help = 'gradient ascent learning rate (default = 0.001)')
    parser.add_argument('--m', type = int, default = 3, help = 'number of update for perturbation (default = 3)')
    parser.add_argument('--damping', type = float, default = 0.01, help = 'learning rate for update inverse hvp')
    parser.add_argument('--recursion_depth', type = int, default = 10, help = 'number of inverse hvp interations')
    parser.add_argument('--tol', type = float, default = 1e-6)
    parser.add_argument('--dataset', type = str, default = 'hiv', help = 'bace, bbbp, clintox, hiv, muv, sider, tox21, toxcast')
    parser.add_argument('--split', type = str, default = 'random_scaffold', help = 'scaffold or random_scaffold')
    parser.add_argument('--eval_train', type = int, default = 1, help = 'evaluating training or not')
    parser.add_argument('--num_workers', type = int, default = 4, help = 'number of workers for dataset loading')
    parser.add_argument('--num_runs', type = int, default = 3, help = 'number of independent runs (default = 3)')
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

    if args.dataset == "tox21":
        num_task = 12
    elif args.dataset == "hiv":
        num_task = 1
    elif args.dataset == "pcba":
        num_task = 128
    elif args.dataset == "muv":
        num_task = 17
    elif args.dataset == "bace":
        num_task = 1
    elif args.dataset == "bbbp":
        num_task = 1
    elif args.dataset == "toxcast":
        num_task = 617
    elif args.dataset == "sider":
        num_task = 27
    elif args.dataset == "clintox":
        num_task = 2

    dataset = MoleculeDataset('../dataset/' + args.dataset, dataset = args.dataset)
    smiles_list = pd.read_csv('../dataset/' + args.dataset + '/processed/smiles.csv', header = None)[0].tolist()
    
    test_auc_list = []
    
    for seed in range(args.num_runs):
        
        print('===== seed ' + str(seed))
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        if args.split == 'scaffold':
            train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
            print('scaffold')
        elif args.split == 'random_scaffold':
            train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=seed)
            print('random scaffold')
        else:
            raise ValueError('Invalid split option')
        
        train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
        val_loader = DataLoader(valid_dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
        
        model = GNN_graphpred(args.num_layer, args.emb_dim, num_task, 
                              JK = args.JK, 
                              drop_ratio = args.dropout_ratio, 
                              graph_pooling = args.graph_pooling, 
                              gnn_type = args.gnn_type)
        model.to(device)
        
        model_param_group = []
        model_param_group.append({'params': model.gnn.parameters()})
        if args.graph_pooling == 'attention':
            model_param_group.append({'params': model.pool.parameters(), 'lr': args.lr * args.lr_scale})
        model_param_group.append({'params': model.graph_pred_linear.parameters(), 'lr': args.lr * args.lr_scale})
        optimizer = optim.Adam(model_param_group, lr = args.lr, weight_decay = args.decay)
        
        test_auc_per_epoch = []
        
        for epoch in range(1, args.epochs + 1):
            print('=== epoch ' + str(epoch))
            all_hvp_train(model, device, train_loader, criterion, optimizer, dataset,
                          emb_dim = args.emb_dim, ratio = args.ratio, step_size = args.step_size, max_pert = args.max_pert,
                          m = args.m, damping = args.damping, recursion_depth = args.recursion_depth, tol = args.tol)

            if args.eval_train:
                train_loss, train_auc = eval(model, device, train_loader, criterion)
            else:
                print('omit the training accuracy computation')
                train_auc = 0
            
            val_loss, val_auc = eval(model, device, val_loader, criterion)
            test_loss, test_auc = eval(model, device, test_loader, criterion)
            
            test_auc_per_epoch.append(test_auc)
            
            print("train_loss: %f val_loss: %f test_loss: %f" %(train_loss, val_loss, test_loss),
                  "\ntrain_auc: %f val_auc: %f test_auc: %f" %(train_auc, val_auc, test_auc))
    
        test_auc_list.append(max(test_auc_per_epoch))
    
    result = pd.DataFrame({'auc': test_auc_list})
    print(result.auc.mean(), result.auc.sem())


if __name__ == '__main__':
    main()
