
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
from train.train import sigmoid, eval
from module.common import MoleculeDataset, scaffold_split, random_scaffold_split
from module.lbfgs_influence import *

criterion = nn.BCEWithLogitsLoss(reduction = "none")

def lbfgs_train(model, 
                device, 
                loader, 
                criterion, 
                optimizer, 
                dataset, 
                emb_dim: int = 300, 
                ratio: float = 0.3, 
                step_size: float = 0.001, 
                max_pert: float = 0.01, 
                history: int = 2):
    
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        
        y = batch.y.unsqueeze(1).to(torch.float32)
        
        # weight_dict_idx = len(loader)*(epoch-1)+(step+1)
        key_list = ['train_'+str(i) for i in batch.id.cpu().numpy()]
        
        weight_dict = {}
        gradient_dict = {i: {} for i in key_list}
        
        for e in range(history + 1):
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
            is_valid = y**2 > 0
            loss_mat = criterion(pred.double(), (y+1)/2)
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
            # update weight and gradient
            weight_dict.update({e: flat_model_weight(model)})
            for k in range(len(batch.id)):
                per_sample_gradient_dict = {e: flat_grad(compute_grad(model, device, dataset, batch, k, criterion))}
                gradient_dict[key_list[k]].update(per_sample_gradient_dict)

            optimizer.zero_grad()
            
            loss = torch.sum(loss_mat)/torch.sum(is_valid)
            loss.backward()
            
            optimizer.step()
            
        
        k = round(len(batch.id) * ratio)
        
        perturb = torch.FloatTensor(k, emb_dim).uniform_(-max_pert, max_pert).to(device)
        perturb.requires_grad_()
        
        graph_embedding = model.pool(model.gnn(batch.x, batch.edge_index, batch.edge_attr), batch.batch)
        loo_influence = lbfgs_loo_if(key_list, gradient_dict, weight_dict)
        _, topk_idx = torch.topk(loo_influence, k = k, axis = -1)
        
        graph_embedding = graph_embedding[topk_idx, :] + perturb
        pred = model.graph_pred_linear(graph_embedding)

        y = batch.y.view([len(batch.id), 1]).to(torch.float32)[topk_idx, :]

        is_valid = y**2 > 0
        loss_mat = criterion(pred.double(), (y+1)/2)
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

        optimizer.zero_grad()
        
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        
        loss.backward()
        optimizer.step()


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
    parser.add_argument('--history', type = int, default = 2, help = 'number of steps for lbfgs')
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
            
            lbfgs_train(model, device, train_loader, criterion, optimizer, dataset, 
                        ratio=args.ratio, step_size=args.step_size, max_pert=args.max_pert, history=args.history)
            
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


