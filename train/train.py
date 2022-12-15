import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from module.influence import compute_loo_if

def sigmoid(z):
    return 1/(1 + np.exp(-z))


def train(model, device, loader, criterion, optimizer):
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


def adv_train(model, 
              device, 
              loader, 
              criterion, 
              optimizer, 
              ratio: float = 0.3, 
              step_size: float = 0.001, 
              max_pert: float = 0.01, 
              m: int = 3):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        
        k = round(len(batch.id) * ratio)
        
        perturb = torch.FloatTensor(k, 300).uniform_(-max_pert, max_pert).to(device)
        perturb.requires_grad_()
        
        graph_embedding = model.pool(model.gnn(batch.x, batch.edge_index, batch.edge_attr), batch.batch)
        loo_influence = compute_loo_if(model.graph_pred_linear, graph_embedding, batch.y, criterion)
        _, topk_idx = torch.topk(loo_influence, k = k, axis = -1)
        
        graph_embedding = graph_embedding[topk_idx, :] + perturb
        # topk_embedding = graph_embedding[topk_idx, :] + perturb
        pred = model.graph_pred_linear(graph_embedding)
        # pred = model.graph_pred_linear(topk_embedding)

        y = batch.y.view([len(batch.id), 1]).to(torch.float64)[topk_idx, :]

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss /= m
        
        for _ in range(m - 1):
            loss.backward()
            perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
            perturb.data = perturb_data.data
            perturb.grad[:] = 0
            
            tmp_graph_embedding = model.pool(model.gnn(batch.x, batch.edge_index, batch.edge_attr), batch.batch)
            tmp_graph_embedding = tmp_graph_embedding[topk_idx, :] + perturb
            # tmp_topk_embedding = tmp_graph_embedding[topk_idx, :] + perturb
            
            tmp_pred = model.graph_pred_linear(tmp_graph_embedding)
            # tmp_pred = model.graph_pred_linear(tmp_topk_embedding)

            loss = 0
            loss_mat = criterion(tmp_pred.double(), (y+1)/2)
            loss += torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            loss = torch.sum(loss_mat)/torch.sum(is_valid)
            loss /= m

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


@torch.no_grad()
def eval(model, device, loader, criterion):
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
    
    auc_list = []
    
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            auc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_score[is_valid,i]))

    if len(auc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(auc_list))/y_true.shape[1]))

    return sum(loss)/len(loss), sum(auc_list)/len(auc_list)
