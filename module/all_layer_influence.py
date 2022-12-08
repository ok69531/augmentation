import torch
from torch.nn.utils import _stateless
from torch.autograd.functional import hessian
from torch.autograd import grad

from rdkit.Chem import AllChem

from functools import reduce

def sample_in_batch(batch, i):
    batch_id = batch.id[i]
    
    if i == 0:
        smile = smiles_list[batch_id]
        mol = AllChem.MolFromSmiles(smile)
        edge_len = len(mol.GetBonds()) * 2
        
        x = batch.x[batch.batch == i]
        edge_index = batch.edge_index[:, :edge_len]
        edge_attr = batch.edge_attr[:edge_len, :]
        b = batch.batch[batch.batch == i]
    
    else:
        pre_smile = smiles_list[batch.id[i - 1]]
        smile = smiles_list[batch_id]
        pre_mol = AllChem.MolFromSmiles(pre_smile)
        mol = AllChem.MolFromSmiles(smile)
        
        pre_edge_len = len(pre_mol.GetBonds()) * 2
        edge_len = len(mol.GetBonds()) * 2
        
        x = batch.x[batch.batch == i]
        edge_attr = batch.edge_attr[pre_edge_len:pre_edge_len+edge_len, :]
        b = torch.full([(batch.batch == i).sum()], 0).to(device)
        
        if len(mol.GetBonds()) > 0: # mol has bonds
            edges_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edges_list.append((i, j))
                edges_list.append((j, i))

            edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long).to(device)

        else:   # mol has no bonds
            edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
    
    return x, edge_index, edge_attr, b


def compute_grad(model, batch, i, criterion):
    x, edge_index, edge_attr, b = sample_in_batch(batch, i)
    
    prediction = model(x, edge_index, edge_attr, b) 
    y = batch.y[i].view(prediction.shape).to(torch.float32)
    loss = criterion(prediction.double(), (y+1)/2)
    
    return grad(loss, list(model.parameters()))


def compute_sample_grads(model, batch, criterion):
    sample_grads = [compute_grad(model, batch, i, criterion) for i in range(len(batch.id))]
    sample_grads = zip(*sample_grads)
    sample_grads = [torch.stack(shards) for shards in sample_grads]
    return sample_grads

# compute_sample_grads(model, batch, criterion)


def compute_classifier_hessian(model, batch, criterion):
    
    names = list(n for n, _ in model.named_parameters())
    params = list(n for _, n in model.named_parameters())

    def func(*params):
        pred = _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, batch)
        return torch.sum(criterion(pred.double(), (batch.y.view(pred.shape)+1)/2))

    hessian_list = hessian(func, tuple(model.parameters()))

    hessian_tmp = []

    for layer_hes_list in hessian_list:
        
        hes_layer_tmp = []

        for i in range(len(layer_hes_list)):
            row = reduce(lambda x, y: x*y, list(layer_hes_list[i].size()[:2]))
            col = reduce(lambda x, y: x*y, list(layer_hes_list[i].size()[2:]))
            # hes_tmp = layer_hes_list[i].reshape((row, col))
            hes_layer_tmp.append(layer_hes_list[i].reshape((row, col)))
            
        h_tmp = torch.concat(hes_layer_tmp, 1)
        hessian_tmp.append(h_tmp)

    hessian_mat = torch.cat(hessian_tmp)

    return hessian_mat


def compute_loo_if(model, batch, criterion):
    per_sample_grad = compute_sample_grads(model, batch, criterion)
    
    gard_batch_tmp = []
    
    for i in range(len(batch.id)):
        sample_grad_tmp = [x[i].reshape(-1) for x in per_sample_grad]
        
        sample_grad = torch.concat(sample_grad_tmp)
        gard_batch_tmp.append(sample_grad)

    grad_batch = torch.stack(gard_batch_tmp)

    hessian_mat = compute_classifier_hessian(model, batch, criterion)
    
    mask = hessian_mat.diagonal() == 0
    hessian_mat += torch.diag(mask) * 1e-4
    inv_hessian = torch.inverse(hessian_mat)
    # inv_hessian = torch.pinverse(hessian_mat)
    
    influence_list = []
    for i in range(grad_batch.shape[0]):
        tmp_influence = (grad_batch[i].unsqueeze(0) @ inv_hessian) @ grad_batch[i].unsqueeze(0).T
        influence_list.append(tmp_influence) 
    influence_arr = torch.abs(torch.cat(influence_list, axis=0).squeeze())
    
    return influence_arr