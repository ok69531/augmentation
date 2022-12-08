import torch
from torch.nn.utils import _stateless
from torch.autograd.functional import hvp
from torch.autograd import grad

from rdkit.Chem import AllChem

from functools import reduce


def vec_product(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
  elemwise_prod = 0
  
  for v1, v2 in zip(vec1, vec2):
    elemwise_prod += torch.sum(v1*v2)
  
  return elemwise_prod


def tensor_norm(tensor_to_add: torch.Tensor) -> torch.Tensor:
    norm = 0
    
    for t in tensor_to_add:
        norm += torch.linalg.norm(t)
    
    return norm


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


def compute_sample_hvp(x, y, edge_index, edge_attr, b, grads, model, criterion):
    
    names = list(n for n, _ in model.named_parameters())
    params = list(n for _, n in model.named_parameters())
    
    # x, edge_index, edge_attr, b = sample_in_batch(batch, i)

    def func(*params):
        pred = _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, (x, edge_index, edge_attr, b))
        return criterion(pred, (y.view(pred.shape)+1)/2)
    
    # v = compute_grad(model, batch, i, criterion)
    hvp_per_sample = torch.autograd.functional.hvp(func, tuple(model.parameters()), grads)[1]
    
    return hvp_per_sample


def compute_sample_inv_hvp(model, batch, i, criterion, 
                           damping: float = 0.1,
                           recursion_depth: int = 10, 
                           tol: float = 1e-6) -> list:
    
    inverse_hvp = None
    
    x, edge_index, edge_attr, b = sample_in_batch(batch, i)
    y = batch.y[i]
    
    # initialize H0^-1 * v = v begin with each sample
    v = compute_grad(model, batch, i, criterion)
    current_estimate = v
    
    for j in range(recursion_depth):
        hvp_estimate = compute_sample_hvp(x, y, edge_index, edge_attr, b, tuple(current_estimate), model, criterion)
        current_estimate = [a + damping * (b - c) for (a, b, c) in zip(current_estimate, hvp_estimate, v)]
        
        norm = tensor_norm(current_estimate)
        
        if norm <= tol:
            inverse_hvp = current_estimate
            break
        
    inverse_hvp = current_estimate
    
    return inverse_hvp


def compute_loo_if(model, batch, criterion) -> torch.Tensor:
    sample_grads = [compute_grad(model, batch, i, criterion) for i in range(len(batch.id))]
    inv_hvps = [compute_sample_inv_hvp(model, batch, i, criterion) for i in range(len(batch.id))]
    
    influence = []
    for i in range(len(batch.id)):
        _ = vec_product(sample_grads[i], inv_hvps[i])
        influence.append(_)
    influence = torch.stack(influence)
    
    return influence
