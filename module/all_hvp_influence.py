import sys
sys.path.append('../')

import typing
import torch
from torch.nn.utils import _stateless
from torch.autograd.functional import hvp
from torch.autograd import grad
import torch_geometric


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


def sample_in_batch(device: torch.device, 
                    dataset, 
                    batch: torch_geometric.data.batch, 
                    i: int) -> tuple:
    
    one_sample = dataset[batch.id[i]].to(device)
    
    x = one_sample.x
    edge_attr = one_sample.edge_attr
    edge_index = one_sample.edge_index
    y = one_sample.y.to(torch.float32)
    b = torch.zeros(len(x)).to(device).to(torch.int64)
    
    return (x, edge_attr, edge_index, b, y)


def compute_grad(model, 
                 criterion: torch.nn.modules.loss,
                 data: tuple) -> tuple:
    
    x, edge_attr, edge_index, b, y = data
    
    pred = model(x, edge_index, edge_attr, b) 
    y = y.view(pred.shape)
    loss = criterion(pred, (y+1)/2)
    
    return grad(loss, tuple(model.parameters()))


def compute_sample_hvp(model, 
                       criterion: torch.nn.modules.loss, 
                       data: tuple, 
                       grad: tuple):
    
    x, edge_attr, edge_index, b, y = data
    
    names = list(n for n, _ in model.named_parameters())
    params = list(n for _, n in model.named_parameters())

    def func(*params):
        pred = _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, (x, edge_index, edge_attr, b))
        return criterion(pred, (y.view(pred.shape)+1)/2)
    
    hvp_per_sample = hvp(func, tuple(model.parameters()), grad)[1]
    
    return hvp_per_sample


def compute_sample_inv_hvp(model, 
                           device: torch.device,
                           criterion: torch.nn.modules.loss, 
                           dataset,
                           batch: torch_geometric.data.batch, 
                           i: int,
                           damping: float = 0.01,
                           recursion_depth: int = 10,
                           tol: float = 1e-6) -> list:
    
    data = sample_in_batch(device, dataset, batch, i)
    
    inverse_hvp = None
    
    grad = compute_grad(model, criterion, data)
    norm1 = tensor_norm(grad)
    
    current_estimate = grad
    
    for j in range(recursion_depth):
        hvp_estimate = compute_sample_hvp(model, criterion, data, tuple(current_estimate))
        current_estimate = [a + damping * (b - c) for (a, b, c) in zip(current_estimate, grad, hvp_estimate)]
        
        norm2 = tensor_norm(hvp_estimate)
        
        if norm2 - norm1 <= tol:
            inverse_hvp = current_estimate
            break
        
    inverse_hvp = current_estimate
    
    return [grad, inverse_hvp]


def all_hvp_loo_if(model, 
                   device: torch.device,
                   criterion: torch.nn.modules.loss,
                   dataset,
                   batch: torch_geometric.data.batch,
                   damping: float = 0.01,
                   recursion_depth: int = 10,
                   tol: float = 1e-6) -> torch.Tensor:
    
    influence = []
    
    for i in range(len(batch.id)):
        sample_grad, inv_hvp = compute_sample_inv_hvp(model, device, criterion, dataset, batch, i, damping, recursion_depth, tol)
        influence.append(vec_product(sample_grad, inv_hvp))
    
    influence = torch.stack(influence)    
        
    return influence
