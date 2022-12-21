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


def compute_grad(model, 
                 criterion: torch.nn.modules.loss,
                 embedding: torch.tensor,
                 target: torch.tensor) -> tuple:
    
    pred = model(embedding) 
    target = target.view(pred.shape).to(torch.float32)
    loss = criterion(pred, (target+1)/2)
    
    return grad(loss, tuple(model.parameters()))


def compute_sample_hvp(model, 
                       criterion: torch.nn.modules.loss, 
                       embedding: torch.tensor,
                       target: torch.tensor,
                       grad: tuple):
    
    names = list(n for n, _ in model.named_parameters())
    params = list(n for _, n in model.named_parameters())

    def func(*params):
        pred = _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, embedding)
        return criterion(pred, (target.view(pred.shape).to(torch.float32)+1)/2)
    
    hvp_per_sample = hvp(func, tuple(model.parameters()), grad)[1]
    
    return hvp_per_sample


def compute_sample_inv_hvp(model, 
                           criterion: torch.nn.modules.loss, 
                           embedding: torch.tensor,
                           target: torch.tensor,
                           damping: float = 0.01,
                           recursion_depth: int = 10,
                           tol: float = 1e-6) -> list:
    
    inverse_hvp = None
    
    grad = compute_grad(model, criterion, embedding, target)
    norm1 = tensor_norm(grad)
    
    current_estimate = grad
    
    for j in range(recursion_depth):
        hvp_estimate = compute_sample_hvp(model, criterion, embedding, target, tuple(current_estimate))
        current_estimate = [a + damping * (b - c) for (a, b, c) in zip(current_estimate, grad, hvp_estimate)]
        
        norm2 = tensor_norm(hvp_estimate)
        
        if norm2 - norm1 <= tol:
            inverse_hvp = current_estimate
            break
        
    inverse_hvp = current_estimate
    
    return [grad, inverse_hvp]


def lin_hvp_loo_if(model, 
                   criterion: torch.nn.modules.loss,
                   embeddings: torch.tensor,
                   targets: torch.tensor,
                   damping: float = 0.01,
                   recursion_depth: int = 10,
                   tol: float = 1e-6) -> torch.Tensor:
    
    influence = []
    
    for i in range(len(targets)):
        sample_grad, inv_hvp = compute_sample_inv_hvp(model, criterion, embeddings[i], targets[i], damping, recursion_depth, tol)
        influence.append(vec_product(sample_grad, inv_hvp))
    
    influence = torch.stack(influence)    
        
    return influence
