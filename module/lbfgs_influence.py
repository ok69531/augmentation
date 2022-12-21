import torch


def sample_in_batch(device, dataset, batch, i):
    one_sample = dataset[batch.id[i]].to(device)
    x = one_sample.x
    edge_attr = one_sample.edge_attr
    edge_index = one_sample.edge_index
    y = one_sample.y.to(torch.float32)
    b = torch.zeros(len(x)).to(device).to(torch.int64)
    return (x, edge_attr, edge_index, b, y.to(torch.float32))


def compute_grad(model, device, dataset, batch, i, criterion):
    x, edge_attr, edge_index, b, y = sample_in_batch(device, dataset, batch, i)
    
    pred = model(x, edge_index, edge_attr, b) 
    y = y.view(pred.shape)
    loss = criterion(pred, (y+1)/2)
    
    return torch.autograd.grad(loss, tuple(model.parameters()))


def flat_grad(grad_list):
    gradient = torch.cat([x.view(-1) for x in grad_list])
    return gradient
    

def flat_model_weight(model):
    parameter_list = [x.view(-1) for x in model.parameters()]
    params = torch.cat(parameter_list)
    return params


def lbfgs_inv_hvp_per_sample(sample_gradient_dict, weight_dict, history=2):
    q = sample_gradient_dict[history]
    
    rho_list = []
    alpha_list = []
    # beta_list = []
    
    # back track
    for i in range(1, history+1):
        y_i = sample_gradient_dict[history] - sample_gradient_dict[history-i]
        s_i = weight_dict[history] - weight_dict[history-i]
        
        rho = 1/(y_i.dot(s_i))
        rho_list.append(rho)
        
        alpha_i = rho * s_i.dot(q)
        alpha_list.append(alpha_i)
        
        q = q - alpha_i * y_i
    
    gamma = weight_dict[history-1].dot(sample_gradient_dict[history-1])/(sample_gradient_dict[history-1].dot(sample_gradient_dict[history-1]))
    r = gamma * q
    
    # forward track
    for i in range(history, 0, -1):
        beta_i = rho_list[i-1] * sample_gradient_dict[history-i].dot(r)
        r = r + (alpha_list[i-1] - beta_i)*(weight_dict[history-i+1] - weight_dict[history-i])
        
        # beta_list.append(beta_i)
    
    return r


def lbfgs_inv_hvp_batch(key_list, gradient_dict, weight_dict, history = 2):
    # key_list = ['train_'+str(i) for i in batch.id.cpu().numpy()]
    hv_list = [lbfgs_inv_hvp_per_sample(gradient_dict[k], weight_dict, history) for k in key_list]
    hv_mat = torch.stack(hv_list)
    return hv_mat


def lbfgs_loo_if(key_list, gradient_dict, weight_dict, history = 2):
    # key_list = ['train_'+str(i) for i in batch.id.cpu().numpy()]
    per_sample_grads = torch.stack([gradient_dict[k][history] for k in key_list])
    
    hv = lbfgs_inv_hvp_batch(key_list, gradient_dict, weight_dict, history)
    
    influence_list = []
    for i in range(per_sample_grads.size(0)):
        tmp_influence = per_sample_grads[i].dot(hv[i])
        influence_list.append(tmp_influence) 
    influence_arr = torch.abs(torch.stack(influence_list))
    
    return influence_arr