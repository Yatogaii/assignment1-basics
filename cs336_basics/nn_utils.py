import torch
from typing import Iterable
   
def run_softmax(x: torch.Tensor,dimension: int) -> torch.Tensor:
    max_val = torch.max(x, dim=dimension, keepdim=True).values
    x -= max_val

    return torch.exp(x) / torch.sum(x.exp(), dim=dimension, keepdim=True)
    
def run_cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    max_val = torch.max(inputs, dim=-1, keepdim=True).values
    shifted = inputs - max_val
    
    exp_shifted = torch.exp(shifted)
    sumed = torch.sum(input=exp_shifted, dim=1, keepdim=True)
    loged = torch.log(sumed)
    
    loss = torch.tensor(0.0) 
    for i in range(targets.shape[0]):
        p_correct = shifted[i][targets[i]]
        loss += loged[i][0] - p_correct
    return loss / targets.shape[0]

def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    total_sq = torch.tensor(0.0)
    for p in parameters:
        if p.grad is None:
            continue

        g = p.grad.detach()

        total_sq += g.float().pow(2).sum().item()

    norm = total_sq ** 0.5
    
    # if norm bigger than given max value, adjust it in-place
    if norm > max_l2_norm:
        scale = max_l2_norm / norm
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.data.mul_(scale)
            
    return None