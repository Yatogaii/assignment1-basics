import torch
   
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