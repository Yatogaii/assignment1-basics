import torch
   
def run_softmax(x: torch.Tensor,dimension: int) -> torch.Tensor:
    max_val = torch.max(x, dim=dimension, keepdim=True).values
    x -= max_val

    return torch.exp(x) / torch.sum(x.exp(), dim=dimension, keepdim=True)
    