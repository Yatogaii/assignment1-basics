from numpy import typing as npt
import numpy as np
import torch

def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(dataset)
    m = context_length
    
    starts = np.random.randint(0, n-m, batch_size)

    xb = np.stack([dataset[s:s+m] for s in starts], axis=0)
    yb = np.stack([dataset[s+1:s+m+1] for s in starts], axis=0)

    xb = torch.from_numpy(xb).to(device=device, dtype=torch.long)
    yb = torch.from_numpy(yb).to(device=device, dtype=torch.long)

    return xb, yb