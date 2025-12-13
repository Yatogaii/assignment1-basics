import torch
import os
import typing

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str|os.PathLike|typing.BinaryIO|typing.IO[bytes]
):
    total_dict = {}
    total_dict["model"] = model.state_dict()
    total_dict["optimizer"] = optimizer.state_dict()
    total_dict["iteration"] = iteration
    
    torch.save(total_dict, out)


def load_checkpoint(
    src: str|os.PathLike|typing.BinaryIO|typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
):
    total_dict = torch.load(src)

    model.load_state_dict(total_dict["model"])
    optimizer.load_state_dict(total_dict["optimizer"])

    return total_dict["iteration"]