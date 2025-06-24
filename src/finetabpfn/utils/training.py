from __future__ import annotations

import torch
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from torch.types import _dtype



def to_cuda(*args: torch.Tensor, device: str | torch.device) -> list[torch.Tensor]:
    '''
    Returns a list of the object pushed to cuda in the same order in which they appear in input.
    If device is different from "cuda" it returns args as a list.
    '''
    if str(device) == "cuda":
        return [obj.to(device="cuda") for obj in args]
    else:
        return list(args)



def is_mixed_precision(
    inference_precision: _dtype | Literal['autocast', 'auto'],
    device: str | torch.device | Literal["auto"]
) -> bool:
    '''
    Light weight utility to determine if tabpfn uses mixed precision.
    Tabpfn uses mixed precision if directly specified or if auto 
    and cuda is enabled.
    '''
    device = resolve_device(device)
    if device != "cpu" and not isinstance(inference_precision, _dtype):
        return True
    return False



def resolve_device(device: str | torch.device | Literal["auto"]) -> str:
    '''Return the device as a string'''
    device = str(device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


