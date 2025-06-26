from __future__ import annotations

import torch
from sklearn.model_selection import train_test_split
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.random import RandomState
    from torch.types import _dtype



def _train_test_split(X, y, train_size: float | None, random_state: int | RandomState | None):
    '''
    train_test_split wrapper to use in get_preprocessed_datasets to ensure stratification.
    Here X and y are explicity defined as parameters.
    '''
    return train_test_split(X, y, train_size=train_size, random_state=random_state, stratify=y)



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


