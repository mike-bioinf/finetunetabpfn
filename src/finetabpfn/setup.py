from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Literal, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import device
    from torch.types import _dtype
    from pathlib import Path
    from tabpfn.config import ModelInterfaceConfig
    from numpy.random import RandomState, Generator
    


@dataclass
class FineTuneSetup:
    '''
    Dataclass for the finetune setup.

    train_percentage (float): 
        Float indicating the percentage of data to use as training. 
        The remaining is used as validation.
        Must be a number in (0, 1).
    
    train_contest_percentage (float):
        Float indicating the percentage of train data to use as context.
        The remaining is used as test. Must be a number in (0, 1).
    
    max_steps (int): 
        Maximum number of finetune steps.
    
    time_limit (int):
        Maximum finetune time in seconds.
    
    validation_metric (Literal["roc_auc", "log_loss"]):
        Validation metric used in the early stopping.
    
    monitor_validation_metric (Literal["log_loss", "roc_auc"] | None):
        Additional metric that is computed on the validation set and monitored 
        during the finetune process. Note that this metric is used only for 
        monitoring purposes and therefore it does not affect the early stopping 
        mechanisms. Cannot be the same metric specified in "validation_metric".
        If None no monitoring is done.
    '''
    train_percentage: float = 0.67
    train_contest_percentage: float = 0.75
    max_steps: int = 100
    time_limit: int = 10000
    validation_metric: Literal["roc_auc", "log_loss"] = "log_loss"
    monitor_validation_metric: Literal["log_loss", "roc_auc"] | None = "roc_auc"

    @classmethod
    def from_dict(cls, dict: dict):
        return cls(**dict)
    
    def _check_percentage(self, attr: str) -> None:
        attr_percentage = getattr(self, attr)
        if attr_percentage <= 0 or attr_percentage >= 1:
            raise ValueError(f"{attr} must be a float in (0, 1).")
        
    def _check_validation_metrics(self) -> None:
        supported_metrics = ["log_loss", "roc_auc"]
        if self.validation_metric not in supported_metrics:
            raise ValueError("validation_metric must be one of 'log_loss' or 'roc_auc'.")
        if self.monitor_validation_metric is not None and self.monitor_validation_metric not in supported_metrics:
            raise ValueError("monitor_validation_metric must be one of 'log_loss', 'roc_auc' or None.")

    def _check_difference_validation_metrics(self) -> None:
       if self.monitor_validation_metric == self.validation_metric:
           raise ValueError("The monitor_validation_metric and validation_metric cannot be the same.") 

    def _general_check(self) -> "FineTuneSetup":
        self._check_percentage("train_percentage")
        self._check_percentage("train_contest_percentage")
        self._check_validation_metrics()
        self._check_difference_validation_metrics()
        return self



@dataclass
class AESetup:
    '''
    Dataclass for the adaptive eraly stopping setup.

    adaptive_rate (float): The rate of increase in patience. 
        Set to 0 to disable, or negative to shrink patience during training.
    adaptive_offset (int): The base value in patience computation.
    min_patience (int): The minimum value of patience.
    max_patience (int): The maximum value of patience.
    '''
    adaptive_rate: float = 0.3 
    adaptive_offset: int = 20
    min_patience: int = 20
    max_patience: int = 100

    @classmethod
    def from_dict(cls, dict: dict):
        return cls(**dict)
    


@dataclass
class TabPFNClassifierParams:
    '''Dataclass for the TabPFNClassifier parameters'''
    n_estimators: int = 4
    softmax_temperature: float = 0.9
    balance_probabilities: bool = False
    average_before_softmax: bool = False
    ignore_pretraining_limits: bool = False
    device: str | device | Literal['auto'] = "auto"
    inference_config: dict | ModelInterfaceConfig | None = None
    random_state: int | RandomState | Generator | None = 0

    @classmethod
    def from_dict(cls, dict: dict):
        return cls(**dict)



def build_instance_setup(
    cls: FineTuneSetup | AESetup | TabPFNClassifierParams,
    input: Any,
    allow_none = False
) -> FineTuneSetup | AESetup | TabPFNClassifierParams:
    '''
    Helps to build the setup instance with the 
    different possible inputs and then returns it. 

    Parameters:
        cls: Setup class.
        input: Input from which try to build the setup class.
        allow_none: Whether to allow None in input.
            If False and input is None an error is raised.

    Returns:
        The setup class.
    '''
    if input is None and not allow_none:
        raise ValueError("input can be None in setup instance building process.")
    elif input is None and allow_none:
        return None
    elif input == "default":
        return cls()
    elif isinstance(input, dict):
        return cls.from_dict(input)
    elif isinstance(input, (FineTuneSetup, AESetup, TabPFNClassifierParams)):
        return input
    else:
        raise ValueError("Input of wrong type to build a setup dataclass.")
