import sys
import logging
import warnings
from functools import wraps
from typing import Any, Literal



def enlist(obj: Any) -> list:
    '''Wrap an object inside a list unless the object is already a list'''
    return obj if isinstance(obj, list) else [obj]



def create_logger(stream=sys.stderr) -> logging.Logger:
    '''Creates and returns the logger instance with the input stream handler'''
    logger = logging.getLogger("finetune")
    logger.setLevel(logging.DEBUG)

    # to avoid repetition of the same handler in the logger 
    # with multiple instances of the finetuner classes
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == stream:
            return logger
        
    console_handler = logging.StreamHandler(stream)
    console_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        fmt="{asctime} \t {message}", 
        datefmt="%Y-%m-%d %H:%M", 
        style="{"
    )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger



def suppress_sklearn_and_tabpfn_warnings(func):
    '''
    Decorator to filter sklearn future deprecation warnings,
    and tabpfn loading and ignore limits warning.
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="sklearn", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*", module=".*tabpfn.*loading")
            warnings.filterwarnings(
                action="ignore", 
                message=".*is greater than the maximum Number of features 500 supported by the model.*",
                category=UserWarning
            )
            return func(*args, **kwargs)
    return wrapper



def get_metric_name(metric: Literal["log_loss", "roc_auc"] | None) -> str | None:
    '''
    Get a nice formatted metric string for printing/logging.
    If None it is returned as is.
    '''
    if metric is None:
        metric_name = None
    elif metric == "roc_auc":
        metric_name = "ROC AUC"
    elif metric == "log_loss":
        metric_name = "Log Loss"
    else:
        raise NotImplementedError("Unsupported metric passed in input.")
        
    return metric_name
