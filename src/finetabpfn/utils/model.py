import sys
import torch
from pathlib import Path
from typing import Literal
from tabpfn.model_loading import _user_cache_dir
from tabpfn.architectures.base import PerFeatureTransformer



def resolve_model_path(model_path: str | Path | Literal['default', 'old_default']) -> str:
    '''
    Resolve the model_path argument deriving a string to pass to the tabpfn machinery.
    Returns a string.
    '''
    if model_path == "default":
        return "auto"
    elif model_path == "old_default":
        cache_folder = _user_cache_dir(sys.platform)
        return str(cache_folder/"tabpfn-v2-classifier.ckpt") # old default checkpoint name
    elif isinstance(model_path, (str, Path)):
        return str(model_path)
    else:
        raise ValueError("Unsupported model_path.")



def save_model(model: PerFeatureTransformer, file: str | Path, checkpoint_config: dict) -> None:
    '''Save the finetuned model to disk in a TabPFN-readable checkpoint format'''
    torch.save(
        dict(state_dict=model.state_dict(), config=checkpoint_config),
        f=str(file)
    )


