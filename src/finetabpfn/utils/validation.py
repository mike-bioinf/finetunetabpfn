from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from sklearn.metrics import log_loss, roc_auc_score

if TYPE_CHECKING:
    import numpy as np
    from tabpfn.constants import XType, YType
    from tabpfn import TabPFNClassifier
    from tabpfn.architectures.base import PerFeatureTransformer
    


## TODO (??): since we fit the classifier always on the same data one can generate
## multiple of them caching the preprocessing and then call predict on all of them 
def get_validation_pred_probas(
    clf: TabPFNClassifier,
    model: PerFeatureTransformer,
    X_trains: list[XType],
    y_trains: list[YType],
    X_vals: list[XType | None]
) -> list[np.ndarray]:
    '''
    Perform the forward pass on the splitted train/val datasets,
    using the classifier instance and the transformer model in input.
    Returns a list of predicted probabilities as numpy arrays.
    '''
    list_pred_probas = []
    
    for X_train, y_train, X_val in zip(X_trains, y_trains, X_vals):
        # skip dataset on which the validation is not requested
        if X_val is None: continue
        clf.fit(X_train, y_train)
        clf.model_ = model
        clf.executor_.model = model
        preds = clf.predict_proba(X_val)
        list_pred_probas.append(preds)
    
    return list_pred_probas



def compute_validation_metric(
    pred_probas: list[np.ndarray],
    y_vals: list[YType | None],
    n_classes: list[int | None],
    metric: Literal["log_loss", "roc_auc"]
) -> float:
    '''
    Computes and returns the mean validation metric from the predicted probabilities, 
    the truth labels and the number of classes of each dataset.
    These three components are taken in order from the respective lists.
    The roc auc score is returned negative (-1*auc).
    '''
    # remove Nones from y_vals and n_classes
    y_vals = [y_val for y_val in y_vals if y_val is not None]
    n_classes = [nc for nc in n_classes if nc is not None]

    val_metric_value = 0
    
    for pred_proba, y_val, nc in zip(pred_probas, y_vals, n_classes):
        if metric == "log_loss":
            val_metric_value += log_loss(y_val, pred_proba)
        elif metric == "roc_auc":
            pred_proba = pred_proba[:, 1] if nc == 2 else pred_proba
            val_metric_value += -1 * roc_auc_score(
                y_val, 
                pred_proba,
                average="macro",
                multi_class="raise" if nc == 2 else "ovr"
            )
        else:
            raise ValueError("Unsupported validation metric.")

    return val_metric_value/len(pred_probas)