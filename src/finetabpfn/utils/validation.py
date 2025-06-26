from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from sklearn.metrics import log_loss, roc_auc_score

if TYPE_CHECKING:
    from tabpfn.constants import XType, YType
    from tabpfn import TabPFNClassifier
    from tabpfn.model.transformer import PerFeatureTransformer
    


## TODO: since we fit the classifier always on the same data one can generate
## multiple of them caching the preprocessing and then call predict on all of them
def iter_validate(
    *,
    clf: TabPFNClassifier,
    model: PerFeatureTransformer,
    X_trains: list[XType],
    y_trains: list[YType],
    X_vals: list[XType | None],
    y_vals: list[YType | None],
    n_classes: list[int | None],
    validation_metric: Literal["log_loss", "roc_auc"],
    print_first_preds: bool     ## for debugging purposes and to remove in production
) -> float:
    '''
    Validates the tabpfn classifier using the input model
    on all the train/val splits, using the input validation metric.
    Note that for "roc_auc" metric the value is returned negative.
    Returns the mean validation metric.
    '''
    val_metric_value = 0
    n_validations = 0

    for X_train, y_train, X_val, y_val, nc in zip(
        X_trains, 
        y_trains,
        X_vals,
        y_vals,
        n_classes
    ):
        # skip dataset on which the validation is not requested
        if X_val is None:
            continue 

        clf.fit(X_train, y_train)
        clf.model_ = model
        clf.executor_.model = model
        preds = clf.predict_proba(X_val)

        # for debugging
        if print_first_preds and n_validations == 0:
            print(preds)
        
        if validation_metric == "log_loss":
            val_metric_value += log_loss(y_val, preds)

        elif validation_metric == "roc_auc":
            preds = preds[:, 1] if nc == 2 else preds
            val_metric_value += -1 * roc_auc_score(
                y_val, 
                preds,
                average="macro",
                multi_class="raise" if nc == 2 else "ovr"
            )

        else:
            raise ValueError("Used a non supported validation metric.")

        n_validations += 1
    
    return val_metric_value / n_validations
