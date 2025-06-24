from __future__ import annotations

import warnings
import math
import time
import numpy as np
import torch
from dataclasses import asdict
from typing import Literal, TYPE_CHECKING
from functools import partial, wraps
from copy import deepcopy
from sklearn.utils import check_X_y
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
from tabpfn.constants import XType, YType
from tabpfn.utils import meta_dataset_collator
from torch.utils.data import DataLoader
from torch.optim import Adam
from finetabpfn.setup import build_instance_setup
from finetabpfn.utils.training import resolve_device
from finetabpfn.utils.general import enlist, create_logger, get_metric_name
from finetabpfn.utils.validation import iter_validate
from finetabpfn.adaptive_early_stopping import AdaptiveEarlyStopping
from finetabpfn.setup import FineTuneSetup, AESetup, TabPFNClassifierParams

if TYPE_CHECKING:
    from tabpfn.model.transformer import PerFeatureTransformer



class AesFineTuner:
    '''
    Class that implements a simple finetune strategy with an adaptive early stopping.

    Parameters:
        Xs: finetune datas.
        
        ys: finetune datas labels.
        
        use_for_validation: 
            Indicates which data should be used in validation 
            in the early stopping procedure. 
            If a list must be of the same lenght as Xs and made of booleans.
            If None all datas are used for validation.ù
        
        learning_rate: learning rate.
        
        batch_size: batch size (for now enforced by tabpfn to 1).
        
        n_accumulation_steps:
            Number of inner step in which the grads are accumulated.
            If "accumulate_grads_over_datasets" is False then it refers to
            the absolute number of batch on which the accumulation is done.
            If "accumulate_grads_over_datasets" is True then refers to the 
            number of entire rounds over the metabatch.
            Note that in both cases a single batch can have a size greater than 1.

        accumulate_grads_over_datasets: 
            Whether to accumulate the grads over the entire "metabatch".
        
        finetune_setup: Finetune specs.
        
        aes_setup: Specs for the adaptive early stopping procedure.
        
        optimizer: optimizer.
        
        device: device.
        
        inference_precision: 
            Precision to use in the finetune process.
            Can be a torch dtype or "autocast" or "auto".
        
        seed: integer used for reproducibility.
        
        log: Whether to log at debug level the finetune step metrics.
    '''
    def __init__(
        self,
        Xs: XType | list[XType],
        ys: YType | list[YType],
        use_for_validation: None | bool | list[bool] = None,
        ## TODO: add categorical features list of lists or None ??.
        learning_rate: float = 1e-5,
        batch_size: int = 1,
        n_accumulation_steps: int = 1,
        accumulate_grads_over_datasets: bool = True,
        tabpfn_classifier_params: Literal["default"] | dict | TabPFNClassifierParams = "default", 
        finetune_setup: Literal["default"] | dict | FineTuneSetup = "default",
        aes_setup: Literal["default"] | dict | AESetup = "default",
        optimizer: Literal["adam"] = "adam",
        seed: int = 0,
        log = True
    ):
        self.Xs, self.ys = self._build_Xys(Xs, ys)
        self.use_for_validation = self._build_use_for_validation(use_for_validation, Xs)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_accumulation_steps = n_accumulation_steps
        self.accumulate_grads_over_datasets = accumulate_grads_over_datasets
        self.tcp = build_instance_setup(TabPFNClassifierParams, tabpfn_classifier_params)
        self.fts = build_instance_setup(FineTuneSetup, finetune_setup)._general_check()
        self.aes_setup = build_instance_setup(AESetup, aes_setup)
        self.optimizer = optimizer
        self.seed = seed
        self.logger = create_logger() if log else None
        # learned finetune attrs
        self.best_model_: PerFeatureTransformer = None
        self.checkpoint_config_: dict= None
        self.init_val_metric_: float = None
        self.best_val_metric_: float = None
        self.n_steps_finetune_: int = 0
        self.best_step_: int = 0
        self.steps_train_losses_ : list[float] = []
        self.steps_val_metrics_: list[float] = []
        self.steps_monitor_val_metrics_: list[float] | None = None
        self.stopping_criteria_ = None



    def _suppress_sklearn_and_loading_warnings(func):
        '''
        Decorator to filter sklearn future deprecation warnings,
        and tabpfn loading warning.
        '''
        @wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", module="sklearn")
                warnings.filterwarnings("ignore", message=".*", module=".*tabpfn.*loading")
                return func(*args, **kwargs)
        return wrapper



    @_suppress_sklearn_and_loading_warnings
    def fit(self) -> float:
        '''
        Finetune the model. 
        Returns the final/best validation loss for optimization scenario.
        '''
        n_training_datasets = len(self.Xs)
        device = resolve_device(self.tcp.device)
        val_metric_name = get_metric_name(self.fts.validation_metric)
        monitor_val_metric_name = get_metric_name(self.fts.monitor_validation_metric)

        aes = AdaptiveEarlyStopping(**asdict(self.aes_setup))
        clf = TabPFNClassifier(**asdict(self.tcp), fit_mode="batched")
        clf_validation = TabPFNClassifier(**asdict(self.tcp), fit_mode="low_memory")

        datasets_n_classes = self._get_datasets_n_classes()
        X_trains, y_trains, X_vals, y_vals = self._split_xys_in_train_val()
        
        ## TODO: here we have no stratification and no folds (WHAT TO DO ?)
        split_fn = partial(
            train_test_split,
            train_size=self.fts.train_contest_percentage,
            random_state=self.seed, 
            shuffle=True
        )

        # this load the model in the "model_" attribute
        datasets_collection = clf.get_preprocessed_datasets(
            X_raw=X_trains,
            y_raw=y_trains,
            split_fn=split_fn
        )
        
        # needed to save the model on disk
        checkpoint_config_ = vars(clf.config_)

        # meta_dataset_collator avoid the default torch collate_fn
        # that is unable to manage our "samples".
        # With batch size of 1 this collate fn has no real effect.
        # When the batch size is unlocked maybe one can implement
        # a second collate function to retain the current situation 
        # of single size batch with no padding
        dl = DataLoader(
            dataset=datasets_collection, 
            batch_size=self.batch_size,
            collate_fn=meta_dataset_collator,
        )

        optim_impl = Adam(
            params=clf.model_.parameters(), 
            lr=self.learning_rate
        )

        loss_fn = torch.nn.NLLLoss()

        partial_iter_validate = partial(
            iter_validate,
            clf=clf_validation,
            X_trains=X_trains,
            y_trains=y_trains,
            X_vals=X_vals,
            y_vals=y_vals,
            n_classes=datasets_n_classes,
            print_first_preds=False
        )

        self.steps_monitor_val_metrics_ = None if self.fts.monitor_validation_metric is None else []
        self.best_model_ = deepcopy(clf.model_)
        
        self.init_val_metric_ = partial_iter_validate(
            model=self.best_model_, 
            validation_metric=self.fts.validation_metric
        )
        
        self.best_val_metric_ = self.init_val_metric_
        
        total_n_accumulation_steps = n_training_datasets * self.n_accumulation_steps \
            if self.accumulate_grads_over_datasets \
            else self.n_accumulation_steps
        
        inner_step_train_losses = []
        total_inner_steps = 0
        start_time = time.time()

        # finetune loop
        while not self._signal_stop_finetune(aes, self.n_steps_finetune_, start_time):
            for batch in dl:
                ## TODO: spostare a cuda i tensori ??
                X_trains, X_tests, y_trains, y_tests, cat_inxs, confs = batch
                clf.fit_from_preprocessed(X_trains, y_trains, cat_inxs, confs)
                preds = clf.forward(X_tests)
                
                # converting preds from (B, C, N) --> (N, C)
                preds = preds.movedim(1, 2)
                preds = preds.reshape(-1, preds.shape[-1])
                
                # converting y_tests from (B, N) --> N
                y_tests = y_tests.flatten()
                
                # computing loss and backward
                loss: torch.Tensor = loss_fn(torch.log(preds + 1e-8), y_tests)
                loss = loss / total_n_accumulation_steps
                loss.backward()

                inner_step_train_losses.append(loss.item())
                total_inner_steps += 1
                
                # manage grads accumulation
                if total_inner_steps % total_n_accumulation_steps == 0:
                    self.n_steps_finetune_ += 1
                    self.steps_train_losses_.append(sum(inner_step_train_losses))
                    inner_step_train_losses = []
                    
                    # grad clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        parameters=clf.model_.parameters(),
                        max_norm=1,
                        error_if_nonfinite=True
                    )

                    # updating params and cleaning grads
                    optim_impl.step()
                    optim_impl.zero_grad()

                    # deepcopy for safety as suggested by the "official finetune example" by priorlabs
                    val_metric = partial_iter_validate(
                        model=deepcopy(clf.model_),
                        validation_metric=self.fts.validation_metric
                    )

                    self.steps_val_metrics_.append(val_metric)
                    
                    if val_metric < self.best_val_metric_:
                        self.best_val_metric_ = val_metric
                        self.best_model_ = deepcopy(clf.model_)
                        self.best_step_ = self.n_steps_finetune_
                        aes.set_best_round(self.best_step_)
                        aes.update_patience()
                    
                    # deepcopy for safety as suggested by the "official finetune example" by priorlabs
                    if self.fts.monitor_validation_metric is not None:
                        self.steps_monitor_val_metrics_.append(
                            partial_iter_validate(
                                model=deepcopy(clf.model_),
                                validation_metric=self.fts.monitor_validation_metric
                            )
                        )
                    
                    if self.logger:
                        self._log_finetune_step(
                            start_time, 
                            val_metric_name, 
                            monitor_val_metric_name
                        )

        return self.best_val_metric_
                    


    def _collect_finetune_attrs() -> dict:
        '''
        Collect the finetune "learned" attributes into a single dict.
        Revert the negative sign of validation matrics if some.
        Returns the dict.
        '''
        pass

    

    def _log_finetune_step(
        self, 
        start_time: float,
        name_val_metric: str,
        name_monitor_val_metric: str | None
    ) -> None:    
        '''Log the finetune step metrics. The log is raises at DEBUG level.'''
        total_time_spent = round(time.time() - start_time, 2)
        time_remaining = round(max(0, self.fts.time_limit - total_time_spent), 2)

        base_log = (
            f"Step {self.n_steps_finetune_}/{self.fts.max_steps} | "
            f"Train Loss: {round(self.steps_train_losses_[-1], 5)} | "
            f"Initial {name_val_metric}: {abs(round(self.init_val_metric_, 5))} | "
            f"Step {name_val_metric}: {abs(round(self.steps_val_metrics_[-1], 5))} | "
            f"Best {name_val_metric}: {abs(round(self.best_val_metric_, 5))}"
        )

        if name_monitor_val_metric is not None:
            base_log += (
                f" | Step Monitor {name_monitor_val_metric}:"
                f" {abs(round(self.steps_monitor_val_metrics_[-1], 5))}"
            )

        base_log += f" | Total Time Spent: {total_time_spent} | Time Remaining: {time_remaining}"
        self.logger.debug(base_log)



    def _signal_stop_finetune(
        self,
        aes: AdaptiveEarlyStopping,
        current_step: int,
        start_time: float
    ) -> bool:
        '''
        Evaluate whether any stop condition is met.
        In positive cases sets the "stopping_criteria_" attribute and returns True. 
        In negative cases leaves  "stopping_criteria_" as None and returns False.
        '''
        stopping_criteria = None

        if aes.get_remaining_patience(current_step) <= 0:
            stopping_criteria = "patience_termination"
        elif (time.time() - start_time) >= self.fts.time_limit:
            stopping_criteria = "time_termination"
        elif current_step == self.fts.max_steps:
            stopping_criteria = "steps_termination"
        elif self.fts.validation_metric == "roc_auc" and math.isclose(-1, self.best_val_metric_, rel_tol=0, abs_tol=1e-5):
            stopping_criteria = "max_roc_auc"
        
        self.stopping_criteria_ = stopping_criteria
        must_finetune_stop = False if stopping_criteria is None else True
        return must_finetune_stop



    def _get_datasets_n_classes(self, mask_training_only = True) -> list[int | None]:
        '''
        Get the number of classes for each dataset.
        If "mask_training_only" is True the number of classes for the training only 
        datasets is masked to None.
        The output is a list of integers and/or None that follow the datasets order.
        '''
        datasets_n_classes = []

        for y, use_in_validation in zip(self.ys, self.use_for_validation):
            if use_in_validation or (not use_in_validation and not mask_training_only):
                # y should be a np array of pandas series
                datasets_n_classes.append(y.unique().size)
            else:
                datasets_n_classes.append(None)

        return datasets_n_classes

    

    def _split_xys_in_train_val(self) -> tuple[list[XType], list[YType], list[XType | None], list[YType | None]]:
        '''
        Generates the training/validation splits on the input Xs and ys.
        The datasets to use only in training have the validation sets counterparts
        set to None. The resulting lists have therefore the same lenght in all scenarios.
        Returns the lists of splits in the X/y-train/val order.
        '''
        X_trains = []
        y_trains = []
        X_vals = []
        y_vals = []

        for X, y, use_in_validation in zip(self.Xs, self.ys, self.use_for_validation):
            if use_in_validation:
                X_train, X_val, y_train, y_val = train_test_split(
                    X,
                    y, 
                    train_size=self.fts.train_percentage, 
                    random_state=self.seed, 
                    stratify=y
                )
            else:
                X_train = X
                y_train = y
                X_val = None
                y_val = None

            X_trains.append(X_train)
            X_vals.append(X_val)
            y_trains.append(y_train)
            y_vals.append(y_val)
            
        return X_trains, y_trains, X_vals, y_vals



    @staticmethod
    def _build_Xys(Xs: XType, ys: YType) -> tuple[list[XType], list[YType]]:
        '''
        Builds and checks the Xs and ys arguments.
        Returns Xs and ys lists.
        '''
        Xs = enlist(Xs)
        ys = enlist(ys)

        if len(Xs) != len(ys):
            raise ValueError("Xs and ys must have the same lenght.")
        
        for X, y in zip(Xs, ys):
            check_X_y(X, y, ensure_all_finite=False)
        
        return Xs, ys



    @staticmethod
    def _build_use_for_validation(use_for_validation, Xs) -> list[bool]:
        '''
        Construct and check the use_for_validation parameter.
        Returns a list of booleans.
        '''
        if use_for_validation is None:
            do_lenght_check = False
            do_check_one_true = False
            use_for_validation = [True for i in range(len(enlist(Xs)))]
        elif isinstance(use_for_validation, bool):
            do_lenght_check = True
            do_check_one_true = True
            use_for_validation = enlist(use_for_validation)
        elif isinstance(use_for_validation, list):
            do_lenght_check = True
            do_check_one_true = True
            for element in use_for_validation:
                if not isinstance(element, bool):
                    raise ValueError("All elements in use_for_validation list must be booleans.")
        else:
            raise ValueError("use_for_validation argument must be None, bool or a list.")
        
        if do_lenght_check:
            if len(use_for_validation) != len(Xs):
                raise ValueError("use_for_validation_list must have the same lenght of Xs.")

        if do_check_one_true:
            if not np.array(use_for_validation).any():
                raise ValueError("At least one value in use_for_validation list must be True.")

        return use_for_validation
