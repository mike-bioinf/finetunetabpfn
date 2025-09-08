from __future__ import annotations

import math
import time
import torch
import numpy as np
import pandas as pd
from dataclasses import asdict
from typing import Literal, TYPE_CHECKING
from functools import partial
from copy import deepcopy
from sklearn.utils import check_X_y
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
from tabpfn.constants import XType, YType
from tabpfn.utils import meta_dataset_collator
from torch.utils.data import DataLoader
from torch.optim import Adam
from finetabpfn.setup import build_instance_setup
from finetabpfn.utils.training import resolve_device, _train_test_split
from finetabpfn.utils.model import save_model, resolve_model_path

from finetabpfn.utils.general import (
    enlist, 
    create_logger, 
    get_metric_name,
    suppress_sklearn_and_tabpfn_warnings
)

from finetabpfn.utils.validation import get_validation_pred_probas, compute_validation_metric
from finetabpfn.adaptive_early_stopping import AdaptiveEarlyStopping
from finetabpfn.setup import FineTuneSetup, AESetup, TabPFNClassifierParams

if TYPE_CHECKING:
    from pathlib import Path
    from torch import device
    from tabpfn.architectures.base import PerFeatureTransformer



class AesFineTunerTabPFNClassifier:
    '''
    Finetune TabPFN classifier models with a simple adaptive early stopping strategy.

    Parameters:
        Xs (XType | list[XType]): 
            Datasets on which the model is finetuned.
        
        ys (YType | list[YType]): 
            Labels of the datasets on which the model is finetuned.
        
        use_for_validation (None | bool | list[bool], optional): 
            Indicates which datasets are used in validation.
            If a list must be of the same lenght as Xs and made of booleans.
            If None all datasets are used for validation.
            Note that the datasets not used in validation are 
            enterely used for training.
        
        model_path (str | Path | Literal['default', 'old_default'], optional):
            Filepath of the tabpfn model to finetune.
            - If "default" or 'old_default' the model will be downloaded upon first use in your system cache directory.
            "default" points to the new current default that is a model post-trained on real data.
            "old_default" points to the old default trained on synthetic data only. 
            - If a path or a string of a path, the model will be loaded from the user-specified 
            location if available, otherwise it will be downloaded to this location.
         
        learning_rate (float, optional): 
            Learning rate to use.
        
        batch_size (int, optional): 
            Batch size to use (for now enforced to 1 by tabpfn).
        
        n_accumulation_steps (int, optional):
            Number of training steps in which the gradients are accumulated.
            Keep in mind that at each step we train on batch obtained from 
            one or multiple datasets, depending on the number of training datasets
            and the batch_size parameter. 

        tabpfn_classifier_params (Literal["default"] | dict | TabPFNClassifierParams, optional):
            Parameters of the tabpfn classifier instance used in finetuning. 
        
        finetune_setup (Literal["default"] | dict | TabPFNClassifierParams, optional): 
            Finetune specifics.
        
        aes_setup (Literal["default"] | dict | AESetup, optional): 
            Specifics of the adaptive early stopping procedure.
        
        optimizer (Literal["adam"], optional): 
            Optimizer to use.
         
        seed (int, optional): 
            Seed used for reproducibility.

        device (str | device | Literal['auto'], optional):
            The device to use for finetuning. 
            If "auto", the device is "cuda" if available, otherwise "cpu".
        
        log (bool, optional): 
            Whether to log the finetune metrics.
            The log informs about the finetune metrics computed at each step.
            The log is emitted at debug level and directed to stderr.
    '''
    def __init__(
        self,
        Xs: XType | list[XType],
        ys: YType | list[YType],
        use_for_validation: None | bool | list[bool] = None,
        model_path: str | Path | Literal['default', 'old_default'] = "default",
        learning_rate: float = 1e-5,
        batch_size: int = 1,
        n_accumulation_steps: int = 1,
        tabpfn_classifier_params: Literal["default"] | dict | TabPFNClassifierParams = "default", 
        finetune_setup: Literal["default"] | dict | FineTuneSetup = "default",
        aes_setup: Literal["default"] | dict | AESetup = "default",
        optimizer: Literal["adam"] = "adam",
        seed: int = 0,
        device: str | device | Literal['auto'] = "auto",
        log = True
    ):
        self.Xs, self.ys = self._build_Xys(Xs, ys)
        self.use_for_validation = self._build_use_for_validation(use_for_validation, self.Xs)
        self.model_path = resolve_model_path(model_path)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_accumulation_steps = n_accumulation_steps
        self.tcp = build_instance_setup(TabPFNClassifierParams, tabpfn_classifier_params)
        self.fts = build_instance_setup(FineTuneSetup, finetune_setup)._general_check()
        self.aes_setup = build_instance_setup(AESetup, aes_setup)
        self.optimizer = optimizer
        self.seed = seed
        self.device = resolve_device(device)
        self.logger = create_logger() if log else None
        # learned finetune attrs
        self.is_fitted_ = False
        self.name_val_metric_: str = None
        self.name_monitor_val_metric_: str | None = None
        self.best_model_: PerFeatureTransformer = None
        self.checkpoint_config_: dict = None
        self.n_steps_finetune_: int = 0
        self.best_step_: int = 0
        self.best_val_metric_: float = None
        self.init_val_metric_: float = None
        self.steps_train_losses_: list[float] = []
        self.steps_val_metrics_: list[float] = []
        self.steps_monitor_val_metrics_: list[float] | None = None
        self.steps_remaining_patience_: list[int] = []
        self.steps_grad_norms_: list[float] = []
        self.stopping_criteria_: str = None



    @suppress_sklearn_and_tabpfn_warnings
    def fit(self) -> float:
        '''
        Finetune the model. 
        Returns the final/best validation loss for optimization scenario.
        '''
        self.name_val_metric_ = get_metric_name(self.fts.validation_metric)
        self.name_monitor_val_metric_ = get_metric_name(self.fts.monitor_validation_metric)
        self.steps_monitor_val_metrics_ = None if self.fts.monitor_validation_metric is None else []
        
        datasets_n_classes = self._get_datasets_n_classes()
        X_trains, y_trains, X_vals, y_vals = self._split_xys_in_train_val()
        
        clf_finetuned, clf_validation = self._prepare_classifiers()
        optim_impl = Adam(clf_finetuned.model_.parameters(), self.learning_rate)
        data_loader = self._prepare_data_loader(clf_finetuned, X_trains, y_trains)
        aes = AdaptiveEarlyStopping(**asdict(self.aes_setup))
        loss_fn = torch.nn.CrossEntropyLoss()

        partial_get_validation_pred_probas = partial(
            get_validation_pred_probas,
            clf=clf_validation,
            X_trains=X_trains,
            y_trains=y_trains,
            X_vals=X_vals
        )

        partial_compute_validation_metric = partial(
            compute_validation_metric,
            y_vals=y_vals,
            n_classes=datasets_n_classes
        )

        self.best_model_ = deepcopy(clf_finetuned.model_)
  
        self.init_val_metric_ = partial_compute_validation_metric(
            pred_probas=partial_get_validation_pred_probas(model=self.best_model_),
            metric=self.fts.validation_metric
        )
        
        self.best_val_metric_ = self.init_val_metric_
           
        if self.logger:
            self.logger.debug(
                f"Finetuning with learning rate of {self.learning_rate}," +
                f" batch size of {self.batch_size}," +
                f" and number of accumulation steps of {self.n_accumulation_steps}." +
                "\n" 
            )

        inner_step_train_losses = []
        total_inner_steps = 0
        start_time = time.time()

        # finetune loop
        while not self._signal_stop_finetune(aes, self.n_steps_finetune_, start_time):
            for batch in data_loader:
                # the tensors are on cpu
                X_trains, X_tests, y_trains, y_test, cat_inxs, confs = batch
                clf_finetuned.fit_from_preprocessed(X_trains, y_trains, cat_inxs, confs)
                logits = clf_finetuned.forward(X_tests, return_logits=True)
                
                # computing loss and backward
                loss: torch.Tensor = loss_fn(logits, y_test.to(self.device))
                loss = loss / self.n_accumulation_steps
                loss.backward()

                inner_step_train_losses.append(loss.item())
                total_inner_steps += 1
                
                # manage grads accumulation
                if total_inner_steps % self.n_accumulation_steps == 0:
                    self.n_steps_finetune_ += 1
                    self.steps_train_losses_.append(sum(inner_step_train_losses))
                    inner_step_train_losses = []
                    
                    # grad clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        parameters=clf_finetuned.model_.parameters(),
                        max_norm=1,
                        error_if_nonfinite=True
                    )

                    self.steps_grad_norms_.append(grad_norm.item())

                    # updating params and cleaning grads
                    optim_impl.step()
                    optim_impl.zero_grad()

                    # deepcopy for safety as "suggested" by the finetune example by priorlabs
                    val_pred_probas = partial_get_validation_pred_probas(
                        model=deepcopy(clf_finetuned.model_)
                    )
                    
                    val_metric = partial_compute_validation_metric(
                        pred_probas=val_pred_probas, 
                        metric=self.fts.validation_metric
                    )
                    
                    self.steps_val_metrics_.append(val_metric)
                    
                    if self.fts.monitor_validation_metric is not None:
                        self.steps_monitor_val_metrics_.append(
                            partial_compute_validation_metric(
                                pred_probas=val_pred_probas,
                                metric=self.fts.monitor_validation_metric
                            )
                        )

                    if val_metric < self.best_val_metric_:
                        self.best_val_metric_ = val_metric
                        self.best_model_ = deepcopy(clf_finetuned.model_)
                        self.best_step_ = self.n_steps_finetune_
                        aes.set_best_round(self.best_step_)
                        aes.update_patience()
                    
                    self.steps_remaining_patience_.append(
                        aes.get_remaining_patience(self.n_steps_finetune_)
                    )
                    
                    if self.logger:
                        self._log_finetune_step(
                            start_time, 
                            self.name_val_metric_, 
                            self.name_monitor_val_metric_
                        )
                        
        self.checkpoint_config_ = vars(clf_finetuned.config_)
        self.best_model_ = self.best_model_.to("cpu")
        self.is_fitted_ = True
        return self.best_val_metric_
                    


    def _prepare_classifiers(self) -> tuple[TabPFNClassifier]:
        '''
        We use two tabpfn classifiers in the finetune process.
        One is finetuned and the other is used to validate the finetuned transformer
        on the validation set(s). This separation is needed since the the "fit_mode" 
        and "memory_saving_mode" parameters must be set to different options in the 
        two scenarios.
        '''
        # we set precision to float32 to avoid the slow computation
        # of higher precisions and gradients underflow at lower/mixed precisions.
        # here implementing gradient scaling is not simple due to the metabatch
        clf_finetuned = TabPFNClassifier(
            **asdict(self.tcp),
            model_path=self.model_path, 
            fit_mode="batched",
            memory_saving_mode=False, 
            inference_precision=torch.float32,
            device=self.device
        )

        # this load the model in the instance
        clf_finetuned._initialize_model_variables()
        
        clf_for_validation = TabPFNClassifier(
            **asdict(self.tcp),
            model_path=self.model_path, 
            fit_mode="low_memory",
            inference_precision=torch.float32,
            device=self.device
        )

        return clf_finetuned, clf_for_validation
    


    def _prepare_data_loader(
        self, 
        clf_finetuned: TabPFNClassifier,
        X_trains: list[XType],
        y_trains: list[YType]
    ) -> DataLoader:
        '''Prepare and return the torch DataLoader'''
        # passing a RandomState to ensure variance in splits
        split_fn = partial(
            _train_test_split,
            train_size=self.fts.train_contest_percentage,
            random_state=np.random.RandomState(self.seed)
        )

        datasets_collection = clf_finetuned.get_preprocessed_datasets(
            X_raw=X_trains,
            y_raw=y_trains,
            split_fn=split_fn
        )

        # meta_dataset_collator avoid the default torch collate_fn
        # that is unable to manage our "samples".
        # With batch size of 1 this collate fn has no effect.
        data_loader = DataLoader(
            dataset=datasets_collection, 
            batch_size=self.batch_size,
            collate_fn=meta_dataset_collator,
        )

        return data_loader



    def save_finetuned_model(self, file: str | Path) -> None:
        '''Save the finetuned model to disk in TabPFN-readable checkpoint format'''
        save_model(self.best_model_, file, self.checkpoint_config_)
 


    def collect_finetune_stats(self) -> dict:
        '''
        Collect the finetune statistics into a single dict.
        Revert the negative sign of validation metrics if some.
        Returns the dict.
        '''
        if not self.is_fitted_:
            raise ValueError("The instance is not fitted.")
        
        val_monitor_array = np.full(self.n_steps_finetune_, np.nan)\
            if self.steps_monitor_val_metrics_ is None\
            else np.abs(np.array(self.steps_monitor_val_metrics_))
        
        # we start from step 1 with 0 corresponding to the base model
        df_summary = pd.DataFrame({
            "step": list(range(1, self.n_steps_finetune_ + 1)),
            "train_loss": self.steps_train_losses_,
            "val_metric": np.abs(np.array(self.steps_val_metrics_)),
            "val_monitor_metric": val_monitor_array,
            "gradient_norm_unclipped": self.steps_grad_norms_,
            "remaining_patience": self.steps_remaining_patience_
        })

        summary_dict = {
            "df_finetune": df_summary,
            "name_val_metric": self.name_val_metric_,
            "name_monitor_val_metric": self.name_monitor_val_metric_, 
            "best_step": self.best_step_,
            "init_val_metric": abs(self.init_val_metric_),
            "best_val_metric": abs(self.best_val_metric_),
            "stopping_criteria": self.stopping_criteria_,
        }
        
        return summary_dict
    


    def _log_finetune_step(
        self, 
        start_time: float,
        name_val_metric: str,
        name_monitor_val_metric: str | None
    ) -> None:    
        '''Log the finetune step metrics. The log is raises at debug level.'''
        total_time_spent = round(time.time() - start_time, 2)
        time_remaining = round(max(0, self.fts.time_limit - total_time_spent), 2)

        base_log = (
            f"Step {self.n_steps_finetune_}/{self.fts.max_steps} | "
            #f"Train Loss: {round(self.steps_train_losses_[-1], 5)} | "
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
        elif (
            self.fts.validation_metric == "roc_auc" and 
            math.isclose(-1, self.best_val_metric_, rel_tol=0, abs_tol=1e-5)
        ):
            stopping_criteria = "max_roc_auc"
        
        self.stopping_criteria_ = stopping_criteria
        must_finetune_stop = False if stopping_criteria is None else True
        return must_finetune_stop



    def _get_datasets_n_classes(self, mask_training_only = True) -> list[int | None]:
        '''
        Get the number of classes for each training dataset.
        If "mask_training_only" is True the number of classes for the training only 
        datasets is masked to None.
        The output is a list of integers and/or None that follow the datasets order.
        '''
        datasets_n_classes = []

        for y, to_use_in_validation in zip(self.ys, self.use_for_validation):
            if to_use_in_validation or (not to_use_in_validation and not mask_training_only):
                # y is a np array of pandas series
                datasets_n_classes.append(y.unique().size)
            else:
                datasets_n_classes.append(None)

        return datasets_n_classes

    

    def _split_xys_in_train_val(self) -> tuple[list[XType], list[YType], list[XType | None], list[YType | None]]:
        '''
        Generates the training/validation splits on the input Xs and ys.
        The datasets that are used only in training have the validation sets counterparts
        set to None. The resulting lists have therefore the same lenght in all scenarios.
        Returns the lists of splits in X/y-train/val order.
        '''
        X_trains = []
        y_trains = []
        X_vals = []
        y_vals = []

        for X, y, to_use_in_validation in zip(self.Xs, self.ys, self.use_for_validation):
            if to_use_in_validation:
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
            use_for_validation = [True for _ in range(len(enlist(Xs)))]
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
                raise ValueError("'use_for_validation' must have the same lenght of Xs.")

        if do_check_one_true:
            if not np.array(use_for_validation).any():
                raise ValueError("At least one value in 'use_for_validation' must be True.")

        return use_for_validation
