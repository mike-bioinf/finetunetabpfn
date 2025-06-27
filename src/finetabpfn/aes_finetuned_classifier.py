from __future__ import annotations

from typing import Literal, Sequence, TYPE_CHECKING
from dataclasses import asdict
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from tabpfn import TabPFNClassifier
from finetabpfn.utils.training import resolve_device
from finetabpfn import TabPFNClassifierParams
from finetabpfn.setup import build_instance_setup
from finetabpfn.aes_finetuner_classifier import AesFineTunerTabPFNClassifier

if TYPE_CHECKING:
    import numpy as np
    from torch import dtype
    from torch import device
    from pathlib import Path
    from tabpfn.constants import XType, YType
    from finetabpfn import FineTuneSetup, AESetup



class AesFineTunedTabPFNClassifier(ClassifierMixin, BaseEstimator):
    '''
    Finetune TabPFN classifiers models with a simple adaptive early stopping strategy,
    and then use it to predict on other/new data.
    
    Note: we do not respect the usual signature of sklearn predict methods.
    This is because tabpfn models are in-context learners and so one must
    also pass the contest along the test data. Secondly we allow to freely
    specify some of the tabpfn classifier parameters for testing.
    

    Parameters
    ----------------------

    model_path : str | Path | Literal[auto], optional
        Filepath of the TabPFN model to finetune.
        - If "auto", the model will be downloaded upon first use
        into your system cache directory.
        - If a path or a string representing a path, the model will be loaded 
        from the specified location if available, otherwise it will be downloaded 
        to this location.

    learning_rate : float, optional
        Learning rate to use.

    batch_size : int, optional
        Batch size to use (currently enforced to 1 by TabPFN).

    n_accumulation_steps : int, optional
        Number of inner steps for gradient accumulation.
        If `accumulate_grads_over_datasets` is False, this refers to the absolute 
        number of batches for accumulation.
        If `accumulate_grads_over_datasets` is True, this refers to the number of 
        rounds over the full metabatches.
        Note that a single batch may still have a size greater than 1 in either case.

    accumulate_grads_over_datasets : bool, optional
        Whether to accumulate gradients over the entire metabatch.

    tabpfn_classifier_params : Literal["default"] | dict | TabPFNClassifierParams, optional
        Parameters for the TabPFN classifier instance used in finetuning.

    finetune_setup : Literal["default"] | dict | TabPFNClassifierParams, optional
        Configuration for the finetuning process.

    aes_setup : Literal["default"] | dict | AESetup, optional
        Configuration for the adaptive early stopping procedure.

    optimizer : Literal["adam"], optional
        Optimizer to use.

    seed : int, optional
        Seed for reproducibility.
    
    device: str | device | Literal['auto'], optional
        The device to use for finetuning. 
        If "auto", the device is "cuda" if available, otherwise "cpu".

    log : bool, optional
        Whether to log finetuning metrics.
        Logging provides details about metrics computed at each step. 
        The log is emitted at debug level and directed to stderr.

        
    Attributes
    --------------------

    finetuned_model_ : PerFeatureTransformer
        The finetuned model.
        
    checkpoint_config_ : dict
        The configuration of the base model that has been finetuned.

    stats_finetune_ : dict
        The dictionary containing all the "statistics" of the finetune process.
    '''
    def __init__(
        self,
        model_path: str | Path | Literal['auto'] = "auto",
        learning_rate: float = 1e-5,
        batch_size: int = 1,
        n_accumulation_steps: int = 1,
        accumulate_grads_over_datasets: bool = True,
        tabpfn_classifier_params: Literal["default"] | dict | TabPFNClassifierParams = "default", 
        finetune_setup: Literal["default"] | dict | FineTuneSetup = "default",
        aes_setup: Literal["default"] | dict | AESetup = "default",
        optimizer: Literal["adam"] = "adam",
        seed: int = 0,
        device: str | device | Literal['auto'] = "auto",
        log = True
    ):
        self.model_path = model_path
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_accumulation_steps = n_accumulation_steps
        self.accumulate_grads_over_datasets = accumulate_grads_over_datasets
        self.tabpfn_classifier_params = tabpfn_classifier_params
        self.finetune_setup = finetune_setup
        self.aes_setup = aes_setup
        self.optimizer = optimizer
        self.seed = seed
        self.device = device
        self.log = log

    

    def fit(
        self,
        Xs: XType | list[XType],
        ys: YType | list[YType],
        use_for_validation: None | bool | list[bool] = None
    ) -> "AesFineTunedTabPFNClassifier":
        '''
        Finetune the TabPFN classifier model.

        Parameters
        -------------
        Xs: XType | list[XType] 
            Datasets on which the model is finetuned.
        
        ys: YType | list[YType] 
            Labels of the datasets on which the model is finetuned.
        
        use_for_validation: None | bool | list[bool], optional
            Indicates which datasets are used in validation.
            If a list must be of the same lenght as Xs and made of booleans.
            If None all datasets are used for validation.
            Note that the datasets not used in validation are 
            enterely used for training.

        Returns
        -------------
        AesFineTunedTabPFNClassifier:
            The instance.
        '''
        finetuner = AesFineTunerTabPFNClassifier(
            Xs=Xs,
            ys=ys,
            use_for_validation=use_for_validation,
            model_path=self.model_path,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            n_accumulation_steps=self.n_accumulation_steps,
            accumulate_grads_over_datasets=self.accumulate_grads_over_datasets,
            tabpfn_classifier_params=self.tabpfn_classifier_params,
            finetune_setup=self.finetune_setup,
            aes_setup=self.aes_setup,
            optimizer=self.optimizer,
            seed=self.seed,
            device=self.device,
            log=self.log
        )

        _ = finetuner.fit()
        self.finetuned_model_ = finetuner.best_model_
        self.checkpoint_config_ = finetuner.checkpoint_config_
        self.stats_finetune_ = finetuner.collect_finetune_stats()
        return self

    
    
    def predict(
        self, 
        X_test: XType,
        X_contest: XType, 
        y_contest: YType, 
        categorical_features_indices: Sequence[int] | None = None,
        balance_probabilities: bool = False,
        inference_precision: dtype | Literal['autocast', 'auto'] = "auto",
        memory_saving_mode: bool | float | int | Literal['auto'] = "auto",
        device: str | device | Literal['auto'] = "auto"
    ) -> np.ndarray:
        '''
        Predict the class labels of X_test samples.

        Parameters
        --------------
        X_test : XType
            X test, usually passed to the base tabpfn classifiers in the predict methods.
        X_contest : XType
            X contest, usually passed to the base tabpfn classifiers in the fit method.
        y_contest : YType
            y contest, usually passed to the base tabpfn classifiers in the fit method.
        categorical_features_indices : Sequence[int] | None, optional
            See TabPFNClassifier documentation.
        inference_precision : dtype | Literal['autocast', 'auto'], optional
            See TabPFNClassifier documentation.
        memory_saving_mode : bool | float | int | Literal['auto'], optional
            See TabPFNClassifier documentation.
        device: str | device | Literal['auto'], optional
            See TabPFNClassifier documentation.
        
        Returns
        -------------
        np.ndarray
            The predicted class labels.
        '''
        check_is_fitted(self, attributes="finetuned_model_")
        
        clf = self._set_tabpfn_for_testing(
            categorical_features_indices,
            balance_probabilities, 
            inference_precision, 
            memory_saving_mode,
            device
        )

        clf.fit(X_contest, y_contest)
        resolved_device = resolve_device(device)
        clf.model_ = self.finetuned_model_.to(resolved_device)
        clf.executor_.model = self.finetuned_model_.to(resolved_device)
        return clf.predict(X_test)



    def predict_proba(
        self, 
        X_test: XType,
        X_contest: XType, 
        y_contest: YType, 
        categorical_features_indices: Sequence[int] | None = None,
        balance_probabilities: bool = False,
        inference_precision: dtype | Literal['autocast', 'auto'] = "auto",
        memory_saving_mode: bool | float | int | Literal['auto'] = "auto",
        device: str | device | Literal['auto'] = "auto"
    ) -> np.ndarray:
        '''
        Predict the probabilities of the classes of X_test samples.

        Parameters
        --------------
        X_test : XType
            X test, usually passed to the base tabpfn classifiers in the predict methods.
        X_contest : XType
            X contest, usually passed to the base tabpfn classifiers in the fit method.
        y_contest : YType
            y contest, usually passed to the base tabpfn classifiers in the fit method.
        categorical_features_indices : Sequence[int] | None, optional
            See TabPFNClassifier documentation.
        inference_precision : dtype | Literal['autocast', 'auto'], optional
            See TabPFNClassifier documentation.
        memory_saving_mode : bool | float | int | Literal['auto'], optional
            See TabPFNClassifier documentation.
        device: str | device | Literal['auto'], optional
            See TabPFNClassifier documentation.

        Returns
        ---------------
        np.ndarray
            The predicted class probabilities.
        '''
        check_is_fitted(self, attributes="finetuned_model_")

        clf = self._set_tabpfn_for_testing(
            categorical_features_indices,
            balance_probabilities, 
            inference_precision, 
            memory_saving_mode,
            device
        )

        clf.fit(X_contest, y_contest)
        resolved_device = resolve_device(device)
        clf.model_ = self.finetuned_model_.to(resolved_device)
        clf.executor_.model = self.finetuned_model_.to(resolved_device)
        return clf.predict_proba(X_test)



    def _set_tabpfn_for_testing(
        self,
        categorical_features_indices,
        balance_probabilities,
        inference_precision,
        memory_saving_mode,
        device
    ) -> TabPFNClassifier:
        '''
        Set the tabpfn classifier with the same specs used in finetuning,
        plus the ones that can be freely selected in testing.
        '''
        clf_params = asdict(
            build_instance_setup(TabPFNClassifierParams, self.tabpfn_classifier_params)
        )
        
        # overwriting the params shared between fit and predict
        clf_params["balance_probabilities"] = balance_probabilities

        return TabPFNClassifier(
            **clf_params, 
            fit_mode="low_memory",
            categorical_features_indices=categorical_features_indices,
            inference_precision=inference_precision,
            memory_saving_mode=memory_saving_mode,
            device=device
        )    