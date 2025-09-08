'''
In this file we test our expectation on some aspects of tabpfn internal machinery on which we rely.
'''

import numpy as np
from tabpfn import TabPFNClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from finetabpfn.setup import FineTuneSetup
from finetabpfn.aes_finetuner_classifier import AesFineTunerTabPFNClassifier



def test_that_predict_methods_uses_executor_model(tmp_path):
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(X, y, train_size=0.8, random_state=10, stratify=y)
    
    clf = TabPFNClassifier()
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_test)

    # download the model in the tmp folder
    clf2 = TabPFNClassifier(model_path=tmp_path/"tabpfn-v2-classifier.ckpt")
    clf2.fit(X_train, y_train)
    clf2.model_ = clf.model_
    clf2.executor_.model = clf.model_
    preds2 = clf2.predict_proba(X_test)
   
    assert np.isclose(preds, preds2, rtol=0, atol=1e-10).all(), "The same tabpfn model does not give the same predictions on the same setting."



def test_that_data_loader_loops_over_datasets_in_order():
    X, y = load_iris(as_frame=True, return_X_y=True)
    Xs = [X.iloc[:-i, :] for i in range(10, 40, 10)]
    ys = [y.iloc[:-i] for i in range(10, 40, 10)]

    finetuner = AesFineTunerTabPFNClassifier(Xs, ys)
    split_value = finetuner.fts.train_contest_percentage
    list_expected_n_rows = [int((X.shape[0]-i) * split_value) for i in range(10, 40, 10)]

    clf_finetune, _ = finetuner._prepare_classifiers()
    data_loader = finetuner._prepare_data_loader(clf_finetune, Xs, ys)

    for i, batch in enumerate(data_loader):
        X_trains, *_ = batch
        actual_n_rows = X_trains[0].shape[1] # (B,R,F)
        expected_n_rows = list_expected_n_rows[i]
        assert actual_n_rows == expected_n_rows, "The dataloader does not respect the training datasets order."
        if i == 2:
            break



def test_that_save_model_utility_works(tmp_path):
    X, y = load_iris(as_frame=True, return_X_y=True)
    fts = FineTuneSetup(max_steps=1)
    finetuner = AesFineTunerTabPFNClassifier(X, y, finetune_setup=fts)
    finetuner.fit()
    file_ckpt = tmp_path / "model_finetuned.ckpt"
    finetuner.save_finetuned_model(file_ckpt)
    assert file_ckpt.exists(follow_symlinks=False), "The save_model utility is not working."