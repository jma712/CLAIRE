from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn import svm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import numpy as np
import torch
import CausalModel_law
from utils import *


# constant predictor
def run_constant(x, y, trn_idx, tst_idx, type='linear'):
    y_const = np.mean(y[trn_idx], axis=0)  # the average y in train data
    if type == 'logistic':
        y_const = 1.0 if y_const >= 0.5 else 0.0
    y_pred_tst = np.full(len(tst_idx), y_const)
    return y_pred_tst

# full predictor
def run_full(x, y, env, trn_idx, tst_idx, type='linear'):
    if type == 'linear':
        clf = LinearRegression()  # linear ? logistic ?
    else:
        clf = LogisticRegression(class_weight='balanced')

    features_full = np.concatenate([x, env], axis=1)
    features_full_trn = features_full[trn_idx]
    features_full_tst = features_full[tst_idx]

    clf.fit(features_full_trn, y[trn_idx])  # train

    # test
    y_pred_tst = clf.predict(features_full_tst)
    return y_pred_tst, clf

# unaware predictor
def run_unaware(x, y, trn_idx, tst_idx, type='linear'):
    if type == 'linear':
        clf = LinearRegression()
    else:
        clf = LogisticRegression(class_weight='balanced')

    clf.fit(x[trn_idx], y[trn_idx])  # train
    # test
    y_pred_tst = clf.predict(x[tst_idx])

    return y_pred_tst, clf