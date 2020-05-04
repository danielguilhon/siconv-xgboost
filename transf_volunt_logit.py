#!/usr/local/anaconda/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:04:54 2020

@author: Daniel Guilhon
"""
#%% imports
import pickle
import statistics
import numpy as np
import pandas as pd
from scipy.stats import randint
from numpy.random import RandomState

import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import load_svmlight_files

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import classification_report

from hyperopt import fmin, tpe, hp, anneal, Trials

#%%
def CalculaScores(y_true, model_prediction):
    tn, fp, fn, tp = confusion_matrix(y_true, model_prediction).ravel()
    specificity = tn / (tn+fp)
    accuracy =  (tp+tn) / (tp+tn+fp+fn)
    precision =  tp / (tp+fp)
    recall =  tp / (tp+fn)
    f_measure = 2*precision*recall/(precision+recall)

    return accuracy, precision, recall, specificity, f_measure

def logit_cv(params, random_state=seed, cv=kf, X=X_train_cv, y=y_train_cv):
    # the function gets a set of variable parameters in "param"
    
    params = {  'C': params['C'], 
                'penalty': params['penalty']
             }
    
    # we use this params to create a new LGBM Regressor
    model = LogisticRegression(
                **params,
                random_state=seed,
                solver='liblinear',
                class_weight = 'balanced'
                )
    
    # and then conduct the cross validation with the same folds as before
    #opcoes de scoring: average_precision (simula aucpr), roc_auc, recall, precision, f1
    score = -cross_val_score(model, X, y, cv=cv, scoring="average_precision", n_jobs=-1).mean()

    return score

#%%%%%%%%%%%%%%%%% TRAIN %%%%%%%%%%%%%%%%%%%%%%
#seed a ser utilizada para replicação
seed = 6439
kf = StratifiedKFold(n_splits=10)
n_iter = 20


# %%
feature_names = pickle.load(open('feature_names_onehot_all.pkl', 'rb'))
X_data, y_data = load_svmlight_file('desbalanceado_onehot_all.svm', n_features = len(feature_names))# pylint: disable=unbalanced-tuple-unpacking

scaler = StandardScaler(with_mean=False)
X_data = scaler.fit_transform(X_data)

X_train_cv, X_test, y_train_cv, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=seed, stratify=y_data)

# %%

clf_logit = LogisticRegression(
                random_state=seed,
                n_jobs =-1,
                solver='liblinear',
                class_weight = 'balanced'
                )

#score = cross_validate(clf_logit, X_train_cv, y_train_cv, cv=kf, scoring=['roc_auc','precision','recall'], n_jobs=-1)
clf_logit.fit(X_train_cv, y_train_cv)

clf_pred_test = clf_logit.predict(X_test)
clf_pred_proba_test = clf_logit.predict_proba(X_test)

acc, prec, rec, spec, f_m = CalculaScores(y_test, clf_pred_test)

print("############RESULTADOS DO TESTE SEM OTIMIZACAO#################")
print("Logit AUC: {:.3f}".format(roc_auc_score(y_test, clf_pred_proba_test[:,1])))
print("Logit Accuracy: {:.3f}".format(acc))
print("Logit Precision: {:.3f}".format(prec))
print("Logit Recall: {:.3f}".format(rec))
print("Logit Specificity: {:.3f}".format(spec))
print("Logit F-Measure: {:.3f}".format(f_m))
# %%
logit_space = {
    'C': hp.loguniform('C', low=-4*np.log(10), high=4*np.log(10)),
    'penalty': hp.choice('penalty', ['l1', 'l2']),
    'l1_ratio': hp.uniform('l1_ratio', 0,1)
}

logit_trials = Trials()

#opcoes para otimizacao sao tpe.suggest e anneal.suggest
logit_otimiza = anneal.suggest

logit_best = fmin(fn=logit_cv, # function to optimize
    space=logit_space, #search space
    algo=logit_otimiza, # optimization algorithm, hyperotp will select its parameters automatically
    max_evals=n_iter, # maximum number of iterations
    trials=logit_trials, # logging
    rstate= RandomState(seed) # fixing random state for the reproducibility
    )

# %%
best_params = { 'C': logit_best['C'], 
                'penalty': logit_best['penalty'],
                'l1_ratio': logit_best['l1_ratio']
             }
model = LogisticRegression(**best_params,
                        random_state=seed,
                        solver='liblinear',
                        class_weight = 'balanced',
                        n_jobs=-1
                    )

model.fit(X_train_cv, y_train_cv)
clf_pred_test = model.predict(X_test)
clf_pred_proba_test = model.predict_proba(X_test)

acc, prec, rec, spec, f_m = CalculaScores(y_test, clf_pred_test)
auc = roc_auc_score(y_test, clf_pred_proba_test[:,1])

print("############RESULTADOS DO TESTE APÓS OTIMIZACAO#################")
print("Logit AUC: {:.3f}".format(roc_auc_score(y_test, clf_pred_proba_test[:,1])))
print("Logit Accuracy: {:.3f}".format(acc))
print("Logit Precision: {:.3f}".format(prec))
print("Logit Recall: {:.3f}".format(rec))
print("Logit Specificity: {:.3f}".format(spec))
print("Logit F-Measure: {:.3f}".format(f_m))