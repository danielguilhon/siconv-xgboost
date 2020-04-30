#!/usr/local/anaconda/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:04:54 2020

@author: guilhon
"""
#%% imports
import pickle
import statistics

import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import load_svmlight_files

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

#%% funcoes
## salva objetos com pickle
def SaveObj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

## carrega objetos que foram salvos com pickle
def LoadObj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

#%% treinamento
clf = XGBClassifier(
                learning_rate=0.1,
                max_depth = 7,
                subsample=0.5,
                colsample_bytree=0.5,
                min_child_weight = 3,
                scale_pos_weight = 20,
                objective = 'binary:logistic',
                eval_metric= 'auc',
                n_estimators=200,
                nthread=4,
                seed=6439
)

#seed a ser utilizada para replicação
seed = 6439

feature_names = pickle.load(open('feature_names_cat_all.pkl', 'rb'))
X_data, y_data = load_svmlight_file('desbalanceado_cat_all.svm', n_features = len(feature_names))# pylint: disable=unbalanced-tuple-unpacking

X_train_cv, X_test, y_train_cv, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=seed, stratify=y_data)

kf = StratifiedKFold(n_splits=10)

num_round = 200
clf.set_params(n_estimators=num_round)


accuracy = []
precision = []
specificity = []
recall = []
f_measure = []
auc = []

for train_index, val_index in kf.split(X_train_cv, y_train_cv):
    X_train, X_val = X_train_cv[train_index], X_train_cv[val_index]
    y_train, y_val = y_train_cv[train_index], y_train_cv[val_index]
    clf.fit(X_train, y_train, eval_metric='map')

    clf_pred_proba = clf.predict_proba(X_val)
    clf_pred = clf.predict(X_val)

    tn, fp, fn, tp = confusion_matrix(y_val, clf_pred).ravel()
    
    specificity.append(tn / (tn+fp))
    accuracy.append( (tp+tn) / (tp+tn+fp+fn) )
    precision.append( tp / (tp+fp) )
    recall.append( tp / (tp+fn) )
    f_measure.append( 2*( tp / (tp+fp) )*( tp / (tp+fn) )/(( tp / (tp+fp) )+( tp / (tp+fn) )) )
    auc.append(roc_auc_score(y_val, clf_pred_proba[:,1]))

print('XGBoost AUC: \n\tMédia: %r\n\tDesvio: %r' % (statistics.mean(auc), statistics.stdev(auc)))
print('XGBoost Accuracy: \n\tMédia: %r\n\tDesvio: %r' % (statistics.mean(accuracy), statistics.stdev(accuracy)))
print('XGBoost Precision: \n\tMédia: %r\n\tDesvio: %r' % (statistics.mean(precision), statistics.stdev(precision)))
print('XGBoost Recall: \n\tMédia: %r\n\tDesvio: %r' % (statistics.mean(recall), statistics.stdev(recall)))
print('XGBoost Specificity: \n\tMédia: %r\n\tDesvio: %r' % (statistics.mean(specificity), statistics.stdev(specificity)))
print('XGBoost F-Measure: \n\tMédia: %r\n\tDesvio: %r' % (statistics.mean(f_measure), statistics.stdev(f_measure)))

# %% teste
clf_pred_test = clf.predict(X_test)
clf_pred_proba_test = clf.predict_proba(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, clf_pred_test).ravel()

#print('XGBoost AUC: %r' % (roc_auc_score(y_test, clf_pred_proba[:,1])))
print('XGBoost Accuracy: %r' % ((tp+tn) / (tp+tn+fp+fn)))
print('XGBoost Precision: %r' % (tp / (tp+fp)))
print('XGBoost Recall: %r' % (tp / (tp+fn)))
print('XGBoost Specificity: %r' % (tn / (tn+fp)))
print('XGBoost F-Measure: %r' % (2*( tp / (tp+fp) )*( tp / (tp+fn) )/(( tp / (tp+fp) )+( tp / (tp+fn) ))))
# %%
