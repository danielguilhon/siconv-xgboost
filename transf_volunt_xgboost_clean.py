#!/usr/local/anaconda/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:04:54 2020

@author: guilhon
"""
#%% imports
import pickle
import statistics
import numpy as np
from scipy.stats import randint

import xgboost as xgb
from xgboost import XGBClassifier

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

from hyperopt import fmin, tpe, hp, anneal, Trials

#%% funcoes
## salva objetos com pickle
def SaveObj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

## carrega objetos que foram salvos com pickle
def LoadObj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

#%%%%%%%%%%%%%%%%% TRAIN %%%%%%%%%%%%%%%%%%%%%%
#seed a ser utilizada para replicação
seed = 6439

feature_names = pickle.load(open('feature_names_cat_all.pkl', 'rb'))
X_data, y_data = load_svmlight_file('desbalanceado_cat_all.svm', n_features = len(feature_names))# pylint: disable=unbalanced-tuple-unpacking

X_train_cv, X_test, y_train_cv, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=seed, stratify=y_data)

#%% main loop 
num_round = 200
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
                seed=seed
)
clf.set_params(n_estimators=num_round)

## esse codigo eh mais simples, porem temos menos controle sobre as metricas
# score = cross_val_score(clf, X_train_cv, y_train_cv, cv=kf, scoring='roc_auc', n_jobs=-1)
# print('XGBoost AUC: %r' % (score.mean()))
# nesse as metricas sao mais flexiveis, mas ainda pre-definidas
# score = cross_validate(clf, X_train_cv, y_train_cv, cv=kf, scoring=['roc_auc','precision','recall'], n_jobs=-1)
# print('XGBoost AUC: %r' % (score['test_roc_auc'].mean()))

accuracy = []
precision = []
specificity = []
recall = []
f_measure = []
auc = []

kf = StratifiedKFold(n_splits=10)
%%timeit -n1 -r1
for train_index, val_index in kf.split(X_train_cv, y_train_cv):
    X_train, X_val = X_train_cv[train_index], X_train_cv[val_index]
    y_train, y_val = y_train_cv[train_index], y_train_cv[val_index]
    
    clf.fit(X_train, y_train, eval_metric='auc')

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

#%%%%%%%%%%%%%%%%%% TEST %%%%%%%%%%%%%%%%%%%%%%%%
clf_pred_test = clf.predict(X_test)
clf_pred_proba_test = clf.predict_proba(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, clf_pred_test).ravel()

print('XGBoost AUC: %r' % (roc_auc_score(y_test, clf_pred_proba_test[:,1])))
print('XGBoost Accuracy: %r' % ((tp+tn) / (tp+tn+fp+fn)))
print('XGBoost Precision: %r' % (tp / (tp+fp)))
print('XGBoost Recall: %r' % (tp / (tp+fn)))
print('XGBoost Specificity: %r' % (tn / (tn+fp)))
print('XGBoost F-Measure: %r' % (2*( tp / (tp+fp) )*( tp / (tp+fn) )/(( tp / (tp+fp) )+( tp / (tp+fn) ))))
#%%%%%%%%%%%%%%%%%% OTIMIZAÇÃO %%%%%%%%%%%%%%%%%%
##########   GRID SEARCH   ###################
kf = StratifiedKFold(n_splits=10)

param_grid={'learning_rate': np.logspace(-3, -1, 3),
            'max_depth':  np.linspace(5,12,6,dtype = int),
            'n_estimators': np.linspace(50,250,5, dtype = int),
            'random_state': [seed]}

gs=GridSearchCV(clf, param_grid, scoring='roc_auc', n_jobs=-1, cv=kf, verbose=False)

%%timeit -n1 -r1
gs.fit(X_train_cv, y_train_cv)

#%%###################   RANDOM SEARCH   ################
n_iter=10
param_grid_rand={'learning_rate': np.logspace(-3, 0, 10),
                 'max_depth':  randint(2,14),
                 'n_estimators': randint(50,250),
                 'random_state': [seed]}

rs=RandomizedSearchCV(clf, param_grid_rand, n_iter = n_iter, scoring='recall', n_jobs=-1, cv=kf, verbose=True, random_state=seed)
#%%
%timeit -n1 -r1 rs.fit(X_train_cv, y_train_cv)
#rs_test_score=accuracy_score(y_test, rs.predict(X_test))

rs_results_df=pd.DataFrame(np.transpose([rs.cv_results_['mean_test_score'],
                                         rs.cv_results_['param_learning_rate'].data,
                                         rs.cv_results_['param_max_depth'].data,
                                         rs.cv_results_['param_n_estimators'].data]),
                           columns=['score', 'learning_rate', 'max_depth', 'n_estimators'])
rs_results_df.plot(subplots=True,figsize=(10, 10))

#%%###################   Tree-structured Parzen Estimator   ################
def gb_recall_cv(params, random_state=seed, cv=kf, X=X_train_cv, y=y_train_cv):
    # the function gets a set of variable parameters in "param"
    params = {  'n_estimators': int(params['n_estimators']), 
                'max_depth': int(params['max_depth']), 
                'learning_rate': params['learning_rate'],
                'min_child_weight': int(params['min_child_weight']),
                'subsample': params['subsample'],
                'gamma': params['gamma'],
                'colsample_bytree': params['colsample_bytree'],
                'alpha': int(params['alpha']),
                'lambda': params['lambda']
             }
    
    # we use this params to create a new LGBM Regressor
    model = XGBClassifier(**params,
                scale_pos_weight = 20,
                objective = 'binary:logistic',
                eval_metric= 'auc',
                nthread=4,
                seed=random_state)
    
    # and then conduct the cross validation with the same folds as before
    score = -cross_val_score(model, X, y, cv=cv, scoring="recall", n_jobs=-1).mean()

    return score

# %%
# possible values of parameters
space={ 'n_estimators': hp.uniformint('n_estimators', 50, 250),
        'max_depth' : hp.uniformint('max_depth', 1, 14),
        'learning_rate': hp.loguniform('learning_rate', -5, 0),
        'min_child_weight': hp.uniformint('min_child_weight', 1, 10),
        'subsample': hp.uniform('subsample', 0.7, 1.0),
        'gamma': hp.uniform('gamma', 0.5, 1.2),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1.0),
        'alpha': hp.uniformint('alpha', 1, 2),
        'lambda': hp.uniform('lambda', 1.0, 2.0)
      }

# trials will contain logging information
trials = Trials()
#%%
##%%timeit -n1 -r1 
best=fmin(fn=gb_recall_cv, # function to optimize
    space=space, #search space
    algo=tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
    max_evals=n_iter, # maximum number of iterations
    trials=trials, # logging
    rstate=np.random.RandomState(seed) # fixing random state for the reproducibility
    )
#%%
# computing the score on the test set
model = XGBClassifier(random_state=seed, n_estimators=int(best['n_estimators']),
                      max_depth=int(best['max_depth']),learning_rate=best['learning_rate'])

score = cross_validate(model, X_train_cv, y_train_cv, cv=kf, scoring=['roc_auc','precision','recall'], n_jobs=-1)
print('XGBoost AUC: %r' % (score['test_roc_auc'].mean()))
print('XGBoost Precision: %r' % (score['test_precision'].mean()))
print('XGBoost Recall: %r' % (score['test_recall'].mean()))

print("Best Recall {:.3f} params {}".format( gb_recall_cv(best), best))