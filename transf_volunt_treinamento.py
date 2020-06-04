#!/usr/local/anaconda/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 03 10:04:54 2020

@author: Daniel Guilhon

Arquivo utiliza os dados gerados pelo transf_volunt_features.py
para treinar os modelos, gerar os X_test e y_test (final), salvar as previsoes
e um dump do modelo para posterior geração dos resultados
"""
#%% imports
import pickle
import statistics
import numpy as np
import pandas as pd
from scipy.stats import randint
from numpy.random import RandomState
from functools import partial
from warnings import filterwarnings

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rcParams

from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file

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

from sklearn.exceptions import ConvergenceWarning

from hyperopt import fmin, tpe, hp, anneal, Trials

import transf_volunt_features as tv

filterwarnings(action='ignore', category=ConvergenceWarning)

### XGBOOST
## funcao a ser minimizada - retorna valor negativo pois trata-se do recall
def xgb_cv(params, random_state, cv, X, y):
    # funcao utilizada pelo Hyperopt fmin
    # recebe os params vindo do espaco de busca de hiperparametros
    params = {  'n_estimators': int(params['n_estimators']), 
                'max_depth': int(params['max_depth']), 
                'learning_rate': params['learning_rate'],
                'min_child_weight': int(params['min_child_weight']),
                'subsample': params['subsample'],
                'gamma': params['gamma'],
                'colsample_bytree': params['colsample_bytree'],
                'alpha': int(params['alpha']),
                'lambda': params['lambda'],
                'scale_pos_weight': params['scale_pos_weight']
             }
    
    # para cada chamada, criamos um classificador com os parametros
    model = XGBClassifier(**params,
                objective = 'binary:logistic',
                eval_metric= 'aucpr',
                nthread=4,
                seed=random_state)
    
    # faz o cross_validation (k-fold, k=10)
    # devemos retornar negativo pois a metrica e minimizada pelo fmin do hyperopt
    # opcoes de scoring: average_precision (simula aucpr), roc_auc, recall, precision, f1
    score = -cross_val_score(model, X, y, cv=cv, scoring="average_precision", n_jobs=-1).mean()

    return score

#### LOGISTIC REGRESSION
## funcao a ser minimizada - retorna valor negativo pois trata-se do recall
def log_cv(params, random_state, cv, X, y):
    # funcao utilizada pelo Hyperopt fmin
    # recebe os params vindo do espaco de busca de hiperparametros
    params = {  'C': params['C'], 
                'intercept_scaling': params['intercept_scaling'],
                'fit_intercept': params['fit_intercept'],
                'penalty': params['penalty'], # normalizacao L1 ou L2 utilizada
                'l1_ratio': params['l1_ratio'] # taxa de normalizacao
             }
    
    # utilizamos os parametros passados para criar o modelo
    model = LogisticRegression(
                **params,
                random_state=seed, # para ser reproduzivel
                solver='saga',  # algoritmo de otimizacao utilizado
                class_weight = 'balanced' #dá o devido peso ao balanceamento entre classes
                )
    
    # faz o cross_validation  (k-fold, k=10)
    # devemos retornar negativo pois a metrica e minimizada pelo fmin do hyperopt
    # opcoes de scoring: average_precision (simula aucpr), roc_auc, recall, precision, f1
    score = -cross_val_score(model, X, y, cv=cv, scoring="average_precision", n_jobs=-1).mean()

    return score


### MLP
## funcao a ser minimizada - retorna valor negativo pois trata-se do recall
def mlp_cv(params, random_state, cv, X, y):
    # funcao utilizada pelo Hyperopt fmin
    # recebe os params vindo do espaco de busca de hiperparametros
    
    params = {
            'hidden_layer_sizes': params['hidden_layer_sizes'],
            'alpha': params['alpha'],
            'activation': params['activation'],
            'solver': params['solver']
             }
    
    # utilizamos os parametros passados para criar o modelo
    model = MLPClassifier(
                **params,
                )
    
    # faz o cross_validation  (k-fold, k=10)
    # devemos retornar negativo pois a metrica eh minimizada pelo fmin do hyperopt
    # opcoes de scoring: average_precision (simula aucpr), roc_auc, recall, precision, f1
    score = -cross_val_score(model, X, y, cv=cv, scoring="average_precision", n_jobs=-1).mean()

    return score
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DEFINICOES INICIAIS %%%%%%%%%%%%%%%%%%%%%%

#seed a ser utilizada para replicação
seed = 6439
# k-fold estratificado, preserva a proposcao de positivos e negativos
kf = StratifiedKFold(n_splits=10)
# controla a quantidade de iteracoes de otimizacao q sera feita pelo Hyperopt
n_iter = 2

desbalanceado = ['onehot_all', 'onehot_sem_municipio', 'onehot_sem_municipio_orgao']

balanceado = ['smote_10_1', 'smote_5_1', 'smote_1_1',
 'nearmiss_10_1', 'nearmiss_5_1', 'nearmiss_1_1'
 'smote_nearmiss_10_1', 'smote_nearmiss_5_1', 'smote_nearmiss_1_1']


##################################################################################

#%%%%%%%%%%%%%%%%%%% TRAIN %%%%%%%%%%%%%%%%%%%%%%
# feature_names e X_data, y_data  dependem da base q sera utilizada
# geramos varias bases com features diferentes, com apenas 1 orgao (22000), etc
# balanceado vs desbalanceado
# categoricos one_hot, features normalizadas, 
# outras mais para igualar a comparação dos testes. 
############ DADOS DESBALANCEADOS #########################################################
############ loop para ler os dados em formato onehot #####################################
for rodada in desbalanceado:
    print("RODADA INICIADA - {}".format(rodada))
    feature_names = tv.Load_Obj('feature_names_' + rodada)
    X_data, y_data = load_svmlight_file('desbalanceado_' + rodada + '.svm', n_features = len(feature_names))# pylint: disable=unbalanced-tuple-unpacking

    # executa a normalizacao dos dados
    scaler = StandardScaler(with_mean=False)
    X_data = scaler.fit_transform(X_data)

    #faz o split entre treino/validacao e teste
    #stratify mantem a proporcao entre classes pos/neg
    X_train_cv, X_test, y_train_cv, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=seed, stratify=y_data)

    tv.Save_Obj(X_test, 'X_test_'+rodada)
    tv.Save_Obj(X_test, 'y_test_'+rodada)
    
    #%% main loop 
################### CLASSIFICACAO INICIAL COM PARAMETROS PADRAO

    clf_xgb = XGBClassifier( #default parameters
                    learning_rate=0.3,
                    gamma=0,
                    max_depth = 6,
                    subsample=1,
                    colsample_bytree=1,
                    min_child_weight = 1,
                    alpha = 0,
                    scale_pos_weight = 20,
                    objective = 'binary:logistic',
                    eval_metric= 'aucpr',
                    n_estimators=150,
                    n_jobs=-1,
                    seed=seed
    )

    clf_log = LogisticRegression(
                    random_state=seed,
                    solver='saga', 
                    class_weight = 'balanced',
                    n_jobs=-1
                    )

    clf_mlp = MLPClassifier(
                    random_state=seed,
                    hidden_layer_sizes=(100,),
                    activation="relu", 
                    solver='adam', 
                    alpha=0.0001, 
                    batch_size='auto', 
                    learning_rate="constant", 
                    learning_rate_init=0.001, 
                    power_t=0.5, 
                    max_iter=200, 
                    shuffle=True, 
                    tol=1e-4, 
                    momentum=0.9, 
                    nesterovs_momentum=True, 
                    validation_fraction=0.1, 
                    beta_1=0.9, 
                    beta_2=0.999, 
                    epsilon=1e-8,
                    early_stopping=True,
                    n_iter_no_change=10
                    )

    classificadores = [clf_xgb, clf_log, clf_mlp]

    previsoes = {}
    for clf in classificadores:
        print("FIT Desbalanceado: {}".format(type(clf).__name__))
        clf.fit(X_train_cv, y_train_cv)
        clf_pred_test = clf.predict(X_test)
        clf_pred_proba_test = clf.predict_proba(X_test)
        acc, prec, rec, spec, f_m = tv.calcula_scores(y_test, clf_pred_test)
        auc = roc_auc_score(y_test, clf_pred_proba_test[:,1])
        previsoes[type(clf).__name__] = {'ACU': "{:.3f}".format(acc),
                                        'SEN':"{:.3f}".format(rec),
                                        'ESP':"{:.3f}".format(spec),
                                        'PRE':"{:.3f}".format(prec),
                                        'F-measure':"{:.3f}".format(f_m),
                                        'AUC': "{:.3f}".format(auc)
                                    }
############## GERA TABELA COM RESULTADOS DESBALANCEADOS #######################################

    previsoes_df = pd.DataFrame(previsoes)
    with open("table_result_1_desbalanc"+rodada+".tex", "w") as f:
        f.write("\\begin{table}[H]\n\\label{table:result:1:desbalanc}\n\\centering\n\\caption{Resumo das métricas para dados desbalanceados sem otimização de hiperparâmetros}\n")
        f.write(previsoes_df.transpose().to_latex())
        f.write("\\end{table}")
    print("Table Latex table_result_1_desbalanc_{}.tex Gerada".format(rodada))

############ GERA FIGURA COM FEATURE IMPORTANCE ###################################################
    tv.Gera_Figura_Feature_Importance(classificadores[0], rodada, feature_names)

############ OTIMIZAÇÃO #########################################################################
    # sum(negative instances) / sum(positive instances)
    # scale_pos_weight eh a proporcao neg/ pos
    # vamos utilizar 20, 10, 5, 1
    previsoes = {}
    for clf in classificadores:
        neg_pos_rate = 20
        # espaco de busca
        if type(clf).__name__ == 'XGBClassifier':
            space={ 'n_estimators': hp.uniformint('n_estimators', 50, 250),
                    'max_depth' : hp.uniformint('max_depth', 1, 14),
                    'learning_rate': hp.loguniform('learning_rate', -5, 0),
                    'min_child_weight': hp.uniformint('min_child_weight', 1, 10),
                    'subsample': hp.uniform('subsample', 0.7, 1.0),
                    'gamma': hp.uniform('gamma', 0.5, 1.2),
                    'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1.0),
                    'alpha': hp.uniformint('alpha', 1, 2),
                    'lambda': hp.uniform('lambda', 1.0, 2.0),
                    'scale_pos_weight': neg_pos_rate
                }
            func_obj = partial(xgb_cv, random_state=seed, cv=kf, X=X_train_cv, y=y_train_cv)

        if type(clf).__name__ == 'LogisticRegression':
            logit_fit = [False, True]
            logit_penalty = ['elasticnet']#['l1','l2'] #elasticnet e mais flexivel, com l1_ratio 0, vira L2, em 1, vira L1
            space = {
                'C': hp.loguniform('C', low=-4*np.log(10), high=4*np.log(10)), #
                'intercept_scaling': hp.loguniform('intercept_scaling', -8*np.log(10), 8*np.log(10)),
                'fit_intercept': hp.choice('fit_intercept', logit_fit),
                'penalty': hp.choice('penalty', logit_penalty), # normalizacao L1 ou L2 utilizada
                'l1_ratio': hp.uniform('l1_ratio', 0, 1) # taxa de normalizacao 0 < l1_ratio <1, combinacao L1/L2.
            }
            func_obj = partial(log_cv, random_state=seed, cv=kf, X=X_train_cv, y=y_train_cv)
        
        if type(clf).__name__ == 'MLPClassifier':
            mlp_activation = ['relu', 'logistic', 'tanh']
            mlp_solver = ['sgd', 'adam']
            space = {
                'learning_rate': hp.loguniform('learning_rate', -6.9, 0.0),
                'hidden_layer_sizes': hp.uniformint('hidden_layer_sizes', 10, 100),
                'alpha': hp.loguniform('alpha', -8*np.log(10), 3*np.log(10)),
                'activation': hp.choice('activation', mlp_activation),
                'solver': hp.choice('solver', mlp_solver)
            }
            func_obj = partial(mlp_cv, random_state=seed, cv=kf, X=X_train_cv, y=y_train_cv)
        # trials will contain logging information
        trials = Trials()
        #opcoes para otimizacao sao tpe.suggest e anneal.suggest
        otimiza = anneal.suggest
 
        best = fmin(fn=func_obj, # function to optimize
                space=space, #search space
                algo=otimiza, # optimization algorithm, hyperotp will select its parameters automatically
                max_evals=n_iter, # maximum number of iterations
                trials=trials, # logging
                rstate= RandomState(seed) # fixing random state for the reproducibility
            )
        
        if type(clf).__name__ == 'XGBClassifier':
            # computing the score on the test set
            best_params = { 
                'n_estimators': int(best['n_estimators']), 
                'max_depth': int(best['max_depth']), 
                'learning_rate': best['learning_rate'],
                'min_child_weight': int(best['min_child_weight']),
                'subsample': best['subsample'],
                'gamma': best['gamma'],
                'colsample_bytree': best['colsample_bytree'],
                'alpha': int(best['alpha']),
                'lambda': best['lambda'],
            }
            model = XGBClassifier(**best_params,
                                    objective = 'binary:logistic',
                                    eval_metric= 'aucpr',
                                    nthread=4,
                                    scale_pos_weight=neg_pos_rate,
                                    seed=seed
                                )
            hyperopt_results=np.array([[-x['result']['loss'],
                      x['misc']['vals']['learning_rate'][0],
                      x['misc']['vals']['max_depth'][0],
                      x['misc']['vals']['min_child_weight'][0],
                      x['misc']['vals']['n_estimators'][0],
                      x['misc']['vals']['colsample_bytree'][0],
                      x['misc']['vals']['subsample'][0],
                      x['misc']['vals']['gamma'][0],
                      x['misc']['vals']['alpha'][0],
                      x['misc']['vals']['lambda'][0]] for x in trials.trials])

        if type(clf).__name__ == 'LogisticRegression':
            
            best_params = { 
                'C': best['C'], 
                'penalty': logit_penalty[best['penalty']],
                'intercept_scaling': best['intercept_scaling'],
                'fit_intercept': logit_fit[best['fit_intercept']],
                'l1_ratio': best['l1_ratio']
            }
            model = LogisticRegression(**best_params,
                                    random_state=seed,
                                    n_jobs = -1,
                                    solver='saga',
                                    class_weight = 'balanced',
                                )

        if type(clf).__name__ == 'MLPClassifier':

            best_params = { 
                'hidden_layer_sizes': int(best['hidden_layer_sizes']),
                'alpha': best['alpha'],
                'activation': mlp_activation[best['activation']],
                'solver': mlp_solver[best['solver']]
            }
            model = MLPClassifier(**best_params,
                        random_state=seed,
                    )

        model.fit(X_train_cv, y_train_cv)
        
        if type(clf).__name__ == 'XGBClassifier':
            tv.Gera_Figura_Feature_Importance(model, rodada, feature_names)

        tv.Save_Obj(model, 'model_'+type(clf).__name__+'_'+rodada+'_otimizado')

        clf_pred_test = model.predict(X_test)
        clf_pred_proba_test = model.predict_proba(X_test)
        acc, prec, rec, spec, f_m = tv.calcula_scores(y_test, clf_pred_test)
        auc = roc_auc_score(y_test, clf_pred_proba_test[:,1])
        previsoes[type(clf).__name__] = {'ACU': "{:.3f}".format(acc),
                                        'SEN':"{:.3f}".format(rec),
                                        'ESP':"{:.3f}".format(spec),
                                        'PRE':"{:.3f}".format(prec),
                                        'F-measure':"{:.3f}".format(f_m),
                                        'AUC': "{:.3f}".format(auc)
                                    }
    
    previsoes_df = pd.DataFrame(previsoes)
    with open("table_result_1_desbalanc"+rodada+"otimizado.tex", "w") as f:
        f.write("\\begin{table}[H]\n\\label{table:result:1:desbalanc}\n\\centering\n\\caption{Resumo das métricas para dados desbalanceados sem otimização de hiperparâmetros}\n")
        f.write(previsoes_df.transpose().to_latex())
        f.write("\\end{table}")
    

    # %%
    # plotting the results of optimization
    tv.Gera_Figura_Hiperopt_Otimizacao(hyperopt_results, rodada)  

# %%
# plot_roc_curve(model, X_test, y_test)
# plot_precision_recall_curve(model, X_test, y_test)