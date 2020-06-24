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

from xgboost.sklearn import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

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
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report

from sklearn.exceptions import ConvergenceWarning

from imblearn.pipeline import Pipeline as imbpipe

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC

from imblearn.under_sampling import NearMiss

from hyperopt import fmin, tpe, hp, anneal, Trials

import transf_volunt_features as tv
import transf_volunt_resultados as tvr

filterwarnings(action='ignore', category=ConvergenceWarning)

### XGBOOST
## funcao a ser minimizada - retorna valor negativo pois trata-se do recall
def xgb_cv(params, random_state, cv, X, y):
    # funcao utilizada pelo Hyperopt fmin
    # recebe os params vindo do espaco de busca de hiperparametros
    new_params = {  'n_estimators': int(params['n_estimators']), 
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
    model = XGBClassifier(**new_params,
                objective = 'binary:logistic',
                eval_metric= 'auc',
                nthread=4,
                seed=random_state)
    
    # faz o cross_validation (k-fold, k=10)
    # devemos retornar negativo pois a metrica e minimizada pelo fmin do hyperopt
    # opcoes de scoring: average_precision (simula aucpr), roc_auc, recall, precision, f1
    score = -cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()

    return score

#### LOGISTIC REGRESSION
## funcao a ser minimizada - retorna valor negativo pois trata-se do recall
def log_cv(params, random_state, cv, X, y):
    # funcao utilizada pelo Hyperopt fmin
    # recebe os params vindo do espaco de busca de hiperparametros
    new_params = {  'C': params['C'], 
                'intercept_scaling': params['intercept_scaling'],
                'fit_intercept': params['fit_intercept'],
                'penalty': params['penalty'], # normalizacao L1 ou L2 utilizada
                'l1_ratio': params['l1_ratio'] # taxa de normalizacao
             }

    # utilizamos os parametros passados para criar o modelo
    model = LogisticRegression(
                **new_params,
                random_state=seed, # para ser reproduzivel
                solver='saga',  # algoritmo de otimizacao utilizado
                class_weight = 'balanced' #dá o devido peso ao balanceamento entre classes
                )
    
    # faz o cross_validation  (k-fold, k=10)
    # devemos retornar negativo pois a metrica e minimizada pelo fmin do hyperopt
    # opcoes de scoring: average_precision (simula aucpr), roc_auc, recall, precision, f1
    score = -cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()

    return score


### MLP
## funcao a ser minimizada - retorna valor negativo pois trata-se do recall
def mlp_cv(params, random_state, cv, X, y):
    # funcao utilizada pelo Hyperopt fmin
    # recebe os params vindo do espaco de busca de hiperparametros
    
    new_params = {
            'learning_rate_init': params['learning_rate_init'],
            'hidden_layer_sizes': params['hidden_layer_sizes'],
            'alpha': params['alpha'],
            'activation': params['activation'],
            'solver': params['solver']
             }

    # utilizamos os parametros passados para criar o modelo
    model = MLPClassifier(
                **new_params,
                )
    
    # faz o cross_validation  (k-fold, k=10)
    # devemos retornar negativo pois a metrica eh minimizada pelo fmin do hyperopt
    # opcoes de scoring: average_precision (simula aucpr), roc_auc, recall, precision, f1
    score = -cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()

    return score
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DEFINICOES INICIAIS %%%%%%%%%%%%%%%%%%%%%%

#seed a ser utilizada para replicação
seed = 6439
# k-fold estratificado, preserva a proposcao de positivos e negativos
kf = StratifiedKFold(n_splits=10)
# controla a quantidade de iteracoes de otimizacao q sera feita pelo Hyperopt
n_iter = 20

dados = ['all', 'sem_municipio', 'sem_municipio_orgao']

##################################################################################

#%%%%%%%%%%%%%%%%%%% TRAIN %%%%%%%%%%%%%%%%%%%%%%
############ DADOS DESBALANCEADOS #########################################################
############ loop para ler os dados em formato onehot #####################################
y = tv.Load_Obj('y_data')

previsoes = {}

for rodada in dados:
    print("RODADA INICIADA - {}".format(rodada))
    
    #dados originais, em formato pandas.DataFrame
    X = tv.Load_Obj('X_data_'+rodada)
 
    #vamos separar colunas categoricas e numericas para aplicar o preprocessamento
    cat_cols = X.columns[X.dtypes == 'O']
    num_cols = X.columns[X.dtypes == 'float64']

    #indice de cada coluna categorica
    cols_idx = []
    col_list = X.columns.tolist()
    for col in num_cols:
        cols_idx.append(col_list.index(col))
    for col in cat_cols:
        cols_idx.append(col_list.index(col))

    feature_names = X.columns.values[cols_idx]

    categories = [
        X[column].unique() for column in X[cat_cols]]

    for cat in categories:
        cat[cat == None] = 'missing'  # noqa

    #transformation para as colunas categoricas
    #valores nulos sao preenchidos com missing
    #ordinalencoder criar categorias numericas
    cat_transf_tree = Pipeline(steps=[
        ('simpleimputer', SimpleImputer(missing_values=None, strategy='constant', fill_value='missing')),
        ('ordinalencoder', OrdinalEncoder(categories=categories))
    ])
    #transformarion para colunas numericas
    #para modelos de arvore, nao usamos o scaler
    num_transf_tree = Pipeline(steps=[
        ('simpleimputer', SimpleImputer(strategy='mean'))
    ])
    #pipeline para executar as transformacao
    column_transf_tree = ColumnTransformer(transformers=[
        ('numericos', num_transf_tree, num_cols),
        ('categoricos', cat_transf_tree, cat_cols)
    ], remainder='passthrough')
        
    #transformatio categoricos para usar com LR e MLP
    #substitui null por missing
    #utiliza onehot enconding
    cat_transf_linear = Pipeline(steps=[
        ('simpleimputer', SimpleImputer(missing_values=None,strategy='constant',fill_value='missing')),
        ('onehotencoder', OneHotEncoder(categories=categories))
    ])
    #transformation numerico
    num_transf_linear = Pipeline(steps=[
        ('simpleimputer', SimpleImputer(strategy='mean')),
        ('standardscaler', StandardScaler())
    ])
    # pipeline da transformacao para LR e MLP
    column_transf_linear = ColumnTransformer(transformers=[
            ('numericos', num_transf_linear, num_cols),
            ('categoricos', cat_transf_linear, cat_cols)
        ], remainder='passthrough')

    #executa a transformacao para tree e LR e MLP
    X_tree = column_transf_tree.fit_transform(X)
    X_linear = column_transf_linear.fit_transform(X)

    #faz o split entre treino/validacao e teste
    #stratify mantem a proporcao entre classes pos/neg
    X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(X_tree, y, test_size=0.1, random_state=seed, stratify=y)
    X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(X_linear, y, test_size=0.1, random_state=seed, stratify=y)

    #%% main loop 
################### CLASSIFICACAO INICIAL COM PARAMETROS PADRAO

    clf_xgb = XGBClassifier( #default parameters
                    learning_rate=0.3,
                    base_score=0.5,
                    gamma=0,
                    max_depth = 6,
                    subsample=1,
                    colsample_bytree=1,
                    min_child_weight = 1,
                    alpha = 0,
                    scale_pos_weight = 20,
                    objective = 'binary:logistic',
                    eval_metric= 'auc',
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
                    max_iter=500, 
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

    previsoes[rodada] = {}
    for clf in classificadores:
        print("FIT Desbalanceado: {}".format(type(clf).__name__))
        if type(clf).__name__ == 'XGBClassifier':
            X_train = X_train_tree.copy()
            y_train = y_train_tree.copy()
            X_test = X_test_tree.copy()
            y_test = y_test_tree.copy()
        else:
            X_train = X_train_linear.copy()
            y_train = y_train_linear.copy()
            X_test = X_test_linear.copy()
            y_test = y_test_linear.copy()
        
        clf.fit(X_train, y_train)
        clf_pred_test = clf.predict(X_test)
        clf_pred_proba_test = clf.predict_proba(X_test)
        acc, prec, rec, spec, f_m = tv.calcula_scores(y_test, clf_pred_test)
        auc = roc_auc_score(y_test, clf_pred_proba_test[:,1])
        previsoes[rodada][type(clf).__name__] = {'ACU': "{:.3f}".format(acc),
                                        'SEN':"{:.3f}".format(rec),
                                        'ESP':"{:.3f}".format(spec),
                                        'PRE':"{:.3f}".format(prec),
                                        'F-measure':"{:.3f}".format(f_m),
                                        'AUC': "{:.3f}".format(auc)
                                    }
    tv.Save_Obj(previsoes, 'previsoes_nao_otimizado_'+rodada)
    tvr.Gera_Tabela_Latex_Previsoes(previsoes, rodada)

    tv.Gera_Figura_Feature_Importance(classificadores[0], rodada, feature_names)
    
    # iteracao nos classificadores, fazendo previsao, calcula precision_recall_curve e roc_curve
    # coloca num dict e passa pra funcao
    #Gera_Figura_Precision_Recall

############ OTIMIZAÇÃO #########################################################################
    # sum(negative instances) / sum(positive instances)
    # scale_pos_weight eh a proporcao neg/ pos
    # vamos utilizar 20, 10, 5, 1

    previsoes[rodada] = {}
    resultados = []
    resultado = {}
    for clf in classificadores:
        neg_pos_rate = 20
        # espaco de busca
        if type(clf).__name__ == 'XGBClassifier':
            
            X_train = X_train_tree.copy()
            y_train = y_train_tree.copy()
            X_test = X_test_tree.copy()
            y_test = y_test_tree.copy()

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
            func = xgb_cv

        if type(clf).__name__ == 'LogisticRegression':
            
            X_train = X_train_linear.copy()
            y_train = y_train_linear.copy()
            X_test = X_test_linear.copy()
            y_test = y_test_linear.copy()

            logit_fit = [False, True]
            logit_penalty = ['elasticnet']#['l1','l2'] #elasticnet e mais flexivel, com l1_ratio 0, vira L2, em 1, vira L1
            space = {
                'C': hp.loguniform('C', low=-4*np.log(10), high=4*np.log(10)), #
                'intercept_scaling': hp.loguniform('intercept_scaling', -8*np.log(10), 8*np.log(10)),
                'fit_intercept': hp.choice('fit_intercept', logit_fit),
                'penalty': hp.choice('penalty', logit_penalty), # normalizacao L1 ou L2 utilizada
                'l1_ratio': hp.uniform('l1_ratio', 0, 1) # taxa de normalizacao 0 < l1_ratio <1, combinacao L1/L2.
            }
            func = log_cv
        
        if type(clf).__name__ == 'MLPClassifier':

            X_train = X_train_linear.copy()
            y_train = y_train_linear.copy()
            X_test = X_test_linear.copy()
            y_test = y_test_linear.copy()

            mlp_activation = ['relu', 'logistic', 'tanh']
            mlp_solver = ['sgd', 'adam']
            space = {
                'learning_rate_init': hp.loguniform('learning_rate_init', -6.9, 0.0),
                'hidden_layer_sizes': hp.uniformint('hidden_layer_sizes', 10, 100),
                'alpha': hp.loguniform('alpha', -8*np.log(10), 3*np.log(10)),
                'activation': hp.choice('activation', mlp_activation),
                'solver': hp.choice('solver', mlp_solver)
            }
            func = mlp_cv
        # trials will contain logging information
        trials = Trials()
        #opcoes para otimizacao sao tpe.suggest e anneal.suggest
        otimiza = anneal.suggest
        func_obj = partial(func, random_state=seed, cv=kf, X=X_train, y=y_train)
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
                'lambda': best['lambda']
            }

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
            
            tv.Save_Obj(hyperopt_results, 'hyperopt_results')

        if type(clf).__name__ == 'LogisticRegression':
            
            best_params = { 
                'C': best['C'], 
                'penalty': logit_penalty[best['penalty']],
                'intercept_scaling': best['intercept_scaling'],
                'fit_intercept': logit_fit[best['fit_intercept']],
                'l1_ratio': best['l1_ratio']
            }

        if type(clf).__name__ == 'MLPClassifier':

            best_params = { 
                'learning_rate_init': best['learning_rate_init'],
                'hidden_layer_sizes': int(best['hidden_layer_sizes']),
                'alpha': best['alpha'],
                'activation': mlp_activation[best['activation']],
                'solver': mlp_solver[best['solver']]
            }

        clf.set_params(**best_params)
        clf.fit(X_train, y_train)
        
        if type(clf).__name__ == 'XGBClassifier':
            tv.Gera_Figura_Feature_Importance(clf, rodada+'_otimizado', feature_names)

        tv.Save_Obj(clf, 'model_'+type(clf).__name__+'_'+rodada+'_otimizado')

        clf_pred_test = clf.predict(X_test)
        clf_pred_proba_test = clf.predict_proba(X_test)
        acc, prec, rec, spec, f_m = tv.calcula_scores(y_test, clf_pred_test)
        auc = roc_auc_score(y_test, clf_pred_proba_test[:,1])
        aucpr = average_precision_score(y_test, clf_pred_proba_test[:,1])
        previsoes[rodada][type(clf).__name__] = {'ACU': "{:.3f}".format(acc),
                                        'SEN':"{:.3f}".format(rec),
                                        'ESP':"{:.3f}".format(spec),
                                        'PRE':"{:.3f}".format(prec),
                                        'F-measure':"{:.3f}".format(f_m),
                                        'AUC': "{:.3f}".format(auc)
                                    }
        resultado['classificador'] = type(clf).__name__
        resultado['auc'] = "{:.3f}".format(auc)
        resultado['aucpr'] = "{:.3f}".format(aucpr)
        resultado['fpr'], resultado['tpr'], _ = roc_curve(y_test, clf_pred_proba_test[:,1])
        resultado['precision'], resultado['recall'], _ = precision_recall_curve(y_test, clf_pred_proba_test[:,1])
        resultados.append(resultado.copy())
########### GERA TABELA COM RESULTADOS DESBALANCEADOS OTIMIZADOS ####################
    tv.Save_Obj(previsoes, 'previsoes_otimizado_'+rodada)
    tvr.Gera_Tabela_Latex_Previsoes(previsoes['all'], rodada+'_otimizado')
    tvr.Gera_Figura_Curva_Roc(resultados, rodada+'_otimizado')
    tvr.Gera_Figura_Curva_Prec_Rec(resultados, rodada+'_otimizado')
    # plotting the results of optimization
    tv.Gera_Figura_Hiperopt_Otimizacao(hyperopt_results, rodada)

    neg_pos = [10, 5, 1]
    sm = SMOTE(random_state=seed)
    nm = NearMiss()
    smnm = imbpipe(steps=[('smote', sm), ('nearmiss', nm) ])
    print("RODADA DE RESAMPLING - {}".format(rodada))
    resamplers=[sm, nm, smnm]
    previsoes[rodada] = {}
    for i in neg_pos:

        taxa = 1/i

        previsoes[rodada][str(i)+'x1'] = {}
        for sampler in resamplers:
            print("RESAMPLER - {}".format(type(sampler).__name__))
            if(type(sampler).__name__ == 'Pipeline'):
                sampler.set_params(steps=[('smote', sm), ('nearmiss', nm)])
            else:
                sampler.set_params(sampling_strategy=taxa)
            
            X_train_tree_res, y_train_tree_res = sampler.fit_resample(X_train_tree, y_train_tree)
            X_train_linear_res, y_train_linear_res = sampler.fit_resample(X_train_linear, y_train_linear)
            
            previsoes[rodada][str(i)+'x1'][type(sampler).__name__] = {}
            for clf in classificadores:
                print("RESAMPLER - {} - CLASSIFICADOR {}".format(type(sampler).__name__, type(clf).__name__))
                if type(clf).__name__ == 'XGBClassifier':
                    
                    clf.set_params(scale_pos_weight=i)
                    
                    X_train = X_train_tree_res.copy()
                    y_train = y_train_tree_res.copy()
                    X_test = X_test_tree.copy()
                    y_test = y_test_tree.copy()
                else:
                    X_train = X_train_linear_res.copy()
                    y_train = y_train_linear_res.copy()
                    X_test = X_test_linear.copy()
                    y_test = y_test_linear.copy()
                
                clf.fit(X_train, y_train)
                clf_pred_test = clf.predict(X_test)
                clf_pred_proba_test = clf.predict_proba(X_test)

                acc, prec, rec, spec, f_m = tv.calcula_scores(y_test, clf_pred_test)
                auc = roc_auc_score(y_test, clf_pred_proba_test[:,1])

                previsoes[rodada][str(i)+'x1'][type(sampler).__name__][type(clf).__name__] = {'ACU': "{:.3f}".format(acc),
                    'SEN':"{:.3f}".format(rec),
                    'ESP':"{:.3f}".format(spec),
                    'PRE':"{:.3f}".format(prec),
                    'F-measure':"{:.3f}".format(f_m),
                    'AUC': "{:.3f}".format(auc)
                }

    tv.Save_Obj(previsoes, 'previsoes_balanceado')
    tvr.Gera_Tabela_Latex_Previsoes(previsoes, rodada)
