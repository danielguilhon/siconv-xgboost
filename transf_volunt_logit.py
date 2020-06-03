#!/usr/local/anaconda/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:04:54 2020

@author: Daniel Guilhon
"""
#%% imports
import pickle
import numpy as np
from numpy.random import RandomState
from functools import partial
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

from sklearn.datasets import load_svmlight_file

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import average_precision_score

from hyperopt import fmin, tpe, hp, anneal, Trials

#%%
def calcula_scores(y_true, model_prediction):
    tn, fp, fn, tp = confusion_matrix(y_true, model_prediction).ravel()
    specificity = tn / (tn+fp)
    accuracy =  (tp+tn) / (tp+tn+fp+fn)
    precision =  tp / (tp+fp)
    recall =  tp / (tp+fn)
    f_measure = 2*precision*recall/(precision+recall)

    return accuracy, precision, recall, specificity, f_measure

def logit_cv(params, random_state, cv, X, y):
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
    
    # faz o cross_validation
    # devemos retornar negativo pois a metrica e minimizada pelo fmin do hyperopt
    # opcoes de scoring: average_precision (simula aucpr), roc_auc, recall, precision, f1
    score = -cross_val_score(model, X, y, cv=cv, scoring="average_precision", n_jobs=-1).mean()

    return score

#%%%%%%%%%%%%%%%%% TRAIN %%%%%%%%%%%%%%%%%%%%%%
#seed a ser utilizada para replicação
seed = 6439
# k-fold estratificado, preserva a proposcao de positivos e negativos
kf = StratifiedKFold(n_splits=10)
# controla a quantidade de iteracoes de otimizacao q sera feita pelo Hyperopt
n_iter = 10

# %%
#le os dados em formato onehot
feature_names = pickle.load(open('feature_names_onehot_all.pkl', 'rb'))
X_data, y_data = load_svmlight_file('desbalanceado_onehot_all.svm', n_features = len(feature_names))# pylint: disable=unbalanced-tuple-unpacking

# executa a normalizacao dos dados
scaler = StandardScaler(with_mean=False)
X_data = scaler.fit_transform(X_data)

#faz o split entre treino/validacao e teste
#stratify mantem a proporcao entre classes pos/neg
X_train_cv, X_test, y_train_cv, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=seed, stratify=y_data)

pickle.dump(X_test, open('x_test_desbalanc_onehot_all.data','wb'))
pickle.dump(y_test, open('y_test_desbalanc_onehot_all.data','wb'))

# %%
# fazemos um teste inicial com o modelo em configuracao padrao
clf_logit = LogisticRegression(
                random_state=seed,
                solver='saga', 
                class_weight = 'balanced'
                )

#score = cross_validate(clf_logit, X_train_cv, y_train_cv, cv=kf, scoring=['roc_auc','precision','recall'], n_jobs=-1)
clf_logit.fit(X_train_cv, y_train_cv)

# pickle.dump(clf_logit, open('clf_logit_default_onehot_all.model','wb'))

# apos o treino, realizamos previsoes com os dados de teste
clf_pred_test = clf_logit.predict(X_test)
clf_pred_proba_test = clf_logit.predict_proba(X_test)

acc, prec, rec, spec, f_m = calcula_scores(y_test, clf_pred_test)

print("############ RESULTADOS DO TESTE LOGIT MODELO PADRÃO #################")
print("Logit AUC: {:.3f}".format(roc_auc_score(y_test, clf_pred_proba_test[:,1])))
print("Logit Accuracy: {:.3f}".format(acc))
print("Logit Precision: {:.3f}".format(prec))
print("Logit Recall: {:.3f}".format(rec))
print("Logit Specificity: {:.3f}".format(spec))
print("Logit F-Measure: {:.3f}".format(f_m))
print("############ RESULTADOS DO TESTE LOGIT MODELO PADRÃO #################")
# %%
# espaco de busca de hiperparametros
logit_fit = [False, True]
logit_penalty = ['l1','l2']

logit_space = {
    'C': hp.loguniform('C', low=-4*np.log(10), high=4*np.log(10)), #
    'intercept_scaling': hp.loguniform('intercept_scaling', -8*np.log(10), 8*np.log(10)),
    'fit_intercept': hp.choice('fit_intercept', [False, True]),
    'penalty': hp.choice('penalty', ['l1', 'l2']), # normalizacao L1 ou L2 utilizada
    'l1_ratio': hp.uniform('l1_ratio', 0, 1) # taxa de normalizacao 0 < l1_ratio <1, combinacao L1/L2.
}

# log das tentativas de otimizacao
logit_trials = Trials()

# opcoes para otimizacao sao tpe.suggest e anneal.suggest
# pela documentacao, anneal tende a convergir melhor, pois faz inferencias com base no historico
logit_otimiza = anneal.suggest

# partial para passar outros parametros para a funcao objetivo a ser otimizada
logit_obj = partial(logit_cv, random_state=seed, cv=kf, X=X_train_cv, y=y_train_cv)

logit_best = fmin(fn=logit_obj, # funcao para otimizar
    space=logit_space, #espaco de busca
    algo=logit_otimiza, # algoritmo de otimizacao que o hyperopt deve utilizar
    max_evals=n_iter, # numero de rodadas de otimizacao
    trials=logit_trials, # logging
    rstate= RandomState(seed) # reprodutibilidade
    )

# %%
#melhores parâmetros encontrados
best_params = { 'C': logit_best['C'], 
                'penalty': logit_penalty[logit_best['penalty']],
                'intercept_scaling': logit_best['intercept_scaling'],
                'fit_intercept': logit_fit[logit_best['fit_intercept']],
                'l1_ratio': logit_best['l1_ratio']
             }

model = LogisticRegression(**best_params,
                        random_state=seed,
                        n_jobs = -1,
                        solver='saga',
                        class_weight = 'balanced',
                    )

# testa os resultados do modelo com os melhores parametros nos dados de teste
model.fit(X_train_cv, y_train_cv)
clf_pred_test = model.predict(X_test)
clf_pred_proba_test = model.predict_proba(X_test)

acc, prec, rec, spec, f_m = calcula_scores(y_test, clf_pred_test)
auc = roc_auc_score(y_test, clf_pred_proba_test[:,1])

print("############ RESULTADOS DO TESTE APÓS OTIMIZACAO #################")
print("Logit AUC: {:.3f}".format(roc_auc_score(y_test, clf_pred_proba_test[:,1])))
print("Logit Accuracy: {:.3f}".format(acc))
print("Logit Precision: {:.3f}".format(prec))
print("Logit Recall: {:.3f}".format(rec))
print("Logit Specificity: {:.3f}".format(spec))
print("Logit F-Measure: {:.3f}".format(f_m))
print("############ RESULTADOS DO TESTE APÓS OTIMIZACAO #################")


fpr, tpr, roc_thresh = roc_curve(y_test, clf_pred_proba_test[:,1])
precision, recall, pr_thresh = precision_recall_curve(y_test, clf_pred_proba_test[:,1])
avg_precision = average_precision_score(y_test, clf_pred_test, average=None)

plt.figure(figsize=(7, 8))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall, precision, color='gold', lw=2)
lines.append(l)
labels.append('Precision-Recall (area = {0:0.2f})'
              ''.format(avg_precision))
fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))


plt.show()

# %%
