#!/usr/local/anaconda/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:04:54 2020

@author: Daniel Guilhon
"""
#%% imports
import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import RandomState
from functools import partial

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

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
    
    # faz o cross_validation
    # devemos retornar negativo pois a metrica eh minimizada pelo fmin do hyperopt
    # opcoes de scoring: average_precision (simula aucpr), roc_auc, recall, precision, f1
    score = -cross_val_score(model, X, y, cv=cv, scoring="average_precision", n_jobs=-1).mean()

    return score


#%%%%%%%%%%%%%%%%% TRAIN %%%%%%%%%%%%%%%%%%%%%%
#seed a ser utilizada para replicação
seed = 6439
# k-fold estratificado, preserva a proposcao de positivos e negativos
kf = StratifiedKFold(n_splits=10)
# controla a quantidade de iteracoes de otimizacao q sera feita pelo Hyperopt
n_iter = 5

# %%
#le os dados em formato onehot
feature_names = pickle.load(open('feature_names_onehot_all.pkl', 'rb'))
X_data, y_data = load_svmlight_file('desbalanceado_onehot_all.svm', n_features = len(feature_names))# pylint: disable=unbalanced-tuple-unpacking

# executa a normalizacao dos dados
scaler = StandardScaler(with_mean=False)
X_data = scaler.fit_transform(X_data)

#faz o split entre treino/validacao e teste (hold-out)
#stratify mantem a proporcao entre classes pos/neg
X_train_cv, X_test, y_train_cv, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=seed, stratify=y_data)

# %%
# fazemos um teste inicial com o modelo em configuracao padrao
clf_mlp = MLPClassifier(
                random_state=seed,
                hidden_layer_sizes=(100,),
                activation="relu", 
                solver='adam', 
                alpha=0.0001, 
                batch_size='auto', 
                learning_rate="constant", 
                learning_rate_init=0.001, 
                power_t=0.5, max_iter=200, 
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

#score = cross_validate(clf_mlp, X_train_cv, y_train_cv, cv=kf, scoring=['roc_auc','precision','recall'], n_jobs=-1)
clf_mlp.fit(X_train_cv, y_train_cv)

# apos o treino, realizamos previsoes com os dados de teste(hold-out)
clf_pred_test = clf_mlp.predict(X_test)
clf_pred_proba_test = clf_mlp.predict_proba(X_test)

acc, prec, rec, spec, f_m = calcula_scores(y_test, clf_pred_test)

print("############  RESULTADOS DO TESTE SEM OTIMIZACAO  #################")
print("MLP AUC: {:.3f}".format(roc_auc_score(y_test, clf_pred_proba_test[:,1])))
print("MLP Accuracy: {:.3f}".format(acc))
print("MLP Precision: {:.3f}".format(prec))
print("MLP Recall: {:.3f}".format(rec))
print("MLP Specificity: {:.3f}".format(spec))
print("MLP F-Measure: {:.3f}".format(f_m))
print("############  RESULTADOS DO TESTE SEM OTIMIZACAO  #################")
# %%
# otimizacao de hiperparametros
# espaco de busca de hiperparametros
mlp_space = {
    'hidden_layer_sizes': hp.uniformint('hidden_layer_sizes', 10, 100),
    'alpha': hp.loguniform('alpha', -8*np.log(10), 3*np.log(10)),
    'activation': hp.choice('activation', ['relu', 'logistic', 'tanh']),
    'solver': hp.choice('solver', ['sgd', 'adam'])
    # 'learning_rate_init': 0.001, 
    #             power_t=0.5,
    #             max_iter=200, 
    #             shuffle=True, 
    #             tol=1e-4, 
    #             momentum=0.9, 
    #             nesterovs_momentum=True, 
    #             validation_fraction=0.1, 
    #             beta_1=0.9, 
    #             beta_2=0.999, 
    #             epsilon=1e-8, 
    #             n_iter_no_change=10
}

# log das tentativas de otimizacao
mlp_trials = Trials()

#opcoes para otimizacao sao tpe.suggest e anneal.suggest
# pela documentacao, anneal tende a convergir melhor, pois faz inferencias com base no historico
mlp_otimiza = anneal.suggest
# partial para passar outros parametros para a funcao objetivo a ser otimizada
mlp_obj = partial(mlp_cv, random_state=seed, cv=kf, X=X_train_cv, y=y_train_cv)

mlp_best = fmin(fn=mlp_obj, # function to optimize
    space=mlp_space, #search space
    algo=mlp_otimiza, # optimization algorithm, hyperotp will select its parameters automatically
    max_evals=n_iter, # maximum number of iterations
    trials=mlp_trials, # logging
    rstate= RandomState(seed) # fixing random state for the reproducibility
    )

# %%
#melhores parâmetros encontrados
best_params = { 
            'hidden_layer_sizes': mlp_best['hidden_layer_sizes'],
            'alpha': mlp_best['alpha'],
            'activation': mlp_best['activation'],
            'solver': mlp_best['solver']
            }
model = MLPClassifier(**best_params,
                        random_state=seed,
                    )

model.fit(X_train_cv, y_train_cv)
# testa os resultados do modelo com os melhores parametros nos dados de teste
clf_pred_test = model.predict(X_test)
clf_pred_proba_test = model.predict_proba(X_test)

acc, prec, rec, spec, f_m = calcula_scores(y_test, clf_pred_test)
auc = roc_auc_score(y_test, clf_pred_proba_test[:,1])

print("############  RESULTADOS DO TESTE APÓS OTIMIZACAO  #################")
print("Logit AUC: {:.3f}".format(roc_auc_score(y_test, clf_pred_proba_test[:,1])))
print("Logit Accuracy: {:.3f}".format(acc))
print("Logit Precision: {:.3f}".format(prec))
print("Logit Recall: {:.3f}".format(rec))
print("Logit Specificity: {:.3f}".format(spec))
print("Logit F-Measure: {:.3f}".format(f_m))

#%% 
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