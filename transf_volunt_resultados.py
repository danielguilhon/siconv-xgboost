#!/usr/local/anaconda/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 19:13:54 2020

@author: Daniel Guilhon

Este arquivo trata os resultados que foram salvos anteriormente no formato pickle
Cada arquivo pickle contem os X_test e y_test usado, as previsoes do modelo, 
previsoes de proba e um dump do modelo em si.
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
'''
plotar a densidade de probabilidade
'''

y_test_df = pd.DataFrame(y_test)

reprovadas_df = y_test_df.loc[y_test_df[0] == 1 ]
reprov_df = clf_proba_test_df.iloc[reprovadas_df.index][1]

aprovadas_df = y_test_df.loc[y_test_df[0] == 0 ]
aprov_df = clf_proba_test_df.iloc[aprovadas_df.index][1]

plt.figure(figsize=(15,6))
plt.hist(aprov_df, histtype='stepfilled', bins=50, cumulative=-1, alpha=0.5,
     label="Aprovadas", color="#D73A30", density=True)
plt.legend(loc="upper left")
plt.hist(reprov_df, histtype='stepfilled', bins=50, cumulative=True,alpha=0.5,
     label="Reprovadas", color="#2A3586", density=True)
plt.legend(loc="upper left")
plt.xlim([0.0,1.0])

plt.title('Distribuição de Probabilidades de Contas Aprovadas/Reprovadas')
plt.xlabel('Probabilidade')
plt.ylabel('Densidade')

#%%
from matplotlib.pyplot import cm
import numpy as np
total = 10
plt.figure(figsize=(5,10))
importances = pd.Series(clf.feature_importances_, feature_names)
features_sorted = importances.sort_values()
total_features = features_sorted[-total:]
total_features.plot.barh(color = iter(cm.rainbow(np.linspace(0,1,total))));