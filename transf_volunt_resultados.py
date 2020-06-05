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
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import rcParams
style.use('fivethirtyeight')

#%% #################################################################################
'''
plotar a densidade de probabilidade
'''
def Gera_Figura_Densidade_Previsoes(y_test, clf_proba_test, nome):

     y_test_df = pd.DataFrame(y_test)
     clf_proba_test_df = pd.DataFrame(clf_proba_test)

     reprovadas_df = y_test_df.loc[y_test_df[0] == 1 ]
     reprov_df = clf_proba_test_df.iloc[reprovadas_df.index][1]

     aprovadas_df = y_test_df.loc[y_test_df[0] == 0 ]
     aprov_df = clf_proba_test_df.iloc[aprovadas_df.index][1]

     fig = plt.figure(figsize=(15,6))
     
     plt.hist(aprov_df, histtype='stepfilled', bins=50, cumulative=-1, alpha=0.5,
          label="Aprovadas", color="#D73A30", density=True)
     plt.legend(loc="upper left")
     plt.hist(reprov_df, histtype='stepfilled', bins=50, cumulative=True,alpha=0.5,
          label="Reprovadas", color="#2A3586", density=True)
     plt.legend(loc="upper left")
     plt.xlim([0.0,1.0])
     plt.axvline(0.5,0,1,ls='--')
     plt.title('Distribuição de Probabilidades de Contas Aprovadas/Reprovadas')
     plt.xlabel('Probabilidade')
     plt.ylabel('Densidade')
     fig.savefig("densidade_previsoes_{}.png".format(nome))
     print("Figura densidade_previsoes_{}.png Gerada".format(nome))


#%% ##############################################################################
'''
gera tabela latex contendo as metricas das previsoes
'''
def Gera_Tabela_Latex_Previsoes(previsoes, nome):
     previsoes_df = pd.DataFrame(previsoes)
     with open("table_result_1_desbalanc"+nome+".tex", "w") as f:
          f.write("\\begin{table}[H]\n\\label{table:result:1:desbalanc}\n\\centering\n\\caption{Resumo das métricas para dados desbalanceados sem otimização de hiperparâmetros}\n")
          f.write(previsoes_df.transpose().to_latex())
          f.write("\\end{table}")
     print("Table Latex table_result_1_desbalanc_{}.tex Gerada".format(nome))

#%%##############################################################################
'''
gera figura com a curva Roc
dados é um vetor no formato [ {'classificador': nome,
                               'fpr': fpr,
                              'tpr': tpr} ]
'''
def Gera_Figura_Curva_Roc(dados, nome):
     style.use('fivethirtyeight')
     rcParams.update({'figure.autolayout': True})
     fig = plt.figure(figsize=(10,5))
     for dado in dados:
          plt.plot(dado['fpr'], dado['tpr'], label=dado['classificador'])
     
     x=np.linspace(0.0, 1.0, num=len(dados[0]['fpr']))
     plt.plot(x,x,'g--')
     plt.legend(loc='lower right')
     #plt.grid()
     plt.ylabel("Taxa de Verdadeiros Positivos")
     plt.xlabel("Taxa de Falsos Positivos")
     plt.title("Curva ROC - Dados Desbalanceados")
     fig.savefig("roc_curve_desbalanceado_{}.png".format(nome))

# %%
