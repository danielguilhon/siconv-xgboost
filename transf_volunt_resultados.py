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
from sqlalchemy import create_engine
from eli5 import show_prediction

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
     with open("table_result_1_desbalanc_"+nome+".tex", "w") as f:
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
###
     fig = plt.figure(figsize=(10,5))
     for dado in dados:
          plt.plot(dado['fpr'], dado['tpr'], label="{} (AUC: {})".format(dado['classificador'], dado['auc']))
###
     x=np.linspace(0.0, 1.0, num=len(dados[0]['fpr']))
     plt.plot(x,x,'g--')
     plt.legend(loc='lower right')
     #plt.grid()
     plt.ylabel("Taxa de Verdadeiros Positivos", fontsize=22)
     plt.xlabel("Taxa de Falsos Positivos", fontsize=22)
     plt.title("Curva ROC - Dados Desbalanceados", fontsize=28)
     fig.savefig("roc_curve_desbalanceado_{}.png".format(nome))
     
#%%##############################################################################
'''
gera figura com a curva Precision-Recall
dados é um vetor no formato [ {'classificador': nome,
                               'precision': precision,
                              'recall': recall} ]
'''
def Gera_Figura_Curva_Prec_Rec(dados, nome):
     fig = plt.figure(figsize=(10,5))
     for dado in dados:
          plt.plot(dado['recall'], dado['precision'], label="{} (AUCPR: {})".format(dado['classificador'], dado['aucpr']))
##     
     x=np.linspace(0.0, 1.0, num=len(dados[0]['precision']))
     y=np.linspace(0.5, 0.5, num=len(dados[0]['precision']))
     plt.plot(x,y,'g--')
     plt.legend(loc='lower right')
     #plt.grid()
     plt.xlabel("Sensibilidade", fontsize=22)
     plt.ylabel("Precisão", fontsize=22)
     plt.title("Curva Precision x Recall - Dados Desbalanceados", fontsize=28)
     fig.savefig("precision_recall_curve_{}.png".format(nome))

# %%
def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.0f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

#%%
def Plot_Analise_Variaveis():
     engine = create_engine("mysql+pymysql://root:Siconv!@localhost/siconv", pool_pre_ping=True,  connect_args = {'use_unicode':True, 'charset':'utf8mb4'})
     total_orgao_df = pd.read_sql("select cod_orgao_sup, count(target) as total, sum(case when target=0 then 1 else 0 end) as negativos, sum(case when target=1 then 1 else 0 end) as positivos from siconv.features GROUP by COD_ORGAO_SUP ORDER  by total DESC;", engine)
     total_instrumento_df = pd.read_sql("select modalidade, count(target) as total, sum(case when target=0 then 1 else 0 end) as negativos, sum(case when target=1 then 1 else 0 end) as positivos from siconv.features GROUP by modalidade ORDER  by total DESC;", engine)
     labels = total_orgao_df.loc[:5,"cod_orgao_sup"]
     labels_instr = total_instrumento_df.loc[:,"modalidade"]
     
     x = np.arange(len(labels)) 
     width = 0.35

     fig = plt.figure(figsize=(10,8))
     
     ax1 = fig.add_subplot(221)
     rects1 = ax1.bar(x - width/2, total_orgao_df.loc[:5,"negativos"], width, label='Negativos')
     rects2 = ax1.bar(x + width/2, total_orgao_df.loc[:5,"positivos"], width, label='Positivos')
     plt.yscale("log")
     # Add some text for labels, title and custom x-axis tick labels, etc.
     ax1.set_ylabel('Total')
     ax1.set_xlabel('Código do Órgão')
     ax1.set_title('Total de Contas por Órgão')
     ax1.set_xticks(x)
     ax1.set_xticklabels(labels)
     ax1.legend()

     autolabel(rects1, ax1)
     autolabel(rects2, ax1)
################################################
     x = np.arange(len(labels_instr)) 
     width = 0.35
     
     ax3 = fig.add_subplot(222)
     
     rects3 = ax3.bar(x - width/2, total_instrumento_df.loc[:5,"negativos"], width, label='Negativos')
     rects4 = ax3.bar(x + width/2, total_instrumento_df.loc[:5,"positivos"], width, label='Positivos')
     plt.yscale("log")
     # Add some text for labels, title and custom x-axis tick labels, etc.
     ax3.set_ylabel('Total')
     ax3.set_xlabel('Instrumento de Repasse')
     ax3.set_title('Total de Contas por Tipo de Instrumento')
     ax3.set_xticks(x)
     ax3.set_xticklabels(labels_instr, rotation=45)
     ax3.legend()

     autolabel(rects3, ax3)
     autolabel(rects4, ax3)

##########################################3

     tbl_features_df = pd.read_sql_table('features',engine) 
     negativos = tbl_features_df.loc[tbl_features_df['TARGET']==0]
     positivos = tbl_features_df.loc[tbl_features_df['TARGET']==1]
     
     ax2 = fig.add_subplot(212)
     ax21 = fig.add_subplot(212)

     ax21.hist(negativos.loc[:,'VL_GLOBAL_CONV'], histtype='stepfilled', bins=1000, alpha=0.5,
               label="Aprovadas", density=False) #color="#2A3586"
     ax21.yaxis.set_label_position("left")
     ax21.yaxis.tick_left()
     ax21.set_yscale("log")

     ax2.hist(positivos.loc[:,'VL_GLOBAL_CONV'], histtype='stepfilled', bins=1000, alpha=0.9,
               label="Reprovadas", density=False)#color="#D73A30"
     ax2.yaxis.set_label_position("right")
     ax2.yaxis.tick_right()
     ax2.set_yscale("log")

     
     #plt.yticks(np.arange(0, 100, 10))
     plt.legend()
     plt.xticks(ticks=[0,1000000,2000000,3000000,4000000,5000000], labels=["0", "R$1 milhão","R$2 milhões", "R$3 milhões", "R$4 milhões", "R$5 milhões"], rotation=45)
     plt.xlim([0.0,5000000])

     ax2.set_ylabel('Frequência')
     ax2.set_xlabel('Valor Global')
     ax2.set_title('Histograma de Valor Global')

     fig.tight_layout()
     fig.savefig("analise_variaveis.png")
     plt.close(fig)




# %%

def Plot_Analise_Previsoes():
     engine = create_engine("mysql+pymysql://root:Siconv!@localhost/siconv", pool_pre_ping=True,  connect_args = {'use_unicode':True, 'charset':'utf8mb4'})
     situacao_conta_df = pd.read_sql("select SITUACAO_CONTA , count(target) as total,  sum(case when target=0 then 1 else 0 end) as negativos, sum(case when target=1 then 1 else 0 end) as positivos from siconv.features f group by SITUACAO_CONTA order by total DESC;", engine)
     projeto_basico_df = pd.read_sql("select SITUACAO_PROJETO_BASICO , count(target) as total,  sum(case when target=0 then 1 else 0 end) as negativos, sum(case when target=1 then 1 else 0 end) as positivos from siconv.features f group by SITUACAO_PROJETO_BASICO order by total DESC;", engine)
##     
     labels_conta = situacao_conta_df.loc[:5,"SITUACAO_CONTA"]
     labels_projeto = projeto_basico_df.loc[:9,"SITUACAO_PROJETO_BASICO"]
##   
     x = np.arange(len(labels_conta)) 
     width = 0.35
##
     fig = plt.figure(figsize=(10,6))
##     
     ax1 = fig.add_subplot(121)
     rects1 = ax1.bar(x - width/2, situacao_conta_df.loc[:5,"negativos"], width, label='Negativos')
     rects2 = ax1.bar(x + width/2, situacao_conta_df.loc[:5,"positivos"], width, label='Positivos')
     plt.yscale("log")
     # Add some text for labels, title and custom x-axis tick labels, etc.
     ax1.set_ylabel('Total' )
     ax1.set_xlabel('Situação da Conta')
     ax1.set_title('Casos por Situação da Conta', fontsize=16)
     ax1.set_xticks(x)
     ax1.set_xticklabels(labels_conta, rotation=60)
     ax1.legend()
##
     autolabel(rects1, ax1)
     autolabel(rects2, ax1)
################################################
     x = np.arange(len(labels_projeto)) 
     width = 0.35
##     
     ax3 = fig.add_subplot(122)
##     
     rects3 = ax3.bar(x - width/2, projeto_basico_df.loc[:9,"negativos"], width, label='Negativos')
     rects4 = ax3.bar(x + width/2, projeto_basico_df.loc[:9,"positivos"], width, label='Positivos')
     plt.yscale("log")
     # Add some text for labels, title and custom x-axis tick labels, etc.
     ax3.set_ylabel('Total')
     ax3.set_xlabel('Situação do Projeto Básico')
     ax3.set_title('Casos por Situação do Projeto Básico', fontsize=16)
     ax3.set_xticks(x)
     ax3.set_xticklabels(labels_projeto, rotation=60)
     ax3.legend()
##
     autolabel(rects3, ax3)
     autolabel(rects4, ax3)
##
     fig.subplots_adjust(bottom=0.35)
     fig.tight_layout()
     fig.savefig("analise_previsoes.png")
     plt.close(fig)


def Acha_Casos_Falsos():
     # acha os casos que errou
     res = clf_pred_test == y_test_tree

     fp_idx = []
     fn_idx = []
     #acha os indices dos casos FN e FP
     i=-1
     for j in res:
          i=i+1
          if j == False:
               if y_test_tree[i] == 0:
                    fp_idx.append(i)
               else:
                    fn_idx.append(i)

     html = show_prediction(xgb, X_test_tree[28], show_feature_values=True, feature_names=feature_names)
     with open('falso_negativo.html', 'w') as f:
          f.write(html_fn.data)