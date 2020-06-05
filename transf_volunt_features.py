#%% imports
import pickle
import pandas as pd
import numpy as np

from sqlalchemy import create_engine

from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.utils import shuffle

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rcParams
from matplotlib import style

style.use('fivethirtyeight')

#%%
## salva objetos com pickle
def Save_Obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


## carrega objetos que foram salvos com pickle
def Load_Obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

## calcula os scores usados para comparar os modelos
def calcula_scores(y_true, model_prediction):
    tn, fp, fn, tp = confusion_matrix(y_true, model_prediction).ravel()
    specificity = tn / (tn+fp)
    accuracy =  (tp+tn) / (tp+tn+fp+fn)
    precision =  tp / (tp+fp)
    recall =  tp / (tp+fn)
    f_measure = 2*precision*recall/(precision+recall)

    return accuracy, precision, recall, specificity, f_measure

#%%
'''
separa aleatoriamente 10% dos dados para ficar pra teste, o que não será impactado pelos
metodos de sampling pois nao serao separados depois.
'''
def Dados_Balanceados_Separa_Teste_Onehot_Sem_Municipio_Orgao():
    feature_names = Load_Obj('feature_names_onehot_sem_municipio_orgao')
    X_data, y_data = load_svmlight_file('desbalanceado_onehot_sem_municipio_orgao.svm', n_features = len(feature_names))# pylint: disable=unbalanced-tuple-unpacking
    
    scaler = StandardScaler(with_mean=False)
    X_data = scaler.fit_transform(X_data)

    X_train_cv, X_test, y_train_cv, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=6439, stratify=y_data)
    Save_Obj(scaler, 'scaler_balanceado')
    dump_svmlight_file(X_train_cv, y_train_cv, 'treino_desbalanceado_onehot_sem_municipio_orgao.svm')
    dump_svmlight_file(X_test, y_test, 'test_desbalanceado_onehot_sem_municipio_orgao.svm')

#%%
'''
cria arquivos svm com features categoricas (sem one hot e sem scaling)
desbalanceadas, já dando um dump nos arquivos.
'''
def Dados_Desbalanceados_Onehot_All():
    engine = create_engine("mysql+pymysql://root:Siconv!@localhost/siconv", pool_pre_ping=True,  connect_args = {'use_unicode':True, 'charset':'utf8mb4'})
    tbl_features_df = pd.read_sql_table('features',engine)
    #cria coluna com faixa de valores relativos ao valor possivel da licitacao
    tbl_features_df['FAIXA_VALOR'] = pd.cut(tbl_features_df.VL_GLOBAL_CONV, 
                                bins=[0,33000,330000,3300000,np.Inf], 
                                labels=["dispensa", "convite", "tomada", "concorrencia"]
                                )
    # cria vetores onehot para as variaveis categoricas
    convenios_dummies_df = pd.get_dummies(tbl_features_df, columns=['UF_PROPONENTE','MODALIDADE',
            'SITUACAO_CONTA','SITUACAO_PROJETO_BASICO', 'COD_MUNIC_IBGE','COD_ORGAO_SUP', 
            'COD_ORGAO', 'FAIXA_VALOR'])
    #embaralha
    convenios_dummies_df_shuffle = shuffle(convenios_dummies_df)
    data_full = convenios_dummies_df_shuffle.copy()
    # e descartamos o resto para não influenciar no treinamento
    X_data = data_full.drop(['index', 'NR_CONVENIO', 'ID_PROPOSTA', 'SIT_CONVENIO', 
            'VL_GLOBAL_CONV', 'VL_REPASSE_CONV', 'VL_CONTRAPARTIDA_CONV', 'VL_DESEMBOLSADO_CONV',
            'TARGET', 'IDENTIF_PROPONENTE'], axis=1)
    feature_names = X_data.columns.values

    Save_Obj(feature_names, 'feature_names_onehot_all')
    dump_svmlight_file(X_data.values, data_full.TARGET.values, 'desbalanceado_onehot_all.svm')    

#%%
def Dados_Desbalanceados_Categoricos_All():
    engine = create_engine("mysql+pymysql://root:Siconv!@localhost/siconv", pool_pre_ping=True,  connect_args = {'use_unicode':True, 'charset':'utf8mb4'})
    tbl_features_df = pd.read_sql_table('features',engine)
    #cria coluna com faixa de valores relativos ao valor possivel da licitacao
    tbl_features_df['FAIXA_VALOR'] = pd.cut(tbl_features_df.VL_GLOBAL_CONV, 
                                bins=[0,33000,330000,3300000,np.Inf], 
                                labels=["dispensa", "convite", "tomada", "concorrencia"]
                                )
    #embaralha
    convenios_df_shuffle = shuffle(tbl_features_df)
    data_full = convenios_df_shuffle.copy()
    # e descartamos o resto para não influenciar no treinamento
    X_data = data_full.drop(['index', 'NR_CONVENIO', 'ID_PROPOSTA', 'SIT_CONVENIO', 
            'VL_GLOBAL_CONV', 'VL_REPASSE_CONV', 'VL_CONTRAPARTIDA_CONV', 'VL_DESEMBOLSADO_CONV',
            'TARGET', 'IDENTIF_PROPONENTE'], axis=1)
    feature_names = X_data.columns.values

    Save_Obj(feature_names, 'feature_names_categoricos_all')
    dump_svmlight_file(X_data.values, data_full.TARGET.values, 'desbalanceado_categoricos_all.svm')    
    
#%%
def Dados_Desbalanceados_Onehot_Sem_Municipio():
    engine = create_engine("mysql+pymysql://root:Siconv!@localhost/siconv", pool_pre_ping=True,  connect_args = {'use_unicode':True, 'charset':'utf8mb4'})
    tbl_features_df = pd.read_sql_table('features',engine)
    #cria coluna com faixa de valores relativos ao valor possivel da licitacao
    tbl_features_df['FAIXA_VALOR'] = pd.cut(tbl_features_df.VL_GLOBAL_CONV, 
                                bins=[0,33000,330000,3300000,np.Inf], 
                                labels=["dispensa", "convite", "tomada", "concorrencia"]
                                )
    convenios_dummies_df = pd.get_dummies(tbl_features_df, columns=['UF_PROPONENTE','MODALIDADE',
            'SITUACAO_CONTA','SITUACAO_PROJETO_BASICO', 'COD_ORGAO_SUP', 'COD_ORGAO', 'FAIXA_VALOR'])
    #embaralha
    convenios_dummies_df_shuffle = shuffle(convenios_dummies_df)
    data_full = convenios_dummies_df_shuffle.copy()
    # e descartamos o resto para não influenciar no treinamento
    X_data = data_full.drop(['index', 'NR_CONVENIO', 'ID_PROPOSTA', 'SIT_CONVENIO', 
            'VL_GLOBAL_CONV', 'VL_REPASSE_CONV', 'VL_CONTRAPARTIDA_CONV', 'VL_DESEMBOLSADO_CONV',
            'TARGET', 'IDENTIF_PROPONENTE','COD_MUNIC_IBGE'], axis=1)
    feature_names = X_data.columns.values

    Save_Obj(feature_names, 'feature_names_onehot_sem_municipio')
    dump_svmlight_file(X_data.values, data_full.TARGET.values, 'desbalanceado_onehot_sem_municipio.svm') 

#%%
def Dados_Desbalanceados_Onehot_Sem_Municipio_Orgao():
    engine = create_engine("mysql+pymysql://root:Siconv!@localhost/siconv", pool_pre_ping=True,  connect_args = {'use_unicode':True, 'charset':'utf8mb4'})
    tbl_features_df = pd.read_sql_table('features',engine)
    #cria coluna com faixa de valores relativos ao valor possivel da licitacao
    tbl_features_df['FAIXA_VALOR'] = pd.cut(tbl_features_df.VL_GLOBAL_CONV, 
                                bins=[0,33000,330000,3300000,np.Inf], 
                                labels=["dispensa", "convite", "tomada", "concorrencia"]
                                )
    convenios_dummies_df = pd.get_dummies(tbl_features_df, columns=['UF_PROPONENTE','MODALIDADE',
            'SITUACAO_CONTA','SITUACAO_PROJETO_BASICO','FAIXA_VALOR'])
    #embaralha
    convenios_dummies_df_shuffle = shuffle(convenios_dummies_df)
    data_full = convenios_dummies_df_shuffle.copy()
    # e descartamos o resto para não influenciar no treinamento
    X_data = data_full.drop(['index', 'NR_CONVENIO', 'ID_PROPOSTA', 'SIT_CONVENIO', 
            'VL_GLOBAL_CONV', 'VL_REPASSE_CONV', 'VL_CONTRAPARTIDA_CONV', 'VL_DESEMBOLSADO_CONV',
            'TARGET', 'IDENTIF_PROPONENTE','COD_MUNIC_IBGE','COD_ORGAO_SUP', 'COD_ORGAO'], axis=1)
    
    feature_names = X_data.columns.values

    Save_Obj(feature_names, 'feature_names_onehot_sem_municipio_orgao')
    dump_svmlight_file(X_data.values, data_full.TARGET.values, 'desbalanceado_onehot_sem_municipio_orgao.svm') 

#%%
'''
já deve ter os arquivos anteriores com as features desbalanceadas para apenas
carregar o arquivo e rebalancear sem conectar no banco
'''
def Dados_Balanceados_SMOTE_Sem_Municipio_Orgao():
    #sampling_strategy = 0.1 ==> 10 x 1
    #sampling_strategy = 0.2 ==> 5 x 1
    #sampling_strategy = 1 ==> 1 x 1
    feature_names = Load_Obj('feature_names_onehot_sem_municipio_orgao')
    X_data, y_data = load_svmlight_file('treino_desbalanceado_onehot_sem_municipio_orgao.svm', n_features = len(feature_names))# pylint: disable=unbalanced-tuple-unpacking

    sm = SMOTE(random_state=6439, sampling_strategy=0.1)
    X_res, y_res = sm.fit_resample(X_data, y_data)
    dump_svmlight_file(X_res, y_res, 'smote_10_1_onehot_sem_municipio_orgao.svm') 

    sm = SMOTE(random_state=6439, sampling_strategy=0.2)
    X_res, y_res = sm.fit_resample(X_data, y_data)
    dump_svmlight_file(X_res, y_res, 'smote_5_1_onehot_sem_municipio_orgao.svm') 

    sm = SMOTE(random_state=6439, sampling_strategy=1.0)
    X_res, y_res = sm.fit_resample(X_data, y_data)
    dump_svmlight_file(X_res, y_res, 'smote_1_1_onehot_sem_municipio_orgao.svm') 


# %%
def Dados_Balanceados_NearMiss_Sem_Municipio_Orgao():
    #sampling_strategy = 0.1 ==> 10 x 1
    #sampling_strategy = 0.2 ==> 5 x 1
    #sampling_strategy = 1 ==> 1 x 1
    feature_names = Load_Obj('feature_names_onehot_sem_municipio_orgao')
    X_data, y_data = load_svmlight_file('treino_desbalanceado_onehot_sem_municipio_orgao.svm', n_features = len(feature_names))# pylint: disable=unbalanced-tuple-unpacking

    nm = NearMiss(sampling_strategy=0.1)
    X_res, y_res = nm.fit_resample(X_data, y_data)
    dump_svmlight_file(X_res, y_res, 'nearmiss_10_1_onehot_sem_municipio_orgao.svm') 

    nm = NearMiss(sampling_strategy=0.2)
    X_res, y_res = nm.fit_resample(X_data, y_data)
    dump_svmlight_file(X_res, y_res, 'nearmiss_5_1_onehot_sem_municipio_orgao.svm') 

    nm = NearMiss(sampling_strategy=1)
    X_res, y_res = nm.fit_resample(X_data, y_data)
    dump_svmlight_file(X_res, y_res, 'nearmiss_1_1_onehot_sem_municipio_orgao.svm') 


# %%
def Dados_Balanceados_SMOTE_NearMiss_Sem_Municipio_Orgao():
    #sampling_strategy = 0.1 ==> 10 x 1
    #sampling_strategy = 0.2 ==> 5 x 1
    #sampling_strategy = 1 ==> 1 x 1
    feature_names = Load_Obj('feature_names_onehot_sem_municipio_orgao')
    X_data, y_data = load_svmlight_file('treino_desbalanceado_onehot_sem_municipio_orgao.svm', n_features = len(feature_names))# pylint: disable=unbalanced-tuple-unpacking

    sm = SMOTE(random_state=6439, sampling_strategy=0.1)
    X_res, y_res = sm.fit_resample(X_data, y_data)
    nm = NearMiss(sampling_strategy=0.1)
    X_res_new, y_res_new = nm.fit_resample(X_res, y_res)
    dump_svmlight_file(X_res_new, y_res_new, 'smote_nearmiss_10_1_onehot_sem_municipio_orgao.svm') 

    sm = SMOTE(random_state=6439, sampling_strategy=0.175)
    X_res, y_res = sm.fit_resample(X_data, y_data)
    nm = NearMiss(sampling_strategy=0.175)

    X_res_new, y_res_new = nm.fit_resample(X_res, y_res)
    dump_svmlight_file(X_res_new, y_res_new, 'smote_nearmiss_5_1_onehot_sem_municipio_orgao.svm') 

    sm = SMOTE(random_state=6439, sampling_strategy=0.5)
    X_res, y_res = sm.fit_resample(X_data, y_data)
    nm = NearMiss(sampling_strategy=0.5)
    X_res_new, y_res_new = nm.fit_resample(X_res, y_res)
    dump_svmlight_file(X_res_new, y_res_new, 'smote_nearmiss_1_1_onehot_sem_municipio_orgao.svm') 

#%% #######################################################################

def Gera_Figura_Feature_Importance(classificador, nome, feature_names):
    total = 10
    rcParams.update({'figure.autolayout': True})
    fig = plt.figure(figsize=(10,5))
    #plot importances do xgb
    importances = pd.Series(classificador.feature_importances_, feature_names)
    features_sorted = importances.sort_values()
    total_features = features_sorted[-total:]
    total_features.plot.barh(color = cm.rainbow(np.linspace(0,1,total)))# pylint: disable=no-member
    plt.ylabel("Características")
    plt.xlabel("Pesos")
    plt.title("Importâncias das Características do Modelo")
    fig.savefig("feature_importance_{}.png".format(nome))
    print("Figura feature_importance_{}.png Gerada".format(nome))

#%% ##############################################################################
def Gera_Figura_Hiperopt_Otimizacao(hyperopt_results, nome):
    
    hyperopt_results_df=pd.DataFrame(hyperopt_results,
                            columns=['score', 'learning_rate', 'max_depth', 'min_child_weight',
                                'n_estimators', 'colsample_bytree', 'subsample', 'gamma', 'alpha','lambda'])

    rcParams.update({'figure.autolayout': True})
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Rodadas de Otimização')
    ax1.set_ylabel('Espaço de Busca', color=color)
    line1 = ax1.plot(hyperopt_results_df['score'], color=color, label='Score')
    ax1.tick_params(axis='y', labelcolor=color)

    color = 'tab:orange'
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Espaço de Busca', color=color)  # we already handled the x-label with ax1
    line2 = ax2.plot(hyperopt_results_df['learning_rate'], color=color, label='Learning Rate')
    ax2.tick_params(axis='y', labelcolor=color)

    color = 'tab:green'
    ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    line3 = ax3.plot(hyperopt_results_df['max_depth'], color=color, label='Max Depth')
    ax3.set_yticks([])

    color = 'tab:red'
    ax4 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    line4 = ax4.plot(hyperopt_results_df['min_child_weight'], color=color, label='Min Child Weight')
    ax4.set_yticks([])

    color = 'tab:purple'
    ax5 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    line5 = ax5.plot(hyperopt_results_df['n_estimators'], color=color, label='#Estimators')
    ax5.set_yticks([])

    color = 'tab:brown'
    ax6 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    line6 = ax6.plot(hyperopt_results_df['colsample_bytree'], color=color, label='ColSample/Tree')
    ax6.set_yticks([])

    color = 'tab:pink'
    ax7 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    line7 = ax7.plot(hyperopt_results_df['subsample'], color=color, label='Subsample')
    ax7.set_yticks([])

    color = 'tab:gray'
    ax8 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    line8 = ax8.plot(hyperopt_results_df['gamma'], color=color, label='Gamma')
    ax8.set_yticks([])

    color = 'tab:olive'
    ax9 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    line9 = ax9.plot(hyperopt_results_df['alpha'], color=color, label='Alpha')
    ax9.set_yticks([])

    color = 'tab:cyan'
    ax10 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    line10 = ax10.plot(hyperopt_results_df['lambda'], color=color, label='Lambda')
    ax10.set_yticks([])

    lines = line1+line2+line3+line4+line5+line6+line7+line8+line9+line10
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc=0)

    fig.savefig("hiperparametros_{}.png".format(nome))
    print("Figura hiperparametros_{}.png Gerada".format(nome))

