#%% imports
import pickle
import pandas as pd

from sqlalchemy import create_engine

from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file

from sklearn.utils import shuffle

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

#%%
## salva objetos com pickle
def Save_Obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


## carrega objetos que foram salvos com pickle
def Load_Obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

#%%
'''
cria arquivos svm com features categoricas (sem one hot e sem scaling)
desbalanceadas, já dando um dump nos arquivos.
'''
def Dados_Desbalanceados_Onehot_All():
    engine = create_engine("mysql+pymysql://root:Siconv!@localhost/siconv", pool_pre_ping=True,  connect_args = {'use_unicode':True, 'charset':'utf8mb4'})
    tbl_features_df = pd.read_sql_table('features',engine)
    # cria vetores onehot para as variaveis categoricas
    convenios_dummies_df = pd.get_dummies(tbl_features_df, columns=['UF_PROPONENTE','MODALIDADE',
            'SITUACAO_CONTA','SITUACAO_PROJETO_BASICO', 'COD_MUNIC_IBGE','COD_ORGAO_SUP', 'COD_ORGAO'])
    #embaralha
    convenios_dummies_df_shuffle = shuffle(convenios_dummies_df)
    data_full = convenios_dummies_df_shuffle.copy()
    # e descartamos o resto para não influenciar no treinamento
    X_data = data_full.drop(['index', 'NR_CONVENIO', 'ID_PROPOSTA', 'SIT_CONVENIO', 'TARGET',
            'IDENTIF_PROPONENTE'], axis=1)
    feature_names = X_data.columns.values

    Save_Obj(feature_names, 'feature_names_onehot_all')
    dump_svmlight_file(X_data.values, data_full.TARGET.values, 'desbalanceado_onehot_all.svm')    

#%%
def Dados_Desbalanceados_Categoricos_All():
    engine = create_engine("mysql+pymysql://root:Siconv!@localhost/siconv", pool_pre_ping=True,  connect_args = {'use_unicode':True, 'charset':'utf8mb4'})
    tbl_features_df = pd.read_sql_table('features',engine)
    
    #embaralha
    convenios_df_shuffle = shuffle(tbl_features_df)
    data_full = convenios_df_shuffle.copy()
    # e descartamos o resto para não influenciar no treinamento
    X_data = data_full.drop(['index', 'NR_CONVENIO', 'ID_PROPOSTA', 'SIT_CONVENIO', 'TARGET',
            'IDENTIF_PROPONENTE'], axis=1)
    feature_names = X_data.columns.values

    Save_Obj(feature_names, 'feature_names_categoricos_all')
    dump_svmlight_file(X_data.values, data_full.TARGET.values, 'desbalanceado_categoricos_all.svm')    
    
#%%
def Dados_Desbalanceados_Onehot_Sem_Municipio():
    engine = create_engine("mysql+pymysql://root:Siconv!@localhost/siconv", pool_pre_ping=True,  connect_args = {'use_unicode':True, 'charset':'utf8mb4'})
    tbl_features_df = pd.read_sql_table('features',engine)
    convenios_dummies_df = pd.get_dummies(tbl_features_df, columns=['UF_PROPONENTE','MODALIDADE',
            'SITUACAO_CONTA','SITUACAO_PROJETO_BASICO', 'COD_ORGAO_SUP', 'COD_ORGAO'])
    #embaralha
    convenios_dummies_df_shuffle = shuffle(convenios_dummies_df)
    data_full = convenios_dummies_df_shuffle.copy()
    # e descartamos o resto para não influenciar no treinamento
    X_data = data_full.drop(['index', 'NR_CONVENIO', 'ID_PROPOSTA', 'SIT_CONVENIO', 'TARGET',
            'IDENTIF_PROPONENTE','COD_MUNIC_IBGE'], axis=1)
    feature_names = X_data.columns.values

    Save_Obj(feature_names, 'feature_names_onehot_sem_municipio')
    dump_svmlight_file(X_data.values, data_full.TARGET.values, 'desbalanceado_onehot_sem_municipio.svm') 

#%%
def Dados_Desbalanceados_Onehot_Sem_Municipio_Orgao():
    engine = create_engine("mysql+pymysql://root:Siconv!@localhost/siconv", pool_pre_ping=True,  connect_args = {'use_unicode':True, 'charset':'utf8mb4'})
    tbl_features_df = pd.read_sql_table('features',engine)
    convenios_dummies_df = pd.get_dummies(tbl_features_df, columns=['UF_PROPONENTE','MODALIDADE',
            'SITUACAO_CONTA','SITUACAO_PROJETO_BASICO'])
    #embaralha
    convenios_dummies_df_shuffle = shuffle(convenios_dummies_df)
    data_full = convenios_dummies_df_shuffle.copy()
    # e descartamos o resto para não influenciar no treinamento
    X_data = data_full.drop(['index', 'NR_CONVENIO', 'ID_PROPOSTA', 'SIT_CONVENIO', 'TARGET',
            'IDENTIF_PROPONENTE','COD_MUNIC_IBGE','COD_ORGAO_SUP', 'COD_ORGAO'], axis=1)
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
    X_data, y_data = load_svmlight_file('desbalanceado_onehot_sem_municipio_orgao.svm', n_features = len(feature_names))# pylint: disable=unbalanced-tuple-unpacking

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
    X_data, y_data = load_svmlight_file('desbalanceado_onehot_sem_municipio_orgao.svm', n_features = len(feature_names))# pylint: disable=unbalanced-tuple-unpacking

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
    X_data, y_data = load_svmlight_file('desbalanceado_onehot_sem_municipio_orgao.svm', n_features = len(feature_names))# pylint: disable=unbalanced-tuple-unpacking

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
