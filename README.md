# siconv-xgboost
## O projeto do mestrado está no Google Drive.
Medtrado/dissertacao/python

Para gerar o banco (MySQL):
### Baixar os dados completos do Siconv no Portal +Brasil
  #### Executar o Notebook ETL Base Sivonv para popular os dados
    ##### Inicialmente, ele executa a criação da estrutura do banco
    ##### Depois faz uma varredura dos arquivos CSV incluindo os dados no banco (algumas importações estão dando erro no CSV)
  #### Executar o script SQL para geracao das Features
