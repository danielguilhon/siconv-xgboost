<h1> siconv-xgboost</h1>
<h2>O projeto do mestrado está no Google Drive.</h2>
<code>
Medtrado/dissertacao/python
</code>
<h2>O projeto é administrado via VS Code, no ambiente local, ~/Mestrado/dissertação/siconv-xgboost, e depois é feita a sincronização com o GitHub (commit/push)</h2>

<h1>ETL para a criação e carga inicial do Banco Siconv</h1>
<h2>1. Inicialmente, devemos baixar os dados do Siconv na Plataforma +Brasil</h2>
<p><a href="http://plataformamaisbrasil.gov.br/download-de-dados">Plataforma +Brasil - Dados Siconv</a></p>
<p>Descompactar os dados em um diretório escolhido</p>
<h2>2. Executar o script de criação do banco e tabelas</h2>
<p><code> mysql -u root -p < create_siconv.sql </code></p>
<h2>3. Executar as células para substituir os ponto e vírgula e inserção dos dados</h2>
<p>
<code>
    SubsPontoVirgula()
    tipos
    
    popular dados [insertion_order]
    convenio
    demais tabelas
</code>
   </p>
<h2>4. Executar o script de geracao das tabelas intermediarias</h2>
<p> <code> mysql -u root -p < convenio_proposta_emenda_consorcio.sql </code> </p>
<h2>5. Executar o script de geracao da tabela de features</h2>
<p><code>
GeraFeatures()    
</code>
</p>