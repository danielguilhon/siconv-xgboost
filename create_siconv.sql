-- primeiramente, criamos a base siconv completa
-- depois, devemos inserir os dados, utilizando o script etl python.
-- após, rodamos o script vw_convenio_proposta_emenda_consorcio.sql para criar as demais tabelas, já com as regras
CREATE DATABASE IF NOT EXISTS siconv;

USE siconv;

-- CREATE TABLE IF NOT EXISTS programa (ID_PROGRAMA VARCHAR(20) COMMENT 'Código Sequencial do Sistema para um Programa',COD_ORGAO_SUP_PROGRAMA VARCHAR(255) COMMENT 'Código do Órgão executor do Programa',DESC_ORGAO_SUP_PROGRAMA TEXT COMMENT 'Nome do Órgão executor do Programa',COD_PROGRAMA VARCHAR(255) COMMENT 'Chave que identifica o programa composta por: (Cód.Órgão+Ano+Cód.Sequencial do Sistema)',NOME_PROGRAMA TEXT COMMENT 'Descrição do Programa de Governo',SIT_PROGRAMA varchar(255) COMMENT 'Situação atual do Programa. Domínio: Cadastrado; Disponibilizado; Inativo',DATA_DISPONIBILIZACAO VARCHAR(20) COMMENT 'Data de disponibilização do Programa',ANO_DISPONIBILIZACAO VARCHAR(10) COMMENT 'Ano de disponibilização do Programa',DT_PROG_INI_RECEB_PROP VARCHAR(20) COMMENT 'Data Início para o recebimento das propostas voluntárias para o Programa',DT_PROG_FIM_RECEB_PROP VARCHAR(20) COMMENT 'Data Fim para o recebimento das propostas voluntárias para o Programa',DT_PROG_INI_EMENDA_PAR VARCHAR(20) COMMENT 'Data Início para o recebimento das propostas de Emenda Parlamentar para o Programa',DT_PROG_FIM_EMENDA_PAR VARCHAR(20) COMMENT 'Data Fim para o recebimento das propostas de Emenda Parlamentar para o Programa',DT_PROG_INI_BENEF_ESP VARCHAR(20) COMMENT 'Data Início para o recebimento das propostas de beneficiário específico para o Programa',DT_PROG_FIM_BENEF_ESP VARCHAR(20) COMMENT 'Data Fim para o recebimento das propostas de beneficiário específico para o Programa',MODALIDADE_PROGRAMA varchar(255) COMMENT 'Modalidade do Programa. Domínio pode ser: CONTRATO DE REPASSE, CONVENIO, TERMO DE COLABORACAO, TERMO DE FOMENTO e TERMO DE PARCERIA',NATUREZA_JURIDICA_PROGRAMA varchar(255) COMMENT 'Natureza Jurídica Atendida pelo Programa. Domínio: Administração Pública Estadual ou do Distrito Federal, Administração Pública Municipal, Consórcio Público, Empresa pública/Sociedade de economia mista e Organização da Sociedade Civil',UF_PROGRAMA varchar(10) COMMENT 'Ufs Habilitadas para o Programa. Domínio: AC, AL, AM, AP, BA, CE, DF, ES, GO, MA, MG, MS, MT, PA, PB, PE, PI, PR, RJ, RN, RO, RR, RS, SC, SE, SP, TO, <null>. Quando o valor é nulo, o programa atende a todo o Brasil',ACAO_ORCAMENTARIA varchar(255) COMMENT 'Número da Ação Orçamentária', INDEX ID_PROGRAMA_IDX (ID_PROGRAMA));

-- CREATE TABLE IF NOT EXISTS PROPONENTES (IDENTIF_PROPONENTE varchar(255) COMMENT 'CNPJ do Proponente',NM_PROPONENTE TEXT COMMENT 'Nome da Entidade Proponente',MUNICIPIO_PROPONENTE TEXT COMMENT 'Município do Proponente',UF_PROPONENTE varchar(10) COMMENT 'UF do Proponente. Domínio: AC, AL, AM, AP, BA, CE, DF, ES, GO, MA, MG, MS, MT, PA, PB, PE, PI, PR, RJ, RN, RO, RR, RS, SC, SE, SP, TO',ENDERECO_PROPONENTE TEXT COMMENT 'Endereço do Proponente',BAIRRO_PROPONENTE varchar(255) COMMENT 'Bairro do Proponente',CEP_PROPONENTE varchar(10) COMMENT 'CEP do Proponente',EMAIL_PROPONENTE varchar(255) COMMENT 'E-mail do Proponente',TELEFONE_PROPONENTE varchar(255) COMMENT 'Telefone do Proponente',FAX_PROPONENTE varchar(255) COMMENT 'Fax do Proponente');

CREATE TABLE IF NOT EXISTS proposta (
    ID_PROPOSTA VARCHAR(20) COMMENT 'Código Sequencial do Sistema para uma Proposta',
    UF_PROPONENTE VARCHAR(10) COMMENT 'UF do Proponente. Domínio: AC, AL, AM, AP, BA, CE, DF, ES, GO, MA, MG, MS, MT, PA, PB, PE, PI, PR, RJ, RN, RO, RR, RS, SC, SE, SP, TO',
    MUNIC_PROPONENTE varchar(255) COMMENT 'Município do Proponente',
    COD_MUNIC_IBGE varchar(255) COMMENT 'Código IBGE do Município',
    COD_ORGAO_SUP varchar(255) COMMENT 'Código do Órgão Superior do Concedente',
    DESC_ORGAO_SUP TEXT COMMENT 'Nome do Órgão Superior do Concedente',
    NATUREZA_JURIDICA varchar(255) COMMENT 'Natureza Jurídica do Proponente. Domínio: Administração Pública Estadual ou do Distrito Federal, Administração Pública Municipal, Consórcio Público, Empresa pública/Sociedade de economia mista e Organização da, Sociedade Civil',
    NR_PROPOSTA varchar(255) COMMENT 'Número da Proposta gerado pelo Siconv',
    DIA_PROP VARCHAR(10) COMMENT 'Dia do cadastro da Proposta',
    MES_PROP VARCHAR(10) COMMENT 'Mês do cadastro da Proposta',
    ANO_PROP VARCHAR(10) COMMENT 'Ano do cadastro da Proposta',
    DIA_PROPOSTA VARCHAR(20) COMMENT 'Data do cadastro da Proposta',
    COD_ORGAO varchar(255) COMMENT 'Código do Órgão ou Entidade Concedente',
    DESC_ORGAO TEXT COMMENT 'Nome do Órgão ou Entidade Concedente',
    MODALIDADE varchar(255) COMMENT 'Modalidade da Proposta. Domínio pode ser: CONTRATO DE REPASSE, CONVENIO, TERMO DE COLABORACAO, TERMO DE FOMENTO e TERMO DE PARCERIA',
    IDENTIF_PROPONENTE varchar(255) COMMENT 'CNPJ do Proponente',
    NM_PROPONENTE VARCHAR(255) COMMENT 'Nome da Entidade Proponente',
    CEP_PROPONENTE varchar(10) COMMENT 'CEP do Proponente',
    ENDERECO_PROPONENTE TEXT COMMENT 'Endereço do Proponente',
    BAIRRO_PROPONENTE varchar(255) COMMENT 'Bairro do Proponente',
    NM_BANCO varchar(255) COMMENT 'Nome do Banco para depósito do recurso da Transferência Voluntária',
    SITUACAO_CONTA varchar(255) COMMENT 'Situação atual da conta bancária do instrumento. Domínio: Aguardando Retorno do Banco, Enviada, Cadastrada, Registrada, Erro na Abertura de Conta, Regularizada, A Verificar, Aguardando Envio e Pendente de Regularização',
    SITUACAO_PROJETO_BASICO VARCHAR(255) COMMENT 'Situação atual do Projeto Básico/Termo de Referência. Domínio: Aguardando Projeto Básico, Não Cadastrado, Projeto Básico Aprovado, Projeto Básico em Análise, Projeto Básico em Complementação, Projeto Básico Rejeitado',
    SIT_PROPOSTA VARCHAR(255) COMMENT 'Situação atual da Proposta. Domínio pode ser: Proposta/Plano de Trabalho Cadastrados, Proposta/Plano de Trabalho em Análise, Proposta/Plano de Trabalho Rejeitados, Proposta/Plano de Trabalho Aprovados, etc',
    DIA_INIC_VIGENCIA_PROPOSTA VARCHAR(20) COMMENT 'Data Início da Vigência da Proposta',
    DIA_FIM_VIGENCIA_PROPOSTA VARCHAR(20) COMMENT 'Data Fim da Vigência da Proposta',
    OBJETO_PROPOSTA TEXT COMMENT 'Descrição do Objeto da Proposta',
    ITEM_INVESTIMENTO VARCHAR(255),
    ENVIADA_MANDATARIA VARCHAR(255),
    VL_GLOBAL_PROP FLOAT COMMENT 'Valor Global da proposta cadastrada (Valor de Repasse Proposta + Valor Contrapartida Proposta)',
    VL_REPASSE_PROP FLOAT COMMENT 'Valor de Repasse do Governo Federal referente a proposta cadastrada',
    VL_CONTRAPARTIDA_PROP FLOAT COMMENT 'Valor da Contrapartida apresentada na proposta pelo convenente');

CREATE TABLE IF NOT EXISTS convenio (
NR_CONVENIO VARCHAR(20) COMMENT 'Número gerado pelo Siconv. Possui faixa de numeração reservada que vai de 700000 a 999999',
ID_PROPOSTA VARCHAR(20) COMMENT 'Código Sequencial do Sistema para uma Proposta',
DIA VARCHAR(10) COMMENT 'Dia em que o Convênio foi assinado',
MES INT COMMENT 'Mês em que o Convênio foi assinado',
ANO VARCHAR(10) COMMENT 'Ano Assinatura do Convênio',
DIA_ASSIN_CONV VARCHAR(20) COMMENT 'Data Assinatura do Convênio',
SIT_CONVENIO varchar(255) COMMENT 'Situação atual do Convênio. Domínio: Em execução, Convênio Anulado, Prestação de Contas enviada para Análise, Prestação de Contas Aprovada, Prestação de Contas em Análise, Prestação de Contas em Complementação, Inadimplente, etc',
SUBSITUACAO_CONV varchar(255) COMMENT 'Sub-Situação atual do Convênio. Domínio: Convênio, Convênio Cancelado, Convênio Encerrado, Proposta, Em aditivação',
SITUACAO_PUBLICACAO varchar(255) COMMENT 'Situação atual da Publicação do instrumento. Domínio: Publicado e Transferido para IN',
INSTRUMENTO_ATIVO varchar(10) COMMENT 'Convênios que ainda não foram finalizados. Domínio: SIM, NÃO',
IND_OPERA_OBTV varchar(10) COMMENT 'Indicativo de que o Convênio opera com OBTV. Domínio: SIM, NÃO',
NR_PROCESSO VARCHAR(255) COMMENT 'Número VARCHAR(10)erno do processo físico do instrumento',
UG_EMITENTE VARCHAR(255) COMMENT 'Número da Unidade Gestora',
DIA_PUBL_CONV VARCHAR(20) COMMENT 'Data da Publicação do Convênio',
DIA_INIC_VIGENC_CONV VARCHAR(20) COMMENT 'Data de Início de Vigência do Convênio',
DIA_FIM_VIGENC_CONV VARCHAR(20) COMMENT 'Data de Fim de Vigência do Convênio',
DIA_FIM_VIGENC_ORIGINAL_CONV VARCHAR(20) COMMENT 'Data de Fim de Vigência Original do Convênio sem os TAs e Prorrogas',
DIAS_PREST_CONTAS VARCHAR(10) COMMENT 'Pazo para a Prestação de Contas do Convênio',
DIA_LIMITE_PREST_CONTAS VARCHAR(20) COMMENT 'Data limite para Prestação de Contas do Convênio',
DATA_SUSPENSIVA VARCHAR(20),
DATA_RETIRADA_SUSPENSIVA VARCHAR(20),
DIAS_CLAUSULA_SUSPENSIVA FLOAT,
SITUACAO_CONTRATACAO varchar(255) COMMENT 'Situação atual da Contratação. Domínio: Cláusula Suspensiva, Liminar Judicial, Normal e Sob Liminar Judicial e Cláusula Suspensiva',
IND_ASSINADO varchar(10) COMMENT 'Indicativo se o Convênio está assinado ou não. Domínio: SIM, NÃO',
MOTIVO_SUSPENSAO TEXT COMMENT 'Descrição do motivo de suspensão referente a cláusula suspensiva',
IND_FOTO varchar(10) COMMENT 'Indicativo se o Convênio possui foto ou não. Domínio: SIM, NÃO',
QTDE_CONVENIOS INT COMMENT 'Quantidade de Instrumentos Assinados',
QTD_TA INT COMMENT 'Quantidade de Termos Aditivos',
QTD_PRORROGA INT COMMENT 'Quantidade de Prorrogas de Ofício',
VL_GLOBAL_CONV FLOAT COMMENT 'Valor global dos Instrumentos assinados (Valor de Repasse + Valor Contrapartida)',
VL_REPASSE_CONV FLOAT COMMENT 'Valor total do aporte do Governo Federal referente a celebração do Instrumento',
VL_CONTRAPARTIDA_CONV FLOAT COMMENT 'Valor total da Contrapartida que será disponibilizada pelo convenente',
VL_EMPENHADO_CONV FLOAT COMMENT 'Valor total empenhado do Governo Federal para os Instrumentos',
VL_DESEMBOLSADO_CONV FLOAT COMMENT 'Valor total desembolsado do Governo Federal para a conta do Instrumento',
VL_SALDO_REMAN_TESOURO FLOAT COMMENT 'Valores devolvidos ao Tesouro ao término do instrumento',
VL_SALDO_REMAN_CONVENENTE FLOAT COMMENT 'Valores devolvidos ao Convenente ao término do instrumento',
VL_RENDIMENTO_APLICACAO FLOAT COMMENT 'Valores referentes aos redimentos de aplicação financeira',
VL_INGRESSO_CONTRAPARTIDA FLOAT COMMENT 'Total de valores referente a ingresso de contrapartida dos Instrumentos',
VL_SALDO_CONTA FLOAT COMMENT 'O Saldo em Conta deve ser entendido como um valor estimado, podendo sofrer variação até o próximo dia útil, principalmente, pelo fato dos rendimentos de aplicação previstos fazerem parte do referido valor.',
VALOR_GLOBAL_ORIGINAL_CONV FLOAT);

-- CREATE TABLE IF NOT EXISTS DESEMBOLSO (ID_DESEMBOLSO VARCHAR(20) COMMENT 'Identificador único gerado pelo Sistema para o Desembolso',NR_CONVENIO VARCHAR(20) COMMENT 'Número gerado pelo Siconv. Possui faixa de numeração reservada que vai de 700000 a 999999',DT_ULT_DESEMBOLSO VARCHAR(20) COMMENT 'Data da última Ordem Bancária gerada',QTD_DIAS_SEM_DESEMBOLSO VARCHAR(10) COMMENT 'Indicador de dias sem desembolso. Domínio: 90,180 e 365 dias',DATA_DESEMBOLSO VARCHAR(20) COMMENT 'Data da Ordem Bancária',ANO_DESEMBOLSO VARCHAR(10) COMMENT 'Ano da Ordem Bancária',MES_DESEMBOLSO VARCHAR(10) COMMENT 'Mês da Ordem Bancária',NR_SIAFI varchar(255) COMMENT 'Número do Documento no SIAFI',VL_DESEMBOLSADO FLOAT COMMENT 'Valor disponibilizado pelo Governo Federal para a conta do instrumento');

-- CREATE TABLE IF NOT EXISTS EMPENHO (ID_EMPENHO VARCHAR(20) COMMENT 'Identificador único gerado pelo Sistema para o Empenho',NR_CONVENIO VARCHAR(20) COMMENT 'Número gerado pelo Siconv. Possui faixa de numeração reservada que vai de 700000 a 999999',NR_EMPENHO varchar(255) COMMENT 'Número da Nota de Empenho',TIPO_NOTA VARCHAR(10) COMMENT 'Código do Tipo de Empenho',DESC_TIPO_NOTA TEXT COMMENT 'Descrição do Tipo de Empenho. Domínio: Empenho Original, Empenho de Despesa Pré-Empenhada, Anulação de Empenho, Reforço de Empenho, Estorno de Anulação de Empenho, etc',DATA_EMISSAO VARCHAR(20) COMMENT 'Data de emissão do Empenho',COD_SITUACAO_EMPENHO VARCHAR(10) COMMENT 'Código da Situação atual do empenho',DESC_SITUACAO_EMPENHO TEXT COMMENT 'Descrição da Situação atual do empenho. Domínio: Registrado no SIAFI e Enviado',VALOR_EMPENHO FLOAT COMMENT 'Valor empenhado');

-- CREATE TABLE IF NOT EXISTS META_CRONO_FISICO (ID_META VARCHAR(20) COMMENT 'Código Sequencial do Sistema para uma Meta',NR_CONVENIO VARCHAR(20) COMMENT 'Número gerado pelo Siconv. Possui faixa de numeração reservada que vai de 700000 a 999999',COD_PROGRAMA VARCHAR(20) COMMENT 'Chave que identifica o programa composta por: (Cód.Órgão+Ano+Cód.Sequencial do Sistema)',NOME_PROGRAMA TEXT COMMENT 'Descrição do Programa de Governo',NR_META VARCHAR(10) COMMENT 'Número da Meta gerada pelo Sistema',TIPO_META varchar(255) COMMENT 'Tipo da Meta: NORMAL/APLICAÇÃO',DESC_META TEXT COMMENT 'Especificação da Meta do Cronograma Físico',DATA_INICIO_META VARCHAR(20) COMMENT 'Data de início da Meta',DATA_FIM_META VARCHAR(20) COMMENT 'Data de término da Meta',UF_META varchar(10) COMMENT 'UF cadastrada para a Meta',MUNICIPIO_META varchar(255) COMMENT 'Município cadastrado para a Meta',ENDERECO_META TEXT COMMENT 'Endereço cadastrado para a Meta',CEP_META varchar(10) COMMENT 'CEP cadastrado para a Meta',QTD_META VARCHAR(15) COMMENT 'Quantidade da Meta',UND_FORNECIMENTO_META TEXT COMMENT 'Unidade de Fornecimento da Meta',VL_META FLOAT COMMENT 'Valor da Meta');

-- CREATE TABLE IF NOT EXISTS PAGAMENTO (NR_MOV_FIN VARCHAR(20) COMMENT 'Número identificador da movimentação financeira',NR_CONVENIO VARCHAR(20) COMMENT 'Número gerado pelo Siconv. Possui faixa de numeração reservada que vai de 700000 a 999999',IDENTIF_FORNECEDOR varchar(255) COMMENT 'CNPJ/CPF do Fornecedor',NOME_FORNECEDOR TEXT COMMENT 'Nome do Fornecedor',TP_MOV_FINANCEIRA VARCHAR(255) COMMENT 'Tipo da movimentação financeira realizada. Domínio: Pagamento a favorecido, Pagamento a favorecido com OBTV',DATA_PAG VARCHAR(20) COMMENT 'Data da realização do pagamento',NR_DL TEXT COMMENT 'Número identificador do Documento de Liquidação',DESC_DL TEXT COMMENT 'Descrição do Documento de Liquidação. Domínio: DIÁRIAS, DUPLICATA, FATURA, FOLHA DE PAGAMENTO, NOTA FISCAL, NOTA FISCAL / FATURA, OBTV PARA EXECUTOR, OBTV PARA O CONVENENTE, etc',VL_PAGO FLOAT COMMENT 'Valor do pagamento');

CREATE TABLE IF NOT EXISTS consorcios (
ID_PROPOSTA VARCHAR(20) COMMENT 'Código Sequencial do Sistema para uma Proposta',
CNPJ_CONSORCIO VARCHAR(255) COMMENT 'CNPJ do Consórcio',
NOME_CONSORCIO TEXT COMMENT 'Razão Social do Consórcio',
CODIGO_CNAE_PRIMARIO VARCHAR(20) COMMENT 'Código do CNAE Primário na Receita Federal',
DESC_CNAE_PRIMARIO TEXT COMMENT 'Descrição do CNAE Primário na Receita Federal',
CODIGO_CNAE_SECUNDARIO VARCHAR(20) COMMENT 'Código do CNAE Secundário na Receita Federal',
DESC_CNAE_SECUNDARIO TEXT COMMENT 'Descrição do CNAE Secundário na Receita Federal',
CNPJ_PARTICIPANTE VARCHAR(255) COMMENT 'CNPJ dos Participantes do Consórcio',
NOME_PARTICIPANTE TEXT COMMENT 'Descrição do nome dos participantes do Consórcio');

-- CREATE TABLE IF NOT EXISTS DESBLOQUEIO_CR (NR_CONVENIO VARCHAR(20) COMMENT 'Número gerado pelo Siconv. Possui faixa de numeração reservada que vai de 700000 a 999999',NR_OB varchar(255) COMMENT 'Número da OB',DATA_CADASTRO VARCHAR(20) COMMENT 'Data de Cadastro',DATA_ENVIO VARCHAR(20) COMMENT 'Data de envio da solicitação de desbloqueio do recurso',TIPO_RECURSO_DESBLOQUEIO varchar(255) COMMENT 'Tipo do Recurso. Domínio: OB, INGRESSO CONTRAPARTIDA e RENDIMENTO APLICAÇÃO',VL_TOTAL_DESBLOQUEIO FLOAT COMMENT 'Valor total de desbloqueio para o Contrato de Repasse',VL_DESBLOQUEADO FLOAT COMMENT 'Valor desbloqueado para o Contrato de Repasse',VL_BLOQUEADO FLOAT COMMENT 'Valor bloqueado para o Contrato de Repasse');

CREATE TABLE IF NOT EXISTS emenda (
ID_PROPOSTA VARCHAR(20) COMMENT 'Código Sequencial do Sistema para uma Proposta',
QUALIF_PROPONENTE TEXT COMMENT 'Qualificação do proponente',
COD_PROGRAMA_EMENDA VARCHAR(20) COMMENT 'Chave que identifica o programa composta por: (Cód.Órgão+Ano+Cód.Sequencial do Sistema)',
NR_EMENDA VARCHAR(20) COMMENT 'Número da Emenda Parlamentar',
NOME_PARLAMENTAR TEXT COMMENT 'Nome do Parlamentar',
BENEFICIARIO_EMENDA VARCHAR(255) COMMENT 'CNPJ do Proponente',
IND_IMPOSITIVO varchar(10) COMMENT 'Indicativo de Orçamento Impositivo (Tipo Parlamentar igual a INDIVIDUAL + Ano de Cadastro da Proposta >= 2014). Domínio: SIM, NÃO',
TIPO_PARLAMENTAR varchar(255) COMMENT 'Tipo do Parlamentar. Domínio pode ser: INDIVIDUAL, COMISSAO, BANCADA',
VALOR_REPASSE_PROPOSTA_EMENDA FLOAT COMMENT 'Valor da Emenda cadastrada na proposta',
VALOR_REPASSE_EMENDA FLOAT COMMENT 'Valor da Emenda assinada');

-- CREATE TABLE IF NOT EXISTS ETAPA_CRONO_FISICO (ID_META VARCHAR(20) COMMENT 'Código Sequencial do Sistema para uma Meta',ID_ETAPA VARCHAR(20) COMMENT 'Código Sequencial do Sistema para uma Etapa',NR_ETAPA VARCHAR(10) COMMENT 'Número da Etapa gerada pelo Sistema',DESC_ETAPA TEXT COMMENT 'Especificação da etapa vinculada a meta do cronograma físico',DATA_INICIO_ETAPA VARCHAR(20) COMMENT 'Data de início prevista para execução da etapa',DATA_FIM_ETAPA VARCHAR(20) COMMENT 'Data fim prevista para execução da etapa',UF_ETAPA varchar(10) COMMENT 'UF cadastrada para a Etapa',MUNICIPIO_ETAPA varchar(255) COMMENT 'Município cadastrado para a Etapa',ENDERECO_ETAPA TEXT COMMENT 'Endereço cadastrado para a Etapa',CEP_ETAPA varchar(10) COMMENT 'CEP cadastrado para a Etapa',QTD_ETAPA VARCHAR(20) COMMENT 'Quantidade da etapa vinculada a meta do cronograma físico.',UND_FORNECIMENTO_ETAPA TEXT COMMENT 'Unidade de fornecimento vinculada a etapa',VL_ETAPA FLOAT COMMENT 'Valor total da etapa vinculada a meta do cronograma físico');

-- CREATE TABLE IF NOT EXISTS INGRESSO_CONTRAPARTIDA (NR_CONVENIO VARCHAR(20) COMMENT 'Número gerado pelo Siconv. Possui faixa de numeração reservada que vai de 700000 a 999999',DT_INGRESSO_CONTRAPARTIDA VARCHAR(20) COMMENT 'Data da disponibilização do recurso por parte do Convenente',VL_INGRESSO_CONTRAPARTIDA FLOAT COMMENT 'Valor disponibilizado pelo Convenente para a conta do instrumento');

-- CREATE TABLE IF NOT EXISTS OBTV_CONVENENTE (NR_MOV_FIN VARCHAR(20) COMMENT 'Número identificador da movimentação financeira',IDENTIF_FAVORECIDO_OBTV_CONV varchar(255) COMMENT 'CNPJ/CPF do Favorecido recebedor do pagamento',NM_FAVORECIDO_OBTV_CONV TEXT COMMENT 'Nome do Favorecido recebedor do pagamento',TP_AQUISICAO TEXT COMMENT 'Tipo de Aquisição',VL_PAGO_OBTV_CONV FLOAT COMMENT 'Valor pago ao favorecido');

-- CREATE TABLE IF NOT EXISTS PLANO_APLICACAO_DETALHADO (ID_PROPOSTA VARCHAR(20) COMMENT 'Código Sequencial do Sistema para uma Proposta',SIGLA varchar(10) COMMENT 'UF cadastrada referente a localidade do item. Domínio: AC, AL, AM, AP, BA, CE, DF, ES, GO, MA, MG, MS, MT, PA, PB, PE, PI, PR, RJ, RN, RO, RR, RS, SC, SE, SP, TO',MUNICIPIO varchar(255) COMMENT 'Município cadastrado referente a localidade do item',NATUREZA_AQUISICAO VARCHAR(10) COMMENT 'Código de natureza de aquisição',DESCRICAO_ITEM TEXT COMMENT 'Descrição do Item',CEP_ITEM varchar(10) COMMENT 'CEP cadastrado referente a localidade do item',ENDERECO_ITEM TEXT COMMENT 'Endereço cadastrado referente a localidade do item',TIPO_DESPESA_ITEM varchar(255) COMMENT 'Tipo da Despesa. Domínio: SERVICO, BEM, OUTROS, TRIBUTO, OBRA e DESPESA_ADMINISTRATIVA',NATUREZA_DESPESA varchar(255) COMMENT 'Natureza da Despesa referente ao item',SIT_ITEM varchar(255) COMMENT 'Situação atual do Item. Domínio: APROVADO',COD_NATUREZA_DESPESA VARCHAR(20) COMMENT 'O campo que se refere à natureza da despesa contém um código composto por oito algarismos, sendo que o 1º dígito representa a categoria econômica, o 2º o grupo de natureza da despesa, o 3º e o 4º dígitos representam a modalidade de aplicação, o 5º e o 6º o elemento de despesa e o 7º e o 8º dígitos representam o desdobramento facultativo do elemento de despesa (subelemento).',QTD_ITEM VARCHAR(20) COMMENT 'Quantidade de Itens',VALOR_UNITARIO_ITEM FLOAT COMMENT 'Valor unitário do item',VALOR_TOTAL_ITEM FLOAT COMMENT 'Valor total do item');

-- CREATE TABLE IF NOT EXISTS PRORROGA_OFICIO (NR_CONVENIO VARCHAR(20) COMMENT 'Número gerado pelo Siconv. Possui faixa de numeração reservada que vai de 700000 a 999999',NR_PRORROGA VARCHAR(255) COMMENT 'Número do Prorroga de Ofício',DT_INICIO_PRORROGA VARCHAR(20) COMMENT 'Data Início de Vigência do Prorroga de Ofício',DT_FIM_PRORROGA VARCHAR(20) COMMENT 'Data Fim de Vigência do Prorroga de Ofício',DIAS_PRORROGA VARCHAR(10) COMMENT 'Dias de prorrogação',DT_ASSINATURA_PRORROGA VARCHAR(20) COMMENT 'Data de assinatura do Prorroga de Ofício',SIT_PRORROGA VARCHAR(255) COMMENT 'Situação atual do Prorroga de Ofício. Domínio: DISPONIBILIZADA, PUBLICADA');

-- CREATE TABLE IF NOT EXISTS TERMO_ADITIVO (NR_CONVENIO VARCHAR(20) COMMENT 'Número gerado pelo Siconv. Possui faixa de numeração reservada que vai de 700000 a 999999',NUMERO_TA VARCHAR(255) COMMENT 'Número do Termo Aditivo',TIPO_TA VARCHAR(255) COMMENT 'Tipo do Termo Aditivo',VL_GLOBAL_TA FLOAT COMMENT 'Valor Global referente ao TA',VL_REPASSE_TA FLOAT COMMENT 'Valor de Repasse referente ao TA',VL_CONTRAPARTIDA_TA FLOAT COMMENT 'Valor de Contrapartida referente ao TA',DT_ASSINATURA_TA VARCHAR(20) COMMENT 'Data da assinatura do Termo Aditivo',DT_INICIO_TA VARCHAR(20) COMMENT 'Data Início de Vigência do Termo Aditivo',DT_FIM_TA VARCHAR(20) COMMENT 'Data Fim de Vigência do Termo Aditivo',JUSTIFICATIVA_TA TEXT COMMENT 'Justificativa para a realização do Termo Aditivo');

-- CREATE TABLE IF NOT EXISTS EMPENHO_DESEMBOLSO (ID_DESEMBOLSO VARCHAR(20) COMMENT 'Identificador único gerado pelo Sistema para o Desembolso',ID_EMPENHO VARCHAR(20) COMMENT 'Identificador único gerado pelo Sistema para o Empenho',VALOR_GRUPO FLOAT COMMENT 'Valor presente nos dados orçamentários da OB');

-- CREATE TABLE IF NOT EXISTS HISTORICO_SITUACAO (ID_PROPOSTA VARCHAR(20) COMMENT 'Código Sequencial do Sistema para uma Proposta',NR_CONVENIO VARCHAR(20) COMMENT 'Número gerado pelo Siconv. Possui faixa de numeração reservada que vai de 700000 a 999999',DIA_HISTORICO_SIT VARCHAR(20) COMMENT 'Data de entrada da situação no sistema',HISTORICO_SIT TEXT COMMENT 'Situação histórica da Proposta/Convênio',DIAS_HISTORICO_SIT VARCHAR(10) COMMENT 'Dias em que a Proposta/Convênio permaneceu na situação',COD_HISTORICO_SIT VARCHAR(10) COMMENT 'Código da situação histórica da Proposta/Convênio, contendo a ordem cronológica do ciclo de vida de um convênio');

-- CREATE TABLE IF NOT EXISTS PROGRAMA_PROPOSTA (ID_PROGRAMA VARCHAR(20) COMMENT 'Código Sequencial do Sistema para um Programa',ID_PROPOSTA VARCHAR(20) COMMENT 'Código Sequencial do Sistema para uma Proposta');

CREATE INDEX idx_nr_convenio ON convenio(NR_CONVENIO);
CREATE INDEX idx_id_proposta ON convenio(ID_PROPOSTA);
CREATE INDEX idx_sit_convenio ON convenio(SIT_CONVENIO);
CREATE INDEX idx_id_proposta ON emenda(ID_PROPOSTA);
CREATE INDEX idx_id_proposta ON consorcios(ID_PROPOSTA);
CREATE INDEX idx_id_proposta ON proposta(ID_PROPOSTA);