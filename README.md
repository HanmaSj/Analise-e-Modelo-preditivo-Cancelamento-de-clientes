# Analise de Dados e Modelo Preditivo para cancelamento de clientes.

Projeto de Ciência dos Dados.

#### Este projeto utiliza técnicas de Ciência de Dados para analisar e prever futuros cancelamentos de clientes ao serviço de uma empresa fictícia.
#### O objetivo é encontrar padrões de cancelamentos, e melhorias internas que podem ser feita na empresa.


##### O projeto foi feito inteiramente com python, utilizando módulos e bibliotecas.

- Para manipulação e tratamento dos Dados foram utilizadas as bibliotecas Pandas e Numpy.

- Para Visualização dos Dados foram utilizadas as bibliotecas Matplotlib, Plotly e seaborn.

- Para criação, treinamento e teste dos Modelos foram usadas as bibliotecas scikit-learn e XGBoost (Foi testado mais de 1 modelo).

- Também foi utilizado o Módulo "utils", um módulo pessoal com códigos reutilizaveis de Machine Learning. (Pode ser visualizado através das pastas: Projeto -> utils)


### Passo a Passo do Projeto:

1 - Limpeza e Tratamento de Dados.

  - Nessa parte foi feito a imputação de valores ausentes, retirada de duplicatas de linhas, checagem de Outliers e aplicações de outras técnicas de limpeza dos Dados. Além do reconhecimento primário dos Dados.
  - Fase essencial para criação futura do modelo e Análise dos Dados.


2 - Análise dos Dados.

  - Nessa fase foi realizada a Análise Exploratória dos Dados, buscando e registrando os melhores Insights presentes na Base de Dados.


3 - Modelo.

  - Sendo a última fase envolvendo código, foi feito a criação de múltiplos Modelos (com os dados tratados), e a avaliação dos modelos criados.
  - Junto também foi feito a escolha e avaliação aprofundada do Melhor modelo.


4 - Relatório.

  - Sendo a última fase, foi relatado os principais pontos do projeto:
    
      Oque foi encontrado nos Dados.
    
      Quais os pontos que a empresa deve focar.

      Oque leva os clientes a cancelar o serviço.
    
      Propostas de intervenção.


## O PROJETO/CÓDIGOS PODEM SER EXECUTADOS ATRAVÉS DE UM NOTEBOOK PYTHON.

- O Arquivo de Dados ("cancelamentos_sample.csv") deve estar no mesmo diretório do Arquivo com final ".ipynb" (Para que a leitura dos Dados sejam feita diretamente).

- Caso não esteja no mesmo diretório, deve ser adicionado na leitura dos Dados - "df = pd.read_csv("caminho/completo/cancelamentos_sample.csv") - o caminho estendido da Base de Dados, assim como no exemplo.
