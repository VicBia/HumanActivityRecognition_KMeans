# Projeto de Agrupamento de Atividades Humanas com K-means

## Objetivo do Projeto

O objetivo deste projeto é implementar e avaliar o algoritmo de K-means usando o dataset "Human Activity Recognition Using Smartphones". O foco está em agrupar atividades humanas com base em dados coletados por sensores, como acelerômetros e giroscópios, embarcados em smartphones. 
 
O projeto abrange desde a análise exploratória dos dados até a determinação do número ideal de clusters, a visualização dos resultados e a elaboração de um relatório técnico detalhado.

## Instruções para Executar o Código

- Instalação:

Clone o repositório:
git clone (https://github.com/VicBia/HumanActivityRecognition_KMeans.git)

- Importe as bibliotecas:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

- Carregue o arquivo CSV e visualize os dados:
treino = pd.read_csv('train.csv')
teste = pd.read_csv('test.csv')
print(treino.head())
print(treino.info())
print(treino.describe())

- Execute o processo de limpeza, normalização e redução de dimensionalidade:
Remova valores ausentes e normalize os dados
Aplique PCA para redução de dimensionalidade

- Realize uma análise e visualize os gráficos:
Método do Cotovelo para escolha de clusters
Visualize os clusters em 2D e 3D

- Execute o agrupamento K-means e avalie os resultados:
Determine o número ideal de clusters pelo Silhouette Score

## Estrutura dos Arquivos
- train.csv : Conjunto de dados de treinamento das atividades humanas.
- test.csv : Conjunto de dados de teste das atividades humanas.
- main_script.py: Script contendo o código principal para análise.
- README.md: Documentação do projeto.

## Tecnologias Utilizadas
- Python
- Bibliotecas : NumPy, pandas, matplotlib, seaborn,sklearn

## Principais Conclusões e Considerações
### Escolha do Número Ideal de Clusters:
O número ideal de clusters foi determinado com base no Silhouette Score , que atingiu o valor máximo para K=2 clusters .
Apesar de existirem mais atividades diferentes no conjunto de dados, os padrões naturais capturados pelo K-means sugerem uma separação mais clara em dois grupos principais.
### Separação dos Clusters:
Os clusters foram bem separados no espaço dimensional limitado pelo PCA , apesar de os dados apresentarem padrões claros que permitem agrupamento eficaz.
### Distribuição dos Dados:
Foi observada uma correspondência significativa entre os clusters formados e algumas atividades humanas apresentadas, o que reforça que os sensores captam variações relevantes nos movimentos.
### Estabilidade do Agrupamento:
O modelo apresentou estabilidade consistente ao longo de múltiplas execuções, com atribuições de clusters confiáveis ​​e reprodutíveis. Isso reforça a robustez do modelo na definição dos grupos.
### Visualização:
As representações em 2D e 3D ilustraram claramente a separação e a compactação dos clusters no espaço dos componentes principais, confirmando a eficácia do K-means para este problema.
### Considerações Gerais:
1. Implicações de K=2:
* Uma simplificação para dois clusters sugere que algumas atividades humanas compartilham padrões muito semelhantes (atividades estáticas e dinâmicas) quando aparecem no espaço limitado pelo PCA.
* Alternativamente, uma técnica de clustering pode não estar capturando toda a complexidade dos dados brutos, ou que pode exigir uma abordagem mais específica.

## Autores e Colaboradores
Anna Miranda e Victoria Reis - Desenvolvimento do código


