# Importando bibliotecas necessárias para análise de dados, visualização e aprendizado de máquina
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

"""# Análise Exploratória

- Examinar as distribuições das variáveis e detectar padrões ou variações nas medições dos sensores.

- Avaliar possíveis correlações entre as variáveis e selecionar aquelas que podem facilitar o agrupamento das atividades.

- Reduzir a dimensionalidade dos dados (por exemplo, utilizando PCA) para facilitar a visualização e interpretação dos clusters.

"""

# Carregando os datasets de treinamento e teste a partir de arquivos CSV
teste = pd.read_csv("test.csv", encoding="utf-8")
treino = pd.read_csv("train.csv", encoding="utf-8")

# Visualizando as primeiras linhas do dataset de treinamento
treino.head()

# Exibindo informações básicas sobre o dataset de treinamento e teste, informações sobre as colunas, tipos de dados e valores nulos
print("Informações do Dataset de Treinamento:")
print(treino.info())
print("\nInformações do Dataset de Teste:")
print(teste.info())

# Examinando as distribuições das variáveis no dataset de treinamento, estatísticas como média, desvio padrão, mínimo, etc.
print("\nEstatísticas Resumidas (Treinamento):")
print(treino.describe())

# Visualizando distribuições de variáveis selecionadas
# Selecionando todas as colunas exceto a última, assumindo que a última é o alvo
colunas_sensores = treino.columns[:-1]

# Exemplo: Visualizando as distribuições das três primeiras colunas de sensores
for coluna in colunas_sensores[:3]:
    plt.figure(figsize=(8, 4))
  # Histograma com curva KDE
    sns.histplot(treino[coluna], kde=True, bins=30, color='blue')
    plt.title(f'Distribuição da variável {coluna}')
    plt.xlabel(coluna)
    plt.ylabel('Frequência')
    plt.show()

# Avaliando correlações entre variáveis
# Calculando a matriz de correlação para as colunas de sensores
matriz_correlacao = treino[colunas_sensores].corr()

# Visualizando a matriz de correlação com um heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(matriz_correlacao, cmap='coolwarm', annot=False, fmt=".2f")
plt.title('Matriz de Correlação (Dataset de Treinamento)')
plt.show()

# Selecionando pares de variáveis com alta correlação
# Ordenando as correlações absolutas e excluindo autocorrelações (valor 1)
pares_alta_correlacao = matriz_correlacao.abs().unstack().sort_values(ascending=False)
pares_alta_correlacao = pares_alta_correlacao[pares_alta_correlacao != 1]

# Exibindo os 10 pares com maior correlação
print("\nTop 10 pares de variáveis com maior correlação:")
print(pares_alta_correlacao[:10])

# Verificando valores ausentes no dataset de treinamento
print("\nNúmero de valores ausentes no dataset de treinamento:")
print(treino.isnull().sum())

# Verificando valores ausentes no dataset de teste
print("\nNúmero de valores ausentes no dataset de teste:")
print(teste.isnull().sum())

# Removendo linhas com valores ausentes (NaN) nos datasets de treinamento e teste
# Remove linhas com valores NaN no dataset de treinamento
treino = treino.dropna()
# Remove linhas com valores NaN no dataset de teste
teste = teste.dropna()

#Redução de dimensionalidade usando PCA (Análise de Componentes Principais)

# Padronizando os dados para ter média 0 e desvio padrão 1
escalador = StandardScaler()
# Treinamento padronizado
dados_treino_escalados = escalador.fit_transform(treino[colunas_sensores])
# Teste padronizado com os mesmos parâmetros
dados_teste_escalados = escalador.transform(teste[colunas_sensores])

# Aplicando PCA para reduzir dimensionalidade, preservando 95% da variância
 # Preserva 95% da variância
pca = PCA(n_components=0.95)
# Treinamento reduzido
dados_treino_pca = pca.fit_transform(dados_treino_escalados)
# Teste reduzido com os mesmos componentes
dados_teste_pca = pca.transform(dados_teste_escalados)

# Variância explicada por cada componente principal
plt.figure(figsize=(8, 4))
plt.plot(
    range(1, len(pca.explained_variance_ratio_) + 1),
    pca.explained_variance_ratio_.cumsum(),
    marker='o'
)
plt.xlabel('Número de Componentes Principais')
plt.ylabel('Variância Explicada Acumulada')
plt.title('PCA - Variância Explicada')
plt.show()

# Criando um DataFrame com os resultados do PCA e as etiquetas de atividade
df_treino_pca = pd.DataFrame(
    dados_treino_pca,
    # Nomes das componentes principais
    columns=[f'CP{i+1}' for i in range(dados_treino_pca.shape[1])]
)
# Assume que a última coluna do dataset é a etiqueta (Atividade)
df_treino_pca['Atividade'] = treino.iloc[:, -1]

# Visualizando os dois primeiros componentes principais em um gráfico de dispersão
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_treino_pca,
    x='CP1', y='CP2',
    hue='Atividade',
    palette='viridis',
    alpha=0.7
)
plt.title('Clusters de Atividades - PCA')
plt.xlabel('Componente Principal 1 (CP1)')
plt.ylabel('Componente Principal 2 (CP2)')
plt.show()

# Resumo dos Componentes do PCA
# Exibindo o número de componentes principais retidos após a redução de dimensionalidade
print(f"\nNúmero de componentes principais retidos: {pca.n_components_}")

# Exibindo a proporção de variância explicada por cada componente principal
print("Proporção de Variância Explicada por Componente Principal:")
print(pca.explained_variance_ratio_)

"""# Implementação do Algoritmo de K-means
- Desenvolvimento do Modelo: Implemente o algoritmo de K-means usando Scikit-Learn. Considerando a alta dimensionalidade dos dados, selecione as variáveis mais relevantes para o agrupamento ou aplique técnicas de redução de dimensionalidade para tornar o processo mais eficiente.

- Escolha do Número de Clusters (K): A escolha do número de clusters é crucial. Utilize métodos como o cotovelo(elbow method) e o silhouette score para encontrar o valor ideal de K que represente as atividades de forma mais natural. Documente o processo e inclua gráficos que suportem sua escolha.

"""

# Determinando o número ideal de clusters usando o Método do Cotovelo

# Lista para armazenar a inércia (soma das distâncias quadradas dentro dos clusters)
inercias = []
# Intervalo de valores de K (número de clusters) a ser avaliado
intervalo_K = range(1, 11)

# Calculando a inércia para diferentes números de clusters
for k in intervalo_K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # Configurações do algoritmo K-Means
    kmeans.fit(dados_treino_pca)  # Ajuste nos dados transformados pelo PCA
    inercias.append(kmeans.inertia_)  # Armazena a inércia do modelo com K clusters

# Plotando o Método do Cotovelo
plt.figure(figsize=(8, 5))
plt.plot(intervalo_K, inercias, marker='o')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inércia')
plt.title('Método do Cotovelo para Escolha de K')
plt.grid()
plt.show()

# Calculando a pontuação de Silhouette para diferentes valores de K

# Lista para armazenar os scores de Silhouette
scores_silhouette = []

# Calculando Silhouette Score para K entre 2 e 10 (não definido para K=1)
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # Configurações consistentes
    rotulos_kmeans = kmeans.fit_predict(dados_treino_pca)  # Gera rótulos para os clusters
    score = silhouette_score(dados_treino_pca, rotulos_kmeans)  # Calcula o Silhouette Score
    scores_silhouette.append(score)  # Armazena o score para cada K

# Plotando os Scores de Silhouette
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), scores_silhouette, marker='o', color='orange')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Score de Silhouette')
plt.title('Pontuação de Silhouette para Escolha de K')
plt.grid()
plt.show()

# Escolhendo o K ideal com base no Silhouette Score
k_otimo = scores_silhouette.index(max(scores_silhouette)) + 2
print(f"\nNúmero ideal de clusters baseado no Score de Silhouette: {k_otimo}")

# Criação do modelo K-means com o número ideal de clusters determinado anteriormente
kmeans = KMeans(n_clusters=k_otimo, random_state=42, n_init=10)
# Ajusta o modelo aos dados transformados pelo PCA e atribui rótulos aos clusters
rotulos_kmeans = kmeans.fit_predict(dados_treino_pca)

# Adiciona os rótulos dos clusters ao DataFrame contendo os dados transformados pelo PCA
df_treino_pca['Cluster'] = rotulos_kmeans

# Visualizando os clusters usando os dois primeiros componentes principais (PCA)

# Criação de um gráfico de dispersão com os clusters coloridos pelos rótulos atribuídos
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_treino_pca, x='CP1', y='CP2', hue='Cluster', palette='viridis', alpha=0.7
)
# Adiciona os centróides dos clusters ao gráfico
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200, c='red', marker='X', label='Centróides'
)
plt.title('Agrupamento K-means com o Número Ideal de Clusters')
plt.legend()
plt.show()

# Executando o K-means novamente para verificar a consistência dos clusters

# Criação do modelo K-means final com inicialização 'k-means++'
kmeans_final = KMeans(n_clusters=k_otimo, init='k-means++', n_init=10, random_state=42)
# Ajusta o modelo final e obtém os rótulos finais
rotulos_finais = kmeans_final.fit_predict(dados_treino_pca)
# Adiciona os rótulos finais ao DataFrame de PCA para análise visual
df_treino_pca['Cluster'] = rotulos_finais

# Visualizando novamente os clusters com o modelo final
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_treino_pca, x='CP1', y='CP2', hue='Cluster', palette='viridis', alpha=0.7
)
plt.scatter(
    kmeans_final.cluster_centers_[:, 0],
    kmeans_final.cluster_centers_[:, 1],
    s=200, c='red', marker='X', label='Centróides'
)
plt.title(f'Agrupamento Final K-means com K={k_otimo}')
plt.legend()
plt.show()

# Realizando múltiplas execuções para verificar a estabilidade dos clusters

# Define o número de execuções para verificar a consistência
numero_execucoes = 10
# Lista para armazenar as atribuições de clusters de cada execução
clusters_estabilidade = []

# Executa o K-means várias vezes, alterando o estado aleatório em cada execução
for execucao in range(numero_execucoes):
    kmeans_temp = KMeans(
        n_clusters=k_otimo, init='k-means++', n_init=1, random_state=42 + execucao
    )
    clusters_temp = kmeans_temp.fit_predict(dados_treino_pca)
    clusters_estabilidade.append(clusters_temp)

# Converte os resultados em um array NumPy para análise
estabilidade = np.array(clusters_estabilidade)

# Calcula o modo (atribuição mais frequente) para cada ponto entre as execuções
print(f"\nVerificação de estabilidade após {numero_execucoes} execuções:")
modo_clusters = pd.DataFrame(estabilidade).mode(axis=0).iloc[0]
print("Atribuições mais frequentes de clusters:")
print(modo_clusters)

# A estabilidade dos clusters é verificada para garantir que as atribuições de grupos não dependam fortemente das inicializações aleatórias, indicando um agrupamento confiável.

# Avaliação do modelo final com o Silhouette Score

# Calcula o Silhouette Score para o agrupamento final
score_final_silhouette = silhouette_score(dados_treino_pca, rotulos_finais)
print(f"\nSilhouette Score para o agrupamento final com K={k_otimo}: {score_final_silhouette}")

# O Silhouette Score para o modelo final confirma a qualidade do agrupamento.
# Um valor próximo de 1 indica clusters bem separados, enquanto valores próximos de 0 ou negativos  podem indicar clusters sobrepostos ou mal definidos.

# Avaliação do agrupamento final: Cálculo da inércia e do Silhouette Score

# Calcula a inércia do modelo final
inercias_final = kmeans_final.inertia_
# Calcula o Silhouette Score do modelo final
score_silhouette_final = silhouette_score(dados_treino_pca, rotulos_finais)

# Exibe os resultados de inércia e Silhouette Score
print(f"\nInércia do agrupamento final: {inercias_final}")
print(f"Score de Silhouette do agrupamento final: {score_silhouette_final}")

# A inércia mede a compactação dos clusters, com valores menores indicando clusters mais coesos.
# O Silhouette Score complementa, indicando o grau de separação e definição dos clusters.

# Interpretação dos clusters: Análise das características de cada grupo

# Criação de um DataFrame com as coordenadas dos centróides nos componentes principais
df_centroides = pd.DataFrame(
    kmeans_final.cluster_centers_,
    columns=[f'CP{i+1}' for i in range(dados_treino_pca.shape[1])]
)

# Exibição das coordenadas dos centróides
print("\nCentroides dos Clusters:")
print(df_centroides)

# Analisando a distribuição de atividades dentro de cada cluster
df_treino_pca['Atividade'] = treino.iloc[:, -1]  # Supondo que a última coluna é a atividade
distribuicao_clusters = df_treino_pca.groupby(['Cluster', 'Atividade']).size().unstack(fill_value=0)

# Exibição da distribuição de atividades por cluster
print("\nDistribuição de Atividades nos Clusters:")
print(distribuicao_clusters)

# 1. Analisar os centróides ajuda a interpretar as características médias de cada cluster nos componentes principais.
# 2. Ver a distribuição de atividades por cluster permite verificar como os dados foram segmentados,
# identificando possíveis padrões ou sobreposições entre classes.

# Visualização 2D dos clusters usando os dois primeiros componentes principais

# Gráfico de dispersão dos dados nos componentes CP1 e CP2, com cores indicando os clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_treino_pca, x='CP1', y='CP2', hue='Cluster', palette='viridis', alpha=0.7
)
# Adiciona os centróides dos clusters no gráfico
plt.scatter(
    kmeans_final.cluster_centers_[:, 0],
    kmeans_final.cluster_centers_[:, 1],
    s=200, c='red', marker='X', label='Centróides'
)
plt.title(f'Agrupamento K-means (2D) com K={k_otimo}')
plt.legend()
plt.show()

#  A visualização em 2D permite verificar a separação dos clusters em um espaço bidimensional.
# Isso é útil para compreender a estrutura geral dos dados, mesmo que haja mais dimensões envolvidas.

# Visualização 3D dos clusters usando os três primeiros componentes principais

# Configuração da figura para visualização em 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Gráfico de dispersão no espaço tridimensional
ax.scatter(
    df_treino_pca['CP1'], df_treino_pca['CP2'], df_treino_pca['CP3'],
    c=df_treino_pca['Cluster'], cmap='viridis', alpha=0.7
)
# Adiciona os centróides dos clusters no espaço tridimensional
ax.scatter(
    kmeans_final.cluster_centers_[:, 0],
    kmeans_final.cluster_centers_[:, 1],
    kmeans_final.cluster_centers_[:, 2],
    s=200, c='red', marker='X', label='Centróides'
)

# Configuração dos rótulos dos eixos e do título do gráfico
ax.set_xlabel('CP1')
ax.set_ylabel('CP2')
ax.set_zlabel('CP3')
ax.set_title(f'Agrupamento K-means (3D) com K={k_otimo}')
plt.legend()
plt.show()

#  Visualização 3D é útil para explorar dados que foram reduzidos em mais de duas dimensões.
# Isso fornece insights adicionais sobre como os clusters estão distribuídos no espaço de variância reduzida.