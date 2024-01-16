# Scikit-Learn

## Conteúdos
- [Recursos do Scikit-Learn](#recursos-do-scikit-learn)
- [StandardScaler](#standardscaler)
- [Métricas](#métricas)
## Descrição 

### O que é o Scikit-Learn?

Scikit-learn é uma biblioteca de aprendizado de máquina de código aberto amplamente utilizada na linguagem de programação Python. Famosa por sua simplicidade e eficácia, é uma das ferramentas mais populares para análise de dados e modelagem estatística.


# Definições

## Tipos de Modelo

### Classificação


## Tipos de Classes

### Classe Balanceada


### Classe Desbalanceada




# Recursos do Scikit-Learn

## StandardScaler 

StandardScaler é uma ferramenta do Scikit-learn usada para pré-processar dados antes de aplicar algoritmos de aprendizado de máquina. Seu principal objetivo é padronizar os recursos (colunas) dos dados, garantindo que eles tenham média 0 e variância 1. 

Isso é especialmente importante para algoritmos que são sensíveis à escala dos dados, como K-Nearest Neighbors (KNN) e Support Vector Machines (SVM). A padronização é realizada subtraindo a média de cada recurso e depois dividindo pelo desvio padrão dos recursos. Matematicamente, para um recurso 
X, a padronização é dada por:

$$
Z = \frac{X - \mu}{\sigma}
$$

onde:
- `Z` é o valor padronizado.
- `X` é o valor original.
- $`\mu`$ é a média dos valores de `X`.
- $`\sigma`$ é o desvio padrão dos valores de `X`.

# Métricas

## Conteúdos
- [Acurácia](#acurácia)
- [Precisão ](#precisão)
- [Recall (Sensibilidade)](#recall-sensibilidade)
- [F1-Score](#f1-score)
- [Matriz de Confusão](#Matriz-de-Confusão )
- [ROC-AUC](#roc-auc)

## Classificação: 

### Acurácia 
A Acurácia é calculada com 

$$
Acurácia = \frac{Predicões \ Corretas}{Predicões \ Totais}
$$

Portanto, ela responde a pergunta de "o quanto esse modelo acerta?" ou "qual a % de acerto".

###  Precisão 
Para explicar o que é precisão é preciso explicar primeiramente o que é  Verdadeiro Positivo (True Positive) e Falso Positivo (False Positive). Para facilitar irei tomar o termo em inglês. Que pode ser compreendido com a imagem: 

![](img/Table1-2.png.webp)
A imagem foi retirada do site "https://plat.ai/blog/confusion-matrix-in-machine-learning/".

Para calcular a precisão podemos utilizar 

$$
Precisão = \frac{T_p}{T_p + F_p}
$$

- $T_p$ é True Positive.
- $F_p$ é False Positive.

Aumentar a precisão em alguns tipos de modelo é essencial, por exemplo, para modelos de Diagnóstico Médico, Sistemas de Justiça Criminal e Detecção de Fraude são alguns dos modelos que a precisão faz um papel essencial.

### Recall Sensibilidade
O Recall é calculado da seguinte forma: 

$$
Recall = \frac{T_p}{T_p + F_n}
$$

- $T_p$ é True Positive.
- $F_n$ é False Negative.

A importância do recall geralmente se baseia em uma análise de risco e custo associada a falsos negativos.  Em muitos casos, existe um trade-off entre recall e precisão. Melhorar o recall muitas vezes significa aceitar mais falsos positivos.

Em resumo, o recall é uma métrica importante em cenários onde as consequências de não detectar um evento positivo (falso negativo) são mais graves do que detectar erroneamente um evento que não ocorreu (falso positivo).

###  F1-Score 
O F1 Score é a média harmônica entre precisão e recall, oferecendo uma única medida que leva em conta tanto a capacidade do modelo de identificar corretamente os casos positivos (recall) quanto a sua precisão em não classificar incorretamente os casos negativos como positivos. Ele é definido da seguinte forma:

$$
F_1 = 2*\frac{P \ \times R}{P \ + R}
$$

- $P$ é Precisão.
- $R$ é Recall.

Com um único número representando tanto a precisão quanto o recall, o F1 Score facilita a comparação entre diferentes modelos ou abordagens, ajudando na escolha do modelo mais adequado para um problema específico.


###  Matriz de Confusão 
 A matriz de confusão é uma tabela que permite a visualização do desempenho do algoritmo, mostrando a comparação entre os valores reais (verdadeiros) e os valores previstos pelo modelo. Para casos binarios, verdadeiro ou falso podemos entender a matriz de confusão como o calculo de  $\ T_p, \ T_n, \ F_p,\  F_n$. 

Temos também a Aplicação em Classificação Multiclasse, em problemas de classificação multiclasse, a matriz de confusão se expande para acomodar todas as classes. Neste caso, ela tem dimensões n x n, onde n é o número de classes. Cada linha representa as instâncias de uma classe real, e cada coluna representa as instâncias de uma classe prevista.

Em resumo, a matriz de confusão é uma ferramenta essencial para a análise profunda do desempenho de modelos de classificação, fornecendo informações valiosas para aprimorar a precisão do modelo.

###  ROC-AUC
ROC-AUC é uma métrica usada para avaliar o desempenho de modelos de classificação, especialmente em contextos onde as classes são desbalanceadas. ROC significa "Receiver Operating Characteristic" e AUC é a "Area Under the Curve" (Área Sob a Curva). Juntas, essas duas medidas fornecem uma visão abrangente de quão bem um modelo de classificação pode distinguir entre as diferentes classes.

Entendendo a Curva ROC:
Curva ROC: É um gráfico que representa a relação entre a taxa de verdadeiros positivos (sensibilidade ou recall) e a taxa de falsos positivos para diferentes limiares de classificação.

Eixo X (Taxa de Falsos Positivos): Representa a proporção de negativos reais que foram incorretamente classificados como positivos.
Eixo Y (Taxa de Verdadeiros Positivos): Representa a proporção de positivos reais que foram corretamente identificados.
Pontos na Curva: Cada ponto na curva ROC representa um limiar de decisão específico. Um modelo que faz previsões aleatórias terá uma curva ROC que é uma linha diagonal do canto inferior esquerdo para o canto superior direito do gráfico. Modelos com melhor desempenho têm curvas ROC que se inclinam para o canto superior esquerdo.

Entendendo AUC (Área Sob a Curva):
AUC: Mede toda a área bidimensional sob a curva ROC.
AUC = 1: Representa um modelo perfeito que consegue distinguir perfeitamente entre todas as classes positivas e negativas.
0.5 ≤ AUC < 1: Quanto maior o AUC, melhor o modelo é em distinguir entre as classes. Um modelo com AUC de 0.5 não tem capacidade de classificação melhor do que o acaso.
AUC < 0.5: Sugere que o modelo está fazendo pior do que classificações aleatórias.

Ao contrário de outras métricas como precisão e recall, o ROC-AUC fornece uma medida de desempenho que é independente de um limiar de classificação específico. O ROC-AUC é uma ferramenta poderosa na avaliação de modelos de classificação, particularmente útil em situações com desbalanceamento de classes ou quando a escolha de um limiar de decisão específico não é clara. Ele oferece uma visão mais holística do desempenho do modelo em comparação a métricas que dependem de um limiar específico, como precisão e recall.









































