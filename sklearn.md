# Scikit-Learn

## Conteúdos
- [Recursos do Scikit-Learn](#recursos-do-scikit-learn)
- [StandardScaler](#standardscaler)
- [Métricas](#métricas)
## Descrição 

### O que é o Scikit-Learn?

Scikit-learn é uma biblioteca de aprendizado de máquina de código aberto amplamente utilizada na linguagem de programação Python. Famosa por sua simplicidade e eficácia, é uma das ferramentas mais populares para análise de dados e modelagem estatística.

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
Precisão = \frac{True \ Positives}{True \ Positives + False \ Positives}
$$

Aumentar a precisão em alguns tipos de modelo é essencial, por exemplo, para modelos de Diagnóstico Médico, Sistemas de Justiça Criminal e Detecção de Fraude são alguns dos modelos que a precisão faz um papel essencial.

###  Recall (Sensibilidade)

###  F1-Score 

###  Matriz de Confusão 

###  ROC-AUC