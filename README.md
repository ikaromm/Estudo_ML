# Estudo_ML

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
