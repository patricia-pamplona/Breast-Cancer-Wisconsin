# Classificação de Nódulos Mamários - Breast Cancer Wisconsin

Projeto desenvolvido para a disciplina de Pós-Graduação em Ciência de Dados, utilizando o conjunto de dados **Breast Cancer Wisconsin (Original)**, disponibilizado pela UCI Machine Learning Repository.

## 1. Objetivo

Desenvolver um modelo de classificação capaz de prever se um nódulo mamário é **benigno** ou **maligno**, utilizando os atributos disponíveis no conjunto de dados.

O modelo principal solicitado na atividade é um **classificador baseado em centróides médios**. Como comparação, também foi implementado um modelo utilizando o algoritmo **K-Nearest Neighbors (KNN)**.

## 2. Dataset

Foi utilizado o conjunto de dados **Breast Cancer Wisconsin (Original)**.

O dataset possui originalmente:

- 699 instâncias;
- 11 atributos;
- 9 atributos utilizados como variáveis preditoras;
- 1 atributo de identificação (`ID`);
- 1 atributo de diagnóstico (`diagnosis`).

O atributo `diagnosis` possui os seguintes valores:

- `2` = Benigno;
- `4` = Maligno.

## 3. Tratamento dos dados

O conjunto de dados possui valores ausentes representados pelo caractere `?`.

De acordo com o requisito da atividade, as instâncias que possuem dados ausentes foram **descartadas**, não sendo realizada imputação de valores.

O atributo `ID` também foi removido, pois representa apenas a identificação da instância e não deve ser utilizado como variável preditora.

Após a remoção das instâncias com dados ausentes, foram utilizadas **683 instâncias** para a construção dos modelos.

## 4. Modelo 1 - Classificador baseado em centróides médios

O primeiro modelo utiliza um classificador baseado em **centróides médios**.

Para cada classe, é calculado um vetor representando o centro médio dos exemplos pertencentes àquela classe no conjunto de treinamento.

Foram calculados dois centróides:

- Centróide da classe Benigno;
- Centróide da classe Maligno.

Para classificar uma nova instância, é calculada a distância euclidiana entre a instância e cada centróide.

A classe atribuída é aquela cujo centróide apresenta a menor distância.

## 5. Modelo 2 - K-Nearest Neighbors (KNN)

Como modelo adicional para comparação, foi implementado o algoritmo **K-Nearest Neighbors (KNN)**.

Foi utilizado:

# python
KNeighborsClassifier(n_neighbors=5)


# Tecnologias utilizadas
Python
Pandas
NumPy
Scikit-learn
VS Code
Git
GitHub

# Como executar

Clone o repositório e acesse a pasta do projeto.

Instale as dependências:
pip install pandas numpy scikit-learn matplotlib seaborn

Para executar o modelo principal:
python modelo_cancer.py

Para executar a comparação entre os modelos:
python modelo_comparacao.py

