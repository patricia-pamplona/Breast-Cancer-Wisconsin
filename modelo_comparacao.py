import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# ============================================================
# 1. CARREGAMENTO DOS DADOS
# ============================================================

arquivo = "breast-cancer-wisconsin.data"

colunas = [
    "id",
    "clump_thickness",
    "uniformity_cell_size",
    "uniformity_cell_shape",
    "marginal_adhesion",
    "single_epithelial_cell_size",
    "bare_nuclei",
    "bland_chromatin",
    "normal_nucleoli",
    "mitoses",
    "diagnosis"
]

df = pd.read_csv(
    arquivo,
    names=colunas,
    na_values="?"
)

print("=" * 70)
print("BREAST CANCER WISCONSIN (ORIGINAL)")
print("=" * 70)

print("\nDimensão original dos dados:")
print(df.shape)


# ============================================================
# 2. VERIFICAR DADOS AUSENTES
# ============================================================

print("\nDados ausentes por atributo:")
print(df.isnull().sum())


# ============================================================
# 3. DESCARTAR INST�NCIAS COM DADOS AUSENTES
# ============================================================

df = df.dropna()

print("\nDimensão após remover dados ausentes:")
print(df.shape)


# ============================================================
# 4. DESCARTAR O ATRIBUTO ID
# ============================================================

df = df.drop(columns=["id"])

print("\nColunas utilizadas no modelo:")
print(df.columns.tolist())


# ============================================================
# 5. SEPARAR ATRIBUTOS E CLASSE
# ============================================================

X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

print("\nQuantidade de amostras por classe:")
print(y.value_counts())


# ============================================================
# 6. DIVIS�O DOS DADOS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nAmostras de treinamento:", len(X_train))
print("Amostras de teste:", len(X_test))


# ============================================================
# 7. NORMALIZA��O
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# MODELO 1
# CLASSIFICADOR BASEADO EM CENTR�IDES M�DIOS
# ============================================================

print("\n")
print("=" * 70)
print("MODELO 1 - CENTRÓIDES MÉDIOS")
print("=" * 70)


# Separar as amostras de treinamento por classe

X_train_benigno = X_train_scaled[y_train.values == 2]

X_train_maligno = X_train_scaled[y_train.values == 4]


# Calcular os centr�ides

centroide_benigno = np.mean(
    X_train_benigno,
    axis=0
)

centroide_maligno = np.mean(
    X_train_maligno,
    axis=0
)


# Fun��o de classifica��o

def classificar_amostra(amostra):

    distancia_benigno = np.linalg.norm(
        amostra - centroide_benigno
    )

    distancia_maligno = np.linalg.norm(
        amostra - centroide_maligno
    )

    if distancia_benigno < distancia_maligno:
        return 2

    return 4


# Fazer previsões

y_pred_centroides = []

for amostra in X_test_scaled:

    classe = classificar_amostra(amostra)

    y_pred_centroides.append(classe)

y_pred_centroides = np.array(y_pred_centroides)


# ============================================================
# FUNÇÃO PARA CALCULAR AS ACURÁCIAS
# ============================================================

def calcular_acuracias(y_real, y_pred):

    acuracia_geral = np.mean(
        y_pred == y_real
    )

    mascara_benigno = y_real == 2

    acuracia_benigno = np.mean(
        y_pred[mascara_benigno] ==
        y_real[mascara_benigno]
    )

    mascara_maligno = y_real == 4

    acuracia_maligno = np.mean(
        y_pred[mascara_maligno] ==
        y_real[mascara_maligno]
    )

    return (
        acuracia_geral,
        acuracia_benigno,
        acuracia_maligno
    )


# Resultados dos centr�ides

resultados_centroides = calcular_acuracias(
    y_test.values,
    y_pred_centroides
)

print("\nResultados - Centróides Médios:")

print(
    f"Acurácia geral: "
    f"{resultados_centroides[0] * 100:.2f}%"
)

print(
    f"Acurácia Benigno: "
    f"{resultados_centroides[1] * 100:.2f}%"
)

print(
    f"Acurácia Maligno: "
    f"{resultados_centroides[2] * 100:.2f}%"
)


# ============================================================
# MODELO 2
# K-NEAREST NEIGHBORS (KNN)
# ============================================================

print("\n")
print("=" * 70)
print("MODELO 2 - K-NEAREST NEIGHBORS (KNN)")
print("=" * 70)


# Criar o modelo

modelo_knn = KNeighborsClassifier(
    n_neighbors=5
)


# Treinar o modelo

modelo_knn.fit(
    X_train_scaled,
    y_train
)


# Fazer previs�es

y_pred_knn = modelo_knn.predict(
    X_test_scaled
)


# Calcular resultados

resultados_knn = calcular_acuracias(
    y_test.values,
    y_pred_knn
)


print("\nResultados - KNN:")

print(
    f"Acurácia geral: "
    f"{resultados_knn[0] * 100:.2f}%"
)

print(
    f"Acurácia Benigno: "
    f"{resultados_knn[1] * 100:.2f}%"
)

print(
    f"Acurácia Maligno: "
    f"{resultados_knn[2] * 100:.2f}%"
)


# ============================================================
# COMPARA��O DOS MODELOS
# ============================================================

print("\n")
print("=" * 70)
print("COMPARAÇÃO DOS MODELOS")
print("=" * 70)

print(
    f"\n{'Métrica':<25}"
    f"{'Centróides':>15}"
    f"{'KNN':>15}"
)

print("-" * 55)

print(
    f"{'Acurácia geral':<25}"
    f"{resultados_centroides[0] * 100:>14.2f}%"
    f"{resultados_knn[0] * 100:>14.2f}%"
)

print(
    f"{'Acurácia Benigno':<25}"
    f"{resultados_centroides[1] * 100:>14.2f}%"
    f"{resultados_knn[1] * 100:>14.2f}%"
)

print(
    f"{'Acurácia Maligno':<25}"
    f"{resultados_centroides[2] * 100:>14.2f}%"
    f"{resultados_knn[2] * 100:>14.2f}%"
)