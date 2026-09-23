import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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

# 1.1 "?" será interpretado pelo Pandas como NaN

df = pd.read_csv(
    arquivo,
    names=colunas,
    na_values="?"
)


print("=" * 60)
print("BREAST CANCER WISCONSIN (ORIGINAL)")
print("=" * 60)

print("\nDimensão original dos dados:")
print(df.shape)


# ============================================================
# 2. VERIFICAR DADOS AUSENTES
# ============================================================

print("\nDados ausentes por atributo:")
print(df.isnull().sum())


# ============================================================
# 3. DESCARTAR INSTÂNCIAS COM DADOS AUSENTES
# ============================================================

df = df.dropna()

print("\nDimensão após remover dados ausentes:")
print(df.shape)


# ============================================================
# 4. DESCARTAR O ATRIBUTO ID
# ============================================================

df = df.drop(columns=["id"])


print("\nColunas utilizadas:")
print(df.columns.tolist())


# ============================================================
# 5. SEPARAR ATRIBUTOS E CLASSE
# ============================================================

X = df.drop(columns=["diagnosis"])

y = df["diagnosis"]


print("\nQuantidade de amostras por classe:")
print(y.value_counts())


# ============================================================
# 6. DIVISÃO ENTRE TREINAMENTO E TESTE
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nQuantidade de amostras para treinamento:")
print(len(X_train))

print("\nQuantidade de amostras para teste:")
print(len(X_test))


# ============================================================
# 7. NORMALIZAÇÃO DOS ATRIBUTOS
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# 8. CALCULAR OS CENTRÓIDES MÉDIOS
# ============================================================

# Separar as amostras de cada classe

X_train_benigno = X_train_scaled[y_train.values == 2]

X_train_maligno = X_train_scaled[y_train.values == 4]


# Calcular a média de cada atributo

centroide_benigno = np.mean(
    X_train_benigno,
    axis=0
)

centroide_maligno = np.mean(
    X_train_maligno,
    axis=0
)


print("\nCentróide da classe BENIGNA:")
print(centroide_benigno)

print("\nCentróide da classe MALIGNA:")
print(centroide_maligno)


# ============================================================
# 9. FUNÇÃO DE CLASSIFICAÇÃO
# ============================================================

def classificar_amostra(amostra):

    distancia_benigno = np.linalg.norm(
        amostra - centroide_benigno
    )

    distancia_maligno = np.linalg.norm(
        amostra - centroide_maligno
    )

    if distancia_benigno < distancia_maligno:
        return 2

    else:
        return 4


# ============================================================
# 10. REALIZAR AS PREVISÕES
# ============================================================

y_pred = []

for amostra in X_test_scaled:

    classe = classificar_amostra(amostra)

    y_pred.append(classe)


y_pred = np.array(y_pred)


# ============================================================
# 11. ACURÁCIA GERAL
# ============================================================

acuracia_geral = np.mean(
    y_pred == y_test.values
)


print("\n" + "=" * 60)
print("RESULTADOS")
print("=" * 60)

print(
    f"\nAcurácia geral: "
    f"{acuracia_geral * 100:.2f}%"
)


# ============================================================
# 12. ACURÁCIA POR CLASSE
# ============================================================

# Classe benigna

mascara_benigno = y_test.values == 2

acuracia_benigno = np.mean(
    y_pred[mascara_benigno] ==
    y_test.values[mascara_benigno]
)


# Classe maligna

mascara_maligno = y_test.values == 4

acuracia_maligno = np.mean(
    y_pred[mascara_maligno] ==
    y_test.values[mascara_maligno]
)


print(
    f"Acurácia - Benigno: "
    f"{acuracia_benigno * 100:.2f}%"
)

print(
    f"Acurácia - Maligno: "
    f"{acuracia_maligno * 100:.2f}%"
)