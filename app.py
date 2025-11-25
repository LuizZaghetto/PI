import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler


MODEL_DIR = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "model.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
METRICS_FILE = os.path.join(MODEL_DIR, "metrics.json")


def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


def load_scaler():
    if not os.path.exists(SCALER_FILE):
        return None
    with open(SCALER_FILE, "rb") as f:
        return pickle.load(f)


def calculate_rmse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return float(np.sqrt(mse))


def reset_model_files():
    if os.path.exists(MODEL_DIR):
        for f in os.listdir(MODEL_DIR):
            os.remove(os.path.join(MODEL_DIR, f))


class Node:
    def __init__(self, value, symbol=None):
        self.value = value
        self.symbol = symbol
        self.left = None
        self.right = None


def criar_codigos(root):
    codigos = {}

    def passear(node, cod_atual):
        if node is None:
            return
        if node.symbol is not None:
            codigos[node.symbol] = cod_atual
            return
        passear(node.left, cod_atual + "0")
        passear(node.right, cod_atual + "1")

    passear(root, "")
    return codigos


def construir_arvore_huffman(df_norm):
    lista_caracteres = []
    for col in df_norm.columns:
        for valor in df_norm[col]:
            for ch in str(valor):
                lista_caracteres.append(ch)

    df_chars = pd.DataFrame(lista_caracteres, columns=["Caractere"])
    freq_series = df_chars["Caractere"].value_counts(ascending=True)
    freq = freq_series.reset_index()
    freq.columns = ["Caractere", "Frequencia"]

    dicionario = dict(zip(freq["Caractere"], freq["Frequencia"]))
    lista_nodes = []
    for caractere, f in dicionario.items():
        lista_nodes.append((f, Node(f, caractere)))
    lista_nodes.sort(key=lambda x: x[0])

    while len(lista_nodes) > 1:
        (f1, n1) = lista_nodes.pop(0)
        (f2, n2) = lista_nodes.pop(0)

        soma = f1 + f2
        pai = Node(soma)
        pai.left = n1
        pai.right = n2

        lista_nodes.append((soma, pai))
        lista_nodes.sort(key=lambda x: x[0])

    raiz = lista_nodes[0][1]
    codigos = criar_codigos(raiz)
    return raiz, codigos


def trans_string(s, codes):
    s = str(s)
    return "".join(codes[ch] for ch in s)


def codificar_com_huffman(df):
    df_num = df.select_dtypes(include=[np.number]).copy()
    if df_num.empty:
        return None

    scaler = MinMaxScaler(feature_range=(1, 10))
    normalized_data = scaler.fit_transform(df_num)
    df_norm = pd.DataFrame(normalized_data, columns=df_num.columns)

    raiz, codigos = construir_arvore_huffman(df_norm)

    df_cod = pd.DataFrame()
    for col in df_norm.columns:
        df_cod[f"cod_huf_{col}"] = df_norm[col].apply(lambda x: trans_string(x, codigos))

    return df_cod


def decodificar(encoded, root):
    resultado = ""
    node = root
    for b in encoded:
        if b == "0":
            node = node.left
        else:
            node = node.right

        if node.symbol is not None:
            resultado += str(node.symbol)
            node = root
    return resultado


st.title("Projeto Integrador – Treino e Teste de Modelo")
st.markdown("---")

# Escolha genérica da coluna alvo
target_col = st.text_input("Nome da coluna alvo", value="time")


st.header("📘 Treinar modelo")

train_file = st.file_uploader(
    f"Envie o CSV de treino (precisa conter a coluna '{target_col}')",
    type=["csv"],
    key="train"
)

if train_file:
    df = pd.read_csv(train_file)

    if target_col not in df.columns:
        st.error(f"O arquivo precisa conter a coluna '{target_col}'.")
    else:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LinearRegression()
        model.fit(X_scaled, y)

        os.makedirs(MODEL_DIR, exist_ok=True)

        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model, f)
        with open(SCALER_FILE, "wb") as f:
            pickle.dump(scaler, f)

        y_pred = model.predict(X_scaled)
        mse = float(np.mean((y - y_pred) ** 2))
        rmse = float(np.sqrt(mse))

        metrics = {"mse": mse, "rmse": rmse}
        with open(METRICS_FILE, "w") as f:
            json.dump(metrics, f, indent=4)

        st.success("Modelo treinado com sucesso!")
        st.write("MSE:", mse)
        st.write("RMSE:", rmse)

        df_cod_treino = codificar_com_huffman(df)
        if df_cod_treino is not None:
            csv_cod_treino = df_cod_treino.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Baixar CSV de treino codificado (Huffman)",
                csv_cod_treino,
                "treino_codificado_huffman.csv"
            )


st.markdown("---")


st.header("🧪 Testar modelo")

test_file = st.file_uploader(
    "Envie o CSV de teste",
    type=["csv"],
    key="test"
)

has_label = st.checkbox(f"Este CSV contém coluna '{target_col}'? (opcional)")

model = load_model()
scaler = load_scaler()

if test_file:
    if model is None or scaler is None:
        st.error("Nenhum modelo encontrado. Treine antes de testar.")
        st.stop()

    df = pd.read_csv(test_file)
    df.columns = [c.strip() for c in df.columns]

    expected_cols = scaler.feature_names_in_

    if has_label:
        if target_col not in df.columns:
            st.error(f"Você marcou que o CSV tem rótulo, mas não existe coluna '{target_col}'.")
            st.stop()
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        y = None
        X = df.copy()

    missing = [c for c in expected_cols if c not in X.columns]
    if missing:
        st.error(f"Faltam colunas no CSV: {missing}")
        st.stop()

    X = X[expected_cols]

    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    df["predicted"] = predictions

    st.write("Resultado:")
    st.dataframe(df)

    if y is not None:
        rmse = calculate_rmse(y, predictions)
        st.success(f"RMSE no teste: {rmse:.4f}")

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Baixar CSV com predições", csv_bytes, "resultado.csv")

    df_cod_teste = codificar_com_huffman(df)
    if df_cod_teste is not None:
        csv_cod_teste = df_cod_teste.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Baixar CSV de teste codificado (Huffman)",
            csv_cod_teste,
            "teste_codificado_huffman.csv"
        )


st.markdown("---")


st.header("🔄 Resetar Modelo")

if st.button("Resetar modelo"):
    reset_model_files()
    st.success("Todos os arquivos do modelo foram removidos.")
