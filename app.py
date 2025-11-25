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


# ---------------------------
# Utils (modelo)
# ---------------------------

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


# ---------------------------
# Huffman + Min-Max (compactação/codificação)
# ---------------------------

class Node:
    def __init__(self, value, symbol=None):
        self.value = value      # frequência
        self.symbol = symbol    # caractere (ex: '1', '2', '.', '-')
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
    # junta todos os caracteres de todos os valores normalizados
    lista_caracteres = []
    for col in df_norm.columns:
        for valor in df_norm[col]:
            for ch in str(valor):
                lista_caracteres.append(ch)

    # conta frequência de cada caractere
    df_chars = pd.DataFrame(lista_caracteres, columns=["Caractere"])
    freq_series = df_chars["Caractere"].value_counts(ascending=True)
    freq = freq_series.reset_index()
    freq.columns = ["Caractere", "Frequencia"]

    # cria nodes iniciais
    dicionario = dict(zip(freq["Caractere"], freq["Frequencia"]))
    lista_nodes = []
    for caractere, f in dicionario.items():
        lista_nodes.append((f, Node(f, caractere)))
    lista_nodes.sort(key=lambda x: x[0])

    # monta a árvore
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
    """
    Recebe um DataFrame (como o de treino ou o de teste),
    seleciona apenas colunas numéricas, aplica MinMaxScaler(1,10)
    e depois codifica com Huffman.

    Retorna um novo DataFrame contendo somente colunas 'cod_huf_<nome_col_original>'.
    """
    # seleciona apenas colunas numéricas
    df_num = df.select_dtypes(include=[np.number]).copy()
    if df_num.empty:
        return None  # nada pra codificar

    # normalização min-max
    scaler = MinMaxScaler(feature_range=(1, 10))
    normalized_data = scaler.fit_transform(df_num)
    df_norm = pd.DataFrame(normalized_data, columns=df_num.columns)

    # constrói árvore de Huffman e codigos
    raiz, codigos = construir_arvore_huffman(df_norm)

    # codificar cada coluna numérica em bits
    df_cod = pd.DataFrame()
    for col in df_norm.columns:
        df_cod[f"cod_huf_{col}"] = df_norm[col].apply(lambda x: trans_string(x, codigos))

    return df_cod


# (Opcional) Função de decodificação, caso queiram demonstrar depois
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


# ---------------------------
# UI
# ---------------------------

st.title("Projeto Integrador – Treino e Teste de Modelo")
st.markdown("---")


# ---------------------------
# SEÇÃO: TREINO
# ---------------------------

st.header("📘 Treinar modelo")

train_file = st.file_uploader(
    "Envie o CSV de treino (precisa conter coluna 'time')",
    type=["csv"],
    key="train"
)

if train_file:
    # OBS: sep=';' porque seus arquivos usam ponto-e-vírgula
    df = pd.read_csv(train_file)

    if "time" not in df.columns:
        st.error("O arquivo precisa conter a coluna 'time'.")
    else:
        X = df.drop(columns=["time"])
        y = df["time"]

        # Normalização para o modelo (StandardScaler)
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

        # 1) GERAR E DISPONIBILIZAR A BASE DE TREINO CODIFICADA (MinMax + Huffman)
        df_cod_treino = codificar_com_huffman(df)
        if df_cod_treino is not None:
            csv_cod_treino = df_cod_treino.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Baixar CSV de treino codificado (Huffman)",
                csv_cod_treino,
                "treino_codificado_huffman.csv"
            )


st.markdown("---")


# ---------------------------
# SEÇÃO: TESTE
# ---------------------------

st.header("🧪 Testar modelo")

test_file = st.file_uploader(
    "Envie o CSV de teste",
    type=["csv"],
    key="test"
)

has_label = st.checkbox("Este CSV contém coluna 'time'? (opcional)")

model = load_model()
scaler = load_scaler()

if test_file:
    if model is None or scaler is None:
        st.error("Nenhum modelo encontrado. Treine antes de testar.")
        st.stop()

    df = pd.read_csv(test_file, sep=';')
    df.columns = [c.strip() for c in df.columns]

    # colunas esperadas (mesmas do treino)
    expected_cols = scaler.feature_names_in_

    if has_label:
        if "time" not in df.columns:
            st.error("Você marcou que o CSV tem rótulo, mas não existe coluna 'time'.")
            st.stop()
        y = df["time"]
        X = df.drop(columns=["time"])
    else:
        y = None
        X = df.copy()

    missing = [c for c in expected_cols if c not in X.columns]
    if missing:
        st.error(f"Faltam colunas no CSV: {missing}")
        st.stop()

    X = X[expected_cols]

    # previsões
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    df["predicted"] = predictions

    st.write("Resultado:")
    st.dataframe(df)

    if y is not None:
        rmse = calculate_rmse(y, predictions)
        st.success(f"RMSE no teste: {rmse:.4f}")

    # CSV "normal" com predições
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Baixar CSV com predições", csv_bytes, "resultado.csv")

    # 2) GERAR E DISPONIBILIZAR A BASE DE TESTE CODIFICADA (MinMax + Huffman)
    # Aqui eu codifico o DataFrame completo já com 'predicted'
    df_cod_teste = codificar_com_huffman(df)
    if df_cod_teste is not None:
        csv_cod_teste = df_cod_teste.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Baixar CSV de teste codificado (Huffman)",
            csv_cod_teste,
            "teste_codificado_huffman.csv"
        )


st.markdown("---")


# ---------------------------
# SEÇÃO: RESET
# ---------------------------

st.header("🔄 Resetar Modelo")

if st.button("Resetar modelo"):
    reset_model_files()
    st.success("Todos os arquivos do modelo foram removidos.")
