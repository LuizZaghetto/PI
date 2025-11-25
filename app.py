import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


MODEL_DIR = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "model.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
METRICS_FILE = os.path.join(MODEL_DIR, "metrics.json")


# ---------------------------
# Utils
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
    df = pd.read_csv(train_file, sep=';')

    if "time" not in df.columns:
        st.error("O arquivo precisa conter a coluna 'time'.")
    else:
        X = df.drop(columns=["time"])
        y = df["time"]

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

    # Identificar colunas esperadas no treino
    expected_cols = scaler.feature_names_in_

    # Se CSV tem rótulo
    if has_label:
        if "time" not in df.columns:
            st.error("Você marcou que o CSV tem rótulo, mas não existe coluna 'time'.")
            st.stop()

        y = df["time"]
        X = df.drop(columns=["time"])
    else:
        y = None
        X = df.copy()

    # Verificar se colunas esperadas existem
    missing = [c for c in expected_cols if c not in X.columns]
    if missing:
        st.error(f"Faltam colunas no CSV: {missing}")
        st.stop()

    # Manter somente colunas esperadas e na ordem certa
    X = X[expected_cols]

    # Escalar e prever
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


st.markdown("---")


# ---------------------------
# SEÇÃO: RESET
# ---------------------------

st.header("🔄 Resetar Modelo")

if st.button("Resetar modelo"):
    reset_model_files()
    st.success("Todos os arquivos do modelo foram removidos.")
