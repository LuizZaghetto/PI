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


st.title("Projeto Integrador")

st.markdown("---")


st.header("Treinar modelo")
train_file = st.file_uploader(
    "Envie o CSV de treino (deve conter coluna 'time')",
    type=["csv"],
    key="train"
)

if train_file:
    df = pd.read_csv(train_file)

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
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)

        metrics = {"mse": float(mse), "rmse": float(rmse)}
        with open(METRICS_FILE, "w") as f:
            json.dump(metrics, f, indent=4)

        st.success("Modelo treinado com sucesso!")
        st.write("MSE:", mse)
        st.write("RMSE:", rmse)

st.markdown("---")

st.header("Testar modelo")

test_file = st.file_uploader(
    "Envie o CSV de teste",
    type=["csv"],
    key="test"
)

has_label = st.checkbox("O CSV contém a coluna 'time'?")

if has_label and "time" not in df.columns:
    st.error("Você marcou que o arquivo tem rótulos, mas a coluna 'time' não existe.")
    st.stop()

if test_file:
    model = load_model()
    scaler = load_scaler()

    if model is None:
        st.error("Nenhum modelo encontrado. Treine antes.")
    else:
        df = pd.read_csv(test_file)
        if has_label:
            y = df["time"]
            X = df.drop(columns=["time"])
        else:
            X = df
            y = None

        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)

        df["predicted"] = pred

        st.write("Resultado:")
        st.dataframe(df)

        if y is not None:
            rmse = calculate_rmse(y, pred)
            st.success(f"RMSE: {rmse:.4f}")

        csv_download = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Baixar resultado CSV", csv_download, "predicoes.csv")


st.markdown("---")

st.header("🔄 Resetar Modelo")

if st.button("Resetar"):
    reset_model_files()
    st.success("Modelo resetado com sucesso! Todos os arquivos foram apagados.")
