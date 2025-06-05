import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, recall_score

st.set_page_config(page_title="Clasificador de Atletas", layout="wide")

@st.cache_data
def cargar_modelos():
    log_model = joblib.load("modelo_logistico.pkl")
    tree_model = joblib.load("modelo_arbol.pkl")
    try:
        svm_model = joblib.load("modelo_svm.pkl")
    except FileNotFoundError:
        svm_model = None
    scaler = joblib.load("escalador.pkl")
    columnas = joblib.load("columnas_modelo.pkl")
    return log_model, tree_model, svm_model, scaler, columnas

def mostrar_imagenes_png(carpeta="figuras"):
    if os.path.isdir(carpeta):
        imagenes = [img for img in os.listdir(carpeta) if img.endswith(".png")]
        for img in imagenes:
            with st.expander(f"📷 {img}"):
                ruta = os.path.join(carpeta, img)
                st.image(ruta, caption=img, use_container_width=True)

log_model, tree_model, svm_model, scaler, columnas = cargar_modelos()

st.sidebar.title("Menú")
seccion = st.sidebar.radio("Ir a:", [
    "📁 Preprocesamiento",
    "📈 Métricas del Modelo",
    "🤖 Predicción"
])

# Preprocesamiento
if seccion == "📁 Preprocesamiento":
    st.title("📁 Análisis Preliminar y Preprocesamiento")
    df = pd.read_csv("datos_atletas.csv")

    st.subheader("👁️‍🗨️ Vista previa de los datos")
    st.dataframe(df.head())

    st.subheader("🔍 Valores faltantes (NaN)")
    nan_totales = df.isna().sum()
    st.write(nan_totales[nan_totales > 0])
    st.info("❌ Se eliminan filas con valores faltantes para evitar sesgos.")

    st.subheader("📊 Distribución de variables numéricas")
    for col in df.select_dtypes(include=np.number).columns:
        with st.expander(f"📊 Histograma: {col}"):
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            st.pyplot(fig)

    st.subheader("📦 Outliers (Boxplots)")
    with st.expander("📦 Ver Outliers (Boxplots)"):
        st.image("figuras/outliers.png", use_container_width=True)

    st.subheader("🔗 Mapa de correlación")
    with st.expander("🔗 Ver Mapa de Correlación"):
        st.image("figuras/correlacion.png", use_container_width=True)

    st.subheader("⚖️ Balance de clases")
    fig, ax = plt.subplots()
    sns.set_theme(style="whitegrid")
    sns.barplot(x=df["Tipo de Atleta"].value_counts().index,
                y=df["Tipo de Atleta"].value_counts().values,
                palette="pastel", ax=ax)
    ax.set_ylabel("Cantidad")
    ax.set_xlabel("Tipo de Atleta")
    st.pyplot(fig)

    st.subheader("📷 Imágenes PNG guardadas")
    mostrar_imagenes_png()

# Métricas
elif seccion == "📈 Métricas del Modelo":
    st.title("📈 Evaluación de Modelos Clasificadores")

    modelos = {
        "Regresión Logística": log_model,
        "Árbol de Decisión": tree_model
    }
    if svm_model:
        modelos["SVM"] = svm_model

    X_test = pd.read_csv("X_test.csv")
    y_test = pd.read_csv("y_test.csv").squeeze()

    for nombre, modelo in modelos.items():
        st.subheader(f"📊 Resultados para: {nombre}")
        y_pred = modelo.predict(X_test)

        st.write("**Accuracy:**", round(accuracy_score(y_test, y_pred), 2))
        st.write("**Recall:**", round(recall_score(y_test, y_pred, average='weighted'), 2))

        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

# Predicción
elif seccion == "🤖 Predicción":
    st.title("🤖 Predicción del Tipo de Atleta")
    st.sidebar.markdown("🎚️ Introduce los datos del atleta:")

    valores = {}
    usados = set()
    contador = 0

    for col in columnas:
        if "_" in col:
            base = col.split("_")[0]
            if base not in usados:
                usados.add(base)
                opciones = [c for c in columnas if c.startswith(base + "_")]
                opciones_nombres = [o.split("_")[1] for o in opciones]
                seleccion = st.sidebar.selectbox(f"{base}", opciones_nombres, key=f"select_{base}_{contador}")
                contador += 1
                for o in opciones:
                    valores[o] = 1.0 if o.endswith(seleccion) else 0.0
        else:
            max_value = 100.0
            if "edad" in col.lower():
                max_value = 50.0
            elif "imc" in col.lower():
                max_value = 40.0
            valores[col] = st.sidebar.slider(
                col, min_value=0.0, max_value=max_value, value=25.0, step=0.5, key=f"slider_{col}"
            )

    if st.sidebar.button("Predecir"):
        entrada = pd.DataFrame([valores])
        for col in columnas:
            if col not in entrada:
                entrada[col] = 0.0
        entrada = entrada[columnas]
        entrada_esc = scaler.transform(entrada)

        st.subheader("📈 Resultados de Predicción")

        def mostrar_resultado(modelo, nombre_modelo):
            pred = modelo.predict(entrada_esc)[0]
            st.success(f"📌 {nombre_modelo}: `{pred}`")

            if hasattr(modelo, "predict_proba") and len(modelo.classes_) > 1:
                probs = modelo.predict_proba(entrada_esc)[0]
                if np.isclose(probs.max(), 1.0) and np.isclose(probs.min(), 0.0):
                    st.info("ℹ️ Este modelo realiza predicciones categóricas duras. Las probabilidades pueden ser 0% o 100%.")

                clases = modelo.classes_
                st.markdown("### 🔢 Probabilidades del tipo de atleta")
                df_probs = pd.DataFrame({
                    'Tipo de Atleta': clases,
                    'Probabilidad': probs * 100
                }).sort_values("Probabilidad", ascending=True)

                st.dataframe(df_probs.style.format({"Probabilidad": "{:.2f}%"}))

                fig, ax = plt.subplots()
                ax.barh(df_probs["Tipo de Atleta"], df_probs["Probabilidad"], color='skyblue')
                ax.set_xlabel("Probabilidad (%)")
                ax.set_xlim(0, 100)
                ax.set_title(f"Distribución de Probabilidades - {nombre_modelo}")
                for i, v in enumerate(df_probs["Probabilidad"]):
                    ax.text(v + 1, i, f"{v:.2f}%", va='center')
                st.pyplot(fig)
            else:
                st.warning("⚠️ Este modelo no soporta predicción probabilística.")

        mostrar_resultado(log_model, "Regresión Logística")
        mostrar_resultado(tree_model, "Árbol de Decisión")
        if svm_model:
            mostrar_resultado(svm_model, "SVM")

