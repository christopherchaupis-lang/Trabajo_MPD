import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Prototipo - Ventas de Pizza", layout="wide")

# -----------------------------------------------------------------------
# 1. CARGA DE DATOS Y FILTRO AL TOP 5 (EN CANTIDAD)
# -----------------------------------------------------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_csv("pizza_sales_ordenado.csv", sep=";")
    df["order_date"] = pd.to_datetime(df["order_date"], dayfirst=True)
    df["order_time"] = pd.to_datetime(df["order_time"], format="%H:%M:%S").dt.time
    return df

@st.cache_data
def calcular_top5(df_completo):
    """Ranking de productos por unidades (cantidad) vendidas — criterio único del análisis."""
    return df_completo.groupby("pizza_name")["quantity"].sum().sort_values(ascending=False).head(5)

@st.cache_data
def agregar_semanal(df):
    """Agrega las ventas a nivel semanal para el análisis de series de tiempo."""
    serie = (
        df.set_index("order_date")
        .resample("W")
        .agg(ventas=("total_price", "sum"), pedidos=("order_id", "nunique"), unidades=("quantity", "sum"))
        .reset_index()
        .rename(columns={"order_date": "fecha"})
    )
    precio = (
        df.set_index("order_date")
        .resample("W")["unit_price"]
        .mean()
        .reset_index()
        .rename(columns={"order_date": "fecha", "unit_price": "precio"})
    )
    serie = serie.merge(precio, on="fecha")
    return serie

def crear_features(df):
    out = df.copy()
    out["semana"] = out["fecha"].dt.isocalendar().week.astype(int)
    out["mes"] = out["fecha"].dt.month
    out["lag_1"] = out["unidades"].shift(1)
    out["lag_2"] = out["unidades"].shift(2)
    out["lag_4"] = out["unidades"].shift(4)
    out["media_4"] = out["unidades"].shift(1).rolling(4).mean()
    return out.dropna().reset_index(drop=True)

def calcular_moda(serie):
    moda = serie.mode()
    if moda.empty:
        return np.nan
    return moda.iloc[0]

def descriptivos_numericos(df, columnas):
    """Media, moda, desviación estándar, mínimo y máximo por variable numérica."""
    filas = []
    for col in columnas:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        filas.append(
            {
                "Variable": col,
                "Media": s.mean(),
                "Moda": calcular_moda(s),
                "Desv. Estándar": s.std(),
                "Mínimo": s.min(),
                "Máximo": s.max(),
            }
        )
    return pd.DataFrame(filas).set_index("Variable")

def descriptivos_categoricos(df, columnas):
    """Para variables categóricas se reporta la moda (valor más frecuente), ya que la media
    no está definida en variables no numéricas."""
    filas = []
    for col in columnas:
        s = df[col].dropna()
        if s.empty:
            continue
        moda = calcular_moda(s)
        freq_relativa = (s == moda).mean() * 100
        filas.append(
            {
                "Variable": col,
                "Moda (valor más frecuente)": moda,
                "Frecuencia de la moda (%)": freq_relativa,
                "N° categorías distintas": s.nunique(),
            }
        )
    return pd.DataFrame(filas).set_index("Variable")

def entrenar_modelo(df_feat):
    variables = ["precio", "semana", "mes", "lag_1", "lag_2", "lag_4", "media_4"]
    variables = [v for v in variables if v in df_feat.columns]
    # evitamos usar 'ventas' y 'pedidos' de la MISMA semana como si fueran conocidos a priori
    # (se determinan junto con 'unidades', no antes)
    corte = int(len(df_feat) * 0.80)
    train = df_feat.iloc[:corte]
    test = df_feat.iloc[corte:]
    model = RandomForestRegressor(n_estimators=250, max_depth=8, random_state=42)
    model.fit(train[variables], train["unidades"])
    pred = model.predict(test[variables])
    mae = mean_absolute_error(test["unidades"], pred)
    rmse = mean_squared_error(test["unidades"], pred) ** 0.5
    mape = np.mean(np.abs((test["unidades"].to_numpy() - pred) / test["unidades"].to_numpy())) * 100
    return model, variables, train, test, pred, mae, rmse, mape

# Carga completa, se identifica el Top 5 en cantidad, y TODO lo demás
# (EDA y pronóstico) se calcula solo sobre esos 5 productos.
df_completo = cargar_datos()
top5 = calcular_top5(df_completo)
productos_top5 = top5.index.tolist()
df = df_completo[df_completo["pizza_name"].isin(productos_top5)].copy()

serie = agregar_semanal(df)
df_feat = crear_features(serie)
model, variables, train, test, pred, mae, rmse, mape = entrenar_modelo(df_feat)

# -----------------------------------------------------------------------
# INTERFAZ
# -----------------------------------------------------------------------
st.title("Prototipo · Análisis y pronóstico de ventas de pizza")
st.caption("Aplicación navegable: del problema de negocio al pronóstico de demanda, con datos reales de venta 2015")
st.warning(
    f"⚠️ Todo el análisis (exploratorio y pronóstico) se calcula **únicamente** sobre los 5 productos "
    f"más demandados en cantidad: {', '.join(productos_top5)}. Los demás productos fueron excluidos."
)

etapa = st.sidebar.radio(
    "Recorrer etapas",
    [
        "1. Problema",
        "2. Datos",
        "3. Top 5 productos",
        "4. Preparación (series de tiempo)",
        "5. Modelamiento",
        "6. Evaluación",
        "7. Conclusiones",
    ],
)

# ---------------------------------------------------------------
if etapa == "1. Problema":
    st.header("1. Comprensión del problema")
    st.write(
        """
    Una pizzería necesita entender cuáles son sus productos más demandados y anticipar
    su demanda semanal para planificar compras de insumos y personal. Este prototipo recorre dos
    preguntas de negocio complementarias, **limitadas a los 5 productos más vendidos en cantidad**:

    1. **¿Qué productos se venden más?** (análisis descriptivo — Top 5 en cantidad)
    2. **¿Cuánto se venderá en las próximas semanas, solo de esos 5 productos?** (pronóstico de demanda con machine learning)
    """
    )
    st.info("Fuente de datos: registro real de pedidos de pizza durante el año 2015 (48,620 líneas de pedido originales).")
    st.markdown("**Pregunta para el alumno:** ¿qué decisiones de negocio cambiarían según el producto top y según el pronóstico semanal?")

# ---------------------------------------------------------------
elif etapa == "2. Datos":
    st.header("2. Comprensión de los datos (solo Top 5 en cantidad)")
    st.dataframe(df.head(15), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Líneas de pedido (Top 5)", f"{len(df):,}")
    c2.metric("Pedidos únicos (Top 5)", f"{df['order_id'].nunique():,}")
    c3.metric("Unidades vendidas (Top 5)", f"{df['quantity'].sum():,}")
    c4.metric("Ingresos (Top 5)", f"${df['total_price'].sum():,.0f}")

    st.markdown("**Distribución por categoría (dentro del Top 5)**")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    df.groupby("pizza_category")["quantity"].sum().sort_values(ascending=False).plot(kind="bar", ax=ax)
    ax.set_ylabel("Unidades vendidas")
    ax.set_xlabel("")
    st.pyplot(fig)

    st.markdown("**Distribución por tamaño (dentro del Top 5)**")
    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    df.groupby("pizza_size")["quantity"].sum().reindex(["S", "M", "L", "XL", "XXL"]).plot(kind="bar", ax=ax2, color="orange")
    ax2.set_ylabel("Unidades vendidas")
    ax2.set_xlabel("")
    st.pyplot(fig2)

# ---------------------------------------------------------------
elif etapa == "3. Top 5 productos":
    st.header("3. Top 5 productos (por cantidad)")
    st.write("Ranking de los 5 productos con más unidades vendidas. Este ranking define qué productos entran en todo el análisis.")

    resumen = (
        df.groupby("pizza_name")
        .agg(unidades=("quantity", "sum"), ingresos=("total_price", "sum"), pedidos=("order_id", "nunique"))
        .reindex(productos_top5)
        .reset_index()
    )
    resumen.index = resumen.index + 1

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Tabla Top 5**")
        st.dataframe(
            resumen.rename(
                columns={"pizza_name": "Producto", "unidades": "Unidades", "ingresos": "Ingresos ($)", "pedidos": "N° Pedidos"}
            ).style.format({"Ingresos ($)": "${:,.0f}", "Unidades": "{:,}", "N° Pedidos": "{:,}"}),
            use_container_width=True,
        )

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(resumen["pizza_name"][::-1], resumen["unidades"][::-1], color="tomato")
        ax.set_xlabel("Unidades vendidas")
        ax.set_title("Top 5 productos por cantidad")
        st.pyplot(fig)

    st.markdown("**Detalle del producto #1**")
    lider = resumen.iloc[0]
    st.success(
        f"**{lider['pizza_name']}** — {lider['unidades']:,.0f} unidades vendidas, "
        f"${lider['ingresos']:,.0f} en ingresos, en {lider['pedidos']:,.0f} pedidos distintos."
    )

# ---------------------------------------------------------------
elif etapa == "4. Preparación (series de tiempo)":
    st.header("4. Preparación e ingeniería de variables — Serie de tiempo")
    st.write(
        """
    Las líneas de pedido de los **5 productos más demandados en cantidad** se agregan a nivel **semanal**
    (suma de ingresos, unidades y pedidos) para construir la serie de tiempo. La variable a pronosticar es la
    **cantidad de unidades vendidas** de esos 5 productos en conjunto.
    Luego se generan variables de calendario y rezagos de la demanda,
    transformando la serie en una tabla supervisada compatible con algoritmos de machine learning.
    """
    )
    st.dataframe(serie.head(10), use_container_width=True)
    st.markdown("**Variables construidas para el modelo:**")
    st.dataframe(df_feat[["fecha", "unidades"] + variables].head(12), use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Análisis descriptivo de las variables de entrada")

    st.markdown("**Variables numéricas** (media, moda, desviación estándar, mínimo y máximo):")
    columnas_numericas = ["unidades"] + variables
    tabla_num = descriptivos_numericos(df_feat, columnas_numericas)
    st.dataframe(tabla_num.style.format("{:.2f}"), use_container_width=True)

    st.markdown(
        "**Variables categóricas** (la media no aplica; se reporta la moda como medida de tendencia central):"
    )
    columnas_categoricas = [c for c in ["pizza_name", "pizza_category", "pizza_size"] if c in df.columns]
    tabla_cat = descriptivos_categoricos(df, columnas_categoricas)
    st.dataframe(
        tabla_cat.style.format({"Frecuencia de la moda (%)": "{:.1f}%"}),
        use_container_width=True,
    )

    importancias_prev = pd.Series(model.feature_importances_, index=variables).sort_values(ascending=False)
    variables_clave = ["unidades"] + [v for v in importancias_prev.head(3).index.tolist() if v != "unidades"]

    st.markdown(f"**Histogramas y diagramas de caja de las variables más importantes** ({', '.join(variables_clave)}):")
    for var in variables_clave:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 3))
            df_feat[var].dropna().hist(ax=ax, bins=15, color="steelblue", edgecolor="white")
            ax.set_title(f"Histograma — {var}")
            ax.set_xlabel(var)
            ax.set_ylabel("Frecuencia")
            st.pyplot(fig)
        with c2:
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            ax2.boxplot(df_feat[var].dropna(), vert=False)
            ax2.set_title(f"Diagrama de caja — {var}")
            ax2.set_xlabel(var)
            ax2.set_yticks([])
            st.pyplot(fig2)

    st.markdown("---")
    st.code(
        """
serie = df.set_index("order_date").resample("W").agg(
    ventas=("total_price", "sum"), pedidos=("order_id", "nunique"), unidades=("quantity", "sum")
)
serie["lag_1"] = serie["unidades"].shift(1)
serie["lag_4"] = serie["unidades"].shift(4)
serie["media_4"] = serie["unidades"].shift(1).rolling(4).mean()
""",
        language="python",
    )

# ---------------------------------------------------------------
elif etapa == "5. Modelamiento":
    st.header("5. Entrenamiento del modelo")
    st.write(
        """
    Se utiliza Random Forest porque permite modelar relaciones no lineales entre precio, calendario
    y comportamiento pasado de la demanda de los 5 productos Top. La separación entrenamiento/prueba respeta el orden temporal:
    80% para entrenamiento y 20% para prueba (sin mezclar el futuro con el pasado).
    """
    )
    c1, c2 = st.columns(2)
    c1.metric("Semanas de entrenamiento", len(train))
    c2.metric("Semanas de prueba", len(test))

    imp = pd.DataFrame({"variable": variables, "importancia": model.feature_importances_}).sort_values(
        "importancia", ascending=False
    )
    st.markdown("**Importancia de variables**")
    st.dataframe(imp, use_container_width=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(imp["variable"][::-1], imp["importancia"][::-1])
    st.pyplot(fig)

# ---------------------------------------------------------------
elif etapa == "6. Evaluación":
    st.header("6. Evaluación")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:,.0f} unidades")
    c2.metric("RMSE", f"{rmse:,.0f} unidades")
    c3.metric("MAPE", f"{mape:.2f}%")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(test["fecha"], test["unidades"], label="Real", marker="o")
    ax.plot(test["fecha"], pred, label="Predicción", marker="o")
    ax.legend()
    ax.set_title("Unidades vendidas semanales del Top 5: reales vs. pronosticadas")
    ax.set_ylabel("Unidades")
    st.pyplot(fig)

    resultados = test[["fecha", "unidades"]].copy()
    resultados["prediccion"] = pred.round(1)
    resultados["error_abs"] = np.abs(resultados["unidades"] - resultados["prediccion"]).round(1)
    st.dataframe(resultados, use_container_width=True)

# ---------------------------------------------------------------
else:
    st.header("7. Conclusiones y discusión")
    st.success(f"El modelo obtuvo un MAPE aproximado de {mape:.2f}% en el conjunto de prueba, sobre el Top 5 de productos en cantidad.")
    st.write(
        """
    El prototipo combina dos vistas complementarias, ambas restringidas a los 5 productos más demandados
    en cantidad: un análisis descriptivo (Top 5, útil para decisiones de menú y abastecimiento) y un
    análisis predictivo (pronóstico semanal de la cantidad de unidades vendidas de esos 5 productos,
    útil para planificación de insumos y personal). El alumno recorre la trazabilidad completa entre problema,
    datos, transformación, entrenamiento, métricas y decisión.
    """
    )
    st.markdown(
        """
    **Preguntas de cierre**
    - ¿Cómo debería el negocio priorizar insumos según el Top 5 de productos en cantidad?
    - ¿Por qué no sería correcto dividir aleatoriamente una serie temporal?
    - ¿Qué eventos (feriados, campañas) podrían explicar semanas atípicas no capturadas por el modelo?
    - ¿Qué se pierde al excluir del análisis a los productos fuera del Top 5?
    """
    )

st.sidebar.markdown("---")
st.sidebar.caption("Caso académico adaptado para fines docentes — Top 5 productos por cantidad, datos de ventas de pizza 2015.")
