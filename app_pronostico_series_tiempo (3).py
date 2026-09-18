import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
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
def construir_serie_diaria(df):
    """Serie de tiempo diaria de unidades vendidas, sumando los 5 productos Top."""
    serie = df.set_index("order_date")["quantity"].resample("D").sum().sort_index()
    freq = pd.infer_freq(serie.index)
    serie = serie.asfreq(freq) if freq else serie.asfreq("D")
    serie = serie.ffill().bfill()
    return serie

def calcular_moda(serie):
    moda = serie.mode()
    return moda.iloc[0] if not moda.empty else np.nan

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

def calcular_metricas(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE (%)": mape}

# Carga completa, se identifica el Top 5 en cantidad, y TODO lo demás
# (EDA y pronóstico) se calcula solo sobre esos 5 productos.
df_completo = cargar_datos()
top5 = calcular_top5(df_completo)
productos_top5 = top5.index.tolist()
df = df_completo[df_completo["pizza_name"].isin(productos_top5)].copy()
y = construir_serie_diaria(df)

# -----------------------------------------------------------------------
# INTERFAZ
# -----------------------------------------------------------------------
st.title("Prototipo · Análisis y pronóstico de ventas de pizza")
st.caption("Aplicación navegable: del problema de negocio al pronóstico de demanda con series de tiempo, datos reales de venta 2015")
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
        "4. Preparación y análisis descriptivo",
        "5. Modelamiento (series de tiempo)",
        "6. Evaluación y pronóstico",
        "7. Conclusiones",
    ],
)

# ---------------------------------------------------------------
if etapa == "1. Problema":
    st.header("1. Comprensión del problema")
    st.write(
        """
    Una pizzería necesita entender cuáles son sus productos más demandados y anticipar
    su demanda diaria para planificar compras de insumos y personal. Este prototipo recorre dos
    preguntas de negocio complementarias, **limitadas a los 5 productos más vendidos en cantidad**:

    1. **¿Qué productos se venden más?** (análisis descriptivo — Top 5 en cantidad)
    2. **¿Cuánto se venderá en los próximos días, solo de esos 5 productos?** (pronóstico con modelos de series de tiempo)
    """
    )
    st.info("Fuente de datos: registro real de pedidos de pizza durante el año 2015 (48,620 líneas de pedido originales).")
    st.markdown("**Pregunta para el alumno:** ¿qué decisiones de negocio cambiarían según el producto top y según el pronóstico?")

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
elif etapa == "4. Preparación y análisis descriptivo":
    st.header("4. Preparación de la serie de tiempo y análisis descriptivo")
    st.write(
        """
    Las líneas de pedido de los **5 productos más demandados en cantidad** se agregan por **día**
    (suma de unidades vendidas) para construir la serie de tiempo que se va a pronosticar.
    """
    )
    st.markdown("**Serie diaria de unidades vendidas (primeros registros):**")
    st.dataframe(y.rename("unidades").reset_index().rename(columns={"order_date": "fecha"}).head(10), use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Análisis descriptivo de las variables")

    st.markdown("**Variables numéricas** (media, moda, desviación estándar, mínimo y máximo):")
    columnas_numericas = ["quantity", "unit_price", "total_price"]
    tabla_num = descriptivos_numericos(df, columnas_numericas)
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

    st.markdown("**Histogramas y diagramas de caja de las variables numéricas:**")
    for var in columnas_numericas:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 3))
            df[var].dropna().hist(ax=ax, bins=20, color="steelblue", edgecolor="white")
            ax.set_title(f"Histograma — {var}")
            ax.set_xlabel(var)
            ax.set_ylabel("Frecuencia")
            st.pyplot(fig)
        with c2:
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            ax2.boxplot(df[var].dropna(), vert=False)
            ax2.set_title(f"Diagrama de caja — {var}")
            ax2.set_xlabel(var)
            ax2.set_yticks([])
            st.pyplot(fig2)

# ---------------------------------------------------------------
elif etapa == "5. Modelamiento (series de tiempo)":
    st.header("5. Modelamiento con técnicas de series de tiempo")
    st.write(
        """
    Se pronostica la **cantidad diaria de unidades vendidas** de los 5 productos Top combinando tres
    enfoques clásicos de series de tiempo: **Suavización Exponencial Triple (Holt-Winters)**, **ARIMA**
    y **SARIMA**, precedidos de una **descomposición** de la serie en tendencia, estacionalidad y residuo.
    """
    )

    st.subheader("Descomposición de la serie")
    seasonal_period = st.number_input(
        "Período estacional (7 = semanal en datos diarios):", min_value=2, value=7, step=1
    )
    model_type = st.radio("Tipo de descomposición:", ["additive", "multiplicative"], horizontal=True)

    try:
        decomp = seasonal_decompose(y, model=model_type, period=seasonal_period)
        fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        axs[0].plot(y.index, decomp.observed); axs[0].set_ylabel("Observado")
        axs[1].plot(y.index, decomp.trend); axs[1].set_ylabel("Tendencia")
        axs[2].plot(y.index, decomp.seasonal); axs[2].set_ylabel("Estacionalidad")
        axs[3].plot(y.index, decomp.resid); axs[3].set_ylabel("Residuo")
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"No se pudo realizar la descomposición con los parámetros actuales: {e}")

    st.subheader("Parámetros de ARIMA / SARIMA")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("**ARIMA / SARIMA (no estacional)**")
        p = st.number_input("AR (p):", 0, 5, 1)
        d = st.number_input("Diferenciación (d):", 0, 2, 1)
        q = st.number_input("MA (q):", 0, 5, 1)
    with col_p2:
        st.markdown("**Componente estacional de SARIMA**")
        P = st.number_input("SAR (P):", 0, 5, 1)
        D = st.number_input("Diferenciación estacional (D):", 0, 2, 1)
        Q = st.number_input("SMA (Q):", 0, 5, 1)
        s = seasonal_period

    st.session_state["ts_params"] = dict(seasonal_period=seasonal_period, model_type=model_type, p=p, d=d, q=q, P=P, D=D, Q=Q, s=s)
    st.info("Los parámetros configurados aquí se usan en la etapa **6. Evaluación y pronóstico**.")

# ---------------------------------------------------------------
elif etapa == "6. Evaluación y pronóstico":
    st.header("6. Evaluación y pronóstico a 5 períodos futuros")

    params = st.session_state.get(
        "ts_params",
        dict(seasonal_period=7, model_type="additive", p=1, d=1, q=1, P=1, D=1, Q=1, s=7),
    )
    seasonal_period = params["seasonal_period"]
    p, d, q = params["p"], params["d"], params["q"]
    P, D, Q, s = params["P"], params["D"], params["Q"], params["s"]
    forecast_horizon = 5

    predictions_in_sample = {}
    forecasts_future = {}

    try:
        hw_model = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=seasonal_period).fit()
        predictions_in_sample["Holt-Winters (HW)"] = hw_model.fittedvalues
        forecasts_future["Holt-Winters (HW)"] = hw_model.forecast(forecast_horizon)
    except Exception as e:
        st.error(f"Error en Holt-Winters: {e}")

    try:
        arima_model = ARIMA(y, order=(p, d, q)).fit()
        predictions_in_sample["ARIMA"] = arima_model.fittedvalues
        forecasts_future["ARIMA"] = arima_model.forecast(forecast_horizon)
    except Exception as e:
        st.error(f"Error en ARIMA: {e}")

    try:
        sarima_model = SARIMAX(y, order=(p, d, q), seasonal_order=(P, D, Q, s)).fit(disp=False)
        predictions_in_sample["SARIMA"] = sarima_model.fittedvalues
        forecasts_future["SARIMA"] = sarima_model.forecast(forecast_horizon)
    except Exception as e:
        st.error(f"Error en SARIMA: {e}")

    metrics_list = []
    for name, pred in predictions_in_sample.items():
        valid_idx = y.index.intersection(pred.index)
        metrics = calcular_metricas(y.loc[valid_idx], pred.loc[valid_idx])
        metrics["Modelo"] = name
        metrics_list.append(metrics)

    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list).set_index("Modelo")
        st.markdown("**Métricas de error en entrenamiento (in-sample):**")
        st.dataframe(metrics_df.style.highlight_min(axis=0, color="lightgreen"), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y.index[-30:], y.iloc[-30:], label="Datos históricos", color="black")
    colores = {"Holt-Winters (HW)": "blue", "ARIMA": "orange", "SARIMA": "green"}
    for name, fcast in forecasts_future.items():
        ax.plot(fcast.index, fcast, "--", label=f"Pronóstico {name}", color=colores.get(name, "red"))
    ax.set_title("Comparativa de pronóstico — unidades diarias (Top 5 productos)")
    ax.set_ylabel("Unidades")
    ax.legend()
    st.pyplot(fig)

    st.markdown("**Valores proyectados (próximos 5 días):**")
    df_future = pd.DataFrame(forecasts_future)
    st.dataframe(df_future, use_container_width=True)

# ---------------------------------------------------------------
else:
    st.header("7. Conclusiones y discusión")
    st.write(
        """
    El prototipo combina dos vistas complementarias, ambas restringidas a los 5 productos más demandados
    en cantidad: un análisis descriptivo (Top 5, distribución por categoría/tamaño, estadísticos de las
    variables clave) y un análisis predictivo con **técnicas de series de tiempo** (descomposición,
    Holt-Winters, ARIMA y SARIMA) para anticipar la demanda diaria de esos 5 productos.
    """
    )
    st.markdown(
        """
    **Preguntas de cierre**
    - ¿Cómo debería el negocio priorizar insumos según el Top 5 de productos en cantidad?
    - ¿Qué modelo (Holt-Winters, ARIMA o SARIMA) tuvo mejor desempeño según las métricas de error, y por qué podría ser así dado el patrón de la serie?
    - ¿Qué eventos (feriados, campañas) podrían explicar días atípicos no capturados por los modelos?
    - ¿Qué se pierde al excluir del análisis a los productos fuera del Top 5?
    """
    )

st.sidebar.markdown("---")
st.sidebar.caption("Caso académico adaptado para fines docentes — Top 5 productos por cantidad, datos de ventas de pizza 2015.")
