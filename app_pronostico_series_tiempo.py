import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Prototipo - Ventas de Pizza", layout="wide")

# -----------------------------------------------------------------------
# 1. CARGA Y PREPARACIÓN DE DATOS
# -----------------------------------------------------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_csv("pizza_sales_ordenado.csv", sep=";")
    df["order_date"] = pd.to_datetime(df["order_date"], dayfirst=True)
    df["order_time"] = pd.to_datetime(df["order_time"], format="%H:%M:%S").dt.time
    return df

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
    # precio promedio ponderado por semana y proporción de promoción (tamaños grandes L/XL/XXL como proxy)
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
    out["lag_1"] = out["ventas"].shift(1)
    out["lag_2"] = out["ventas"].shift(2)
    out["lag_4"] = out["ventas"].shift(4)
    out["media_4"] = out["ventas"].shift(1).rolling(4).mean()
    return out.dropna().reset_index(drop=True)

def entrenar_modelo(df_feat):
    variables = ["precio", "pedidos", "unidades", "semana", "mes", "lag_1", "lag_2", "lag_4", "media_4"]
    variables = [v for v in variables if v in df_feat.columns and v not in ["ventas"]]
    # evitamos usar 'pedidos' y 'unidades' de la MISMA semana como si fueran conocidos a priori
    variables = [v for v in variables if v not in ["pedidos", "unidades"]]
    corte = int(len(df_feat) * 0.80)
    train = df_feat.iloc[:corte]
    test = df_feat.iloc[corte:]
    model = RandomForestRegressor(n_estimators=250, max_depth=8, random_state=42)
    model.fit(train[variables], train["ventas"])
    pred = model.predict(test[variables])
    mae = mean_absolute_error(test["ventas"], pred)
    rmse = mean_squared_error(test["ventas"], pred) ** 0.5
    mape = np.mean(np.abs((test["ventas"].to_numpy() - pred) / test["ventas"].to_numpy())) * 100
    return model, variables, train, test, pred, mae, rmse, mape

df = cargar_datos()
serie = agregar_semanal(df)
df_feat = crear_features(serie)
model, variables, train, test, pred, mae, rmse, mape = entrenar_modelo(df_feat)

# -----------------------------------------------------------------------
# INTERFAZ
# -----------------------------------------------------------------------
st.title("Prototipo · Análisis y pronóstico de ventas de pizza")
st.caption("Aplicación navegable: del problema de negocio al pronóstico de demanda, con datos reales de venta 2015")

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
    Una pizzería necesita entender qué productos son sus mayores generadores de ventas y anticipar
    la demanda semanal para planificar compras de insumos y personal. Este prototipo recorre dos
    preguntas de negocio complementarias:

    1. **¿Qué productos venden más?** (análisis descriptivo — Top 5)
    2. **¿Cuánto se venderá en las próximas semanas?** (pronóstico de demanda con machine learning)
    """
    )
    st.info("Fuente de datos: registro real de pedidos de pizza durante el año 2015 (48,620 líneas de pedido).")
    st.markdown("**Pregunta para el alumno:** ¿qué decisiones de negocio cambiarían según el producto top y según el pronóstico semanal?")

# ---------------------------------------------------------------
elif etapa == "2. Datos":
    st.header("2. Comprensión de los datos")
    st.dataframe(df.head(15), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Líneas de pedido", f"{len(df):,}")
    c2.metric("Pedidos únicos", f"{df['order_id'].nunique():,}")
    c3.metric("Productos distintos", df["pizza_name"].nunique())
    c4.metric("Ingresos totales", f"${df['total_price'].sum():,.0f}")

    st.markdown("**Distribución por categoría**")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    df.groupby("pizza_category")["quantity"].sum().sort_values(ascending=False).plot(kind="bar", ax=ax)
    ax.set_ylabel("Unidades vendidas")
    ax.set_xlabel("")
    st.pyplot(fig)

    st.markdown("**Distribución por tamaño**")
    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    df.groupby("pizza_size")["quantity"].sum().reindex(["S", "M", "L", "XL", "XXL"]).plot(kind="bar", ax=ax2, color="orange")
    ax2.set_ylabel("Unidades vendidas")
    ax2.set_xlabel("")
    st.pyplot(fig2)

# ---------------------------------------------------------------
elif etapa == "3. Top 5 productos":
    st.header("3. Top 5 productos")
    st.write("Ranking de los productos con mejor desempeño, medido en unidades vendidas e ingresos generados.")

    criterio = st.radio("Ordenar por:", ["Unidades vendidas", "Ingresos generados"], horizontal=True)

    resumen = (
        df.groupby("pizza_name")
        .agg(unidades=("quantity", "sum"), ingresos=("total_price", "sum"), pedidos=("order_id", "nunique"))
        .reset_index()
    )

    col_orden = "unidades" if criterio == "Unidades vendidas" else "ingresos"
    top5 = resumen.sort_values(col_orden, ascending=False).head(5).reset_index(drop=True)
    top5.index = top5.index + 1

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Tabla Top 5**")
        st.dataframe(
            top5.rename(
                columns={"pizza_name": "Producto", "unidades": "Unidades", "ingresos": "Ingresos ($)", "pedidos": "N° Pedidos"}
            ).style.format({"Ingresos ($)": "${:,.0f}", "Unidades": "{:,}", "N° Pedidos": "{:,}"}),
            use_container_width=True,
        )

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(top5["pizza_name"][::-1], top5[col_orden][::-1], color="tomato")
        ax.set_xlabel(criterio)
        ax.set_title("Top 5 productos")
        st.pyplot(fig)

    st.markdown("**Detalle del producto #1**")
    lider = top5.iloc[0]
    st.success(
        f"**{lider['pizza_name']}** — {lider['unidades']:,.0f} unidades vendidas, "
        f"${lider['ingresos']:,.0f} en ingresos, en {lider['pedidos']:,.0f} pedidos distintos."
    )

# ---------------------------------------------------------------
elif etapa == "4. Preparación (series de tiempo)":
    st.header("4. Preparación e ingeniería de variables — Serie de tiempo")
    st.write(
        """
    Las 48,620 líneas de pedido se agregan a nivel **semanal** (suma de ingresos, unidades y pedidos)
    para construir la serie de tiempo. Luego se generan variables de calendario y rezagos de la demanda,
    transformando la serie en una tabla supervisada compatible con algoritmos de machine learning.
    """
    )
    st.dataframe(serie.head(10), use_container_width=True)
    st.markdown("**Variables construidas para el modelo:**")
    st.dataframe(df_feat[["fecha", "ventas"] + variables].head(12), use_container_width=True)
    st.code(
        """
serie = df.set_index("order_date").resample("W").agg(
    ventas=("total_price", "sum"), pedidos=("order_id", "nunique"), unidades=("quantity", "sum")
)
serie["lag_1"] = serie["ventas"].shift(1)
serie["lag_4"] = serie["ventas"].shift(4)
serie["media_4"] = serie["ventas"].shift(1).rolling(4).mean()
""",
        language="python",
    )

# ---------------------------------------------------------------
elif etapa == "5. Modelamiento":
    st.header("5. Entrenamiento del modelo")
    st.write(
        """
    Se utiliza Random Forest porque permite modelar relaciones no lineales entre precio, calendario
    y comportamiento pasado de las ventas. La separación entrenamiento/prueba respeta el orden temporal:
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
    c1.metric("MAE", f"${mae:,.0f}")
    c2.metric("RMSE", f"${rmse:,.0f}")
    c3.metric("MAPE", f"{mape:.2f}%")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(test["fecha"], test["ventas"], label="Real", marker="o")
    ax.plot(test["fecha"], pred, label="Predicción", marker="o")
    ax.legend()
    ax.set_title("Ventas semanales reales vs. pronosticadas")
    ax.set_ylabel("Ingresos ($)")
    st.pyplot(fig)

    resultados = test[["fecha", "ventas"]].copy()
    resultados["prediccion"] = pred.round(1)
    resultados["error_abs"] = np.abs(resultados["ventas"] - resultados["prediccion"]).round(1)
    st.dataframe(resultados, use_container_width=True)

# ---------------------------------------------------------------
else:
    st.header("7. Conclusiones y discusión")
    st.success(f"El modelo obtuvo un MAPE aproximado de {mape:.2f}% en el conjunto de prueba.")
    st.write(
        """
    El prototipo combina dos vistas complementarias: un análisis descriptivo (Top 5 de productos, útil
    para decisiones de menú y abastecimiento) y un análisis predictivo (pronóstico semanal de ingresos,
    útil para planificación operativa). El alumno recorre la trazabilidad completa entre problema,
    datos, transformación, entrenamiento, métricas y decisión.
    """
    )
    st.markdown(
        """
    **Preguntas de cierre**
    - ¿Cómo debería el negocio priorizar insumos según el Top 5 de productos?
    - ¿Por qué no sería correcto dividir aleatoriamente una serie temporal?
    - ¿Qué eventos (feriados, campañas) podrían explicar semanas atípicas no capturadas por el modelo?
    """
    )

st.sidebar.markdown("---")
st.sidebar.caption("Caso académico adaptado para fines docentes — datos de ventas de pizza 2015.")
