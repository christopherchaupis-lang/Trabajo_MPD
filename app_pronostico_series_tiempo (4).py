import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Prototipo - Ventas de Pizza", page_icon="🍕", layout="wide")

# -----------------------------------------------------------------------
# 0. ESTILO VISUAL (TEMA PIZZERÍA)
# -----------------------------------------------------------------------
PALETA_PIZZA = ["#C1272D", "#F4A300", "#6B8E23", "#8B4513", "#F5D061"]  # tomate, queso, albahaca, corteza, mozzarella

def aplicar_estilo_pizza():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #FFF8EE;
        }
        h1, h2, h3 {
            color: #7A2E0E;
            font-family: 'Trebuchet MS', sans-serif;
        }
        h1 {
            border-bottom: 4px solid #C1272D;
            padding-bottom: 8px;
        }
        [data-testid="stSidebar"] {
            background-color: #7A2E0E;
        }
        [data-testid="stSidebar"] * {
            color: #FFF8EE !important;
        }
        [data-testid="stMetricValue"] {
            color: #C1272D;
        }
        [data-testid="stMetricLabel"] {
            color: #7A2E0E;
        }
        div.stButton > button, .stRadio label {
            font-family: 'Trebuchet MS', sans-serif;
        }
        .stAlert {
            border-radius: 10px;
        }
        hr {
            border-top: 2px dashed #F4A300;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def estilo_ejes_pizza(ax):
    """Aplica el toque visual de pizzería a un eje de matplotlib."""
    ax.set_facecolor("#FFF8EE")
    ax.figure.set_facecolor("#FFF8EE")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#7A2E0E")
    return ax

aplicar_estilo_pizza()

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
    """Agrega las ventas a nivel semanal para el análisis de series de tiempo.

    CORRECCIÓN CLAVE: elimina las semanas INCOMPLETAS (la primera y la última
    del año). El resample 'W' crea un último bin (28/12–03/01) que solo contiene
    4 días de ventas (el dataset termina el 31/12): sus unidades caen ~40-50%
    artificialmente y el modelo, que predice semanas completas, genera un error
    gigante en ese punto (en el gráfico: real≈105 vs predicción≈227).
    """
    tmp = df.set_index("order_date")
    serie = (
        tmp.resample("W")
        .agg(ventas=("total_price", "sum"), pedidos=("order_id", "nunique"), unidades=("quantity", "sum"))
        .reset_index()
        .rename(columns={"order_date": "fecha"})
    )
    precio = (
        tmp.resample("W")["unit_price"]
        .mean()
        .reset_index()
        .rename(columns={"order_date": "fecha", "unit_price": "precio"})
    )
    # n° de días con ventas dentro de cada semana (detecta semanas cortas)
    dias = (
        df.assign(dia=df["order_date"].dt.normalize())
        .drop_duplicates("dia")
        .set_index("dia")
        .resample("W")
        .size()
        .reset_index(name="dias_con_venta")
        .rename(columns={"dia": "fecha"})
    )
    serie = serie.merge(precio, on="fecha").merge(dias, on="fecha")
    n_incompletas = int((serie["dias_con_venta"] < 7).sum())
    serie = serie[serie["dias_con_venta"] >= 7].drop(columns="dias_con_venta").reset_index(drop=True)
    serie.attrs["semanas_incompletas"] = n_incompletas
    return serie

def crear_features(df):
    out = df.copy()
    out["t"] = np.arange(len(out))                                   # tendencia
    doy = out["fecha"].dt.dayofyear
    out["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)                # estacionalidad anual
    out["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    out["semana"] = out["fecha"].dt.isocalendar().week.astype(int)
    out["mes"] = out["fecha"].dt.month
    out["lag_1"] = out["unidades"].shift(1)
    out["lag_2"] = out["unidades"].shift(2)
    out["lag_3"] = out["unidades"].shift(3)
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

VARIABLES = ["precio", "semana", "mes", "t", "sin_doy", "cos_doy",
             "lag_1", "lag_2", "lag_3", "lag_4", "media_4"]

def _mape(reales, preds):
    reales = np.asarray(reales, dtype=float)
    preds = np.asarray(preds, dtype=float)
    return np.mean(np.abs((reales - preds) / np.maximum(reales, 1))) * 100

def entrenar_modelo(df_feat):
    corte = int(len(df_feat) * 0.80)  # split temporal: 80% train / 20% test, sin mezclar
    train = df_feat.iloc[:corte]
    test = df_feat.iloc[corte:]
    model = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42)
    model.fit(train[VARIABLES], train["unidades"])
    pred = model.predict(test[VARIABLES])
    mae = mean_absolute_error(test["unidades"], pred)
    rmse = mean_squared_error(test["unidades"], pred) ** 0.5
    mape = _mape(test["unidades"], pred)
    return model, train, test, pred, mae, rmse, mape

@st.cache_data
def backtest_temporal(df_feat, min_train=28, horizonte=4, paso=4):
    """Validación con ventana expansiva: en vez de un solo split (11 semanas,
    muy inestable), simula varios 'finales de período' y promedia el error.
    Devuelve métricas globales y tabla real vs. predicción de todos los folds."""
    preds, reales, fechas = [], [], []
    for corte in range(min_train, len(df_feat) - horizonte + 1, paso):
        tr = df_feat.iloc[:corte]
        te = df_feat.iloc[corte:corte + horizonte]
        m = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42)
        m.fit(tr[VARIABLES], tr["unidades"])
        preds.extend(m.predict(te[VARIABLES]))
        reales.extend(te["unidades"].tolist())
        fechas.extend(te["fecha"].tolist())
    preds = np.array(preds)
    reales = np.array(reales, dtype=float)
    resumen = pd.DataFrame({"fecha": fechas, "real": reales, "prediccion": np.round(preds, 1)})
    return (mean_absolute_error(reales, preds),
            mean_squared_error(reales, preds) ** 0.5,
            _mape(reales, preds), resumen)

@st.cache_data
def entrenar_modelo_completo(df_feat):
    """Reentrena el mismo Random Forest, pero usando el 100% del historial disponible
    (no solo el 80% de entrenamiento). Este es el modelo que se usa para pronosticar
    semanas que todavía no existen, ya que ahí no hay nada que 'reservar' para prueba."""
    modelo_final = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42)
    modelo_final.fit(df_feat[VARIABLES], df_feat["unidades"])
    return modelo_final

@st.cache_data
def pronosticar_futuro(df_feat, _modelo_final, n_semanas=8):
    """Pronóstico RECURSIVO a n_semanas hacia adelante con el mismo modelo.

    Como el modelo necesita lag_1..lag_4 y media_4 (valores de semanas anteriores)
    y esas semanas todavía no existen, cada predicción se usa como si fuera un dato
    real para calcular los rezagos de la semana siguiente (encadenado paso a paso).
    Supuesto de negocio: el precio se mantiene igual al promedio de las últimas 4
    semanas observadas (no se proyectan cambios de precio)."""
    historial_unidades = df_feat["unidades"].tolist()
    ultima_fecha = df_feat["fecha"].iloc[-1]
    precio_referencia = df_feat["precio"].tail(4).mean()
    t_actual = df_feat["t"].iloc[-1]

    fechas_futuras, predicciones = [], []
    for paso in range(1, n_semanas + 1):
        fecha_f = ultima_fecha + pd.Timedelta(weeks=paso)
        t_actual += 1
        doy = fecha_f.dayofyear
        fila = pd.DataFrame([{
            "precio": precio_referencia,
            "semana": int(fecha_f.isocalendar()[1]),
            "mes": fecha_f.month,
            "t": t_actual,
            "sin_doy": np.sin(2 * np.pi * doy / 365.25),
            "cos_doy": np.cos(2 * np.pi * doy / 365.25),
            "lag_1": historial_unidades[-1],
            "lag_2": historial_unidades[-2],
            "lag_3": historial_unidades[-3],
            "lag_4": historial_unidades[-4],
            "media_4": np.mean(historial_unidades[-4:]),
        }])[VARIABLES]
        pred = float(_modelo_final.predict(fila)[0])
        historial_unidades.append(pred)
        fechas_futuras.append(fecha_f)
        predicciones.append(pred)

    return pd.DataFrame({"fecha": fechas_futuras, "prediccion": np.round(predicciones, 1)})

# Carga completa, se identifica el Top 5 en cantidad, y TODO lo demás
# (EDA y pronóstico) se calcula solo sobre esos 5 productos.
df_completo = cargar_datos()
top5 = calcular_top5(df_completo)
productos_top5 = top5.index.tolist()
df = df_completo[df_completo["pizza_name"].isin(productos_top5)].copy()

serie = agregar_semanal(df)
n_semanas_incompletas = serie.attrs.get("semanas_incompletas", 0)
df_feat = crear_features(serie)
model = entrenar_modelo(df_feat)[0]  # el split 80/20 ya no se usa en evaluación
bt_mae, bt_rmse, bt_mape, bt_tabla = backtest_temporal(df_feat)

N_SEMANAS_FUTURO = 8
modelo_final = entrenar_modelo_completo(df_feat)
pronostico_futuro = pronosticar_futuro(df_feat, modelo_final, n_semanas=N_SEMANAS_FUTURO)

# -----------------------------------------------------------------------
# INTERFAZ
# -----------------------------------------------------------------------
st.title("🍕 Prototipo · Análisis y pronóstico de ventas de pizza")
st.caption("Aplicación navegable: del problema de negocio al pronóstico de demanda, con datos reales de venta 2015")
st.warning(
    f"⚠️ Todo el análisis (exploratorio y pronóstico) se calcula **únicamente** sobre los 5 productos "
    f"más demandados en cantidad: {', '.join(productos_top5)}. Los demás productos fueron excluidos."
)

ICONOS_ETAPA = {
    "1. Problema": "🎯",
    "2. Datos": "📋",
    "3. Top 5 productos": "🏆",
    "4. Preparación (series de tiempo)": "🍳",
    "5. Modelamiento": "🤖",
    "6. Evaluación": "📈",
    "7. Pronóstico a futuro": "🔮",
    "8. Conclusiones": "🧑‍🍳",
}

st.sidebar.markdown("## 🍕 Menú de la app")
etapa = st.sidebar.radio(
    "Recorrer etapas",
    list(ICONOS_ETAPA.keys()),
    format_func=lambda e: f"{ICONOS_ETAPA[e]}  {e}",
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

    st.markdown("🍅 **Distribución por categoría (dentro del Top 5)**")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    df.groupby("pizza_category")["quantity"].sum().sort_values(ascending=False).plot(kind="bar", ax=ax, color=PALETA_PIZZA)
    ax.set_ylabel("Unidades vendidas")
    ax.set_xlabel("")
    estilo_ejes_pizza(ax)
    st.pyplot(fig)

    st.markdown("🧀 **Distribución por tamaño (dentro del Top 5)**")
    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    df.groupby("pizza_size")["quantity"].sum().reindex(["S", "M", "L", "XL", "XXL"]).plot(kind="bar", ax=ax2, color="#F4A300")
    ax2.set_ylabel("Unidades vendidas")
    ax2.set_xlabel("")
    estilo_ejes_pizza(ax2)
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
        ax.barh(resumen["pizza_name"][::-1], resumen["unidades"][::-1], color="#C1272D")
        ax.set_xlabel("Unidades vendidas")
        ax.set_title("🏆 Top 5 productos por cantidad")
        estilo_ejes_pizza(ax)
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
    if n_semanas_incompletas:
        st.info(
            f"🧹 Se descartaron **{n_semanas_incompletas} semana(s) incompleta(s)** (con menos de 7 días de ventas). "
            "Son artefactos del corte del año (por ejemplo, la última semana solo tiene 4 días de datos) "
            "y harían que el modelo pareciera mucho peor de lo que es: predice semanas completas, "
            "pero la realidad reportada está 'incompleta'."
        )
    st.markdown("**Variables construidas para el modelo:**")
    st.dataframe(df_feat[["fecha", "unidades"] + VARIABLES].head(12), use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Análisis descriptivo de las variables de entrada")

    st.markdown("**Variables numéricas** (media, moda, desviación estándar, mínimo y máximo):")
    columnas_numericas = ["unidades"] + VARIABLES
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

    importancias_prev = pd.Series(model.feature_importances_, index=VARIABLES).sort_values(ascending=False)
    variables_clave = ["unidades"] + [v for v in importancias_prev.head(3).index.tolist() if v != "unidades"]

    st.markdown(f"**Histogramas y diagramas de caja de las variables más importantes** ({', '.join(variables_clave)}):")
    for var in variables_clave:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 3))
            df_feat[var].dropna().hist(ax=ax, bins=15, color="#F4A300", edgecolor="#7A2E0E")
            ax.set_title(f"Histograma — {var}")
            ax.set_xlabel(var)
            ax.set_ylabel("Frecuencia")
            estilo_ejes_pizza(ax)
            st.pyplot(fig)
        with c2:
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            caja = ax2.boxplot(df_feat[var].dropna(), vert=False, patch_artist=True)
            for patch in caja["boxes"]:
                patch.set_facecolor("#C1272D")
                patch.set_alpha(0.6)
            ax2.set_title(f"Diagrama de caja — {var}")
            estilo_ejes_pizza(ax2)
            ax2.set_xlabel(var)
            ax2.set_yticks([])
            st.pyplot(fig2)

    st.markdown("---")
    st.code(
        """
serie = df.set_index("order_date").resample("W").agg(
    ventas=("total_price", "sum"), pedidos=("order_id", "nunique"), unidades=("quantity", "sum")
)
# Eliminar semanas con < 7 días de ventas (artefactos de inicio/fin de año)
serie = serie[serie["dias_con_venta"] >= 7]
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
    c1.metric("Semanas de entrenamiento (cada fold del backtest)", f"{len(df_feat) - 4} aprox.")
    c2.metric("Semanas de la serie completa", len(df_feat) + 4)

    imp = pd.DataFrame({"variable": VARIABLES, "importancia": model.feature_importances_}).sort_values(
        "importancia", ascending=False
    )
    st.markdown("**Importancia de variables**")
    st.dataframe(imp, use_container_width=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(imp["variable"][::-1], imp["importancia"][::-1], color="#6B8E23")
    estilo_ejes_pizza(ax)
    st.pyplot(fig)

# ---------------------------------------------------------------
elif etapa == "6. Evaluación":
    st.header("6. Evaluación")

    st.subheader("Validación temporal (backtesting — métrica principal)")
    st.write(
        "En lugar de confiar en un único split de ~11 semanas, se simulan varios cortes temporales "
        "con ventana expansiva y se promedia el error. Es una estimación mucho más estable del MAPE."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (backtest)", f"{bt_mae:,.0f} unidades")
    c2.metric("RMSE (backtest)", f"{bt_rmse:,.0f} unidades")
    c3.metric("MAPE (backtest)", f"{bt_mape:.2f}%")

    fig_bt, ax_bt = plt.subplots(figsize=(10, 4))
    ax_bt.plot(bt_tabla["fecha"], bt_tabla["real"], label="Real 🍕", marker="o", color="#C1272D")
    ax_bt.plot(bt_tabla["fecha"], bt_tabla["prediccion"], label="Predicción 🤖", marker="o", color="#F4A300")
    ax_bt.legend()
    ax_bt.set_title("Backtesting: reales vs. pronosticadas (varios folds)")
    ax_bt.set_ylabel("Unidades")
    estilo_ejes_pizza(ax_bt)
    st.pyplot(fig_bt)

    st.markdown("**Detalle por semana (folds del backtesting):**")
    bt_detalle = bt_tabla.copy()
    bt_detalle["Error (%)"] = (
        np.abs(bt_detalle["real"] - bt_detalle["prediccion"])
        / np.maximum(bt_detalle["real"], 1) * 100
    ).round(2)
    bt_detalle = bt_detalle.rename(
        columns={"fecha": "Semana", "real": "Real (unidades)", "prediccion": "Predicción"}
    )
    st.dataframe(bt_detalle, use_container_width=True)

# ---------------------------------------------------------------
elif etapa == "7. Pronóstico a futuro":
    st.header("🔮 7. Pronóstico a futuro — próximas 8 semanas")
    st.write(
        f"""
    Se usa el **mismo modelo Random Forest**, pero reentrenado con el **100% del historial**
    (no solo el 80% de entrenamiento), porque para pronosticar el futuro ya no tiene sentido
    reservar datos para prueba. El pronóstico es **recursivo**: como el modelo necesita conocer
    las ventas de semanas anteriores (`lag_1`...`lag_4`, `media_4`) y esas semanas futuras todavía
    no existen, cada predicción se reutiliza como si fuera un dato real para poder calcular los
    rezagos de la semana siguiente, uno a uno hasta llegar a las {N_SEMANAS_FUTURO} semanas.

    **Supuesto de negocio:** se asume que el precio promedio se mantiene igual al de las últimas
    4 semanas observadas (no se proyectan cambios de precio ni promociones).
    """
    )

    tabla_pronostico = pronostico_futuro.rename(
        columns={"fecha": "Semana", "prediccion": "Unidades pronosticadas"}
    )
    st.dataframe(tabla_pronostico, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Promedio semanal proyectado", f"{pronostico_futuro['prediccion'].mean():,.0f} unidades")
    c2.metric("Semana con más demanda", f"{pronostico_futuro['prediccion'].max():,.0f} unidades")
    c3.metric("Semana con menos demanda", f"{pronostico_futuro['prediccion'].min():,.0f} unidades")

    st.markdown("**Historial reciente + pronóstico a 8 semanas**")
    historico_reciente = df_feat[["fecha", "unidades"]].tail(16)

    fig_fc, ax_fc = plt.subplots(figsize=(10, 4))
    ax_fc.plot(historico_reciente["fecha"], historico_reciente["unidades"],
               label="Histórico real 🍕", marker="o", color="#7A2E0E")
    ax_fc.plot(pronostico_futuro["fecha"], pronostico_futuro["prediccion"],
               label=f"Pronóstico {N_SEMANAS_FUTURO} semanas 🔮", marker="o", linestyle="--", color="#C1272D")
    ax_fc.axvline(df_feat["fecha"].iloc[-1], color="#F4A300", linestyle=":", linewidth=2)
    ax_fc.legend()
    ax_fc.set_title("Demanda semanal: histórico vs. pronóstico")
    ax_fc.set_ylabel("Unidades")
    estilo_ejes_pizza(ax_fc)
    st.pyplot(fig_fc)

    st.info(
        f"📌 El error esperado de este pronóstico es similar al del backtesting (MAPE ≈ {bt_mape:.2f}%), "
        "pero en la práctica suele crecer cuanto más lejos se proyecta, porque los rezagos usados "
        "en las últimas semanas del horizonte ya no son datos reales, sino predicciones previas del "
        "propio modelo (el error se puede ir acumulando)."
    )

# ---------------------------------------------------------------
elif etapa == "8. Conclusiones":
    st.header("8. Conclusiones y discusión")
    st.success(
        f"El modelo obtuvo un MAPE aproximado de **{bt_mape:.2f}%** en validación temporal (backtesting), "
        f"sobre el Top 5 de productos en cantidad."
    )
    st.write(
        """
    El prototipo combina dos vistas complementarias, ambas restringidas a los 5 productos más demandados
    en cantidad: un análisis descriptivo (Top 5, útil para decisiones de menú y abastecimiento) y un
    análisis predictivo (pronóstico semanal de la cantidad de unidades vendidas de esos 5 productos,
    útil para planificación de insumos y personal). El alumno recorre la trazabilidad completa entre problema,
    datos, transformación, entrenamiento, métricas y decisión.

    **Lección clave de preparación:** las semanas incompletas por el corte del calendario (la última del año
    tenía solo 4 días de ventas) inflaban artificialmente el error del modelo. Detectarlas y descartarlas
    —o modelarlas con la variable 'días de venta'— es parte del trabajo de preparación de datos.
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
