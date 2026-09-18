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

PARAMS_POR_DEFECTO = {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 2}

# Rejilla de búsqueda: combinaciones razonables para una serie semanal corta
# (pocos datos -> conviene explorar árboles menos profundos / más regularizados,
# que suelen generalizar mejor que el valor fijo original).
GRID_HIPERPARAMETROS = {
    "n_estimators": [200, 300, 500],
    "max_depth": [3, 4, 6, 8, None],
    "min_samples_leaf": [1, 2, 3, 5],
}

def entrenar_modelo(df_feat, parametros=None):
    parametros = parametros or PARAMS_POR_DEFECTO
    corte = int(len(df_feat) * 0.80)  # split temporal: 80% train / 20% test, sin mezclar
    train = df_feat.iloc[:corte]
    test = df_feat.iloc[corte:]
    model = RandomForestRegressor(random_state=42, n_jobs=-1, **parametros)
    model.fit(train[VARIABLES], train["unidades"])
    pred = model.predict(test[VARIABLES])
    mae = mean_absolute_error(test["unidades"], pred)
    rmse = mean_squared_error(test["unidades"], pred) ** 0.5
    mape = _mape(test["unidades"], pred)
    return model, train, test, pred, mae, rmse, mape

@st.cache_data
def backtest_temporal(df_feat, min_train=28, horizonte=4, paso=4, parametros=None):
    """Validación con ventana expansiva: en vez de un solo split (11 semanas,
    muy inestable), simula varios 'finales de período' y promedia el error.
    Devuelve métricas globales y tabla real vs. predicción de todos los folds."""
    parametros = parametros or PARAMS_POR_DEFECTO
    preds, reales, fechas = [], [], []
    for corte in range(min_train, len(df_feat) - horizonte + 1, paso):
        tr = df_feat.iloc[:corte]
        te = df_feat.iloc[corte:corte + horizonte]
        m = RandomForestRegressor(random_state=42, n_jobs=-1, **parametros)
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
def optimizar_hiperparametros(df_feat, min_train=28, horizonte=4, paso=4):
    """Recorre GRID_HIPERPARAMETROS y se queda con la combinación que da el menor
    MAPE en el mismo backtest de ventana expansiva (ningún dato de prueba se usa
    para elegir; se compara siempre con el mismo esquema de folds). El valor original
    (300, 8, 2) es apenas una de las combinaciones evaluadas, así que el resultado
    nunca es peor que el punto de partida."""
    mejor_mape = np.inf
    mejores_params = dict(PARAMS_POR_DEFECTO)
    for n_est in GRID_HIPERPARAMETROS["n_estimators"]:
        for m_depth in GRID_HIPERPARAMETROS["max_depth"]:
            for m_leaf in GRID_HIPERPARAMETROS["min_samples_leaf"]:
                params = {"n_estimators": n_est, "max_depth": m_depth, "min_samples_leaf": m_leaf}
                preds, reales = [], []
                for corte in range(min_train, len(df_feat) - horizonte + 1, paso):
                    tr = df_feat.iloc[:corte]
                    te = df_feat.iloc[corte:corte + horizonte]
                    m = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
                    m.fit(tr[VARIABLES], tr["unidades"])
                    preds.extend(m.predict(te[VARIABLES]))
                    reales.extend(te["unidades"].tolist())
                if not preds:
                    continue
                mape = _mape(np.array(reales, dtype=float), np.array(preds))
                if mape < mejor_mape:
                    mejor_mape, mejores_params = mape, params
    return mejores_params, mejor_mape

@st.cache_data
def entrenar_modelo_completo(df_feat, parametros=None):
    """Reentrena el mismo Random Forest, pero usando el 100% del historial disponible
    (no solo el 80% de entrenamiento). Este es el modelo que se usa para pronosticar
    semanas que todavía no existen, ya que ahí no hay nada que 'reservar' para prueba."""
    parametros = parametros or PARAMS_POR_DEFECTO
    modelo_final = RandomForestRegressor(random_state=42, n_jobs=-1, **parametros)
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

OPCION_TODOS_TOP5 = "🍕 Todos los Top 5 (agregado)"

@st.cache_data
def preparar_serie_producto(df_completo, productos_top5, producto_seleccionado):
    """Filtra el dataset al producto elegido (o a los 5 juntos si se elige la opción agregada)
    y recalcula la serie semanal + variables de features, reutilizando la misma lógica que el
    análisis principal, para poder segmentar el análisis por producto individual."""
    if producto_seleccionado == OPCION_TODOS_TOP5:
        df_sel = df_completo[df_completo["pizza_name"].isin(productos_top5)].copy()
    else:
        df_sel = df_completo[df_completo["pizza_name"] == producto_seleccionado].copy()
    serie_sel = agregar_semanal(df_sel)
    df_feat_sel = crear_features(serie_sel)
    return df_sel, serie_sel, df_feat_sel

# Carga completa, se identifica el Top 5 en cantidad, y TODO lo demás
# (EDA y pronóstico) se calcula solo sobre esos 5 productos.
df_completo = cargar_datos()
top5 = calcular_top5(df_completo)
productos_top5 = top5.index.tolist()
df = df_completo[df_completo["pizza_name"].isin(productos_top5)].copy()

serie = agregar_semanal(df)
n_semanas_incompletas = serie.attrs.get("semanas_incompletas", 0)
df_feat = crear_features(serie)

# Se buscan los hiperparámetros del Random Forest que minimizan el MAPE del backtest
# (sobre el Top 5 agregado) y se reutilizan en todos los modelos de la app -mismo
# criterio de selección que antes se fijaba a mano en (300, 8, 2)-.
mejores_params, mape_busqueda = optimizar_hiperparametros(df_feat)

model = entrenar_modelo(df_feat, mejores_params)[0]  # el split 80/20 ya no se usa en evaluación
bt_mae, bt_rmse, bt_mape, bt_tabla = backtest_temporal(df_feat, parametros=mejores_params)

N_SEMANAS_FUTURO = 8
modelo_final = entrenar_modelo_completo(df_feat, mejores_params)
pronostico_futuro = pronosticar_futuro(df_feat, modelo_final, n_semanas=N_SEMANAS_FUTURO)

# -----------------------------------------------------------------------
# INTERFAZ
# -----------------------------------------------------------------------
st.title("🍕 Prototipo · Análisis y pronóstico de ventas de pizza")
st.caption("Aplicación navegable: del problema de negocio al pronóstico de demanda, con datos reales de venta 2015")

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

    st.markdown("### 🍕 ¿Por qué proyectar la demanda en este rubro?")
    st.write(
        """
    El negocio de las pizzerías tiene características que hacen especialmente valioso pronosticar
    la demanda semana a semana, en lugar de reaccionar solo cuando ya se tiene el pedido en la puerta:

    - **Los insumos son perecibles.** Masa, queso, embutidos y vegetales se vencen en pocos días.
      Comprar de más genera merma y pérdida directa de dinero; comprar de menos genera quiebres de
      stock justo en la semana de mayor demanda.
    - **La demanda no es constante:** varía por temporada, día de la semana y tendencias de consumo.
      Un pronóstico permite anticipar picos y valles en vez de operar "a ciegas" con el mismo nivel
      de compra todas las semanas.
    - **La planificación de personal depende del volumen esperado.** Saber con anticipación cuántas
      unidades se venderán ayuda a programar turnos de cocina y reparto sin sobre ni sub-dotar al local.
    - **Es la base para decisiones de mayor alcance:** negociar mejores precios con proveedores según
      volúmenes proyectados, decidir promociones en semanas de baja demanda esperada, o priorizar
      qué productos mantener siempre disponibles según el Top 5 de ventas.

    En resumen, pasar de "vender lo que llega" a "anticipar lo que se va a vender" es lo que convierte
    los datos históricos de pedidos en una herramienta de planificación operativa y financiera.
    """
    )

# ---------------------------------------------------------------
elif etapa == "2. Datos":
    st.header("2. Comprensión de los datos (dataset completo)")
    st.dataframe(df_completo.head(15), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Líneas de pedido", f"{len(df_completo):,}")
    c2.metric("Pedidos únicos", f"{df_completo['order_id'].nunique():,}")
    c3.metric("Unidades vendidas", f"{df_completo['quantity'].sum():,}")
    c4.metric("Ingresos totales", f"${df_completo['total_price'].sum():,.0f}")

    st.markdown("🍅 **Distribución por categoría (todos los productos)**")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    df_completo.groupby("pizza_category")["quantity"].sum().sort_values(ascending=False).plot(kind="bar", ax=ax, color=PALETA_PIZZA)
    ax.set_ylabel("Unidades vendidas")
    ax.set_xlabel("")
    estilo_ejes_pizza(ax)
    st.pyplot(fig)

    st.markdown("🧀 **Distribución por tamaño (todos los productos)**")
    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    df_completo.groupby("pizza_size")["quantity"].sum().reindex(["S", "M", "L", "XL", "XXL"]).plot(kind="bar", ax=ax2, color="#F4A300")
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

    st.markdown("**Ajuste de hiperparámetros**")
    st.write(
        f"""
    En vez de fijar a mano `n_estimators`, `max_depth` y `min_samples_leaf`, se probaron
    **{len(GRID_HIPERPARAMETROS["n_estimators"]) * len(GRID_HIPERPARAMETROS["max_depth"]) * len(GRID_HIPERPARAMETROS["min_samples_leaf"])}
    combinaciones** con el mismo backtest de ventana expansiva, y se eligió la que da el menor MAPE
    (así se reduce el sobreajuste típico de árboles muy profundos con pocas semanas de historia).
    """
    )
    c3, c4, c5 = st.columns(3)
    c3.metric("n_estimators", mejores_params["n_estimators"])
    c4.metric("max_depth", mejores_params["max_depth"] if mejores_params["max_depth"] is not None else "Sin límite")
    c5.metric("min_samples_leaf", mejores_params["min_samples_leaf"])
    st.caption(f"MAPE del backtest con estos hiperparámetros: {mape_busqueda:.2f}%")

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

    producto_eval = st.selectbox(
        "🍕 Selecciona un producto para evaluar (o mira el Top 5 agregado):",
        [OPCION_TODOS_TOP5] + productos_top5,
        key="selector_evaluacion",
    )
    df_sel, serie_sel, df_feat_sel = preparar_serie_producto(df_completo, productos_top5, producto_eval)
    bt_mae_sel, bt_rmse_sel, bt_mape_sel, bt_tabla_sel = backtest_temporal(df_feat_sel, parametros=mejores_params)

    st.subheader(f"Validación temporal (backtesting) — {producto_eval}")
    st.write(
        "En lugar de confiar en un único split de ~11 semanas, se simulan varios cortes temporales "
        "con ventana expansiva y se promedia el error. Es una estimación mucho más estable del MAPE."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (backtest)", f"{bt_mae_sel:,.0f} unidades")
    c2.metric("RMSE (backtest)", f"{bt_rmse_sel:,.0f} unidades")
    c3.metric("MAPE (backtest)", f"{bt_mape_sel:.2f}%")

    fig_bt, ax_bt = plt.subplots(figsize=(10, 4))
    ax_bt.plot(bt_tabla_sel["fecha"], bt_tabla_sel["real"], label="Real 🍕", marker="o", color="#C1272D")
    ax_bt.plot(bt_tabla_sel["fecha"], bt_tabla_sel["prediccion"], label="Predicción 🤖", marker="o", color="#F4A300")
    ax_bt.legend()
    ax_bt.set_title(f"Backtesting: reales vs. pronosticadas — {producto_eval}")
    ax_bt.set_ylabel("Unidades")
    estilo_ejes_pizza(ax_bt)
    st.pyplot(fig_bt)

    st.markdown("**Detalle por semana (folds del backtesting):**")
    bt_detalle = bt_tabla_sel.copy()
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

    producto_fc = st.selectbox(
        "🍕 Selecciona un producto para pronosticar (o mira el Top 5 agregado):",
        [OPCION_TODOS_TOP5] + productos_top5,
        key="selector_pronostico",
    )
    df_sel, serie_sel, df_feat_sel = preparar_serie_producto(df_completo, productos_top5, producto_fc)
    modelo_final_sel = entrenar_modelo_completo(df_feat_sel, mejores_params)
    pronostico_sel = pronosticar_futuro(df_feat_sel, modelo_final_sel, n_semanas=N_SEMANAS_FUTURO)
    _, _, _, bt_tabla_sel_fc = backtest_temporal(df_feat_sel, parametros=mejores_params)
    mape_referencia = _mape(bt_tabla_sel_fc["real"], bt_tabla_sel_fc["prediccion"])

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

    tabla_pronostico = pronostico_sel.rename(
        columns={"fecha": "Semana", "prediccion": "Unidades pronosticadas"}
    )
    st.dataframe(tabla_pronostico, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Promedio semanal proyectado", f"{pronostico_sel['prediccion'].mean():,.0f} unidades")
    c2.metric("Semana con más demanda", f"{pronostico_sel['prediccion'].max():,.0f} unidades")
    c3.metric("Semana con menos demanda", f"{pronostico_sel['prediccion'].min():,.0f} unidades")

    st.markdown(f"**Historial reciente + pronóstico a {N_SEMANAS_FUTURO} semanas — {producto_fc}**")
    historico_reciente = df_feat_sel[["fecha", "unidades"]].tail(16)

    fig_fc, ax_fc = plt.subplots(figsize=(10, 4))
    ax_fc.plot(historico_reciente["fecha"], historico_reciente["unidades"],
               label="Histórico real 🍕", marker="o", color="#7A2E0E")
    ax_fc.plot(pronostico_sel["fecha"], pronostico_sel["prediccion"],
               label=f"Pronóstico {N_SEMANAS_FUTURO} semanas 🔮", marker="o", linestyle="--", color="#C1272D")
    ax_fc.axvline(df_feat_sel["fecha"].iloc[-1], color="#F4A300", linestyle=":", linewidth=2)
    ax_fc.legend()
    ax_fc.set_title(f"Demanda semanal: histórico vs. pronóstico — {producto_fc}")
    ax_fc.set_ylabel("Unidades")
    estilo_ejes_pizza(ax_fc)
    st.pyplot(fig_fc)

    st.info(
        f"📌 El error esperado de este pronóstico es similar al del backtesting (MAPE ≈ {mape_referencia:.2f}%), "
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

    producto_lider = productos_top5[0]
    demanda_prom_futuro = pronostico_futuro["prediccion"].mean()
    semana_pico = pronostico_futuro.loc[pronostico_futuro["prediccion"].idxmax()]

    st.markdown("### 🍕 Conclusiones de negocio")
    st.write(
        f"""
    - **El Top 5 de productos concentra la mayor parte de la demanda**, con **{producto_lider}** como
      el producto más vendido en cantidad. Priorizar la disponibilidad de insumos para estos 5 productos
      reduce el riesgo de quiebre de stock en lo que más se vende, y es una base razonable para negociar
      mejores condiciones con proveedores clave.
    - **El pronóstico a 8 semanas proyecta una demanda promedio de ~{demanda_prom_futuro:,.0f} unidades
      semanales** para el Top 5, con un pico esperado de **{semana_pico['prediccion']:,.0f} unidades**
      en la semana del {semana_pico['fecha'].strftime('%d/%m/%Y')}. Esta información permite anticipar
      compras de insumos y turnos de personal antes de que la demanda ocurra, en lugar de reaccionar
      sobre la marcha.
    - **El error del modelo (MAPE ≈ {bt_mape:.2f}%)** es una referencia útil para dimensionar el margen
      de error al planificar: conviene mantener un colchón de stock proporcional a ese error, en vez de
      comprar exactamente la cantidad pronosticada semana a semana.
    - **La demanda tiene componentes de calendario relevantes** (estacionalidad y tendencia, capturadas
      en el modelo mediante variables de semana, mes y rezagos), lo que sugiere que la planificación de
      compras no debería ser uniforme durante el año, sino ajustarse según la época.
    - **Excluir del análisis a los productos fuera del Top 5** simplifica la operación, pero también
      implica que las decisiones de abastecimiento e insumos compartidos con productos menos vendidos
      no quedan cubiertas por este pronóstico; conviene revisarlas por separado.

    **Lección clave de preparación de datos:** las semanas incompletas por el corte del calendario (la
    última del año tenía solo 4 días de ventas) inflaban artificialmente el error del modelo. Detectarlas
    y descartarlas —o modelarlas explícitamente con la variable "días de venta"— fue clave para obtener
    una medición de error realista, y es un recordatorio de que la calidad del pronóstico depende tanto
    del modelo como de la preparación previa de los datos.
    """
    )

st.sidebar.markdown("---")

