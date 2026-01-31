import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Data Science Energy Analytics", layout="wide", page_icon="🔬")

# Estilo personalizado para mejorar la UI
st.markdown("""
    <style>
    .stMetric { border: 1px solid #e6e9ef; padding: 15px; border-radius: 10px; background-color: #ffffff; }
    .main { background-color: #f8f9fa; }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER Y NARRATIVA ---
st.title("🔬 Renewable Energy Intelligence Portal")
st.markdown("""
    Bienvenido al portal de analítica avanzada. Esta herramienta transforma datos crudos en 
    **historias accionables** mediante tres bloques de análisis y un motor de segmentación.
""")

# --- CARGA DE DATOS Y SIDEBAR ---
st.sidebar.header("📁 Control de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo .csv", type=["csv"])

if uploaded_file is not None:
    @st.cache_data
    def load_and_preprocess(file):
        df = pd.read_csv(file)
        if 'Fecha_Entrada_Operacion' in df.columns:
            df['Fecha_Entrada_Operacion'] = pd.to_datetime(df['Fecha_Entrada_Operacion'])
        return df

    df_raw = load_and_preprocess(uploaded_file)

    # Filtros Dinámicos que afectan a toda la historia
    with st.sidebar.expander("🎯 Segmentación del Mercado", expanded=True):
        tech_list = sorted(df_raw['Tecnologia'].unique())
        selected_tech = st.multiselect("Tecnología:", tech_list, default=tech_list)
        
        op_list = sorted(df_raw['Operador'].unique())
        selected_op = st.multiselect("Operador:", op_list, default=op_list)

    # DataFrame Filtrado
    df = df_raw[(df_raw['Tecnologia'].isin(selected_tech)) & (df_raw['Operador'].isin(selected_op))]

    # --- ESTRUCTURA DE BLOQUES ---
    tab_cual, tab_cuan, tab_graf, tab_ml = st.tabs([
        "📊 Bloque Cualitativo", "🔢 Bloque Cuantitativo", "📈 Bloque Gráfico", "🤖 Segmentación (ML)"
    ])

    # --- 1. BLOQUE CUALITATIVO (¿Quién y Qué?) ---
    with tab_cual:
        st.header("Análisis de Composición y Categorías")
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Cuota de Mercado por Tecnología")
            fig_pie = px.pie(df, names='Tecnologia', values='Capacidad_Instalada_MW', hole=0.4,
                            color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with c2:
            st.subheader("Estado Operativo de las Plantas")
            fig_sun = px.sunburst(df, path=['Estado_Actual', 'Tecnologia'], values='Capacidad_Instalada_MW')
            st.plotly_chart(fig_sun, use_container_width=True)

    # --- 2. BLOQUE CUANTITATIVO (¿Cuánto?) ---
    with tab_cuan:
        st.header("Estadística Descriptiva Avanzada")
        
        # KPIs Dinámicos
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Capacidad Total", f"{df['Capacidad_Instalada_MW'].sum():,.0f} MW")
        m2.metric("Inversión Acumulada", f"${df['Inversion_Inicial_MUSD'].sum():,.1f} M")
        m3.metric("Eficiencia Promedio", f"{df['Eficiencia_Planta_Pct'].mean():.1f}%")
        m4.metric("Generación Diaria", f"{df['Generacion_Diaria_MWh'].sum():,.0f} MWh")

        st.divider()
        
        col_sel = st.selectbox("Analizar distribución de:", 
                              ['Capacidad_Instalada_MW', 'Generacion_Diaria_MWh', 'Eficiencia_Planta_Pct', 'Inversion_Inicial_MUSD'])
        
        sc1, sc2 = st.columns([1, 2])
        with sc1:
            st.write("**Métricas de Forma y Tendencia Central**")
            stats = pd.Series({
                "Media": df[col_sel].mean(),
                "Mediana": df[col_sel].median(),
                "Desviación Est.": df[col_sel].std(),
                "Asimetría (Skew)": df[col_sel].skew(),
                "Kurtosis": df[col_sel].kurtosis()
            })
            st.table(stats)
        
        with sc2:
            fig_hist = px.histogram(df, x=col_sel, color="Tecnologia", marginal="rug",
                                   title=f"Histograma de {col_sel} por Tecnología")
            st.plotly_chart(fig_hist, use_container_width=True)

    # --- 3. BLOQUE GRÁFICO (Storytelling Visual) ---
    with tab_graf:
        st.header("Visualización de Relaciones Críticas")
        
        st.subheader("¿Cómo se traduce la inversión en eficiencia?")
        fig_scatter = px.scatter(
            df, x="Inversion_Inicial_MUSD", y="Eficiencia_Planta_Pct",
            color="Tecnologia", size="Capacidad_Instalada_MW",
            hover_name="ID_Proyecto", trendline="ols",
            title="Relación Inversión vs Eficiencia (con Línea de Tendencia)"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.subheader("Comparativa de Rendimiento por Operador")
        fig_box = px.box(df, x="Operador", y="Generacion_Diaria_MWh", color="Tecnologia",
                        title="Dispersión de Generación por Operador")
        st.plotly_chart(fig_box, use_container_width=True)

    # --- 4. BLOQUE MACHINE LEARNING (Descubrimiento) ---
    with tab_ml:
        st.header("🤖 Segmentación Automática mediante K-Means")
        st.markdown("Agrupamos los proyectos basándonos en sus similitudes estadísticas (Inversión vs Capacidad).")
        
        clusters = st.slider("Selecciona número de clusters (grupos):", 2, 6, 3)
        
        # Preparación para Clustering
        X = df[['Inversion_Inicial_MUSD', 'Capacidad_Instalada_MW']].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = KMeans(n_clusters=clusters, random_state=42)
        df.loc[X.index, 'Cluster'] = model.fit_predict(X_scaled).astype(str)
        
        fig_ml = px.scatter(
            df, x="Inversion_Inicial_MUSD", y="Capacidad_Instalada_MW",
            color="Cluster", symbol="Tecnologia",
            size="Eficiencia_Planta_Pct",
            title=f"Resultados de la Segmentación en {clusters} Grupos"
        )
        st.plotly_chart(fig_ml, use_container_width=True)
        st.info("💡 Los clusters revelan qué proyectos tienen comportamientos atípicos o son 'líderes de eficiencia' en su rango de inversión.")

else:
    st.warning("⚠️ Esperando carga de datos para iniciar el motor de análisis.")
    st.image("https://images.unsplash.com/photo-1466611653911-95282fc3656b?q=80&w=2070&auto=format&fit=crop", 
             caption="Energía para el futuro a través de los datos")
