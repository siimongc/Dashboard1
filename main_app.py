import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from groq import Groq

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Data Science Energy Analytics + AI", layout="wide", page_icon="🔬")

# Estilo personalizado para mejorar la UI
st.markdown("""
    <style>
    .stMetric { border: 1px solid #e6e9ef; padding: 15px; border-radius: 10px; background-color: #ffffff; }
    .main { background-color: #f8f9fa; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0px 0px; gap: 1px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #ff4b4b; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: CONFIGURACIÓN Y API KEY ---
st.sidebar.header("🔑 Acceso Inteligente")
groq_key = st.sidebar.text_input("Groq API Key:", type="password", help="Obtén tu llave en console.groq.com")

st.sidebar.divider()
st.sidebar.header("📁 Control de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo .csv", type=["csv"])

# --- FUNCIÓN DE ASISTENTE IA ---
def consultar_asistente_groq(prompt_content, api_key):
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Eres un Consultor Senior en Energía y Científico de Datos. Tu objetivo es interpretar métricas técnicas y dar recomendaciones estratégicas claras."},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.5,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error en la conexión con la IA: {str(e)}"

# --- LÓGICA PRINCIPAL ---
if uploaded_file is not None:
    @st.cache_data
    def load_and_preprocess(file):
        df = pd.read_csv(file)
        if 'Fecha_Entrada_Operacion' in df.columns:
            df['Fecha_Entrada_Operacion'] = pd.to_datetime(df['Fecha_Entrada_Operacion'])
        return df

    df_raw = load_and_preprocess(uploaded_file)

    # Filtros Dinámicos
    with st.sidebar.expander("🎯 Segmentación del Mercado", expanded=True):
        tech_list = sorted(df_raw['Tecnologia'].unique())
        selected_tech = st.multiselect("Tecnología:", tech_list, default=tech_list)
        
        op_list = sorted(df_raw['Operador'].unique())
        selected_op = st.multiselect("Operador:", op_list, default=op_list)

    df = df_raw[(df_raw['Tecnologia'].isin(selected_tech)) & (df_raw['Operador'].isin(selected_op))]

    # --- HEADER ---
    st.title("🔬 Renewable Energy Intelligence Portal")
    st.markdown(f"Analizando **{len(df)}** proyectos filtrados.")

    # --- ESTRUCTURA DE PESTAÑAS ---
    tab_cual, tab_cuan, tab_graf, tab_ml, tab_ai = st.tabs([
        "📊 Bloque Cualitativo", 
        "🔢 Bloque Cuantitativo", 
        "📈 Bloque Gráfico", 
        "🤖 Segmentación (ML)",
        "🧠 Asistente de Análisis IA"
    ])

    # --- 1. BLOQUE CUALITATIVO ---
    with tab_cual:
        st.header("Análisis de Composición")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Cuota de Mercado")
            fig_pie = px.pie(df, names='Tecnologia', values='Capacidad_Instalada_MW', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            st.subheader("Jerarquía Operativa")
            fig_sun = px.sunburst(df, path=['Estado_Actual', 'Tecnologia'], values='Capacidad_Instalada_MW')
            st.plotly_chart(fig_sun, use_container_width=True)

    # --- 2. BLOQUE CUANTITATIVO ---
    with tab_cuan:
        st.header("Estadística Descriptiva")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Capacidad Total", f"{df['Capacidad_Instalada_MW'].sum():,.0f} MW")
        m2.metric("Inversión Total", f"${df['Inversion_Inicial_MUSD'].sum():,.1f} M")
        m3.metric("Eficiencia Promedio", f"{df['Eficiencia_Planta_Pct'].mean():.1f}%")
        m4.metric("Generación Diaria", f"{df['Generacion_Diaria_MWh'].sum():,.0f} MWh")

        col_sel = st.selectbox("Distribución de:", ['Capacidad_Instalada_MW', 'Generacion_Diaria_MWh', 'Eficiencia_Planta_Pct', 'Inversion_Inicial_MUSD'])
        sc1, sc2 = st.columns([1, 2])
        with sc1:
            st.table(df[col_sel].describe())
        with sc2:
            fig_hist = px.histogram(df, x=col_sel, color="Tecnologia", marginal="box")
            st.plotly_chart(fig_hist, use_container_width=True)

    # --- 3. BLOQUE GRÁFICO ---
    with tab_graf:
        st.header("Visualización de Historias")
        st.subheader("Inversión vs Eficiencia")
        fig_scatter = px.scatter(df, x="Inversion_Inicial_MUSD", y="Eficiencia_Planta_Pct", 
                                 color="Tecnologia", size="Capacidad_Instalada_MW", trendline="ols")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # --- 4. BLOQUE ML ---
    with tab_ml:
        st.header("Clustering de Proyectos")
        n_clusters = st.slider("Clusters:", 2, 6, 3)
        X = df[['Inversion_Inicial_MUSD', 'Capacidad_Instalada_MW']].dropna()
        X_scaled = StandardScaler().fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_scaled)
        df.loc[X.index, 'Cluster'] = kmeans.labels_.astype(str)
        fig_ml = px.scatter(df, x="Inversion_Inicial_MUSD", y="Capacidad_Instalada_MW", color="Cluster", size="Eficiencia_Planta_Pct")
        st.plotly_chart(fig_ml, use_container_width=True)

    # --- 5. BLOQUE ASISTENTE IA (GROQ) ---
    with tab_ai:
        st.header("🧠 Consultoría Inteligente Llama 3.3")
        if not groq_key:
            st.info("💡 Por favor, ingresa tu API Key en la barra lateral para habilitar el análisis de la IA.")
        else:
            st.subheader("Análisis Automático de Hallazgos")
            if st.button("🚀 Ejecutar Análisis con IA"):
                with st.spinner("Procesando datos con Llama 3.3 Versatile..."):
                    # Crear el resumen técnico para la IA
                    stats_summary = f"""
                    DATOS ACTUALES:
                    - Proyectos totales: {len(df)}
                    - Tecnologías: {', '.join(selected_tech)}
                    - Inversión Total: {df['Inversion_Inicial_MUSD'].sum():.2f} MUSD
                    - Eficiencia Promedio: {df['Eficiencia_Planta_Pct'].mean():.2f}%
                    - Correlación Inversión-Eficiencia: {df['Inversion_Inicial_MUSD'].corr(df['Eficiencia_Planta_Pct']):.4f}
                    - Operador con mayor capacidad: {df.groupby('Operador')['Capacidad_Instalada_MW'].sum().idxmax()}
                    """
                    
                    insight = consultar_asistente_groq(stats_summary, groq_key)
                    st.markdown("### 📋 Informe Estratégico")
                    st.markdown(insight)

else:
    st.warning("⚠️ Sube el archivo 'energia_renovable.csv' para comenzar.")
