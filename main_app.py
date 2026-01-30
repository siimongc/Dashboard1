import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 1. Configuración de la página
st.set_page_config(
    page_title="EDA Energías Renovables",
    page_icon="🌱",
    layout="wide"
)

# Estilos CSS personalizados para mejorar la apariencia
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("⚡ Dashboard de Análisis: Sector de Energía Renovable")
st.markdown("Carga tu archivo CSV para desglosar el análisis en componentes cualitativos, cuantitativos y visuales.")

# --- BARRA LATERAL (SIDEBAR) ---
st.sidebar.header("📂 Configuración de Datos")
archivo_subido = st.sidebar.file_uploader("Sube tu archivo .csv aquí", type=["csv"])

if archivo_subido is not None:
    # Función de carga con caché
    @st.cache_data
    def cargar_datos(file):
        df = pd.read_csv(file)
        # Limpieza básica: convertir fechas
        if 'Fecha_Entrada_Operacion' in df.columns:
            df['Fecha_Entrada_Operacion'] = pd.to_datetime(df['Fecha_Entrada_Operacion'])
        return df

    df = cargar_datos(archivo_subido)

    # Filtros Dinámicos
    st.sidebar.divider()
    st.sidebar.subheader("Filtros Globales")
    
    # Filtro de Tecnología
    tecnologias = sorted(df['Tecnologia'].unique())
    tech_sel = st.sidebar.multiselect("Tecnologías:", tecnologias, default=tecnologias)

    # Filtro de Operadores
    operadores = sorted(df['Operador'].unique())
    op_sel = st.sidebar.multiselect("Operadores:", operadores, default=operadores)

    # Filtrado del DataFrame
    df_filtrado = df[(df['Tecnologia'].isin(tech_sel)) & (df['Operador'].isin(op_sel))]

    # --- ORGANIZACIÓN EN TABS (BLOQUES) ---
    tab_cual, tab_cuan, tab_graf = st.tabs(["📊 Bloque Cualitativo", "🔢 Bloque Cuantitativo", "📈 Bloque Gráfico"])

    # --- 1. BLOQUE CUALITATIVO (Variables Categóricas) ---
    with tab_cual:
        st.header("Análisis de Variables Cualitativas")
        st.info("Este bloque analiza las categorías, estados y distribución de los proyectos.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribución por Tecnología")
            conteo_tech = df_filtrado['Tecnologia'].value_counts().reset_index()
            conteo_tech.columns = ['Tecnología', 'Cantidad de Proyectos']
            st.table(conteo_tech)

        with col2:
            st.subheader("Estado Actual de Proyectos")
            conteo_estado = df_filtrado['Estado_Actual'].value_counts().reset_index()
            conteo_estado.columns = ['Estado', 'Frecuencia']
            st.dataframe(conteo_estado, use_container_width=True)

        st.divider()
        st.subheader("Exploración de Operadores")
        st.write(f"Se identifican **{df_filtrado['Operador'].nunique()}** operadores activos en la selección actual.")
        st.dataframe(df_filtrado[['ID_Proyecto', 'Tecnologia', 'Operador', 'Estado_Actual', 'Conectado_SIN']].head(15), use_container_width=True)

    # --- 2. BLOQUE CUANTITATIVO (Números y Estadísticas) ---
    with tab_cuan:
        st.header("Análisis Numérico y Descriptivo")
        
        # KPIs Superiores
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Capacidad Total", f"{df_filtrado['Capacidad_Instalada_MW'].sum():,.0f} MW")
        kpi2.metric("Inversión Total", f"${df_filtrado['Inversion_Inicial_MUSD'].sum():,.1f} M")
        kpi3.metric("Eficiencia Promedio", f"{df_filtrado['Eficiencia_Planta_Pct'].mean():,.1f}%")
        kpi4.metric("Generación Media", f"{df_filtrado['Generacion_Diaria_MWh'].mean():,.1f} MWh")

        st.divider()
        
        st.subheader("Resumen Estadístico")
        # Seleccionamos solo columnas numéricas para el resumen
        resumen = df_filtrado.describe().T
        st.dataframe(resumen.style.format("{:,.2f}"), use_container_width=True)

        st.subheader("Análisis de Correlación")
        # Matriz de correlación para entender relaciones entre inversión, capacidad y eficiencia
        corr = df_filtrado.select_dtypes(include=['number']).corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='Viridis', title="Correlación entre Variables")
        st.plotly_chart(fig_corr, use_container_width=True)

    # --- 3. BLOQUE GRÁFICO (Visualizaciones Avanzadas) ---
    with tab_graf:
        st.header("Visualización Dinámica de Datos")
        
        tipo_grafico = st.radio(
            "Selecciona el tipo de análisis visual:",
            ["Relación Inversión vs Generación", "Composición de Capacidad", "Distribución de Eficiencia"],
            horizontal=True
        )

        if tipo_grafico == "Relación Inversión vs Generación":
            fig = px.scatter(
                df_filtrado, 
                x="Inversion_Inicial_MUSD", 
                y="Generacion_Diaria_MWh",
                color="Tecnologia",
                size="Capacidad_Instalada_MW",
                hover_name="ID_Proyecto",
                trendline="ols",
                title="¿Influye la inversión en la generación diaria?"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif tipo_grafico == "Composición de Capacidad":
            fig = px.sunburst(
                df_filtrado, 
                path=['Tecnologia', 'Estado_Actual'], 
                values='Capacidad_Instalada_MW',
                title="Jerarquía de Capacidad por Tecnología y Estado"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif tipo_grafico == "Distribución de Eficiencia":
            fig = px.box(
                df_filtrado, 
                x="Tecnologia", 
                y="Eficiencia_Planta_Pct", 
                color="Tecnologia",
                points="all",
                title="Variabilidad de Eficiencia por Fuente de Energía"
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    # Estado inicial: Sin archivo
    st.info("👋 ¡Bienvenido! Por favor, usa la barra lateral para subir tu archivo `energia_renovable.csv`.")
    st.image("https://images.unsplash.com/photo-1473341304170-971dccb5ac1e?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", 
             caption="Análisis de Energía Limpia para un futuro sostenible")
