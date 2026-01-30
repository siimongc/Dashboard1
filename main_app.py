import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(page_title="EDA Energía Renovable", layout="wide")

# Título principal
st.title("📊 Análisis Exploratorio de Datos: Sector Energético")
st.markdown("Esta aplicación permite analizar la capacidad, generación y eficiencia de proyectos de energía renovable.")

# Función para cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv("energia_renovable.csv")
    df['Fecha_Entrada_Operacion'] = pd.to_datetime(df['Fecha_Entrada_Operacion'])
    return df

try:
    df = load_data()

    # --- SIDEBAR: FILTROS ---
    st.sidebar.header("Filtros de Análisis")
    
    tecnologia = st.sidebar.multiselect(
        "Selecciona Tecnología:",
        options=df["Tecnologia"].unique(),
        default=df["Tecnologia"].unique()
    )

    operador = st.sidebar.multiselect(
        "Selecciona Operador:",
        options=df["Operador"].unique(),
        default=df["Operador"].unique()
    )

    # Aplicar filtros
    df_selection = df.query(
        "Tecnologia == @tecnologia & Operador == @operador"
    )

    # --- MÉTRICAS CLAVE (KPIs) ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Proyectos", f"{df_selection.shape[0]}")
    with col2:
        st.metric("Capacidad Total", f"{df_selection['Capacidad_Instalada_MW'].sum():,.1f} MW")
    with col3:
        st.metric("Generación Promedio", f"{df_selection['Generacion_Diaria_MWh'].mean():,.1f} MWh")
    with col4:
        st.metric("Inversión Total", f"${df_selection['Inversion_Inicial_MUSD'].sum():,.1f} MUSD")

    st.divider()

    # --- VISUALIZACIONES ---
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.subheader("Capacidad Instalada por Tecnología")
        fig_cap = px.bar(
            df_selection.groupby("Tecnologia")["Capacidad_Instalada_MW"].sum().reset_index(),
            x="Tecnologia",
            y="Capacidad_Instalada_MW",
            color="Tecnologia",
            template="plotly_white"
        )
        st.plotly_chart(fig_cap, use_container_width=True)

    with row1_col2:
        st.subheader("Estado Actual de los Proyectos")
        fig_pie = px.pie(
            df_selection, 
            names='Estado_Actual', 
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.subheader("Relación: Inversión vs Eficiencia")
        fig_scatter = px.scatter(
            df_selection,
            x="Inversion_Inicial_MUSD",
            y="Eficiencia_Planta_Pct",
            color="Tecnologia",
            size="Capacidad_Instalada_MW",
            hover_name="ID_Proyecto",
            trendline="ols" # Requiere statsmodels
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with row2_col2:
        st.subheader("Distribución de Generación Diaria")
        fig_box = px.box(
            df_selection,
            x="Tecnologia",
            y="Generacion_Diaria_MWh",
            color="Tecnologia"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # --- TABLA DE DATOS ---
    with st.expander("Ver conjunto de datos filtrado"):
        st.dataframe(df_selection, use_container_width=True)

except Exception as e:
    st.error(f"Error al cargar el archivo: {e}")
    st.info("Asegúrate de que el archivo 'energia_renovable.csv' esté en la misma carpeta que este script.")
