import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 1. Configuración de la página
st.set_page_config(
    page_title="EDA - Energía Renovable",
    page_icon="⚡",
    layout="wide"
)

# Estilo personalizado para el título
st.title("⚡ Dashboard de Análisis: Energías Renovables")
st.markdown("""
Esta herramienta realiza un **Análisis Exploratorio de Datos (EDA)** automático sobre conjuntos de datos 
de plantas de energía. Sube tu archivo CSV para comenzar.
""")

# 2. Sidebar para carga de datos y filtros
st.sidebar.header("📂 Carga de Datos")
archivo_subido = st.sidebar.file_uploader("Sube tu archivo .csv", type=["csv"])

if archivo_subido is not None:
    # Función para cargar datos con caché para mejorar el rendimiento
    @st.cache_data
    def cargar_datos(file):
        df = pd.read_csv(file)
        # Convertir fecha si existe la columna
        if 'Fecha_Entrada_Operacion' in df.columns:
            df['Fecha_Entrada_Operacion'] = pd.to_datetime(df['Fecha_Entrada_Operacion'])
        return df

    df = cargar_datos(archivo_subido)

    # --- SECCIÓN DE FILTROS EN SIDEBAR ---
    st.sidebar.divider()
    st.sidebar.subheader("Filtros de Análisis")
    
    # Filtro por Tecnología
    lista_tech = df['Tecnologia'].unique().tolist()
    tech_seleccionada = st.sidebar.multiselect("Selecciona Tecnologías:", lista_tech, default=lista_tech)

    # Filtro por Operador
    lista_op = df['Operador'].unique().tolist()
    op_seleccionado = st.sidebar.multiselect("Selecciona Operadores:", lista_op, default=lista_op)

    # Aplicar filtros al DataFrame
    df_filtrado = df[(df['Tecnologia'].isin(tech_seleccionada)) & (df['Operador'].isin(op_seleccionado))]

    # 3. Métricas Principales (KPIs)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Proyectos", f"{len(df_filtrado)}")
    with col2:
        cap_total = df_filtrado['Capacidad_Instalada_MW'].sum()
        st.metric("Capacidad Total", f"{cap_total:,.2f} MW")
    with col3:
        gen_promedio = df_filtrado['Generacion_Diaria_MWh'].mean()
        st.metric("Gen. Diaria Promedio", f"{gen_promedio:,.2f} MWh")
    with col4:
        inv_total = df_filtrado['Inversion_Inicial_MUSD'].sum()
        st.metric("Inversión Total", f"${inv_total:,.1f} MUSD")

    st.divider()

    # 4. Visualizaciones
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.subheader("Capacidad por Tecnología (MW)")
        fig_bar = px.bar(
            df_filtrado.groupby("Tecnologia")["Capacidad_Instalada_MW"].sum().reset_index(),
            x="Tecnologia",
            y="Capacidad_Instalada_MW",
            color="Tecnologia",
            text_auto='.2s',
            template="plotly_white"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with row1_col2:
        st.subheader("Distribución por Estado de Proyecto")
        fig_pie = px.pie(
            df_filtrado, 
            names='Estado_Actual', 
            hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.subheader("Inversión vs. Eficiencia")
        fig_scatter = px.scatter(
            df_filtrado,
            x="Inversion_Inicial_MUSD",
            y="Eficiencia_Planta_Pct",
            color="Tecnologia",
            size="Capacidad_Instalada_MW",
            hover_name="ID_Proyecto",
            marginal_x="box"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with row2_col2:
        st.subheader("Análisis de Eficiencia por Operador")
        fig_box = px.box(
            df_filtrado,
            x="Operador",
            y="Eficiencia_Planta_Pct",
            color="Operador"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # 5. Tabla de Datos
    with st.expander("🔍 Explorar Datos Filtrados"):
        st.dataframe(df_filtrado, use_container_width=True)
        st.download_button(
            label="Descargar CSV Filtrado",
            data=df_filtrado.to_csv(index=False),
            file_name="datos_filtrados.csv",
            mime="text/csv"
        )

else:
    # Pantalla de bienvenida si no hay archivo
    st.info("💡 Por favor, carga el archivo 'energia_renovable.csv' en el panel de la izquierda para visualizar el análisis.")
    
    # Ejemplo de estructura esperada
    with st.expander("Estructura requerida del CSV"):
        st.write("""
        El archivo debe contener al menos las siguientes columnas:
        - `Tecnologia` (Solar, Eólica, etc.)
        - `Operador` (Nombre de la empresa)
        - `Capacidad_Instalada_MW` (Numérico)
        - `Generacion_Diaria_MWh` (Numérico)
        - `Eficiencia_Planta_Pct` (Numérico)
        - `Inversion_Inicial_MUSD` (Numérico)
        - `Estado_Actual` (Operación, Mantenimiento, etc.)
        """)
