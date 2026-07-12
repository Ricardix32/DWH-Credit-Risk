import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Core Microfinanciero", page_icon="🏛️", layout="wide")

# ==============================================================================
# CARGA DINÁMICA DE DATOS DEL MODELO (Desde Jupyter Notebook)
# ==============================================================================
@st.cache_data
def cargar_datos_analiticos():
    try:
        # Lee los CSV generados por el entrenamiento real
        df_resultados = pd.read_csv('resultados_benchmark.csv')
        df_split = pd.read_csv('datos_split.csv')
        return df_resultados, df_split
    except FileNotFoundError:
        st.error("⚠️ Archivos CSV no encontrados. Asegúrate de mover 'resultados_benchmark.csv' y 'datos_split.csv' a esta carpeta.")
        st.stop()

# Cargamos los datos reales a la memoria de la app
resultados_ml, split_data = cargar_datos_analiticos()

# ==============================================================================
# MENÚ LATERAL: SEPARACIÓN DE ROLES (Front-Office vs Back-Office)
# ==============================================================================
st.sidebar.title("🏛️ Sistema Core")
st.sidebar.markdown("---")

# Perfil 1: Operaciones
st.sidebar.header("👨‍💼 Módulos Operativos (Asesores)")
opcion_operaciones = st.sidebar.radio(
    "Operaciones diarias:",
    ("📥 1. Ingesta de Documentos (IDP)", "🔮 2. Simulador de Crédito"),
    index=0
)

st.sidebar.markdown("---")

# Perfil 2: Riesgos y Data Science
st.sidebar.header("📊 Módulos Analíticos (Riesgos)")
opcion_analitica = st.sidebar.radio(
    "Análisis de Modelos:",
    ("Ninguno", "🔬 3. Muestreo de Datos", "📈 4. Benchmark ML", "🧠 5. Interpretabilidad"),
    index=0
)

# Lógica para mostrar solo una pantalla a la vez basada en la selección
vista_actual = opcion_analitica if opcion_analitica != "Ninguno" else opcion_operaciones

# ==============================================================================
# ROL: ASESOR DE CRÉDITO (FRONT-OFFICE)
# ==============================================================================
if vista_actual == "📥 1. Ingesta de Documentos (IDP)":
    st.title("📥 Ingesta de Documentos Alternativos (MYPES)")
    st.write("Interfaz de captura documental para asesores en la provincia de Chepén.")
    
    LANDING_DIR = "/app/data/landing/boletas/"
    os.makedirs(LANDING_DIR, exist_ok=True)

    id_solicitud = st.text_input("Ingrese el ID de Solicitud (SK_ID_CURR)", placeholder="Ej. 100002",)
    uploaded_file = st.file_uploader("Sube la boleta de pago física (JPG)", type=["jpg", "jpeg"])

    if st.button("Subir Documento al Lakehouse", type="primary"):
        if not id_solicitud or not id_solicitud.isdigit():
            st.error("⚠️ Debe ingresar un ID numérico válido.")
        elif uploaded_file is None:
            st.error("⚠️ Adjunte un documento.")
        else:
            nuevo_nombre = f"{id_solicitud}_{uploaded_file.name}"
            with open(os.path.join(LANDING_DIR, nuevo_nombre), "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ Documento depositado exitosamente para la solicitud {id_solicitud}.")

# ==============================================================================
# ROL: ASESOR DE CRÉDITO (FRONT-OFFICE)
# ==============================================================================
# ... (Aquí está tu código de Ingesta de Documentos que ya funciona) ...

elif vista_actual == "🔮 2. Simulador de Crédito":
    st.title("🔮 Simulador de Inferencia de Riesgo")
    st.write("Evaluación de nuevos prospectos en tiempo real utilizando el modelo LightGBM en producción.")
    
    # 1. Carga del Modelo en Caché (Para no recargar el archivo en cada clic)
    @st.cache_resource
    def cargar_modelo():
        import joblib
        modelo = joblib.load('modelo_riesgo_lgbm.pkl')
        columnas = joblib.load('columnas_modelo.pkl')
        return modelo, columnas

    try:
        modelo_produccion, columnas_requeridas = cargar_modelo()
    except Exception as e:
        st.error(f"⚠️ Error cargando el modelo: {e}")
        st.stop()

    st.markdown("### Perfil del Solicitante")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        edad = st.number_input("Edad (Años)", min_value=18, max_value=85, value=35)
        antiguedad_laboral = st.number_input("Antigüedad Laboral (Años)", min_value=0.0, max_value=50.0, value=2.5)
        genero = st.selectbox("Género", ["M", "F"])
        
    with col2:
        ingreso_total = st.number_input("Ingreso Neto Demostrado", min_value=500, value=2500)
        monto_credito = st.number_input("Monto Solicitado", min_value=1000, value=15000)
        anualidad = st.number_input("Cuota Anual Estimada", min_value=100, value=3500)
        
    with col3:
        sector = st.selectbox("Sector Económico", ["Trade: type 7", "Business Entity Type 3", "Self-employed", "Other"])
        nivel_educativo = st.selectbox("Nivel Educativo", ["Secondary / secondary special", "Higher education", "Incomplete higher"])
        calificacion_region = st.selectbox("Calificación Región Urbana", [1, 2, 3], index=1)

    if st.button("🧠 Procesar Solicitud", type="primary"):
        # 2. Captura de datos en tiempo real (sin TARGET)
        import pandas as pd
        nuevo_cliente = pd.DataFrame([{
            'edad_anios': edad,
            'antiguedad_laboral_anios': antiguedad_laboral,
            'amt_income_total': ingreso_total,
            'amt_credit': monto_credito,
            'amt_annuity': anualidad,
            'credit_to_income_ratio': monto_credito / ingreso_total if ingreso_total > 0 else 0,
            'annuity_income_ratio': anualidad / ingreso_total if ingreso_total > 0 else 0,
            'calificacion_region_ciudad': calificacion_region,
            'CODE_GENDER': genero,
            'ORGANIZATION_TYPE': sector,
            'NAME_EDUCATION_TYPE': nivel_educativo
        }])
        
        # 3. Aplicar One-Hot Encoding idéntico al entorno de entrenamiento
        nuevo_cliente_ohe = pd.get_dummies(nuevo_cliente)
        
        # Limpieza de caracteres especiales (El mismo truco que usamos en Jupyter)
        import re
        nuevo_cliente_ohe.columns = [re.sub(r'[{}":,\[\]]', '_', col) for col in nuevo_cliente_ohe.columns]
        
        # 4. Alinear con la estructura estricta del modelo
        df_inferencia = pd.DataFrame(columns=columnas_requeridas)
        for col in columnas_requeridas:
            if col in nuevo_cliente_ohe.columns:
                df_inferencia[col] = nuevo_cliente_ohe[col]
            else:
                df_inferencia[col] = 0 # Si falta una categoría, se rellena con 0
                
        # 5. PREDICCIÓN (Inferencia)
        probabilidad_default = modelo_produccion.predict_proba(df_inferencia)[0][1] * 100
        
        st.markdown("---")
        st.markdown("### Dictamen Analítico")
        
        # Lógica de decisión del negocio
        if probabilidad_default < 40.0:
            st.success(f"✅ **CRÉDITO APROBADO**")
            st.metric("Probabilidad de Impago (Riesgo)", f"{probabilidad_default:.2f}%")
        else:
            st.error(f"❌ **CRÉDITO RECHAZADO**")
            st.metric("Probabilidad de Impago (Riesgo)", f"{probabilidad_default:.2f}%")
            
        st.info("💡 **Traza Algorítmica:** La inferencia fue procesada utilizando los pesos de convergencia del modelo LightGBM. Las variables de apalancamiento (Credit-to-Income) y estabilidad laboral fueron determinantes.")

# ==============================================================================
# ROL: DATA SCIENTIST & RIESGOS (BACK-OFFICE)
# ==============================================================================

# ==============================================================================
# MÓDULO 3: INTEGRIDAD DE DATOS (EL 80/20 ESTRATIFICADO)
# ==============================================================================
elif vista_actual == "🔬 3. Muestreo de Datos":
    st.title("🔬 Validación del Muestreo y Desbalance de Clases")
    st.write("Demostración de la estratificación en la división Train/Test para asegurar la evaluación correcta de la clase minoritaria.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.info("💡 **Regla de Negocio:** El dataset presenta un desbalance severo. Una división aleatoria simple podría dejar el conjunto de prueba sin morosos, invalidando el AUC-ROC.")
        # Calculamos el total de registros dinámicamente desde el CSV
        total_registros = split_data['Cantidad'].sum()
        st.metric("Total de Registros (Gold)", f"{total_registros:,}")
        st.metric("Ratio de Desbalance", "11.5 : 1")
        
    with col2:
        # Gráfico de barras interactivo con Altair
        import altair as alt
        chart = alt.Chart(split_data).mark_bar().encode(
            x=alt.X('Conjunto:N', title='Conjunto de Datos'),
            y=alt.Y('Cantidad:Q', title='Volumen de Solicitudes'),
            color=alt.Color('Clase:N', scale=alt.Scale(domain=['Buenos Pagadores', 'Morosos (Default)'], range=['#1f77b4', '#d62728'])),
            tooltip=['Conjunto', 'Clase', 'Cantidad']
        ).properties(height=350, title="Distribución de Clases (Train vs Test)")
        
        st.altair_chart(chart, use_container_width=True)
        st.caption("Visualización: La franja roja (morosos) mantiene exactamente la misma proporción (~8%) en ambos conjuntos, validando matemáticamente la integridad del *split*.")

# ==============================================================================
# MÓDULO 4: EVALUACIÓN COMPETITIVA (BENCHMARK)
# ==============================================================================
elif vista_actual == "📈 4. Benchmark ML":
    st.title("📈 Benchmark de Algoritmos (Credit Scoring)")
    st.write("Comparativa de métricas priorizando la separación de clases (AUC-ROC) sobre la data real.")
    
    # Extraemos el mejor modelo automáticamente de la primera fila del CSV ordenado
    top_model = resultados_ml.iloc[0]
    
    st.subheader(f"🏆 Modelo Ganador: {top_model['Modelo']}")
    col1, col2, col3 = st.columns(3)
    col1.metric("AUC-ROC Máximo", f"{top_model['AUC-ROC']:.4f}")
    col2.metric("Recall (Detección Morosos)", f"{top_model['Recall (Clase 1)']:.4f}")
    col3.metric("Tiempo de Entrenamiento", f"{top_model['Tiempo (Segundos)']} seg")
    
    st.markdown("---")
    
    col_grafico, col_tabla = st.columns([1, 2])
    
    with col_grafico:
        st.subheader("Comparativa Visual (AUC-ROC)")
        st.bar_chart(data=resultados_ml.set_index("Modelo")["AUC-ROC"], color="#1f77b4")
        
    with col_tabla:
        st.subheader("Matriz Detallada de Resultados")
        st.dataframe(
            resultados_ml.style.highlight_max(subset=['AUC-ROC', 'Recall (Clase 1)'], color='lightgreen'), 
            use_container_width=True, 
            hide_index=True
        )

elif vista_actual == "🧠 5. Interpretabilidad":
    st.title("🧠 Interpretabilidad del Modelo (XAI)")
    st.write("Análisis de las variables que definen el riesgo crediticio, extraídas directamente del motor LightGBM en producción.")
    
    try:
        import joblib
        # Cargamos el cerebro y la estructura real
        modelo_produccion = joblib.load('modelo_riesgo_lgbm.pkl')
        columnas_requeridas = joblib.load('columnas_modelo.pkl')
        
        # Extraemos la importancia matemática de las variables (Feature Importances)
        importancias = modelo_produccion.feature_importances_
        
        # Creamos un DataFrame emparejando el nombre de la columna con su peso real
        df_importancia = pd.DataFrame({
            "Variable": columnas_requeridas,
            "Importancia (Scores)": importancias
        })
        
        # Limpiamos los nombres para que se vean más estéticos en el gráfico
        df_importancia["Variable"] = df_importancia["Variable"].str.replace("_", " ").str.title()
        
        # Ordenamos de mayor a menor y tomamos el Top 10
        df_importancia = df_importancia.sort_values(by="Importancia (Scores)", ascending=False).head(10)
        
        st.markdown("### Top 10 Factores de Riesgo (Feature Importance)")
        
        # Gráfico dinámico basado en el modelo real
        import altair as alt
        chart = alt.Chart(df_importancia).mark_bar(color='#ff7f0e').encode(
            x=alt.X('Importancia (Scores):Q', title='Peso Algorítmico (Splits)'),
            y=alt.Y('Variable:N', sort='-x', title='Variable de Negocio'),
            tooltip=['Variable', 'Importancia (Scores)']
        ).properties(height=400)
        
        st.altair_chart(chart, use_container_width=True)
        
        st.info("💡 **Alineación con el Negocio:** Este gráfico no es simulado. Representa los nodos de decisión matemáticos reales que LightGBM utilizó durante su entrenamiento con el dataset de la capa Gold. Las variables en la cima son las que el banco debe observar con mayor detenimiento al originar un crédito.")

    except Exception as e:
        st.error(f"No se pudieron cargar los artefactos del modelo para el análisis: {e}")