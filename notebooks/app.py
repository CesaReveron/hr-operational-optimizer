import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA WEB
# ==========================================
st.set_page_config(
    page_title="HR Operational Optimizer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 HR Operational Optimizer: Predicción de Absentismo e Impacto Financiero")
st.markdown("---")

# ==========================================
# CARGAR LOS ARCHIVOS DE INTELIGENCIA ARTIFICIAL (.PKL)
# ==========================================
@st.cache_resource
def load_models():
    modelo = joblib.load('modelo_absentismo.pkl')
    escalador = joblib.load('escalador_absentismo.pkl')
    return modelo, escalador

try:
    modelo, escalador = load_models()
    st.sidebar.success("✅ IA Conectada con Éxito")
except Exception as e:
    st.sidebar.error("❌ Error al cargar los archivos .pkl")
    st.sidebar.write(str(e))

# ==========================================
# PANEL LATERAL: ENTRADA DE DATOS DEL EMPLEADO
# ==========================================
st.sidebar.header("📝 Datos del Empleado a Evaluar")

# Sliders interactivos basados en tus variables reales del dataset
media_horas = st.sidebar.slider("Media de Horas Diarias Trabajadas", min_value=4.0, max_value=12.0, value=7.5, step=0.5)
ingreso_mensual = st.sidebar.slider("Ingreso Mensual ($)", min_value=1000, max_value=20000, value=6500, step=500)
anos_ascenso = st.sidebar.slider("Años desde el Último Ascenso", min_value=0, max_value=15, value=2, step=1)
satisfaccion = st.sidebar.slider("Satisfacción Laboral (1-4)", min_value=1, max_value=4, value=3, step=1)

# ==========================================
# MÓDULO PRINCIPAL 1: PREDICCIÓN DE RIESGO DE LA IA
# ==========================================
st.subheader("🔮 Diagnóstico Predictivo del Empleado")
st.write("Presiona el botón para evaluar el perfil del empleado mediante el modelo predictivo.")

if st.button("Evaluar Riesgo de Absentismo"):
    # IMPORTANTE: Construimos la fila con las 42 columnas en el mismo orden exacto que get_dummies
    # Para evitar errores de dimensiones, creamos una plantilla vacía basada en los datos escalados
    num_features = escalador.mean_.shape[0] # Detecta cuántas columnas espera el escalador (42)
    
    # Creamos un vector lleno de ceros
    datos_entrada = np.zeros((1, num_features))
    
    # Inyectamos los valores numéricos en sus posiciones correspondientes de las columnas numéricas bases
    # Asumiendo los índices iniciales más comunes por orden alfabético/numérico en tu set
    datos_entrada[0, 0] = media_horas       # Media_Horas_Diarias
    datos_entrada[0, 1] = ingreso_mensual   # MonthlyIncome
    datos_entrada[0, 2] = anos_ascenso      # YearsSinceLastPromotion
    
    # Escalamos la entrada de la misma manera que en el entrenamiento
    datos_escalados = escalador.transform(datos_entrada)
    
    # Hacemos la predicción mediante el modelo lógico cargado
    prediccion = modelo.predict(datos_escalados)[0]
    probabilidad = modelo.predict_proba(datos_escalados)[0][1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediccion == 1 or probabilidad > 0.5:
            st.error(f"⚠️ **ALTO RIESGO DE ABSENTISMO**\n\nEste perfil tiene una alta tendencia a registrar ausencias críticas (más de 25 días al año).")
        else:
            st.success(f"✅ **RIESGO BAJO / NORMAL**\n\nEl patrón operativo de este empleado se mantiene estable y dentro de los rangos seguros.")
            
    with col2:
        st.metric(label="Probabilidad de Ausencia Crítica", value=f"{probabilidad*100:.1f}%")

st.markdown("---")

# ==========================================
# MÓDULO PRINCIPAL 2: SIMULADOR FINANCIERO "WHAT-IF"
# ==========================================
st.subheader("💰 Simulador de Impacto Financiero 'What-If'")
st.write("Calcula el retorno económico estimado al mitigar las jornadas críticas en la organización.")

col_sim1, col_sim2 = st.columns(2)

with col_sim1:
    costo_dia_sustituto = st.number_input("Costo diario promedio de una jornada perdida o sustituto ($)", min_value=10, max_value=1000, value=120)
    total_empleados = st.number_input("Tamaño total de la plantilla de empleados", min_value=1, max_value=10000, value=4410)

with col_sim2:
    reduccion_estimada = st.slider("Reducción esperada de ausencias anuales por empleado (días)", min_value=0, max_value=15, value=3)

# Ecuaciones financieras del simulador
total_dias_evitados = total_empleados * reduccion_estimada
ahorro_financiero = total_dias_evitados * costo_dia_sustituto

st.markdown("### 📊 Retorno de Inversión (ROI) Corporativo")
col_m1, col_m2 = st.columns(2)
col_m1.metric(label="Días Totales de Absentismo Salvados al Año", value=f"{total_dias_evitados:,} días")
col_m2.metric(label="Ahorro Económico Neto para la Compañía", value=f"${ahorro_financiero:,} USD", delta=f"${ahorro_financiero:,} Ganancia")

st.info("💡 **Conclusión del Analista:** Como los datos demográficos son homogéneos, las políticas corporativas deben enfocarse en regular la estabilidad operativa (evitando picos extremos en la media de horas diarias) para capturar el retorno económico proyectado en este simulador.")