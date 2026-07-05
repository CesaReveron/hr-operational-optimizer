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
    columnas_modelo = joblib.load('columnas_modelo.pkl')
    return modelo, escalador, columnas_modelo

try:
    modelo, escalador, columnas_modelo = load_models()
    st.sidebar.success("✅ IA Conectada con Éxito (Regresión Real)")
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
# MÓDULO PRINCIPAL 1: PREDICCIÓN DE DÍAS REAL DE LA IA
# ==========================================
st.subheader("🔮 Diagnóstico Predictivo del Empleado")
st.write("Evalúa el perfil del empleado mediante el modelo predictivo de regresión lineal entrenado.")

if st.button("Evaluar Riesgo de Absentismo"):
    try:
        # 1. Creamos el perfil base usando los PROMEDIOS CORPORATIVOS del escalador.
        # Esto garantiza que las 40 columnas tengan valores estables y matemáticamente lógicos.
        datos_base = escalador.mean_.copy().reshape(1, -1)
        df_entrada = pd.DataFrame(datos_base, columns=columnas_modelo)
        
        # 2. Inyección directa y segura usando las llaves exactas mapeadas de tu archivo .pkl
        df_entrada['Media_Horas_Diarias'] = media_horas
        df_entrada['MonthlyIncome'] = ingreso_mensual
        df_entrada['YearsSinceLastPromotion'] = anos_ascenso
        df_entrada['JobSatisfaction'] = satisfaccion
            
        # 3. Escalamos el vector de características de forma idéntica al entrenamiento
        datos_escalados = escalador.transform(df_entrada)
        
        # 4. PREDECIMOS LOS DÍAS EXACTOS CON LA IA DE REGRESIÓN
        dias_predichos = modelo.predict(datos_escalados)[0]
        
        # Acotamos por seguridad matemática al rango real del dataset original (0 a 24 días)
        dias_predichos = np.clip(dias_predichos, 0, 24)
        
        # 5. Despliegue de Resultados de Regresión en Pantalla
        col1, col2 = st.columns(2)
        
        with col1:
            # Si supera la media lógica de ausencias del corporativo (12.7 días), alertamos operativamente
            if dias_predichos > 13.0:
                st.error(f"⚠️ **ALTO IMPACTO OPERATIVO**\n\nEste perfil presenta una tendencia a registrar ausencias elevadas, estimando un volumen superior al promedio corporativo.")
            else:
                st.success(f"✅ **RANGO OPERATIVO BAJO CONTROL**\n\nEl patrón operativo proyectado para este empleado se mantiene estable dentro de los límites normales.")
                
        with col2:
            st.metric(label="Días de Absentismo Proyectados al Año", value=f"{dias_predichos:.1f} días")
            
    except Exception as error_pred:
        st.error(f"❌ Error en el procesamiento de IA: {str(error_pred)}")

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

st.info("💡 **Conclusión del Analista:** Como los datos demográficos son homogéneos, las políticas corporativas deben enfocarse en la estabilidad operativa (evitando picos extremos en la media de horas diarias) para capturar el retorno económico proyectado.")