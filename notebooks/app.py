import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA WEB
# ==========================================
st.set_page_config(
    page_title="HR Operational Optimizer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 HR Operational Optimizer: Predicción de Absentismo")
st.markdown("---")

# ==========================================
# 2. CARGAR LOS ARCHIVOS DE INTELIGENCIA ARTIFICIAL (.PKL)
# ==========================================
@st.cache_resource
def load_models():
    modelo = joblib.load('modelo_absentismo.pkl')
    escalador = joblib.load('escalador_absentismo.pkl')
    columnas_modelo = joblib.load('columnas_modelo.pkl')
    return modelo, escalador, columnas_modelo

try:
    modelo, escalador, columnas_modelo = load_models()
    st.sidebar.success("✅ IA Conectada con Éxito (Modelo de Regresión)")
except Exception as e:
    st.sidebar.error("❌ Error al cargar los archivos .pkl. Asegúrate de haber ejecutado todo el notebook de modelado primero.")
    st.sidebar.write(str(e))
    st.stop()

# ==========================================
# 3. PANEL LATERAL: FORMULARIO DEL EMPLEADO
# ==========================================
st.sidebar.header("📝 Datos del Empleado a Evaluar")

# --- Variables Numéricas Directas ---
edad = st.sidebar.slider("Edad", 18, 60, 35)
distancia = st.sidebar.slider("Distancia a Casa (km)", 1, 30, 9)
educacion = st.sidebar.slider("Nivel Educativo (1-5)", 1, 5, 3)
nivel_puesto = st.sidebar.slider("Nivel del Puesto (1-5)", 1, 5, 2)
salario = st.sidebar.number_input("Sueldo Mensual ($)", min_value=1000, max_value=250000, value=6500)
num_empresas = st.sidebar.slider("Número de Empresas Anteriores", 0, 9, 2)
aumento_salario = st.sidebar.slider("Porcentaje de Aumento Salarial", 11, 25, 15)
opciones_acciones = st.sidebar.slider("Nivel de Opciones de Acciones (0-3)", 0, 3, 1)
anos_trabajados = st.sidebar.slider("Años Totales Trabajados", 0, 40, 10)
capacitaciones = st.sidebar.slider("Capacitaciones el Año Pasado", 0, 6, 2)
anos_compania = st.sidebar.slider("Años en la Compañía", 0, 40, 5)
anos_ascenso = st.sidebar.slider("Años desde el Último Ascenso", 0, 15, 1)
anos_jefe = st.sidebar.slider("Años con el Jefe Actual", 0, 17, 4)
horas_diarias = st.sidebar.slider("Promedio de Horas Diarias Trabajadas", 4.0, 12.0, 7.5)

# --- Variables de Encuesta (Satisfacción) ---
st.sidebar.subheader("⭐ Encuestas de Satisfacción")
satisfaccion_entorno = st.sidebar.slider("Satisfacción con el Entorno (1-4)", 1, 4, 3)
satisfaccion_trabajo = st.sidebar.slider("Satisfacción con el Trabajo (1-4)", 1, 4, 3)
equilibrio_vida = st.sidebar.slider("Equilibrio Vida-Trabajo (1-4)", 1, 4, 3)
implicacion = st.sidebar.slider("Implicación Laboral (1-4)", 1, 4, 3)
desempenio = st.sidebar.slider("Evaluación de Desempeño (3-4)", 3, 4, 3)

# --- Variables Categóricas (Selectores) ---
st.sidebar.subheader("💼 Datos Organizacionales")
rotacion = st.sidebar.selectbox("¿Tiene riesgo de Rotación?", ["No", "Yes"])
viajes = st.sidebar.selectbox("Frecuencia de Viajes de Negocios", ["Non-Travel", "Travel_Frequently", "Travel_Rarely"])
departamento = st.sidebar.selectbox("Departamento", ["Human Resources", "Research & Development", "Sales"])
campo_educacion = st.sidebar.selectbox("Campo de Educación", ["Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"])
genero = st.sidebar.selectbox("Género", ["Female", "Male"])
rol_puesto = st.sidebar.selectbox("Rol del Puesto", ["Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager", "Manufacturing Director", "Research Director", "Research Scientist", "Sales Executive", "Sales Representative"])
estado_civil = st.sidebar.selectbox("Estado Civil", ["Divorced", "Married", "Single"])

# ==========================================
# 4. PROCESAMIENTO DE DATOS Y PREDICCIÓN
# ==========================================

# Creamos un diccionario base con todas las columnas necesarias en 0
datos_empleado = {col: 0 for col in columnas_modelo}

# Llenamos las variables numéricas directas
datos_empleado['Age'] = edad
datos_empleado['DistanceFromHome'] = distancia
datos_empleado['Education'] = educacion
datos_empleado['JobLevel'] = nivel_puesto
datos_empleado['MonthlyIncome'] = salario
datos_empleado['NumCompaniesWorked'] = num_empresas
datos_empleado['PercentSalaryHike'] = aumento_salario
datos_empleado['StockOptionLevel'] = opciones_acciones
datos_empleado['TotalWorkingYears'] = anos_trabajados
datos_empleado['TrainingTimesLastYear'] = capacitaciones
datos_empleado['YearsAtCompany'] = anos_compania
datos_empleado['YearsSinceLastPromotion'] = anos_ascenso
datos_empleado['YearsWithCurrManager'] = anos_jefe
datos_empleado['Media_Horas_Diarias'] = horas_diarias
datos_empleado['EnvironmentSatisfaction'] = satisfaccion_entorno
datos_empleado['JobSatisfaction'] = satisfaccion_trabajo
datos_empleado['WorkLifeBalance'] = equilibrio_vida
datos_empleado['JobInvolvement'] = implicacion
datos_empleado['PerformanceRating'] = desempenio

# Activamos los flags binarios (One-Hot Encoding manual para que coincida exactamente)
if rotacion == "Yes": datos_empleado['Attrition_Yes'] = 1
if viajes == "Travel_Frequently": datos_empleado['BusinessTravel_Travel_Frequently'] = 1
if viajes == "Travel_Rarely": datos_empleado['BusinessTravel_Travel_Rarely'] = 1
if departamento == "Research & Development": datos_empleado['Department_Research & Development'] = 1
if departamento == "Sales": datos_empleado['Department_Sales'] = 1

if campo_educacion == "Life Sciences": datos_empleado['EducationField_Life Sciences'] = 1
if campo_educacion == "Marketing": datos_empleado['EducationField_Marketing'] = 1
if campo_educacion == "Medical": datos_empleado['EducationField_Medical'] = 1
if campo_educacion == "Other": datos_empleado['EducationField_Other'] = 1
if campo_educacion == "Technical Degree": datos_empleado['EducationField_Technical Degree'] = 1

if genero == "Male": datos_empleado['Gender_Male'] = 1

if rol_puesto == "Human Resources": datos_empleado['JobRole_Human Resources'] = 1
if rol_puesto == "Laboratory Technician": datos_empleado['JobRole_Laboratory Technician'] = 1
if rol_puesto == "Manager": datos_empleado['JobRole_Manager'] = 1
if rol_puesto == "Manufacturing Director": datos_empleado['JobRole_Manufacturing Director'] = 1
if rol_puesto == "Research Director": datos_empleado['JobRole_Research Director'] = 1
if rol_puesto == "Research Scientist": datos_empleado['JobRole_Research Scientist'] = 1
if rol_puesto == "Sales Executive": datos_empleado['JobRole_Sales Executive'] = 1
if rol_puesto == "Sales Representative": datos_empleado['JobRole_Sales Representative'] = 1

if estado_civil == "Married": datos_empleado['MaritalStatus_Married'] = 1
if estado_civil == "Single": datos_empleado['MaritalStatus_Single'] = 1

# Convertir a DataFrame respetando el orden exacto de las columnas de entrenamiento
df_input = pd.DataFrame([datos_empleado])[columnas_modelo]

# Escalado estadístico de los datos ingresados
df_scaled = escalador.transform(df_input)

# Ejecutar la predicción matemática asegurando que nunca muestre días negativos
prediccion_cruda = modelo.predict(df_scaled)[0]
dias_predichos = max(0.0, float(prediccion_cruda))

# ==========================================
# 5. PRESENTACIÓN DE RESULTADOS EN LA WEB
# ==========================================
st.subheader("🎯 Resultado de la Evaluación Predictiva")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Jornadas de Absentismo Estimadas al Año", 
        value=f"{dias_predichos:.1f} días",
        delta=f"Impacto Operativo" if dias_predichos > 10 else "Nivel Estable",
        delta_color="inverse" if dias_predichos > 10 else "normal"
    )

with col2:
    if dias_predichos > 12:
        st.error("⚠️ **Alerta:** Este perfil presenta una tendencia alta a ausentarse. Se sugiere revisar las cargas de trabajo, las horas diarias o sus niveles de satisfacción.")
    else:
        st.success("✅ **Normal:** El nivel de ausencias estimado para este perfil se encuentra dentro del rango aceptable de la organización.")

st.markdown("---")

# ==========================================
# 6. SIMULADOR FINANCIERO "WHAT-IF"
# ==========================================
st.subheader("💰 Simulador de Impacto Financiero 'What-If'")
st.write("Calcula el retorno económico estimado al mitigar las jornadas de ausencia en la organización.")

col_sim1, col_sim2 = st.columns(2)

with col_sim1:
    costo_dia_sustituto = st.number_input("Costo diario promedio de una jornada perdida ($)", min_value=10, max_value=1000, value=120)
    total_empleados = st.number_input("Tamaño total de la plantilla de empleados", min_value=1, max_value=10000, value=4410)

with col_sim2:
    reduccion_estimada = st.slider("Reducción esperada de ausencias anuales por empleado (días)", min_value=0, max_value=15, value=3)

# Ecuaciones financieras del simulador
total_dias_evitados = total_empleados * reduccion_estimada
ahorro_financiero = total_dias_evitados * costo_dia_sustituto

st.markdown("### 📊 Retorno de Inversión (ROI) Corporativo")
col_m1, col_m2 = st.columns(2)
col_m1.metric(label="Días Totales de Absentismo Salvados al Año", value=f"{total_dias_evitados:,} días")
col_m2.metric(label="Ahorro Económico Estimado para la Empresa", value=f"${ahorro_financiero:,}")