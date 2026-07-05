import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de página limpia y centrada (Evita el desorden visual)
st.set_page_config(
    page_title="HR Operational Optimizer",
    page_icon="📊",
    layout="centered"
)

st.title("📊 HR Operational Optimizer")
st.markdown("### Control y Predicción de Absentismo Laboral")
st.write("Introduce los datos del empleado y presiona el botón inferior para calcular.")
st.markdown("---")

# Carga segura y automática de los archivos de Inteligencia Artificial
@st.cache_resource
def cargar_artefactos():
    modelo = joblib.load('modelo_absentismo.pkl')
    escalador = joblib.load('escalador_absentismo.pkl')
    columnas = joblib.load('columnas_modelo.pkl')
    return modelo, escalador, columnas

try:
    modelo, escalador, columnas_modelo = cargar_artefactos()
except Exception as e:
    st.error("❌ ERROR: No se encuentran los archivos .pkl en esta carpeta. Recuerda ejecutar el notebook '02_modelado.ipynb' primero.")
    st.stop()

# ==========================================
# 1. FORMULARIO PRINCIPAL (CAMPOS CLAVE)
# ==========================================
st.subheader("📝 Datos del Empleado")

col1, col2 = st.columns(2)

with col1:
    edad = st.number_input("Edad", min_value=18, max_value=60, value=35)
    salario = st.number_input("Sueldo Mensual ($)", min_value=1000, max_value=50000, value=6500)
    horas_diarias = st.number_input("Horas Diarias Promedio", min_value=4.0, max_value=12.0, value=7.5, step=0.5)
    departamento = st.selectbox("Departamento", ["Research & Development", "Sales", "Human Resources"])
    estado_civil = st.selectbox("Estado Civil", ["Married", "Single", "Divorced"])

with col2:
    anos_compania = st.number_input("Años en la Compañía", min_value=0, max_value=40, value=5)
    rol_puesto = st.selectbox("Rol del Puesto", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Research Director", "Sales Representative", "Human Resources"])
    viajes = st.selectbox("Viajes de Negocios", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
    genero = st.selectbox("Género", ["Male", "Female"])
    rotacion = st.selectbox("¿Riesgo de Rotación?", ["No", "Yes"])

# ==========================================
# 2. CAMPOS SECUNDARIOS (OCULTOS PARA NO AGOBIAR)
# ==========================================
with st.expander("⚙️ Modificar Encuestas y Datos Secundarios (Opcional)"):
    c_exp1, c_exp2 = st.columns(2)
    with c_exp1:
        distancia = c_exp1.number_input("Distancia a Casa (km)", 1, 30, 9)
        educacion = c_exp1.number_input("Nivel Educativo (1-5)", 1, 5, 3)
        nivel_puesto = c_exp1.number_input("Nivel del Puesto (1-5)", 1, 5, 2)
        num_empresas = c_exp1.number_input("Empresas Anteriores", 0, 9, 2)
        aumento_salario = c_exp1.number_input("Porcentaje Aumento Salarial", 11, 25, 15)
        opciones_acciones = c_exp1.number_input("Opciones de Acciones (0-3)", 0, 3, 1)
    with c_exp2:
        anos_trabajados = c_exp2.number_input("Años Totales Trabajados", 0, 40, 10)
        capacitaciones = c_exp2.number_input("Capacitaciones Año Pasado", 0, 6, 2)
        anos_ascenso = c_exp2.number_input("Años desde Último Ascenso", 0, 15, 1)
        anos_jefe = c_exp2.number_input("Años con Jefe Actual", 0, 17, 4)
        satisfaccion_entorno = c_exp2.slider("Satisfacción Entorno", 1, 4, 3)
        satisfaccion_trabajo = c_exp2.slider("Satisfacción Trabajo", 1, 4, 3)
        equilibrio_vida = c_exp2.slider("Equilibrio Vida", 1, 4, 3)
        implicacion = c_exp2.slider("Implicación Laboral", 1, 4, 3)
        desempenio = c_exp2.slider("Desempeño", 3, 4, 3)
        campo_educacion = c_exp2.selectbox("Campo de Educación", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])

st.markdown("---")

# ==========================================
# 3. BOTÓN DE EJECUCIÓN Y PROCESAMIENTO
# ==========================================
if st.button("🔮 Calcular Predicción", type="primary", use_container_width=True):
    
    # Creamos la estructura limpia mapeando las 40 columnas requeridas por tu modelo
    datos = {col: 0 for col in columnas_modelo}
    
    # Asignación de variables numéricas
    datos['Age'] = edad
    datos['DistanceFromHome'] = distancia
    datos['Education'] = educacion
    datos['JobLevel'] = nivel_puesto
    datos['MonthlyIncome'] = salario
    datos['NumCompaniesWorked'] = num_empresas
    datos['PercentSalaryHike'] = aumento_salario
    datos['StockOptionLevel'] = opciones_acciones
    datos['TotalWorkingYears'] = anos_trabajados
    datos['TrainingTimesLastYear'] = capacitaciones
    datos['YearsAtCompany'] = anos_compania
    datos['YearsSinceLastPromotion'] = anos_ascenso
    datos['YearsWithCurrManager'] = anos_jefe
    datos['Media_Horas_Diarias'] = horas_diarias
    datos['EnvironmentSatisfaction'] = satisfaccion_entorno
    datos['JobSatisfaction'] = satisfaccion_trabajo
    datos['WorkLifeBalance'] = equilibrio_vida
    datos['JobInvolvement'] = implicacion
    datos['PerformanceRating'] = desempenio

    # Conversión binaria de categorías (One-Hot Encoding Manual Seguro)
    if rotacion == "Yes": datos['Attrition_Yes'] = 1
    if viajes == "Travel_Frequently": datos['BusinessTravel_Travel_Frequently'] = 1
    if viajes == "Travel_Rarely": datos['BusinessTravel_Travel_Rarely'] = 1
    if departamento == "Research & Development": datos['Department_Research & Development'] = 1
    if departamento == "Sales": datos['Department_Sales'] = 1
    if campo_educacion == "Life Sciences": datos['EducationField_Life Sciences'] = 1
    if campo_educacion == "Marketing": datos['EducationField_Marketing'] = 1
    if campo_educacion == "Medical": datos['EducationField_Medical'] = 1
    if campo_educacion == "Other": datos['EducationField_Other'] = 1
    if campo_educacion == "Technical Degree": datos['EducationField_Technical Degree'] = 1
    if genero == "Male": datos['Gender_Male'] = 1
    
    if rol_puesto == "Human Resources": datos['JobRole_Human Resources'] = 1
    if rol_puesto == "Laboratory Technician": datos['JobRole_Laboratory Technician'] = 1
    if rol_puesto == "Manager": datos['JobRole_Manager'] = 1
    if rol_puesto == "Manufacturing Director": datos['JobRole_Manufacturing Director'] = 1
    if rol_puesto == "Research Director": datos['JobRole_Research Director'] = 1
    if rol_puesto == "Research Scientist": datos['JobRole_Research Scientist'] = 1
    if rol_puesto == "Sales Executive": datos['JobRole_Sales Executive'] = 1
    if rol_puesto == "Sales Representative": datos['JobRole_Sales Representative'] = 1
    
    if estado_civil == "Married": datos['MaritalStatus_Married'] = 1
    if estado_civil == "Single": datos['MaritalStatus_Single'] = 1

    # Convertir a DataFrame ordenado y aplicar escalador estadístico
    df_input = pd.DataFrame([datos])[columnas_modelo]
    df_scaled = escalador.transform(df_input)
    
    # Realizar predicción matemática (Imposibilita días negativos)
    prediccion_cruda = modelo.predict(df_scaled)[0]
    dias_predichos = max(0.0, float(prediccion_cruda))

    # ==========================================
    # 4. PRESENTACIÓN DE RESULTADOS
    # ==========================================
    st.markdown("### 🎯 Resultado del Análisis")
    st.metric(label="Jornadas de Absentismo Estimadas al Año", value=f"{dias_predichos:.1f} días")
    
    # FIX FINAL: Condicional limpio que imprime texto de forma segura y obligatoria
    if dias_predichos > 12:
        st.error("Alerta: Este perfil presenta una tendencia alta a ausentarse. Se sugiere revisar las cargas de trabajo y jornadas.")
    else:
        st.success("Normal: El nivel de ausencias estimado para este perfil se encuentra dentro del rango aceptable para la organización.")