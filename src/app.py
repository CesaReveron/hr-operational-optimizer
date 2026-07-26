import streamlit as st
import os
import pandas as pd
import numpy as np
import joblib


# ==========================================
# CONFIGURACIÓN DE LA PLATAFORMA (UI/UX)
# ==========================================
st.set_page_config(
    page_title="HR Operational Optimizer",
    page_icon="📊",
    layout="wide"
)

# Estilos CSS Avanzados para emular una interfaz SaaS Premium limpia y estructurada
st.markdown("""
    <style>
    /* Fondo e interfaz general */
    .stApp {
        background-color: #0F172A;
        color: #E2E8F0;
    }
    
    /* Panel Lateral (Sidebar) con tono pizarra oscuro integrado */
    [data-testid="stSidebar"] {
        background-color: #0B0F19 !important;
        border-right: 1px solid #1E293B !important;
    }
    
    /* Títulos y textos tipográficos */
    h1, h2, h3, h4, h5, h6, label, .stSlider p {
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif;
    }
    
    .subtitle-text {
        color: #94A3B8 !important;
        font-size: 15px;
        margin-top: -15px;
        margin-bottom: 25px;
    }

    /* Diseño moderno de las Pestañas (Tabs) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0B0F19;
        padding: 6px;
        border-radius: 8px;
        border: 1px solid #1E293B;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #94A3B8 !important;
        padding: 10px 20px;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #FFFFFF !important;
        background-color: #1E293B;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
    }

    /* Botón Premium de Acción */
    .stButton>button {
        background: linear-gradient(135deg, #38BDF8 0%, #2563EB 100%) !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 14px 28px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 12px 0 rgba(37, 99, 235, 0.2) !important;
        transition: all 0.25s ease !important;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px 0 rgba(37, 99, 235, 0.4) !important;
    }
    
    /* Bloques de KPI y Reportes Finales */
    .metric-container {
        background: radial-gradient(circle at top left, #1E293B, #0F172A);
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value-large {
        font-size: 42px;
        color: #FFFFFF;
        font-weight: 800;
        text-shadow: 0 0 10px rgba(56, 189, 248, 0.2);
    }
    
    /* Alertas customizadas corporativas */
    .alert-box {
        padding: 16px 20px;
        border-radius: 8px;
        font-size: 14px;
        line-height: 1.5;
        margin-top: 15px;
    }
    .alert-high {
        background-color: rgba(239, 68, 68, 0.1);
        border: 1px solid #EF4444;
        color: #F87171;
    }
    .alert-normal {
        background-color: rgba(16, 185, 129, 0.1);
        border: 1px solid #10B981;
        color: #34D399;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# CONEXIÓN SEGURA DE MODELOS DE IA
# ==========================================
@st.cache_resource
def inicializar_ia():
    # 1. Detectar la ubicación de app.py (carpeta 'src') y subir un nivel a la raíz
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SRC_DIR)
    
    # 2. Construir las rutas exactas apuntando a la carpeta 'models' en la raíz
    ruta_modelo = os.path.join(BASE_DIR, 'models', 'modelo_absentismo.pkl')
    ruta_escalador = os.path.join(BASE_DIR, 'models', 'escalador_absentismo.pkl')
    ruta_columnas = os.path.join(BASE_DIR, 'models', 'columnas_modelo.pkl')
    
    # 3. Cargar los modelos usando las rutas dinámicas calculadas arriba
    modelo = joblib.load(ruta_modelo)
    escalador = joblib.load(ruta_escalador)
    columnas_modelo = joblib.load(ruta_columnas)
    
    return modelo, escalador, columnas_modelo

try:
    modelo, escalador, columnas_modelo = inicializar_ia()
except Exception as e:
    st.error("❌ Archivos de IA no detectados en el servidor.")
    st.stop()

# ==========================================
# SIDEBAR IZQUIERDO: VARIABLES CORE
# ==========================================
st.sidebar.markdown("<h3 style='color: #38BDF8; font-weight: 800; margin-bottom: 5px;'>👨‍💼 Perfil Esencial</h3>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: #64748B; font-size: 12px; margin-bottom: 20px;'>Filtros demográficos primarios.</p>", unsafe_allow_html=True)

edad = st.sidebar.slider("Edad Cronológica", 18, 60, 35)
salario = st.sidebar.slider("Sueldo Mensual Base ($)", 1000, 20000, 6500, step=250)
horas_diarias = st.sidebar.slider("Jornada Diaria Promedio", 4.0, 12.0, 7.5, step=0.5)
anos_compania = st.sidebar.slider("Años en la Organización", 0, 40, 5)

st.sidebar.markdown("<br><hr style='border-color: #1E293B;'><br>", unsafe_allow_html=True)
st.sidebar.markdown("<h4 style='color: #FFFFFF; font-weight: 700;'>Status Contractual</h4>", unsafe_allow_html=True)
estado_civil = st.sidebar.selectbox("Estado Civil", ["Married", "Single", "Divorced"])
genero = st.sidebar.selectbox("Género", ["Male", "Female"])
rotacion = st.sidebar.selectbox("Riesgo Crítico de Rotación", ["No", "Yes"])

# ==========================================
# PANEL CENTRAL: WORKSPACE PRINCIPAL
# ==========================================
st.markdown("<h1>📊 HR OPERATIONAL OPTIMIZER</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Consola Ejecutiva de Inteligencia Artificial para el Control de Absentismo</div>", unsafe_allow_html=True)

st.markdown("<h3 style='font-weight: 700; margin-bottom: 15px; font-size: 18px;'>⚙️ Configuración Modular del Colaborador</h3>", unsafe_allow_html=True)

# Inicializamos las pestañas lógicas para segmentar la información ordenadamente
tab_corp, tab_career, tab_survey = st.tabs([
    "🏢 Estructura Organizacional", 
    "📈 Evolución Profesional", 
    "🧠 Clima Laboral & Evaluaciones"
])

# CONTENIDO TAB 1: ESTRUCTURA ORGANIZACIONAL
with tab_corp:
    st.markdown("<p style='color: #94A3B8; font-size: 13px;'>Especifique los datos geográficos y organizacionales asignados a la plaza.</p>", unsafe_allow_html=True)
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        departamento = st.selectbox("Departamento Técnico", ["Research & Development", "Sales", "Human Resources"])
        rol_puesto = st.selectbox("Rol Corporativo Ejecutado", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Research Director", "Sales Representative", "Human Resources"])
        campo_educacion = st.selectbox("Área de Especialización Académica", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
    with col_c2:
        # SE CORRIGEN LOS ÍTEMS QUITANDO LOS GUIONES BAJOS PARA LA VISTA DEL USUARIO
        viajes = st.selectbox("Frecuencia de Viajes de Negocio", ["Travel Rarely", "Travel Frequently", "Non-Travel"])
        distancia = st.slider("Distancia a Oficina (km)", 1, 30, 9)
        educacion = st.slider("Nivel de Estudios Acreditados (1-5)", 1, 5, 3)

# CONTENIDO TAB 2: EVOLUCIÓN PROFESIONAL
with tab_career:
    st.markdown("<p style='color: #94A3B8; font-size: 13px;'>Historial acumulado, compensación y línea de tiempo del empleado.</p>", unsafe_allow_html=True)
    col_ca1, col_ca2 = st.columns(2)
    with col_ca1:
        anos_trabajados = st.slider("Vida Laboral Total (Años)", 0, 40, 10)
        nivel_puesto = st.slider("Rango de Jerarquía de la Plaza (1-5)", 1, 5, 2)
        num_empresas = st.slider("Compañías Laboradas Anteriormente", 0, 9, 2)
    with col_ca2:
        aumento_salario = st.slider("Último Incremento Salarial (%)", 11, 25, 15)
        anos_ascenso = st.slider("Años transcurridos desde Último Ascenso", 0, 15, 1)
        anos_jefe = st.slider("Años bajo el Mismo Líder Directo", 0, 17, 4)
        opciones_acciones = st.selectbox("Nivel de Stock Options (Acciones)", [0, 1, 2, 3], index=1)

# CONTENIDO TAB 3: CLIMA LABORAL & EVALUACIONES
with tab_survey:
    st.markdown("<p style='color: #94A3B8; font-size: 13px;'>Indicadores psicológicos y de desempeño recopilados en las encuestas internas anuales.</p>", unsafe_allow_html=True)
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        satisfaccion_entorno = col_s1.slider("Percepción del Clima / Entorno (1-4)", 1, 4, 3)
        satisfaccion_trabajo = col_s1.slider("Satisfacción con el Rol / Funciones (1-4)", 1, 4, 3)
        equilibrio_vida = col_s1.slider("Balance Vida - Trabajo (1-4)", 1, 4, 3)
    with col_s2:
        implicacion = col_s2.slider("Grado de Engagement / Implicación (1-4)", 1, 4, 3)
        desempenio = col_s2.slider("Score de Desempeño Técnico Evaluado (3-4)", 3, 4, 3)
        capacitaciones = col_s2.slider("Capacitaciones Cursadas el Año Pasado", 0, 6, 2)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# INFERENCIA Y CÁLCULO DE RESULTADOS
# ==========================================
col_btn, _ = st.columns([2, 2])
with col_btn:
    ejecutar = st.button("🔮 EJECUTAR DIAGNÓSTICO PREDICTIVO")

if ejecutar:
    # Mapeo estructurado idéntico contra las 40 columnas matemáticas del modelo
    datos = {col: 0 for col in columnas_modelo}
    
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

    # One-Hot Encoding Manual Robusto
    if rotacion == "Yes": datos['Attrition_Yes'] = 1
    
    # TRADUCCIÓN INTERNA DE LOS COMPORTAMIENTOS SIN INTERFERIR CON LA VISTA LIMPIA DE LA INTERFAZ
    if viajes == "Travel Frequently": datos['BusinessTravel_Travel_Frequently'] = 1
    if viajes == "Travel Rarely": datos['BusinessTravel_Travel_Rarely'] = 1
    
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

    # Inferencia Estadística
    df_input = pd.DataFrame([datos])[columnas_modelo]
    df_scaled = escalador.transform(df_input)
    prediccion_cruda = modelo.predict(df_scaled)[0]
    dias_predichos = max(0.0, float(prediccion_cruda))

    # Despliegue Visual de Resultados (Layout Limpio)
    st.markdown("<hr style='border-color: #1E293B;'><br>", unsafe_allow_html=True)
    
    col_res1, col_res2 = st.columns([2, 3])
    
    with col_res1:
        st.markdown(f"""
            <div class="metric-container">
                <p style="color: #38BDF8; font-size: 13px; text-transform: uppercase; font-weight: 700; letter-spacing: 1px; margin-bottom: 5px;">Ausencia Anual Estimada</p>
                <div class="metric-value-large">{dias_predichos:.1f} días</div>
                <p style="color: #64748B; font-size: 12px; margin-top: 8px;">Margen de error promedio del modelo: ± 4.7 días</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col_res2:
        if dias_predichos > 12:
            st.markdown("""
                <div class="alert-box alert-high">
                    <strong>⚠️ PERFIL DE RIESGO RELATIVO ALTO:</strong> Este perfil se ubica en el grupo con mayor propensión estimada dentro de la plantilla. Recomendado como candidato a revisión preventiva, no como diagnóstico definitivo.
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="alert-box alert-normal">
                    <strong>✅ RANGO NORMAL COMPROBADO:</strong> Las métricas estimadas reflejan un comportamiento alineado a los estándares estables y saludables corporativos de la organización.
                </div>
            """, unsafe_allow_html=True)
            
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("ℹ️ Nota metodológica: ¿qué tan preciso es este modelo?"):
        st.markdown("""
        Este modelo fue evaluado con **MAE ≈ 4.7 días** y **R² ≈ 0.05** sobre datos de test,
        tras comparar 5 algoritmos distintos (Regresión Lineal, Random Forest, Gradient Boosting,
        HistGradientBoosting y Poisson Regressor) — todos convergieron a un rendimiento similar.

        Esto significa que las variables demográficas, salariales y de encuesta disponibles
        **no predicen con alta precisión el número exacto de días de absentismo**. Es un hallazgo
        legítimo, no una limitación oculta: sugiere que las causas reales del absentismo (salud,
        situaciones personales) no están capturadas en este tipo de datos.

        **Recomendación de uso:** interpretar la predicción como un **ordenador de riesgo relativo**
        entre empleados (para priorizar dónde mirar primero), no como un número exacto de días a
        planificar.
        """)

# ==========================================
# SECCIÓN FINANCIERA: SIMULADOR "WHAT-IF"
# ==========================================
st.markdown("<br><br><hr style='border-color: #1E293B;'><br>", unsafe_allow_html=True)
st.markdown("<h3 style='font-weight: 800; color: #FFFFFF;'>💰 Simulador de Impacto Económico 'What-If'</h3>", unsafe_allow_html=True)
st.markdown("<p style='color: #94A3B8; font-size: 14px;'>Dimensione el impacto financiero global al mitigar jornadas críticas a nivel organizacional.</p>", unsafe_allow_html=True)

with st.container(border=True):
    col_sim1, col_sim2, col_sim3 = st.columns(3)
    with col_sim1:
        costo_dia_sustituto = st.number_input("Costo diario promedio / sustituto ($)", min_value=10, max_value=1000, value=120)
    with col_sim2:
        total_empleados = st.number_input("Tamaño total de la plantilla", min_value=1, max_value=10000, value=4410)
    with col_sim3:
        reduccion_estimada = st.slider("Reducción de ausencias esperada (días/año)", min_value=0, max_value=15, value=3)

    # Ecuaciones financieras del simulador
    total_dias_evitados = total_empleados * reduccion_estimada
    ahorro_financiero = total_dias_evitados * costo_dia_sustituto

    # Despliegue de los retornos financieros de forma ejecutiva
    st.markdown("<br>", unsafe_allow_html=True)
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown(f"""
            <div class="metric-container" style="border-left: 5px solid #38BDF8;">
                <p style="color: #94A3B8; font-size: 13px; margin-bottom: 5px;">Jornadas Laborales Recuperadas</p>
                <div style="font-size: 32px; font-weight: 800; color: #FFFFFF;">{total_dias_evitados:,} días</div>
            </div>
        """, unsafe_allow_html=True)
    with col_f2:
        st.markdown(f"""
            <div class="metric-container" style="border-left: 5px solid #10B981;">
                <p style="color: #94A3B8; font-size: 13px; margin-bottom: 5px;">Ahorro Anual Estimado (ROI)</p>
                <div style="font-size: 32px; font-weight: 800; color: #10B981;">${ahorro_financiero:,} USD</div>
            </div>
        """, unsafe_allow_html=True)