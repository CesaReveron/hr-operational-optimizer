# Mi proyecto final — Plantilla

## 1. Proyecto principal

Título del proyecto:
"HR Operational Optimizer: Modelo de IA para la Predicción de Absentismo Laboral e Impacto Financiero"

¿Qué problema resuelve y a quién le sirve?
Ayuda a Directores de Operaciones (COOs), Directores Financieros (CFOs) y gestores de Recursos Humanos a mitigar y predecir el impacto económico del absentismo laboral no planificado. El modelo analiza el histórico de registros horarios, factores de estrés y datos demográficos para predecir cuántos días de ausencia tendrá un empleado al año, permitiendo a la empresa provisionar personal de retén de forma eficiente y aplicar políticas de bienestar que reduzcan costes operativos.

Dataset principal (enlace):
https://www.kaggle.com/datasets/vjchoudhary7/hr-analytics-case-study

¿Por qué este dataset? (tamaño, calidad, columnas interesantes):
Es un set de datos excelente de más de 4.000 registros dividido en tablas relacionales (`general_data.csv`, `employee_survey_data.csv`, `manager_survey_data.csv` y archivos de control de tiempos `in_time`/`out_time`). Su riqueza radica en que exige realizar un proceso ETL avanzado de unificación (*merge*) de datos y permite calcular, a partir de las marcas horarias brutas, variables críticas de negocio como la media de horas extra, retrasos recurrentes y días de ausencia reales (absentismo) por empleado.

Tipo de Machine Learning:
- [X] Regresión
- [ ] Clasificación
- [ ] Series temporales
- [ ] Clustering
- [ ] Visión / YOLO
- [ ] Otro: ______

¿Qué va a predecir o hacer el modelo?
Predecirá de forma numérica la cantidad estimada de días de absentismo o ausencias no planificadas al año de un empleado (Regresión) basándose en su entorno laboral, volumen de horas extra y encuestas de satisfacción.

## 2. Plan B (otro proyecto distinto)

Título del proyecto plan B:
"FairPay Audit: Sistema Inteligente de Auditoría Salarial y Detección de Anomalías Retributivas"

Dataset plan B (enlace):
https://www.kaggle.com/datasets/vjchoudhary7/hr-analytics-case-study

Tipo de ML del plan B:
Regresión / Detección de Anomalías (Utilizar el mismo ecosistema de datos pero entrenando al modelo para predecir el salario justo de un empleado según sus méritos puros —desempeño, experiencia, educación—, identificando mediante desviaciones estadísticas a los empleados infravalorados en riesgo de fuga).

## 3. Producto / Presentación

Herramienta elegida:
- [X] Streamlit
- [ ] PowerBI

Extras opcionales (solo si aportan):
- [ ] RAG / chatbot sobre mis datos
- [ ] Ollama (LLM local)
- [ ] Mapas interactivos
- [X] Otro: Simulador de Impacto Financiero "What-If" (Una sección interactiva en Streamlit donde el cliente introduce el coste por día de un trabajador sustituto y, moviendo sliders como las horas extra o la flexibilidad horaria, la IA calcula en tiempo real cuántos días de absentismo se reducen y cuánto dinero ahorra la empresa trimestralmente).

## 4. Repositorio

URL de GitHub: https://github.com/_______________________