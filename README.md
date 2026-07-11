# HR Operational Optimizer 📊

## Sistema de Inteligencia Artificial para la Predicción de Absentismo Laboral e Impacto Financiero

Proyecto final de bootcamp de Data Science. Desarrolla un modelo de **regresión** que estima los días de absentismo laboral de un empleado a partir de variables demográficas, de compensación, trayectoria profesional y clima laboral, con el objetivo de dar visibilidad a **COO y CFO** sobre el coste operativo real del absentismo y permitirles simular el ahorro de reducirlo.

🔗 **Repositorio:** https://github.com/CesaReveron/hr-operational-optimizer

✍️ **Autor:** Cesar Germain Reveron Garcia

---

## 🎯 Problema de negocio

El absentismo no planificado genera un coste indirecto difícil de cuantificar (sustituciones, pérdida de productividad, sobrecarga de equipo). Este proyecto:

- Predice cuántos días de absentismo es esperable que tenga un empleado (regresión, no clasificación).
- Identifica qué factores (salario, antigüedad, satisfacción, balance vida-trabajo, etc.) más influyen.
- Traduce esa predicción a un **impacto económico estimado** mediante un simulador "What-If" pensado para perfiles no técnicos (COO/CFO).

## 🗂️ Estructura del proyecto

```
HR-OPERATIONAL-OPTIMIZER/
├── data/
│   ├── raw/                    # Datos originales (no versionados, ver más abajo)
│   └── processed/              # Dataset limpio generado por el ETL
├── models/                     # Modelo, escalador y lista de columnas (joblib)
├── notebooks/
│   ├── 00_prep.ipynb           # ETL: unión de tablas, cálculo del target, imputación
│   ├── 01_eda.ipynb            # Análisis exploratorio
│   └── 02_modelado.ipynb       # Entrenamiento, evaluación y guardado del modelo
├── src/
│   └── app.py                  # App interactiva en Streamlit
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Pipeline

1. **`00_prep.ipynb` (ETL):** carga `general_data`, `employee_survey_data`, `manager_survey_data`, `in_time` y `out_time`. Calcula `Dias_Absentismo` a partir de los NaN en `in_time`, **descontando previamente los días marcados como festivo/cierre para toda la plantilla**. Calcula `Media_Horas_Diarias` a partir de la diferencia `out_time - in_time`. Une todo por `EmployeeID` e imputa nulos (mediana en variables numéricas, moda en encuestas). Exporta `data/processed/df_consolidado_clean.csv`.
2. **`01_eda.ipynb`:** traducción de variables a español, distribución del target, mapa de correlaciones, relación horas trabajadas vs. absentismo, comparativa por departamento y rol.
3. **`02_modelado.ipynb`:** codificación one-hot, escalado (`StandardScaler` solo sobre variables continuas, las dummies quedan intactas), y comparación de `LinearRegression`, `RandomForestRegressor` y `GradientBoostingRegressor`. Selección automática por menor RMSE. Guarda modelo, escalador y lista de columnas en `models/`.
4. **`src/app.py`:** interfaz Streamlit que reconstruye el vector de entrada **por nombre de columna** (no por posición), aplica el mismo escalador y modelo, y muestra la predicción junto a un simulador financiero "What-If".

## 📈 Resultados

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Regresión Lineal | 4.73 días | 5.39 | 0.05 |
| Random Forest Regressor | 4.94 días | 5.69 | -0.06 |
| Gradient Boosting Regressor | 4.79 días | 5.48 | 0.02 |

**Modelo seleccionado:** Regresión Lineal (menor RMSE).

## ⚠️ Limitaciones conocidas

- El poder predictivo es limitado (R² bajo): las variables disponibles (demografía, encuestas, compensación) explican solo una parte pequeña de la variabilidad real del absentismo. El modelo es más útil como **ordenador de riesgo relativo** entre empleados que como predictor exacto del número de días.
- El target excluye los días de cierre total de empresa, pero un NaN individual en `in_time` puede seguir incluyendo vacaciones planificadas, no solo ausencia real no planificada. Es una limitación de los datos de origen, no del pipeline.
- `Media_Horas_Diarias` se deriva de los mismos ficheros (`in_time`/`out_time`) de los que sale el target; no hay fuga directa, pero conviene tenerlo presente al interpretar su importancia en el modelo.

## 🚀 Cómo ejecutar el proyecto

```bash
# 1. Clonar el repositorio
git clone <url-de-tu-repo>
cd HR-OPERATIONAL-OPTIMIZER

# 2. Crear entorno virtual e instalar dependencias
python -m venv .venv
source .venv/bin/activate      # En Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Colocar los datos originales en data/raw/
#    (general_data.csv, employee_survey_data.csv, manager_survey_data.csv,
#     in_time.csv, out_time.csv)

# 4. Ejecutar los notebooks en orden: 00_prep -> 01_eda -> 02_modelado

# 5. Lanzar la app
streamlit run src/app.py
```

## 🛠️ Stack técnico

Python · Pandas · NumPy · Scikit-learn · Streamlit · Matplotlib · Seaborn · Joblib