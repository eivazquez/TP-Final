# Proyecto: Predicción de Precios de Propiedades en Argentina

**Proyecto integrador — 5HP** **Curso:** Programación Avanzada para Ciencia de Datos  
**Universidad:** Universidad de la ciudad de Buenos Aires  
**Equipo 5HP:**
- MATIAS ALEJANDRO BANCHIO
- PABLO GABRIEL CIOCIANO
- PAULA GISELA COCHIMANO
- ANTONIO LUIS EMILIO MARTINEZ
- ENRIQUE IGNACIO VAZQUEZ

---

## 🚀 Resumen del Proyecto

Este proyecto analiza un conjunto de datos de propiedades en Argentina (dataset de Properati) con el objetivo de desarrollar un modelo de *machine learning* capaz de predecir el precio de venta (en USD) de un inmueble en función de sus características principales, como la ubicación, el tipo de propiedad, la superficie, y la cantidad de ambientes y baños.

El pipeline completo incluye la limpieza y preprocesamiento de datos, un análisis exploratorio (EDA), entrenamiento y comparación de múltiples modelos de regresión, y una optimización final mediante ajuste de hiperparámetros.

---

## 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python 3.10+
* **Análisis y Manipulación de Datos:** Pandas, NumPy
* **Bases de Datos:** DuckDB (para persistencia de resultados analíticos)
* **Visualización:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn (para pipelines, preprocesamiento, `train_test_split`, `GridSearchCV`, `LinearRegression` y `RandomForestRegressor`)
* **Modelado Avanzado:** XGBoost (para `XGBRegressor`)
* **Serialización de Modelos:** Joblib
* **Entorno:** Jupyter Notebook / Google Colab

---

## ⚙️ Instalación y Configuración

1.  **Clonar el repositorio:**
    ```bash
    git clone [URL-DEL-REPOSITORIO]
    cd [NOMBRE-DEL-REPOSITORIO]
    ```

2.  **Crear un entorno virtual (recomendado):**
    ```bash
    python -m venv env
    source env/bin/activate  # En Windows: env\Scripts\activate
    ```

3.  **Instalar las dependencias:**
    Se puede crear un archivo `requirements.txt` con el siguiente contenido e instalarlo.

    **requirements.txt:**
    ```
    pandas
    numpy
    duckdb
    matplotlib
    seaborn
    joblib
    scikit-learn
    xgboost
    jupyter
    ```

    **Comando de instalación:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Descargar los datos:**
    Asegúrese de tener el archivo `entrenamiento.csv` ubicado en la carpeta `data/`.

---

## ▶️ Cómo Ejecutar el Pipeline

1.  Inicie Jupyter Notebook en su terminal:
    ```bash
    jupyter notebook
    ```
2.  Abra el archivo `TPFinal_.ipynb`.
3.  Ejecute todas las celdas en orden, desde la importación de librerías hasta la persistencia de datos. Los artefactos generados (dataset limpio, modelo y base de datos) se guardarán en la carpeta `data/`.

---

## 📖 Descripción del Notebook (`TPFinal_.ipynb`)

El notebook está estructurado en 7 secciones principales que siguen un flujo de trabajo estándar de ciencia de datos.

### 1. Carga y Limpieza de Datos 🧹
* Se carga el dataset crudo `entrenamiento.csv`.
* Se aplica un filtro inicial para mantener solo operaciones de **Venta** y en moneda **USD**.
* Se seleccionan las columnas clave: `price`, `surface_covered`, `rooms`, `bathrooms`, `property_type`, y `l2` (ciudad).
* Se eliminan todos los registros con valores nulos en estas columnas.
* Se filtran valores atípicos (outliers) de precio, manteniendo solo propiedades entre $10,000 y $1,000,000 USD para estabilizar el modelo.
* Se renombran las columnas para mayor claridad (ej. `l2` a `city`).
* El dataset limpio resultante se guarda como `data/cleaned_data.csv`.

### 2. Análisis Exploratorio de Datos (EDA) 📊
* Se analiza la distribución de la variable objetivo (`price`) mediante un histograma, mostrando un sesgo a la derecha.
* Se genera una matriz de correlación (heatmap) para las variables numéricas (`price`, `surface`, `rooms`, `bathrooms`), identificando una fuerte correlación positiva entre el precio y la superficie.
* Se utiliza un diagrama de caja (boxplot) para visualizar la distribución de precios según el `property_type`.

### 3. Ingeniería de Features y División de Datos 🔪
* Se definen las variables predictoras (X) y la variable objetivo (y).
* Se realiza una división de los datos en conjuntos de entrenamiento (80%) y prueba (20%) usando `train_test_split`, asegurando la reproducibilidad con `random_state=42`.

### 4. Creación del Pipeline y Modelos Base 🤖
* Se define un `ColumnTransformer` para el preprocesamiento automático:
    * **Variables Numéricas** (`surface`, `rooms`, `bathrooms`): Se escalan con `StandardScaler`.
    * **Variables Categóricas** (`property_type`, `city`): Se codifican con `OneHotEncoder`.
* Se crea un `Pipeline` de Scikit-learn que integra el preprocesador y el modelo.
* Se entrenan y evalúan tres modelos base para comparar rendimiento:
    1.  `LinearRegression`
    2.  `RandomForestRegressor`
    3.  `XGBRegressor`
* Las métricas de evaluación (MAE, RMSE, R²) se almacenan en un DataFrame (`results_df`).

### 5. Ajuste de Hiperparámetros (Hyperparameter Tuning) 🛠️
* Se selecciona el modelo con mejor rendimiento base (en este caso, `XGBoost`) para una optimización más profunda.
* Se utiliza `GridSearchCV` para encontrar la mejor combinación de hiperparámetros (ej. `n_estimators`, `max_depth`, `min_samples_leaf`) para el `RandomForestRegressor` (como alternativa robusta).
* Se identifica y almacena el `best_model` (el pipeline optimizado).

### 6. Evaluación del Modelo Final y Análisis 📈
* Se evalúa el modelo optimizado (`best_model`) contra el conjunto de prueba.
* Se generan gráficos comparativos (barplots) de **RMSE** y **R²** para todos los modelos (base y optimizado), confirmando que XGBoost ofrece el mejor rendimiento (R² ~0.705).
* Se crea un gráfico de dispersión (scatterplot) de **Valores Reales vs. Valores Predichos** para evaluar visualmente la precisión y el sesgo del modelo final.
* Se genera un gráfico de **Importancia de Features** (del modelo Random Forest) para entender qué variables contribuyen más a la predicción del precio.

### 7. Persistencia del Modelo y Resultados 💾
* El pipeline completo del mejor modelo (incluyendo preprocesador y modelo entrenado) se serializa y guarda en un archivo `data/best_model.pkl` usando `joblib`.
* Los resultados clave del análisis se guardan en una base de datos **DuckDB** (`data/properati_models.db`) en tres tablas separadas para consulta futura:
    * `input_data`: El DataFrame limpio usado para el análisis.
    * `model_results`: El DataFrame con las métricas de todos los modelos.
    * `model_config`: Los mejores hiperparámetros encontrados por `GridSearchCV`.