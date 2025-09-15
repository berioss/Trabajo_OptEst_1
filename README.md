Proyecto: Optimización Estocástica aplicada al ruteo y análisis de tiempos de viaje
================================================================================

📌 Descripción
--------------
Este proyecto implementa herramientas de Optimización Estocástica (particularmente el método de 
Sample Average Approximation – SAA) para resolver problemas de ruteo en redes de transporte urbano.  
El contexto de aplicación son datos de tiempos de viaje de taxis en la ciudad de Manhattan, los cuales 
se procesan para:
- Construir pools de observaciones de tiempos de viaje entre nodos de interés.
- Realizar limpieza de valores atípicos (outliers).
- Generar muestras de escenarios para el SAA.
- Resolver el problema de ruteo bajo incertidumbre minimizando el costo esperado.

🗂️ Estructura del proyecto
---------------------------
- Practica1-OptSth-Rios-Brahyan.ipynb
  Notebook principal donde se desarrollan los siguientes pasos:
  1. Importación y preprocesamiento de datos.
  2. Limpieza de valores atípicos usando el criterio de rango intercuartílico (IQR).
  3. Construcción de muestras de escenarios para el SAA.
  4. Modelado del problema de ruteo como un problema de optimización estocástica.
  5. Resolución del modelo y análisis de resultados.

- /data/
  Carpeta (no incluida en este repositorio) destinada a almacenar las bases de datos de viajes en taxi y tiempos de trayecto.

- /images/
  Gráficas y resultados visuales generados en el análisis (por ejemplo, boxplots, histogramas de tiempos, rutas óptimas).

⚙️ Requerimientos
-----------------
El proyecto está implementado en Python 3.x y usa principalmente las siguientes librerías:
- numpy
- pandas
- matplotlib / seaborn
- scipy
- pyomo
- jupyter

Instalación recomendada:
    pip install numpy pandas matplotlib seaborn scipy pyomo jupyter

🚀 Ejecución
------------
1. Clonar o descargar el repositorio.
2. Colocar los datos de viajes en la carpeta data/.
3. Abrir el notebook en Jupyter:
       jupyter notebook Practica1-OptSth-Rios-Brahyan.ipynb
4. Ejecutar las celdas de forma secuencial.

📊 Resultados esperados
-----------------------
- Identificación y depuración de valores atípicos en los tiempos de viaje.
- Construcción de escenarios representativos para el modelo estocástico.
- Obtención de rutas óptimas bajo incertidumbre minimizando el costo esperado.
- Comparación de resultados entre diferentes instancias y tamaños de muestra SAA.

🎯 Objetivo académico
---------------------
Este trabajo corresponde a una práctica del curso de Optimización Estocástica, cuyo propósito es que los 
estudiantes implementen modelos y algoritmos de optimización bajo incertidumbre aplicados a problemas 
reales, en este caso, el Vehicle Routing Problem (VRP) con tiempos de viaje inciertos.

👤 Autor
--------
- Brahyan Ríos
Curso: Optimización Estocástica – 2025
