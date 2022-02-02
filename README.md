# ecobici_forecast
Tiene sentido que la demanda de bicis en CDMX tenga una estacionalidad por horas del día y días de la semana. Pero ¿es posible predecir la demanda futura? 
¡Claro que sí! 🚴🏾

### Estructura del repositorio:
    .
    ├── ...
    ├── media                           # Directorio con imágenes para README
    │   └── ...
    ├── scripts                         # Directorio con el código necesario para analizar y modelar demanda en ventanas de tiempo
    │   ├── ecobici                     # Directorio para ocupar clases y métodos
    │   │   ├── __init__.py             # Para que el folder "ecobici" se puede trabajar de forma modular
    │   │   ├── utils.py                # Clase base, con métodos como importar, creación de variables, reestructuración de datos, etcétera
    │   │   └── models.py               # Clase hija de BaseClass, con métodos adicionales específicos para tratamientos de los datos Ecobici
    │   │
    │   └── EcoBici_Model.ipynb         # Modelo de pronóstico para demanda de viajes en Ecobici
    └── requirements.txt                # Instalar las librerías necesarias con el comando: pip install -r requirements.txt

<br>

# Fuente de datos

Se obtienen los [datos abiertos de Ecobici](https://www.ecobici.cdmx.gob.mx/es/informacion-del-servicio/open-data) mes con mes desde 2018. La estructura de cada archivo csv es la siguiente:

|Genero_Usuario|Edad_Usuario|Bici|Ciclo_Estacion_Retiro|Fecha_Retiro|Hora_Retiro|Ciclo_Estacion_Arribo|Fecha_Arribo|Hora_Arribo|
|---|---|---|---|---|---|---|---|---|
|M|39|10849|430|30/11/20|23:45:01|166|01/12/20|0:27:25|
|F|24|9943|122|01/12/20|5:55:41|326|01/12/20|6:21:13|

<br>

# Importar

Al leer los 48 archivos csv se reestructura por chunks tal que la tabla final indica la demanda de cada estación para cada ventana de tiempo en la respectiva fecha:

|Ciclo_Estacion_Retiro|fecha|00 a 09|10 a 12|13 a 14|15 a 17|18 a 20|>= 21|
|---|---|---|---|---|---|---|---|
|1|01/01/18|1|10|9|15|8|0|
|1|02/01/18|40|30|22|34|54|8|
|1|03/01/18|51|37|43|53|59|8|
|1|04/01/18|67|18|44|46|61|9|
|1|05/01/18|62|36|44|56|27|11|
|1|06/01/18|13|25|16|19|17|7|
|1|07/01/18|12|22|18|29|22|7|
|1|08/01/18|79|38|37|56|95|11|
|1|09/01/18|83|33|33|52|99|18|

<br>

# Transformar

Para cada estación, se "recorre" la demanda que tuvo en cada ventana de tiempo para que el próximo modelo "aprenda" la estacionalidad y tendencia de la demanda, es decir, podrá contestar la pregunta: ¿Cuántos viajes tendré en mi estación hoy? Porque ahora sabemos cómo fueron los viajes de los últimos 40 días en cada ventana de tiempo.

La estructura de esta tabla queda de la siguiente forma, siendo el subíndice "\_n" la demanda hace n-días en la misma ventana de tiempo:

![Alt text](media/shifted.png?raw=true "Time Window Shifted")

<br>

# Modelo

Se compara el score R<sup>2</sup> de tres modelos:

|Modelo|R<sup>2</sup> test|R<sup>2</sup> train|
|---|---|---|
|_Regresión Lineal_|88.70%|88.65%|
|_Bosque Aleatorio_|88.03%|98.26%|
|_Red Neuronal (SKlearn)_|87.65%|96.40%|

Por lo tanto, se elige la Regresión Lineal tanto por contar con el mejor score en el conjunto de test (generaliza correctamente porque es prácticamente el mismo score en el conjunto de entrenamiento) como por la baja complejidad del modelo (y peso del mismo para poner en Producción).

<br>

# Resultado

Aún cuando en el entrenamiento contamos con temporadas atípicas derivadas de la pandemia COVID-19, la demanda estimada es muy parecida a la real:

![Alt text](media/forecast.png?raw=true "Time Window Shifted")

<br>

# Siguientes pasos

Se plantea la posibilidad de predecir no sólo la demanda total de hoy con info de los 40 días pasados, sino segmentada por ventana de tiempo. Pruebas preliminares no resultan en R<sup>2</sup> mayor a 65%, se utilizarán modelos de Deep Learning para mejorar este resultado.


<br><br>

<div align="center"><strong>¡GRACIAS!</strong></div>
<br><br>
