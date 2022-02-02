# ecobici_forecast
Tiene sentido que la demanda de bicis en CDMX tenga una estacionalidad por horas del día y días de la semana. Pero ¿es posible predecir la demanda futura? ¡Claro que sí! 🚴🏾

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
