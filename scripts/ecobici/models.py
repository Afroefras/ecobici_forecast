from .utils import BaseClass


from numpy import array
from re import search, sub
from datetime import datetime
from pandas import DataFrame, to_datetime

class EcoBici(BaseClass):
    def __init__(self, base_dir: str, file_name: str = 'EcoBici') -> None:
        '''
        Hereda los atributos de la clase base
        '''
        super().__init__(base_dir, file_name=file_name)


    def get_by_hour_range(self, file_path, from_year: int, to_year: int, date_col: str='Fecha_Retiro', hour_col: str='Hora_Retiro', date_format: str=r'%d/%m/%Y', hour_format: str=r'%H:%M:%S', **kwargs) -> DataFrame:
        '''
        Importa un csv, lo limpia y transforma según los parámetros de "pandas.pivot_table()" que reciba
        '''
        # Lee una tabla en formato ".csv"
        df = self.get_csv(file_path)

        # Omite los registros completamente nulos
        df = self.rem_nan_rows(df, verbose=False)
        # Omite los registros nulos en los campos de fecha u hora
        df.dropna(subset=[date_col,hour_col], inplace=True)

        # Corregir formatos de hora incorrectos como 18:: --> 18:00:00
        df[hour_col] = df[hour_col].map(lambda x: sub(r'::$', ':00:00', str(x)))
        # Corregir formatos de hora incorrectos como 18:01: --> 18:01:00
        df[hour_col] = df[hour_col].map(lambda x: sub(r':$', ':00', str(x)))

        # Une las columnas de fecha y hora
        df['fecha'] = df[[date_col,hour_col]].apply(' '.join, axis=1)
        # Aplica el método de crear variables de fecha y rangos de hora
        df = self.date_vars(df, dayfirst=True, format=f'{date_format} {hour_format}')

        # Filtrar sólo los años de interés
        df = df[df['fecha_year'].astype(int)>=from_year].copy()
        df = df[df['fecha_year'].astype(int)<=to_year].copy()

        # Estructura la tabla como lo indiquen los parámetros
        df = df.assign(n=1).pivot_table(**kwargs)
        return df


    def read_raw_files(self, from_year: int=2000, to_year: int=datetime.now().year, export_result: bool=True, **kwargs) -> DataFrame:
        ''''
        Obtiene todos los archivos de los años indicados para reestructurarlos según los parámetros de "pandas.pivot_table()" que reciba
        '''
        # Filtra los archivos que cumplan con la condición: de X año a Y año
        filtered_files = sorted([x for x in self.files_list if from_year <= int(search(r'/(\d{4})\-', str(x)).group(1)) <= to_year])

        # DataFrame vacío para ir acumulando los datos
        df = DataFrame()
        for chunk_file in filtered_files:
            # Aplicar el método anterior para agrupar por archivo y no los datos completos
            transformed = self.get_by_hour_range(chunk_file, from_year, to_year, **kwargs)
            # Acumular la tabla anterior con el archivo actual
            df = df.append(transformed)
            # Eliminar objeto para optimizar memoria
            del transformed

        # Volver a agrupar si es que el índice se repite en dos archivos
        df = df.reset_index().pivot_table(index=df.index.names, values=df.columns, aggfunc=sum).reset_index()

        # Tal vez el usuario quiera exportar el resultado
        if export_result: self.export_csv(df, index=False, name_suffix=f'from_{from_year}_to_{to_year}', to_subfolder='transformed')
        return df


    def filter_months(self, X: DataFrame, y: array, months_list: list=range(1,13)) -> DataFrame:
        '''
        Filtra sólo los meses de interés
        '''
        # Guardar los nombres de índices y columnas para reasignar al final
        to_index = X.index.names
        to_columns = X.columns

        # Reasigna la matriz X a un DataFrame
        df = X.reset_index().copy()
        # Une el vector "y" 
        df['real'] = y

        # Obtiene los meses de la columna fecha
        df['month'] = to_datetime(df['fecha']).dt.month
        # Y filtra los meses de interés
        df = df[df['month'].isin(months_list)].copy()

        # Finalmente separa de nuevo la matriz X y el vector "y"
        y = df['real'].values
        X = df.set_index(to_index)[to_columns]
        return X, y


    def ecobici_shifted(self, source, **kwargs) -> tuple:
        '''
        Aplica el método multishift para poder entrenar un modelo de regresión
        '''
        # Verifica si el parámetro es una tabla
        if isinstance(source, DataFrame): df = source.copy()
        # De otro modo, es el nombre del archivo ubicado en el directorio base
        else: df = self.get_csv(source, just_name=True)

        # Aplica el escalonado de la estructura de tabla que reciba como parámetros de pandas.pivot_table()
        X, y = self.apply_multishift(df, **kwargs)
        
        return X, y