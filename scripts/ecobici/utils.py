# Control de datos
from time import sleep
from pathlib import Path, PosixPath
from pickle import dump as save_pkl, load as load_pkl
from IPython.display import clear_output

# Ingeniería de variables
from re import search
from numpy import array, nan
from datetime import datetime
from pandas import DataFrame, Series, read_csv, to_datetime, date_range, cut, options
options.mode.chained_assignment = None

# Modelos
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Gráficas
import cufflinks as cf
cf.go_offline()

class BaseClass:
    def __init__(self, base_dir: str, file_name: str='EcoBici') -> None:
        '''
        Inicializa la clase recibiendo un directorio y opcionalmente un nombre base
        '''
        # Obtiene un directorio como texto y convertirlo a tipo Path para unir directorios, buscar archivos, etc.
        self.base_dir = Path(base_dir)
        # Enlista todos los archivos con formato YYYY_MM en el directorio
        self.files_list = [x for x in self.base_dir.glob('*.csv') if search(r'[\/\\]\d{4}\-\d{2}\.csv',str(x)) is not None]
        # Nombre base para los archivos que vayan a exportarse
        self.file_name = file_name


    def __len__(self) -> int:
        '''
        Cantidad de archivos de interés en el directorio
        '''
        return len(self.files_list)


    def __str__(self) -> str:
        return f'{self.__len__()} archivos en:\n{self.base_dir}'


    def cool_print(self, text: str, sleep_time: float=0.02, show_time: float=1.0, by_word: bool=False) -> None: 
        '''
        Imprimir como si se fuera escribiendo
        '''
        acum = ''
        for x in (text.split() if by_word else text): 
            # Acumular texto
            acum += x+' ' if by_word else x
            # Limpiar pantalla
            clear_output(wait=True)
            # Esperar un poco para emular efecto de escritura
            sleep(sleep_time*(9 if by_word else 1))
            # Imprimir texto acumulado
            print(acum)
        # Mantener el texto en pantalla
        sleep(show_time)


    def get_csv(self, file_path: PosixPath, just_name: bool=False, add_subfolder: bool=False, subfolder_name: str='transformed', **kwargs) -> DataFrame: 
        '''
        Obtener tabla a partir de un archivo .csv
        '''
        # Obtiene el nombre del archivo a partir de un directorio
        file_name = str(file_path).split('/')[-1]
        try: 
            if add_subfolder: get_from = self.base_dir.joinpath(subfolder_name)
            else: get_from = self.base_dir
            # Si el parámetro lo indica, obtiene el csv conectando el directorio con el nombre del archivo
            if just_name: df = read_csv(get_from.joinpath(file_name), low_memory=False, **kwargs)
            # De otro modo, utiliza el directorio que recibe como parámetro
            else: df = read_csv(file_path, low_memory=False, **kwargs)

            # Obtiene e informa del número de renglones y columnas
            df_shape = df.shape
            self.cool_print(f'Archivo con nombre {file_name} fue encontrado en:\n{get_from}\nCon {df_shape[0]} renglones y {df_shape[-1]} columnas')
            return df
        # Imprime que hubo un error al obtener el csv
        except: self.cool_print(f'No se encontró el archivo con nombre {file_name} en:\n{get_from}\nSi el archivo csv existe, seguramente tiene un encoding y/o separador diferente a "utf-8" y "," respectivamente\nIntenta de nuevo!')
    

    def export_csv(self, df: DataFrame, name_suffix: str=None, to_subfolder: str=None, **kwargs) -> None: 
        '''
        Exportar un archivo en formato csv
        '''
        if to_subfolder is not None:
            # Si el parámetro lo indica, se asignará un sub-directorio que parte del base
            export_path = self.base_dir.joinpath(to_subfolder)
            # Intenta crear dicho sub-directorio
            try: export_path.mkdir()
            # No hay problema si dicho sub-directorio ya existe
            except FileExistsError: pass
        # Si no se indica to_subfolder, utilizar el directorio base
        else: export_path = self.base_dir

        # Define si habrá un sufijo para el nombre del archivo
        export_name = f'{self.file_name}.csv' if name_suffix==None else f'{self.file_name}_{name_suffix}.csv'
        # Exporta el archivo en el directorio base
        df.to_csv(export_path.joinpath(export_name), **kwargs)
        # Informa al usuario
        self.cool_print(f'Archivo: {export_name} fue exportado exitosamente en:\n{export_path}')


    def rem_nan_rows(self, df: DataFrame, thres: float=1.0, verbose: bool=True) -> DataFrame:
        '''
        Omitir registros mayor o igual al porcentaje "thres" de valores nulos
        '''
        # Filtrar sólo los registros con algún valor nulo
        has_null = df.loc[df.isnull().sum(axis=1)>0].index

        to_remove = []
        # enumerate(['A','B','C']) == zip(range(len(['A','B','C']),['A','B','C'])) == [(0,'A'), (1,'B'), (2,'C')]
        for row in has_null:
            # Revisar por renglón, transponiéndolo
            sub_df = df.loc[row,:].T
            # Obtener el porcentaje de nulos
            perc_nan = sub_df.isnull().mean()
            # Si dicho porcentaje es mayor, guardar el lugar del renglón en una lista
            if perc_nan >= thres: to_remove.append(row)

        # Omitir los registros de la lista con el porcentaje de valores nulos más grande que el parámetro "thres"
        df = df.loc[~df.index.isin(to_remove),:]
        # Informar cuántos renglones fueron omitidos
        if verbose: self.cool_print(f'{len(to_remove)} renglones con {"{:.1%}".format(thres)} o más de valores nulos fueron eliminados')
        
        return df


    def create_bins(self, df: DataFrame, col: str, bins: list, lower_limit=-1, upper_limit=1000) -> Series:
        '''
        Recibiendo los cortes, recibe una columna numérica y crea rangos tipo "00", "01 a 05", ">=6"
        '''
        # Función para convertir float: 1.0 --> str: '01'
        def two_char(n): return str(int(n)).zfill(2)

        # Crear rangos
        df[f'{col}_range'] = cut(df[col], bins=[lower_limit]+bins+[upper_limit])
        # Convertirlo a texto: [1.0 - 5.0] --> '01 a 05'
        df[f'{col}_range'] = df[f'{col}_range'].map(lambda x: two_char(x.left+1)+' a '+two_char(x.right) if x!=nan else nan)

        # Corregir algunas etiquetas como: '01 a 01'-->'01' y también '03 a upper_limit'-->'>= 03'
        last_cut = two_char(bins[-1]+1)
        df[[f'{col}_range']] = df[[f'{col}_range']].replace({
            **{last_cut+f' a {upper_limit}': '>= '+last_cut},
            **{two_char(x)+' a '+two_char(x): two_char(x) for x in bins}
        })
        # No perder de vista los valores ausentes: "La falta de información también es información"
        df[f'{col}_range'] = df[f'{col}_range'].map(lambda x: nan if str(x)=='nan' else str(x))

        return df[f'{col}_range']


    def date_vars(self, df: DataFrame, date_col: str='fecha', hours_bin: list=[9,12,14,17,20], **kwargs) -> DataFrame: 
        '''
        Crear variables de fecha: año, trimestre, mes, hora y rangos de hora
        '''
        # Convertir a tipo datetime
        df[date_col] = to_datetime(df[date_col], **kwargs)

        # Para extraer la división de año
        df[f'{date_col}_year'] = df[date_col].dt.year.map(int).map(str)
        # Trimestre a dos caracteres
        df[f'{date_col}_quarter'] = df[date_col].dt.quarter.map(lambda x: str(int(x)).zfill(2))
        # Mes a dos caracteres
        df[f'{date_col}_month'] = df[date_col].dt.month.map(lambda x: str(int(x)).zfill(2))

        # Concatenar el año, tanto trimestre como con el mes
        df[f'{date_col}_yearquarter'] = df[f'{date_col}_year']+' - '+df[f'{date_col}_quarter']
        df[f'{date_col}_yearmonth'] = df[f'{date_col}_year']+' - '+df[f'{date_col}_month']

        # Día de la semana, sólo los primeros 3 caracteres
        df[f'{date_col}_month'] = df[date_col].dt.day_name().str[:3]

        # Hora
        df[f'{date_col}_hour'] = df[date_col].dt.hour
        # Crear rangos de hora
        df[f'{date_col}_hour_range'] = self.create_bins(df, f'{date_col}_hour', bins=hours_bin)

        # Mantener sólo la fecha
        df[date_col] = df[date_col].dt.date
        return df


    def outliers(self, df: DataFrame, cols: list=None, rem_perc: float=0.03, rem_outliers: bool=True) -> DataFrame:
        ''''
        Mediante el modelo de sklearn, elimina los outliers analizando los datos de forma multivariada
        '''
        # Instancia el modelo con el % que reciba como parámetro
        outlier = IsolationForest(contamination=rem_perc, n_jobs=-1)

        # Si no se reciben columnas, contemplar todas
        cols = df.columns if cols is None else cols
        # Indica con "-1" si el registro es atípico
        df['outlier'] = outlier.fit_predict(df[cols])

        # Omite dichos registros y la columna que indica si es atípico
        if rem_outliers: df = df[df['outlier']!=-1].drop(columns = 'outlier')
        return df


    def multishift(self, df: DataFrame, id_cols: list, date_col: str='fecha', shifts: list=range(1,22), rem_sum_zero: bool=True, create_counter: bool=False, **pivot_args): 
        '''
        Escalona los valores para crear una Tabla Analítica de Datos con formato: valor hoy, valor 1 día antes, dos días antes, etc
        '''
        # Asegurarse que tiene solamente la fecha
        df[date_col] = df[date_col].map(to_datetime).dt.date

        # Sólo una columna que servirá como ID
        id_col = ','.join(id_cols)
        df[id_col] = df[id_cols].astype(str).apply(','.join, axis=1)

        # Omitir aquellos IDs con menor frequencia que el máximo valor de "shifts", porque inevitablemente tendrán shift vacíos
        freq = df[id_col].value_counts().to_frame()
        omit_idx = freq[freq[id_col]<=max(shifts)].index.to_list()
        if len(omit_idx)>0: 
            df = df[~df[id_col].isin(omit_idx)].copy()
        
        # Columna auxiliar para conteo de registros
        if create_counter: df['n'] = 1

        # Estructurar una tabla pivote, de donde se partirá para "recorrer" los días
        df = df.pivot_table(index=[id_col,date_col], **pivot_args, fill_value=0)
        # Unir las posibles multi-columnas en una
        df.columns = ['_'.join([x for x in col]) if not isinstance(df.columns[0],str) else col for col in df.columns]

        df = df.reset_index()
        total = DataFrame()
        for row in set(df[id_col]): 
            # Para cada grupo de renglones por ID
            df_id = df.set_index(id_col).loc[row,: ]
            # Asegurar todas las fechas
            tot_dates = DataFrame(date_range(start=df_id[date_col].min(), end=df_id[date_col].max()).date, columns=[date_col])
            df_id = df_id.merge(tot_dates, on=date_col, how='right').fillna(0)
            cols = df_id.columns[1: ]

            # Comenzar el "escalonado" de la tabla pivote inicial
            aux = df_id.copy()
            for i in shifts:
                # Renombrar la columna que se acaba de escalonar
                if i > 0: aux = aux.join(df_id.iloc[: ,1: ].shift(i).rename(columns={x: f'{x}_{str(i).zfill(2)}' for x in cols}))
            # No perder de vista el "id" de este subconjunto
            aux[id_col] = row
            # Agregar a la tabla total
            total = total.append(aux, ignore_index=True)

        # Mantener como índice para tener una matriz X de valores continuos
        total.set_index(id_cols+[date_col], inplace=True)

        # Omitir registros que suman 0?
        if rem_sum_zero:
            total['sum'] = total.sum(axis=1)
            total = total[total['sum']>0].drop('sum', axis=1)

        return total

    
    def apply_multishift(self, df: DataFrame, export_shifted: bool=True, **kwargs) -> tuple: 
        '''
        Aplica el método anterior, exporta los resultados y devuelve una matriz X de valores continuos no nulos y el vector "y" objetivo
        '''
        # Aplicar la función "multishift" con los parámetros personalizados
        df = self.multishift(df, **kwargs)
        df.dropna(inplace=True)
        df = df[sorted(df.columns)].copy()

        # Obtener la lista de las columnas de todos los días previos
        prev = df.head(1).filter(regex='_\d+').columns.tolist()
        # Y aquellas originales, sin escalonar
        actual = [x for x in df.columns if x not in prev]

        # Ordena las columnas
        df = df[actual+prev].copy()

        # Tal vez el usuario quiere exportar los resultados
        if export_shifted: self.export_csv(df, name_suffix='shifted', to_subfolder='transformed')

        # Seleccionar los datos para construir f(X)=y
        X = df[prev].copy()
        y = df[actual].sum(axis=1).values
        return X, y

    
    def train_model(self, X: DataFrame, y: array, scaler=RobustScaler, model=LinearRegression, rem_outliers: bool=False, perc_outliers: float=0.03, **kwargs): 
        '''
        Escala y entrena un modelo, devuelve el score, el objeto tipo Pipeline y la relevancia de cada variable
        '''
        if rem_outliers:
            # Unir la tabla
            df = X.join(DataFrame(y, index=X.index, columns=['real']))
            # Para omitir outliers
            df = self.outliers(df, rem_perc=perc_outliers)
            # Y nuevamente separar la matriz X y el vector "y"
            X = df[[x for x in df.columns if x not in ['real']]].copy()
            y = df['real'].values

        # Conjunto de entrenamiento y de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, random_state=7, shuffle=True)

        # Define los pasos del flujo
        pipe_obj = Pipeline(steps=[('prep', scaler()), ('model', model(**kwargs))])

        # Entrena y guarda el score en test
        test_score = pipe_obj.fit(X_train,y_train).score(X_test, y_test)
        # Guarda el score en train, para revisar sobreajuste
        train_score = pipe_obj.score(X_train,y_train)

        # Imprime los scores
        self.cool_print(f"Test score:\t{'{:.2%}'.format(test_score)}\nTrain score:\t{'{:.2%}'.format(train_score)}")

        # Elige la forma de obtener las variables más representativas
        # Ya sea por Regresión Lineal
        try: most_important_features = pipe_obj[-1].coef_ 
        except: 
            # O por Árbol de decisión, Bosque Aleatorio, XGBoost
            try: most_important_features = pipe_obj[-1].feature_importances_
            # De otro modo, solamente asignar un vector de 0s a este objeto
            except: most_important_features = [0]*len(X.columns)

        # Las ordena descendentemente
        coef_var = DataFrame(zip(X.columns, most_important_features)).sort_values(1, ascending=False).reset_index(drop=True)
        coef_var = coef_var.rename(columns={0:'Variable',1:'Peso'}).set_index('Variable')

        # Devuelve el objeto para clustering, la lista de scores tanto en train como en test y la relevancia de cada variable para el modelo 
        return pipe_obj, (test_score,train_score), coef_var


    def choose_best_model(self, X: DataFrame, y: array, models: list, scaler=RobustScaler):
        pass


    def real_vs_est(self, X: DataFrame, y: array, model, omit_zero: bool=True) -> DataFrame:
        '''
        Devuelve una tabla con dos columnas: el valor real y el valor predicho por el modelo
        '''
        # De todo el conjunto de datos...
        df = X.join(DataFrame(y, index=X.index, columns=['real']))
        # Predice el el valor...
        df['estimado'] = model.predict(X)

        # Si el parámetro lo indica, reemplaza negativos por 0
        if omit_zero: df['estimado'] = df['estimado'].map(lambda x: max(0,x))

        # Y devuelve sólo las columna real y la estimada
        return df[['real','estimado']]

    
    def plot_real_vs_est(self, X: DataFrame, y: array, model, id_col: str, top_n: int=5, date_col: str='fecha', from_year: int=1900, to_year: int=datetime.now().year, **kwargs) -> None:
        '''
        Grafica la tendencia real y la predicha por el modelo a través del tiempo
        ''' 
        # Obtener real vs estimado
        pred = self.real_vs_est(X, y, model, **kwargs).reset_index()

        # Filtrar sólo años de interés
        pred['year'] = to_datetime(pred[date_col]).dt.year
        df = pred[(pred['year']>=from_year)&(pred['year']<=to_year)].copy()
        df.drop(columns='year', inplace=True)

        # Establer el topN de ID para comparar gráficamente
        top_ids = df[id_col].value_counts().index.tolist()[:top_n]
        df = df[df[id_col].isin(top_ids)].copy()

        # Mostrar comportamiento real vs estimado
        df.set_index(id_col, inplace=True)
        for x in set(df.index):
            df_id = df.loc[x,: ].reset_index(drop=True).set_index(date_col)
            df_id.iplot(title=x)


    def save_model(self, model, model_name: str) -> None:
        '''
        Exporta el modelo en modo diccionario para que cuando se importe, se conozca de qué trata el objeto
        '''
        models_path = self.base_dir.joinpath('models')
        try: models_path.mkdir()
        except FileExistsError: pass

        # Guarda el pickle con extensión ".xz" para comprimirlo
        with open(models_path.joinpath(f'{model_name}.xz'), 'wb') as f:
            # Como diccionario para conocer su nombre
            save_pkl({model_name:model}, f)
            
        # Confirma que el archivo fue guardado exitosamente
        self.cool_print(f'El modelo {model_name}.xz fue guardado existosamente en:\n{models_path}')

    
    def get_model(self, model_name: str) -> None:
        '''
        Exporta el modelo en modo diccionario para que cuando se importe, se conozca de qué trata el objeto
        '''
        models_path = self.base_dir.joinpath('models')
        # Guarda el pickle con extensión ".xz" para comprimirlo
        with open(models_path.joinpath(f'{model_name}.xz'), 'wb') as f:
            # Como diccionario para conocer su nombre
            model_dict = load_pkl(f)
            
        # Confirma que el archivo fue guardado exitosamente
        self.cool_print(f'El modelo {model_name}.xz fue importado existosamente en:\n{models_path}')
        return model_dict
