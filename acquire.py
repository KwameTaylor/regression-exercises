import pandas as pd
import numpy as np
from pydataset import data
import matplotlib.pyplot as plt
import seaborn as sns
import os
from env import host, user, password

def get_connection(db, user=user, host=host, password=password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_titanic_data():
    if os.path.isfile('titanic.csv') == False:
        sql_query = 'SELECT * FROM passengers'
        df = pd.read_sql(sql_query, get_connection('titanic_db'))
        df.to_csv('titanic.csv')
        return df
    else:
        df = pd.read_csv('titanic.csv', index_col=0)
        return df

def get_iris_data():
    if os.path.isfile('iris.csv') == False:
        sql_query = """
                    SELECT species_id,
                    species_name,
                    sepal_length,
                    sepal_width,
                    petal_length,
                    petal_width
                    FROM measurements
                    JOIN species
                    USING(species_id)
                    """
        df = pd.read_sql(sql_query, get_connection('iris_db'))
        df.to_csv('iris.csv')
        return df
    else:
        df = pd.read_csv('iris.csv', index_col=0)
        return df