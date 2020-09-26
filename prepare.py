import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

def prep_mall_data(df):
    '''
    Takes the acquired mall data, does data prep,
    and returns train, test, and validate data splits.
    '''
    df['is_female'] = (df.gender == 'Female').astype('int')
    train_and_validate, test = train_test_split(df, test_size=.15, random_state=666)
    train, validate = train_test_split(train_and_validate, test_size=.15, random_state=666)
    return train, test, validate