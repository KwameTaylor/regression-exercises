import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

from acquire import get_titanic_data, get_iris_data

# Prep Iris Data
def prep_iris():
    df = get_iris_data()
    df = df.drop(columns='species_id').rename(columns={'species_name': 'species'})
    dummy = pd.get_dummies(df.species, drop_first=True)
    df = pd.concat([df, dummy], axis=1)
    return df

# Prep Titanic Data
def titanic_split(df):
    '''
    This function performs split on titanic data, stratify survived.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=666, 
                                        stratify=df.survived)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=666, 
                                   stratify=train_validate.survived)
    return train, validate, test

def impute_mean_age(train, validate, test):
    '''
    This function imputes the mean of the age column into
    observations with missing values.
    Returns transformed train, validate, and test df.
    '''
    # create the imputer object with mean strategy
    imputer = SimpleImputer(strategy = 'mean')
    
    # fit on and transform age column in train
    train['age'] = imputer.fit_transform(train[['age']])
    
    # transform age column in validate
    validate['age'] = imputer.transform(validate[['age']])
    
    # transform age column in test
    test['age'] = imputer.transform(test[['age']])
    
    return train, validate, test

def prep_titanic():
    '''
    This function reads titanic data into a df from a csv file.
    Returns prepped train, validate, and test dfs
    '''
    # use my acquire function to read data into a df from a csv file
    df = get_titanic_data()
    
    # drop rows where embarked/embark town are null values
    df = df[~df.embarked.isnull()]
    
    # encode embarked & sex using dummy columns
    titanic_dummies = pd.get_dummies(df[['sex', 'embarked']], drop_first=True)
    
    # join dummy columns back to df
    df = pd.concat([df, titanic_dummies], axis=1)
    
    # drop the deck column
    df = df.drop(columns=['passenger_id', 'deck', 'sex', 'embarked', 'class', 'embark_town'])

    # impute missing age values
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imp_mean.fit(df[['age']])
    df[['age']] = imputer.transform(df[['age']])

    return df