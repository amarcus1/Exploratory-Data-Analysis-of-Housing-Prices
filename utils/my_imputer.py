# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 08:05:38 2022

@author: amarcus1
"""
from sklearn.impute import SimpleImputer
import numpy as np

#def check_missing(data):
#    missing_values = data.isnull().sum()
#    print('There are {:n} columns with null values'.format(missing_values[ missing_values != 0 ].count()))
#    print(missing_values[ missing_values != 0 ])
#    return missing_values
    
def my_imputer(data, data_type, strategy):
    # check how many missing_values are there
    missing_values = check_missing(data)
    
    if missing_values[ missing_values != 0 ].count() == 0:
        return
    
    # define imputer
    imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    
    # get the names of columns with missing values
    cols_missing = list(data.columns[ data.isnull().any() ])

    # imputate
    imputer = imputer.fit(data[cols_missing])
    data[cols_missing] = imputer.transform(data[cols_missing])
    
    # Change the columns back to numeric type
    if data_type == 'int':
        cols = list(missing_values[ missing_values > 0 ].index)
        for col in cols:
            data[col] = data[col].astype(data_type)
        
        
def check_missing(data):
    missing_values = data.isnull().sum()
    n_rows = data.shape[0]
    print('There are {:n} columns with null values'.format(missing_values[ missing_values != 0 ].count()))
    missing_values = missing_values[ missing_values != 0 ] / n_rows * 100
    print(missing_values.sort_values(ascending=False).map('{:,.1f}'.format) + '%')
    return missing_values
    
# def my_imputer(data):
#     # check how many missing_values are there
#     missing_values = check_missing(data)
    
#     if missing_values[ missing_values != 0 ].count() == 0:
#         return
    
#     # define imputer
#     imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    
#     # get the names of columns with missing values
#     cols_missing = list(data.columns[ data.isnull().any() ])

#     # imputate
#     imputer = imputer.fit(data[cols_missing])
#     data[cols_missing] = imputer.transform(data[cols_missing])
    
#     return data