import numpy as np
from numpy import nan
import pandas as pd
import gc
from sklearn.preprocessing import MinMaxScaler

from GaussianTargetEncoder import GaussianTargetEncoder

def Preprocessing(ds_raw,Train_Start,Train_End,Test_Start,Test_End):

    mask_train=(ds_raw['Date'] > Train_Start) & (ds_raw['Date'] <= Train_End)
    mask_test = (ds_raw['Date'] > Test_Start) & (ds_raw['Date'] <= Test_End)

    ds_train = ds_raw.loc[mask_train].dropna()
    ds_train.sort_values(by=['Date'], inplace=True)
    scaler = MinMaxScaler()
    ds_train[['Consumption']] = scaler.fit_transform(ds_train[['Consumption']])

    ds_test = ds_raw.loc[mask_test].dropna()
    ds_test.sort_values(by=['Date'], inplace=True)
    ds_test[['Consumption']] = scaler.transform(ds_test[['Consumption']])

    ds_train["target"] = np.log1p(ds_train.Consumption)
    ds_test["target"] = ds_train.target.mean()

    # define groupings and corresponding priors
    groups_and_priors = {
        
        # singe encodings
        ("Hour",):        None,
        ("weekday",):     None,
        ("Month",):       None,
        ("PubHol",):      None,
        # second-order interactions
        ("Hour", "weekday"):        ["gte_Hour", "gte_weekday"],
        ("Hour", "PubHol"):        ["gte_Hour", "gte_PubHol"]}


    PRIOR_PRECISION = 10

    features = []
    for group_cols, prior_cols in groups_and_priors.items():
        features.append(f"gte_{'_'.join(group_cols)}")
        gte = GaussianTargetEncoder(list(group_cols), "target", prior_cols)    
        ds_train[features[-1]] = gte.fit_transform(ds_train, PRIOR_PRECISION)
        ds_test[features[-1]]  = gte.transform(ds_test,  PRIOR_PRECISION)

        drop_cols = ["Hour", "weekday", "Month","PubHol"]
    ds_train.drop(drop_cols, axis=1, inplace=True)
    ds_test.drop(drop_cols, axis=1, inplace=True)
    del  gte
    gc.collect()

    X_train = ds_train.drop(columns=['ObjectName','Consumption','Date','target','m_bool']).copy()
    Y_train = ds_train[['target']].copy()

    X_test = ds_test.drop(columns=['ObjectName','Consumption','Date','target','m_bool']).copy()
    Y_test = ds_test[['Consumption']].copy()
    Results=ds_test[['ObjectName','Date','Consumption','m_bool']]
    Results.reset_index(drop=True,inplace=True)

    return X_train,X_test,Y_train,Y_test, Results