import numpy as np
from numpy import nan
import pandas as pd
import gc
from sklearn.preprocessing import MinMaxScaler
from workalendar.europe import Finland
from GaussianTargetEncoder import GaussianTargetEncoder

def Preprocessing(ds_raw,Train_Start,Train_End,Test_Start,Test_End):
    # Clean from zero and outliers + MinMax scaler
    def replace(group):
        mean, std = group.mean(), group.std()
        outliers = (group - mean).abs() > 4*std #Changed to 4times instead of 3 28072022
        group[outliers] = "outlier"
        return group
    ds_raw['Consumption_corrected']=ds_raw['Consumption']
    ds_raw['Consumption_corrected']=ds_raw['Consumption_corrected'].transform(replace)
    ds_raw['Outlier']=ds_raw.Consumption_corrected=="outlier"
    ds_raw['Consumption']=ds_raw.Consumption_corrected.replace(['outlier'],nan)
    ds_raw=ds_raw.drop(columns=['Consumption_corrected','Outlier'])

    def upper_boundary(group):
        values = group
        median = np.percentile(values, 50)
        std = np.std(values)
        upper_b = median + 6*std
        return upper_b

    def cleaned_data(group, upper_b, window_size=168):
        times_outlier = group[group['Consumption'] > upper_b]['Date']
        ind_to_change = group['Consumption'] > upper_b

        # Calculate the rolling median with the specified window size for previous data points
        rolling_median = group['Consumption'].rolling(window=window_size).median().shift(1)
        
        # Handle the edge case for the start of the data where the rolling median cannot be calculated
        rolling_median.iloc[0] = group['Consumption'].iloc[0]

        # Update the 'Consumption' values with the median of the previous values for the identified outliers
        group['Consumption'].loc[ind_to_change] = rolling_median.loc[ind_to_change]

        return group['Consumption'], times_outlier
    upper_b=upper_boundary(ds_raw['Consumption'])
    ds_raw['Consumption'], times_outlier = cleaned_data(ds_raw, upper_b, window_size=168)
    # Get Public holidays
    cal = Finland()
    start = ds_raw.Date.min()
    start_year = start.year  # Assuming dates are Timestamp objects.
    end = ds_raw.Date.max()
    end_year = end.year 
    holidays = set(holiday[0] 
                for year in range(start_year, end_year + 1)
                for holiday in cal.holidays(year)
                if start <= pd.Timestamp(holiday[0]) <= end)
    ds_raw['PubHol']=ds_raw.Date.dt.date.isin(holidays).astype(int)

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
