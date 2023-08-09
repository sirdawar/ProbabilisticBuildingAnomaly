import pandas as pd
import numpy as np
from datetime import timedelta

def Alarm_matrix(raw_ds):
    def assign_alarm_flags(raw_ds, params = { 'observed_col': 'Consumption', 'predict_col': 'Predicted', 'low_bound': 'Lower bound','up_bound':'Upper bound'} ):
        ## TODO: check if both are having same number of rows
        df = raw_ds.copy()
        coeff=0
        df['alarm_flag'] = df[[params['observed_col'],params['predict_col'],params['low_bound'],params['up_bound']]].apply(
            lambda rec: rec[0] > (1+coeff)*rec[3] or rec[0] < (1-coeff)*rec[2] ,
            axis=1
        )
        return df

    def is_outside_bounds(date):
        data = df[(df['Date'] >= date - timedelta(hours=1)) &
                  (df['Date'] <= date + timedelta(hours=1))]
        return any(data['alarm_flag'])
    df=assign_alarm_flags(raw_ds)
    df.reset_index(drop=True, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['alarm_daily'] = False
    df['alarm_weekly'] = False
    for index, row in df.iterrows():
        if row['alarm_flag']:
            prev_day_hour = row['Date'] - timedelta(days=1)
            day_before_prev_hour = row['Date'] - timedelta(days=2)
            prev_week_hour = row['Date'] - timedelta(days=7)
            day_2week_prev_hour = row['Date'] - timedelta(days=14)
            
            if is_outside_bounds(prev_day_hour) and is_outside_bounds(day_before_prev_hour):
                df.at[index, 'alarm_daily'] = True


            if is_outside_bounds(prev_week_hour) and is_outside_bounds(day_2week_prev_hour):
                df.at[index, 'alarm_weekly'] = True


    return df