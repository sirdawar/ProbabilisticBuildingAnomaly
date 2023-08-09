
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import metrics
from scipy.stats import norm



def Visualization(Results):
    listofBuildings=Results['ObjectName'].unique()
    fig = make_subplots(rows=len(listofBuildings), cols=1,shared_xaxes=False)
    
    CV_RMSE=[]
    RMSE=[]


    i=1 
    alarm_cols = ['alarm_daily','alarm_weekly'] # [ alarm_flag ]
    alarm_colors = ['orange','green','rgb(30,144,255)']
    for unique in listofBuildings:
        tempdf=Results[Results['ObjectName']==unique]
        tempdf=tempdf.groupby(['Date'])[['Consumption','Predicted','Lower bound','Upper bound','m_bool']+ alarm_cols].first()
        rmse = round(np.sqrt(metrics.mean_squared_error(tempdf['Consumption'], tempdf['Predicted'])),4)
        cv_rmse=round(rmse/np.mean(tempdf['Consumption'])*100,2)
   
        CV_RMSE.append(cv_rmse)
        RMSE.append(rmse)

        fig.add_trace(go.Scatter(x=tempdf.index,y=tempdf['Consumption'],name=('Measured '),line=dict(color="gray")),row=i,col=1)
        fig.add_trace(go.Scatter(x=tempdf.index,y=tempdf['Predicted'],name=('Predicted '),line=dict(color='red') ),row=i,col=1)
        fig.add_trace(go.Scatter(x=tempdf.index,y=tempdf['Upper bound'] ,name=('Upper bound '), mode='lines', marker=dict(color='rgba(255, 0, 0, 0.1)'), line=dict(width=0)),row=i,col=1)
        fig.add_trace(go.Scatter(x=tempdf.index,y=tempdf['Lower bound'] ,name=('Lower bound '), mode='lines', marker=dict(color='rgba(255, 0, 0, 0.1)'), line=dict(width=0), fillcolor='rgba(255, 0, 0, 0.1)', fill='tonexty'),row=i,col=1)
        for alrm in range(len(alarm_cols)):
            alarmdf=tempdf.loc[tempdf[alarm_cols[alrm]] == True]
            fig.add_trace(go.Scatter(x=alarmdf.index,y=alarmdf['Consumption'],name=('Alarm [' + str(alrm + 1) + '] ' + unique), mode='markers', marker=dict(color=alarm_colors[alrm],opacity=0.5, size = 12 - (alrm * 3)) ),row=i,col=1)

        artificial_anom=tempdf.loc[tempdf['m_bool'] == True]
        fig.add_trace(go.Scatter(x=artificial_anom.index,y=artificial_anom['Consumption'], mode='markers', marker=dict(color='black', size = 6) ),row=i,col=1)
        fig.add_annotation(xref="x domain",yref="y domain",x=0.5, y=1.1, showarrow=False,
                    text=f'{unique} CV_RMSE: {cv_rmse}, RMSE: {rmse}', row=i, col=1)
        i=i+1
    return fig