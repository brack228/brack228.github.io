import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import pickle
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

pick1=open('2020_era_model.pkl','rb')
erareg=pickle.load(pick1)

def EraPrePro(file):
    current =pd.read_csv(file)
    current=current.rename(columns={'ERA':'ERA_current_year'})
    current=current.reset_index(drop=True)
    for col in current.columns:
        if type(current[col][1])==str:
            current[col]=current[col].str.replace('%','',regex=False)
    numbers=['Age', 'ERA_current_year', 'K/9', 'BB/9', 'K/BB', 'H/9',
       'HR/9', 'AVG', 'WHIP', 'BABIP', 'LOB%', 'GB/FB', 'LD%', 'GB%', 'FB%',
       'IFFB%', 'HR/FB', 'IFH%', 'BUH%', 'O-Swing%', 'Z-Swing%', 'Swing%',
       'O-Contact%', 'Z-Contact%', 'Contact%', 'Zone%', 'F-Strike%', 'SwStr%',
       'K%', 'BB%', 'SIERA', 'RS/9', 'Pull%', 'Cent%', 'Oppo%', 'Soft%',
       'Med%', 'Hard%', 'xFIP', 'playerid']
    for col in numbers:
        current[col]=current[col].astype(float)
    return(current)
   

pitchfeatures=['Age', 'ERA_current_year', 'K/9', 'BB/9',
       'K/BB', 'H/9', 'HR/9', 'AVG', 'WHIP', 'BABIP', 'LOB%', 'GB/FB', 'LD%',
       'GB%', 'FB%', 'IFFB%', 'HR/FB','O-Swing%', 'Z-Swing%',
       'Swing%', 'O-Contact%', 'Z-Contact%', 'Contact%', 'Zone%', 'F-Strike%',
       'SwStr%', 'K%', 'BB%','Pull%', 'Cent%', 'Oppo%',
       'Soft%', 'Med%', 'Hard%']

current=EraPrePro('pitch2019.csv')

XC=current[pitchfeatures]
current['preds']=erareg.predict(XC)
preds=current[['Name','ERA_current_year','SIERA','preds','IP']].sort_values('preds')
preds.to_csv('pitching_leaders.csv')

pick2=open('2020_woba_model.pkl','rb')
wobareg=pickle.load(pick2)

def wobaPrePro(file1):
    current =pd.read_csv(file1)
    
  
    current=current.rename(columns={'wOBA':'wOBA_current_year'})
    current=current.reset_index(drop=True)
    for col in current.columns:
        if type(current[col][1])==str:
            current[col]=current[col].str.replace('%','',regex=False)
    numbers=['Age', 'AVG', 'BB%', 'K%', 'BB/K', 'OBP',
       'SLG', 'OPS', 'ISO', 'BABIP', 'GB/FB', 'LD%', 'GB%', 'FB%', 'IFFB%',
       'HR/FB', 'IFH%', 'BUH%', 'wOBA_current_year', 'O-Swing%', 'Z-Swing%',
       'Swing%', 'O-Contact%', 'Z-Contact%', 'Contact%', 'Zone%', 'F-Strike%',
       'SwStr%', 'Pull%', 'Cent%', 'Oppo%', 'Soft%', 'Med%', 'Hard%']
    for col in numbers:
        current[col]=current[col].astype(float)
    current=current.dropna()
    
    
    return(current)

current2=wobaPrePro('batting2019.csv')

wobafeatures=['AVG', 'BB%', 'K%', 'BB/K', 'OBP',
       'SLG', 'ISO', 'BABIP', 'GB/FB', 'LD%', 'GB%', 'FB%', 'IFFB%',
       'HR/FB', 'IFH%', 'BUH%', 'wOBA_current_year', 'O-Swing%', 'Z-Swing%',
       'Swing%', 'O-Contact%', 'Z-Contact%', 'Contact%', 'Zone%', 'F-Strike%',
       'SwStr%', 'Pull%', 'Cent%', 'Oppo%', 'Soft%', 'Med%', 'Hard%']

XC=current2[wobafeatures]
current2['preds']=wobareg.predict(XC)
preds=current2[['Name','wOBA_current_year','preds','PA']].sort_values('preds',ascending=False)
preds.to_csv('batting_leaders.csv')

pick3=open('2020_reliever_era_model.pkl','rb')
rerareg=pickle.load(pick3)

def EraPrePro(file):
    current =pd.read_csv(file)
    current=current.rename(columns={'ERA':'ERA_current_year'})
    current=current.reset_index(drop=True)
    for col in current.columns:
        if type(current[col][1])==str:
            current[col]=current[col].str.replace('%','',regex=False)
    numbers=['Age', 'ERA_current_year', 'K/9', 'BB/9', 'K/BB', 'H/9',
       'HR/9', 'AVG', 'WHIP', 'BABIP', 'LOB%', 'GB/FB', 'LD%', 'GB%', 'FB%',
       'IFFB%', 'HR/FB', 'IFH%', 'BUH%', 'O-Swing%', 'Z-Swing%', 'Swing%',
       'O-Contact%', 'Z-Contact%', 'Contact%', 'Zone%', 'F-Strike%', 'SwStr%',
       'K%', 'BB%', 'SIERA', 'RS/9', 'Pull%', 'Cent%', 'Oppo%', 'Soft%',
       'Med%', 'Hard%', 'xFIP', 'playerid']
    for col in numbers:
        current[col]=current[col].astype(float)
    return(current)
   

pitchfeatures=['Age', 'ERA_current_year', 'K/9', 'BB/9',
       'K/BB', 'H/9', 'HR/9', 'AVG', 'WHIP', 'BABIP', 'LOB%', 'GB/FB', 'LD%',
       'GB%', 'FB%', 'IFFB%', 'HR/FB','O-Swing%', 'Z-Swing%',
       'Swing%', 'O-Contact%', 'Z-Contact%', 'Contact%', 'Zone%', 'F-Strike%',
       'SwStr%', 'K%', 'BB%','Pull%', 'Cent%', 'Oppo%',
       'Soft%', 'Med%', 'Hard%']

current=EraPrePro('reliefpitch2019.csv')

XC=current[pitchfeatures]
current['preds']=rerareg.predict(XC)
preds=current[['Name','ERA_current_year','SIERA','preds','IP']].sort_values('preds')
preds.to_csv('relief_pitching_leaders.csv')







