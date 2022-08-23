#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 10:45:47 2022
@author: larissa

"""

import os
import pandas as pd
from datetime import *
from functools import reduce
import numpy as np

#Importing the paths for all the stations and their data

def pathFinder(station,pol):
    st = station + '.csv'
    return  os.path.join(os.path.expanduser('~'),'Desktop','Projetos','LACCAN','dados',pol, st)


stationsPol = [['camb_pm','centro_pm','cong_pm','ibi_pm','lapa_pm','mooca_pm','pdp_pm','pin_pm','samar_pm','san_pm','smp_pm'],
 ['ibi_o3','mooca_o3','pdp_o3','pin_o3','smp_o3'],
 ['centro_co','cong_co','ibi_co','lapa_co','mooca_co','pdp_co','pin_co'],
 ['centro_no2','cong_no2','ibi_no2','pin_no2'],
 ['cong_so2','ibi_so2','pdp_so2']]

stationsPath = []
for combo in range(len(stationsPol)):
    if(combo == 0):
        for station in stationsPol[combo]:
            st = pathFinder(station,'PM10')
            stationsPath.append(st)
    elif(combo == 1):
        for station in stationsPol[combo]:
            st = pathFinder(station,'O3')
            stationsPath.append(st)
    elif(combo == 2):
        for station in stationsPol[combo]:
            st = pathFinder(station,'CO')
            stationsPath.append(st)
    elif(combo == 3):
        for station in stationsPol[combo]:
            st = pathFinder(station,'NO2')
            stationsPath.append(st)
    else:
        for station in stationsPol[combo]:
            st = pathFinder(station,'SO2')
            stationsPath.append(st)
            

#Lendo os arquivos csv - por estacao

#OBS: Tirei smp enquanto a API não funciona 
#smp 
#smp_pm = pd.read_csv(stationsPath[10],encoding='ISO-8859-1', sep = ',', parse_dates = {'date': ['Data','Hora']}, dayfirst = True)
#smp_o3 = pd.read_csv(stationsPath[15],encoding='ISO-8859-1', sep = ',', parse_dates = {'date': ['Data','Hora']}, dayfirst = True)


#camb
camb_pm = pd.read_csv(stationsPath[0],encoding='ISO-8859-1', sep = ',', parse_dates = {'date': ['Data','Hora']}, dayfirst = True)


#centro
centro_pm  = pd.read_csv(stationsPath[1],encoding='ISO-8859-1', sep = ',', parse_dates = {'date': ['Data','Hora']}, dayfirst = True)
centro_co = pd.read_csv(stationsPath[16],encoding='ISO-8859-1', sep = ',', parse_dates = {'date': ['Data','Hora']}, dayfirst = True)
centro_no2  = pd.read_csv(stationsPath[23],encoding='ISO-8859-1', sep = ',', parse_dates = {'date': ['Data','Hora']}, dayfirst = True)

#congonhas
cong_pm  = pd.read_csv(stationsPath[2],encoding='ISO-8859-1', sep = ',', parse_dates = {'date': ['Data','Hora']}, dayfirst = True)
cong_co = pd.read_csv(stationsPath[17],encoding='ISO-8859-1', sep = ',', parse_dates = {'date': ['Data','Hora']}, dayfirst = True)
cong_no2 = pd.read_csv(stationsPath[24],encoding='ISO-8859-1', sep = ',', parse_dates = {'date': ['Data','Hora']}, dayfirst = True)
cong_so2 = pd.read_csv(stationsPath[27],encoding='ISO-8859-1', sep = ',', parse_dates = {'date': ['Data','Hora']}, dayfirst = True)

#ibiripuera
ibi_pm = pd.read_csv(stationsPath[3],encoding='ISO-8859-1', sep = ',', parse_dates = {'date': ['Data','Hora']}, dayfirst = True)
ibi_o3 =pd.read_csv(stationsPath[11],encoding='ISO-8859-1', sep = ',', parse_dates = {'date': ['Data','Hora']}, dayfirst = True)
ibi_co = pd.read_csv(stationsPath[18],encoding='ISO-8859-1', sep = ',', parse_dates = {'date': ['Data','Hora']}, dayfirst = True)
ibi_no2 = pd.read_csv(stationsPath[25],encoding='ISO-8859-1', sep = ',', parse_dates = {'date': ['Data','Hora']}, dayfirst = True)
ibi_so2 = pd.read_csv(stationsPath[28],encoding='ISO-8859-1', sep = ',', parse_dates = {'date': ['Data','Hora']}, dayfirst = True)

#OBS: Tirei lapa enquanto a API não funciona 
#lapa 
#lapa_pm = pd.read_csv(stationsPath[4],encoding='ISO-8859-1', sep = ',', parse_dates = {'date': ['Data','Hora']}, dayfirst = True)
#lapa_co = pd.read_csv(stationsPath[19],encoding='ISO-8859-1', sep = ',', parse_dates = {'date': ['Data','Hora']}, dayfirst = True)

#mooca 
mooca_pm = pd.read_csv(stationsPath[5],encoding='ISO-8859-1', sep = ',', parse_dates = {'date': ['Data','Hora']}, dayfirst = True)
mooca_o3 = pd.read_csv(stationsPath[12],encoding='ISO-8859-1', sep = ',', parse_dates = {'date': ['Data','Hora']}, dayfirst = True)
mooca_co = pd.read_csv(stationsPath[20],encoding='ISO-8859-1', sep = ',', parse_dates = {'date': ['Data','Hora']}, dayfirst = True)

#pdp 
pdp_pm = pd.read_csv(stationsPath[6],encoding='ISO-8859-1', sep = ',', parse_dates = {'date': ['Data','Hora']}, dayfirst = True)
pdp_o3 = pd.read_csv(stationsPath[13],encoding='ISO-8859-1', sep = ',', parse_dates = {'date': ['Data','Hora']}, dayfirst = True)
pdp_co = pd.read_csv(stationsPath[21],encoding='ISO-8859-1', sep = ',', parse_dates = {'date': ['Data','Hora']}, dayfirst = True)
pdp_so2 = pd.read_csv(stationsPath[29],encoding='ISO-8859-1', sep = ',', parse_dates = {'date': ['Data','Hora']}, dayfirst = True)

#pin 
pin_pm = pd.read_csv(stationsPath[7],encoding='ISO-8859-1', sep = ',', parse_dates = {'date': ['Data','Hora']}, dayfirst = True)
pin_o3 = pd.read_csv(stationsPath[14],encoding='ISO-8859-1', sep = ',', parse_dates = {'date': ['Data','Hora']}, dayfirst = True)
pin_co = pd.read_csv(stationsPath[22],encoding='ISO-8859-1', sep = ',', parse_dates = {'date': ['Data','Hora']}, dayfirst = True)
pin_no2 = pd.read_csv(stationsPath[26],encoding='ISO-8859-1', sep = ',', parse_dates = {'date': ['Data','Hora']}, dayfirst = True)

#samar 
samar_pm = pd.read_csv(stationsPath[8],encoding='ISO-8859-1', sep = ',', parse_dates = {'date': ['Data','Hora']}, dayfirst = True)

#san 
san_pm = pd.read_csv(stationsPath[9],encoding='ISO-8859-1', sep = ',', parse_dates = {'date': ['Data','Hora']}, dayfirst = True)




#Fixing CO
#coList = [centro_co,cong_co,ibi_co,lapa_co,mooca_co,pdp_co,pin_co]
coList = [centro_co,cong_co,ibi_co,mooca_co,pdp_co,pin_co]

for coDF in range(0,len(coList)):
    for co in range(len(coList[coDF])):
        coList[coDF].iloc[co,1] =  coList[coDF].iloc[co,1] +  (coList[coDF].iloc[co,2]/10)
    coList[coDF] = coList[coDF].drop('Unnamed: 3',axis = 1)

centro_co = coList[0]
cong_co = coList[1]
ibi_co = coList[2]
#lapa_co = coList[3]
mooca_co = coList[3]
pdp_co = coList[4]
pin_co = coList[5]

#Turning multiple dataframes into one

'''df_temp = [smp_pm, smp_o3,camb_pm, centro_pm, centro_co,centro_no2,cong_pm, cong_co, cong_no2, cong_so2, ibi_pm, ibi_o3, 
           ibi_co,ibi_no2, ibi_so2, lapa_pm, lapa_co, mooca_pm, mooca_o3, mooca_co, pdp_pm, pdp_o3, pdp_co, pdp_so2, 
           pin_pm, pin_o3, pin_co, pin_no2, samar_pm, san_pm]
'''
df_temp = [camb_pm, centro_pm, centro_co,centro_no2,cong_pm, cong_co, cong_no2, cong_so2, ibi_pm, ibi_o3, 
           ibi_co,ibi_no2, ibi_so2, mooca_pm, mooca_o3, mooca_co, pdp_pm, pdp_o3, pdp_co, pdp_so2, 
           pin_pm, pin_o3, pin_co, pin_no2, san_pm]

for df in df_temp:
    df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y %H:%M")

stations = reduce(lambda left,right: pd.merge(left,right, on=['date'],
                                              how = 'outer'),df_temp)


#Filling missing values in the station columns 

def fix_na_1(df_col):
    h_matrix = pd.DataFrame()
    hours = 24
    #Mudei de 25 para 24 - 04/02/2022
    for i in range(24):    
        s = list(range(i, len(df_col),hours))
        h = df_col.loc[s]
        h = h.reset_index(drop = True)
        h_matrix = h_matrix.append(h)
    return h_matrix

def fix_na_2(df):
    for i in range(0,len(df)):
        for j in range(0,len(df.columns)):
            if(np.isnan(df.iloc[i,j])):
                #Check if the column and row are empty
                na_row = np.isnan(df.iloc[i,:].std())
                na_col = np.isnan(df.iloc[:,j].std())
                
                if (na_row and na_col):
                    print('Blank row and column. Could not predict')
                    break
                
                # Estimate a random sample based on row (hour) mean/sd
                elif(~na_row):
                    m1 = df.iloc[i,:].mean()
                    sd1 = df.iloc[i,:].std()
                    row_norm = round(np.random.normal(m1,sd1),2)
                    
                    # Avoid negative samples
                    while(row_norm <0):
                        row_norm = round(np.random.normal(m1,sd1),2)
                    df.iloc[i,j] = row_norm
                    
                elif(~na_col):
                    m2 = df.iloc[:,j].mean()
                    sd2 = df.iloc[:,j].std()
                    col_norm = round(np.random.normal(m2,sd2),2)
                    
                    # Avoid negative samples
                    while(col_norm <0):
                        col_norm = round(np.random.normal(m2,sd2),2)
                    df.iloc[i,j] = col_norm
                
                else:
                    m1 = df.iloc[i,:].mean()
                    m2 = df.iloc[:,j].mean()
                    sd1 = df.iloc[i,:].std() 
                    sd2 = df.iloc[:,j].std()
                    norm = abs(round(np.random.normal([m1,m2],[sd1,sd2]),2))
                    while(norm < 0):
                        norm = abs(round(np.random.normal([m1,m2],[sd1,sd2]),2))
                    df.iloc[i,j] = norm
    y = []
    for i in range(0,len(df.columns)):
        x = df.iloc[:,i]
        x = x.to_list()
        y.append(x)
    df_col = [item for sublist in y for item in sublist]
    return df_col

estacoes_t = stations.copy(deep = True)
estacoes_t = pd.DataFrame(estacoes_t.iloc[6888:7056,:])
estacoes_t = estacoes_t.reset_index()

for i in stations:
    if i == 'date':
        continue
    else:
        ts = estacoes_t.loc[:,i]
        df = fix_na_1(ts)
        df_col = fix_na_2(df)
        estacoes_t.loc[:,i] = df_col

#Treating the final dataframe for multivariate predictions
estacoes_t = estacoes_t.dropna(axis = 1, how='all')
estacoes_t = estacoes_t.drop('index',axis=1)