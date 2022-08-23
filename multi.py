#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#from prediction import estacoes_t
import os
import pandas as pd
from datetime import *
from functools import reduce
import numpy as np
from numpy import asarray
import cv2
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages

#Importando os dados já completos de prediction

#from prediction import estacoes_t as airpol
#airpol = pd.read_csv('C:\Users\dwlar\Desktop\Projetos\LACCAN\version_multi01\Multivariate_air_pollution\airpol.csv')
airpol = pd.read_csv('airpol.csv')

def as_grid25(dataMatrix, gridsize = 25, nmax = 1000):
    step = int(nmax/gridsize)
    sectors = np.arange(0,1000,step)
    coverageMatrix = np.zeros((25,25))
    
    #Creating the gridded map
    for i in range(0, len(sectors)):
        for j in range(0, len(sectors)):
            
            #analysing the original sectors and downsizing
            for x in range(sectors[i],sectors[i]+(step-1)):
                for y in range(sectors[j],sectors[j] + (step -1)):
                    #if there's a sample, cover the sector
                    if(dataMatrix[x,y] == 1):
                        coverageMatrix[i,j] = 1
    return coverageMatrix

def sectorize_coord():
    all_stations = np.zeros((1000,1000));
    all_stations[240][540] = 1 # X73 - CONGONHAS
    all_stations[300][780] = 1 # X94 - CENTRO
    all_stations[320][680] = 1 # X90 - CAMBUCI
    all_stations[230][600] = 1 # X83 - IBIRAPUERA
    all_stations[360][720] = 1 # X85 - MOOCA
    all_stations[250][430] = 1 # X72 - P. D. PEDRO II
    all_stations[210][720] = 1 # X99 - PINHEIROS
    all_stations[230][810] = 1 # X63 - SANTANA
    all_stations[250][480] = 1 # X64 - SANTO AMARO
    
    all_stations = as_grid25(all_stations) # 1
    rows, cols = np.where(all_stations == 1)
    all_stations = np.vstack((rows, cols)).T
    #Talvez usar: all_stations[np.arange(condicional- all_stations == 1)]
	#all_stations = all_stations[order(all_stations[,1]),] # 3
    
    return(all_stations)


def map_coords():
    img_path = 'spmap.png'
    img = cv2.imread(img_path,0);
    spmap = img / 255.0;
    
    coords = as_grid25(spmap).astype(int)
    coords = np.where((coords==0)|(coords==1), (coords)^1, coords)
    #print(coords)
    
    return coords

def combination(station_id, var_name):
    df_cols = np.array(np.meshgrid(station_id,var_name)).T
    
    df_cols = np.concatenate((df_cols[0],df_cols[2],df_cols[4],df_cols[7],df_cols[8]),axis=0)
    
    df_cols = np.delete(df_cols,np.where(df_cols[:,1] == 'SO2') ,axis=0)
    df_cols = np.delete(df_cols,np.where(df_cols[:,1] == 'NO2') ,axis=0)
    df_cols = pd.DataFrame(data = df_cols,columns = ['Var2', 'Var1'])
    return df_cols

'''
def snapshot_series(station_coords, st_airpol_nafix, var_name, station_id):
    airpol_snap = st_airpol_nafix
    

def prediction_series(airpol,var_name,station_coords):
    for(i in range(len(airpol))):
        airpol_snapshot = airpol
    return 'ok'
'''

sp_coords = map_coords()
#No original as coordenadas sao convertidas em um csv...talvez fazer
#write.csv(sp_coords, "../environment/spcoords_25x25.csv", row.names=FALSE)

station_coord = sectorize_coord()

station_id = [["73-"],["94-"],["90-"],["83-"],["85-"],["72-"],["99-"],["63-"],["64-"]]
var_name = [["CO"],["PM10"], ["O3"], ["NO2"], ["SO2"]]

station_id_coord = np.append(station_id,station_coord, axis = 1)

df_cols = combination(station_id, var_name)

#PREDICTION - FIRST STEP
var_name_CO_PM10_O3 = ["CO", "MP10", "O3"]
station_id_CO_PM10_03 = ["83-","85-","72-","99-"]

CO_PM10_03_coords = pd.DataFrame(station_id_coord, columns = ['station_id', 'x','y']).drop([0,1,2,7,8]).reset_index()
#CO_PM10_03_snapshot_series = snapshot_series(CO_PM10_03_coords,airpol,var_name_CO_PM10_O3,station_id_CO_PM10_03)
#CO_PM10_03_reconst = predict_series(CO_PM10_03_snapshot_series,var_name_CO_PM10_O3,sp_coords)


