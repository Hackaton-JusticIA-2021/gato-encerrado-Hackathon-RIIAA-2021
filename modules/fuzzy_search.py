import numpy as np
import pandas as pd
import rapidfuzz
import datefinder
import re


#Función para encontrar el nombre de las fichas en que se encuentra un nombre
def find_matches(name,clase,df):
  df["Index"]=df.index
  matchs = rapidfuzz.process.extract(name, df["Texto"], scorer=rapidfuzz.fuzz.token_set_ratio,  score_cutoff=80,limit=900)
  df_aux = pd.DataFrame(np.array(matchs).reshape((len(matchs),3)),columns=["label","confianza","Index"])
  df_aux["Index"] = df_aux["Index"].astype(int)
  df_salida = df_aux.merge(df,  how='inner', on='Index').drop(["Conjunto","Fuente","Texto","MetodoTexto","confianza","Index"],axis = 1)
  df_salida["label"] = name
  df_salida["clase"] = clase
  return df_salida


#Función para encontrar fechas de una ficha 
def get_dates(archivo,df):
  text = df[df["NombreArchivo"]==archivo]["Texto"].values[0]
  print(text)
  df_fechas = pd.DataFrame()
  for i in text.split():
      if i.startswith("Exp"):
          NO_EXPEDIENTE = i
          text = text.replace(i,'')
  for i in text.split():
    if i.startswith("H-") or i.startswith("L-"):
      text = text.replace(i,'')
  matches = datefinder.find_dates(text)
  for match in matches:
    df_fechas = df_fechas.append(pd.DataFrame(np.array([archivo, match]).reshape(1,2),
             columns=["NombreArchivo","fechas"]))
  df_fechas['Expediente'] = NO_EXPEDIENTE
  return (df_fechas)