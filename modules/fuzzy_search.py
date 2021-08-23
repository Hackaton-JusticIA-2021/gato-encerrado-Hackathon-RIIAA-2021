import numpy as np
import pandas as pd
import rapidfuzz
import datefinder
import re


#Función que recibe el nombre de alguna entidad a buscar, una clase ('Organización', 'Lugar', 'Servidor Público', 'Enjuiciados') y un dataframe con las transcripciones. 
#Encuentra los matchs entre la entidad y los textos del 

def find_matches(name,clase,df_transcripciones):
  df_transcripciones["Index"]=df_transcripciones.index
  matchs = rapidfuzz.process.extract(name, df_transcripciones["Texto"], scorer=rapidfuzz.fuzz.token_set_ratio,  score_cutoff=85,limit=900)
  df_aux = pd.DataFrame(np.array(matchs).reshape((len(matchs),3)),columns=["label","confianza","Index"])
  df_aux["Index"] = df_aux["Index"].astype(int)
  df_salida = df_aux.merge(df_transcripciones,  how='inner', on='Index').drop(["Conjunto","Fuente","Texto","MetodoTexto","confianza","Index"],axis = 1)
  df_salida["label"] = name
  df_salida["clase"] = clase
  return df_salida

#Función que toma como argumentos un dataframe de transcripciones en el formato de salida del reto 2A y una lista de peronas, entidades,etc a buscar 
#(Nota:Es importante que las listas tengan el mismo formato, es decir nombres de las columnas, con el que se nos proporcionaron al inicio). y devuelve un dataframe con todas las coincidencias encontradas en el formato del reto 2B
#el parámetro listas debe tener el siguiente formato:
#listas = [organizaciones,lugares,servidores,enjuiciados]

def obtener_matchs(df_transcripciones,listas):  
  df_salida = pd.DataFrame([],columns=["label","NombreArchivo","clase"])
  for lista in listas:
    for i in range(len(lista)):
      df_salida = df_salida.append(find_matches(lista.head(i+1)[lista.columns[0]].values[-1],lista.columns[0],df_transcripciones))
  return df_salida



#Función que recibe como argumentos el nombre de un archivo y un data frame con las transcripciones, devuelve un aviso si el nombre del archivo no se encuentra en la base de transcripciones 
#y devuelve un dataframe con las fechas y número de expediente encontrados en la transcripción. (El formato de salida no corresponde con el reto 2B, este se obtiene más adelante.)

def obtener_df_expedientes_fechas_single(nombre_archivo,df_transcripcion):  
  if (df_transcripcion["NombreArchivo"]==nombre_archivo).sum()==0:
    return 0

  else:
    text = df_transcripcion[df_transcripcion["NombreArchivo"]==nombre_archivo]["Texto"].values[0]
    df_fechas_exp = pd.DataFrame()
    expedientes = []
    NO_EXPEDIENTE = "No encontrado"
    j = 0
    expedientes = []
    for i in text.split():
      j = j+1
      if i.startswith("Exp"):
        expedientes.append(i)
        text = text.replace(i,'')
        NO_EXPEDIENTE = expedientes[0]

    for i in text.split():
      if i.startswith("H-") or i.startswith("L-"):
        text = text.replace(i,'')
    matches = datefinder.find_dates(text)
    for match in matches:
      df_fechas_exp = df_fechas_exp.append(pd.DataFrame(np.array([nombre_archivo, match]).reshape(1,2),
              columns=["NombreArchivo","fechas"]))
    df_fechas_exp['Expediente'] = NO_EXPEDIENTE
  return (df_fechas_exp)

#Función que toma por argumento un archivo de transcripción en el formato que devuelve el reto 2A y devuelve los expedientes y fechas encontrados con el formato del reto 2B. 
def obtener_df_expedientes_fechas(df_transcripcion):
  df_fechas_exp = pd.DataFrame()
  for nombre_archivo in df_transcripcion["NombreArchivo"]:
    df_fechas_exp = df_fechas_exp.append(obtener_df_expedientes_fechas_single(nombre_archivo,transcripciones))

  df_fechas = df_fechas_exp[["NombreArchivo","fechas"]]
  df_fechas["clase"] ="Fechas" 
  df_fechas.columns = ["NombreArchivo","label","clase"]
  df_expedientes = df_fechas_exp[["NombreArchivo","Expediente"]].groupby(['NombreArchivo','Expediente']).size().reset_index().rename(columns={0:'count'}).drop(["count"],axis =1)
  df_expedientes["clase"] ="Expediente" 
  df_expedientes.columns = ["NombreArchivo","label","clase"]
  return df_expedientes.append(df_fechas)
