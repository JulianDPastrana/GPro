#!/home/usuario/Documents/Gpro/gpvenv/bin/python3
# Correction line 147 pydataxm.py file:
# data = pd.concat([data, temporal_data], ignore_index=True)
from pydataxm import *
import datetime as dt
import pandas as pd

def save_by_plant(df):
    df['generacion_diaria [GWh]'] = df.sum(axis=1, skipna=True, numeric_only=True) / 1e6
    df['Date'] = pd.to_datetime(df['Date'])
    df_pivoted = df.pivot(index='Date', columns='Values_Name', values='generacion_diaria [GWh]')
    df_pivoted['Total [Gwh]'] = df_pivoted.sum(axis=1, skipna=True, numeric_only=True)
    return df_pivoted

# Create a pydataxm.ReadDB instance
start, end = dt.date(2000, 1, 1), dt.date(2024, 2, 12)
api_object = pydataxm.ReadDB()

df = api_object.get_collections()
df.to_excel("get_collection.xlsx")

# USEFULL VOLUME 
df_sistema = api_object.request_data(
                                'VoluUtilDiarEner',
                                'Embalse', 
                                start, 
                                end)
pivot_df = df_sistema.pivot(index='Date', columns='Name', values='Value')
pivot_df.to_excel("useful_volume.xlsx")
# GENERATION
df_generacion = api_object.request_data(
                    coleccion="Gene",                 
                    metrica="Recurso",                     
                    start_date=start,       
                    end_date=end,          
                    )

df_sistema = api_object.request_data(
                                'ListadoRecursos',
                                'Sistema', 
                                start, 
                                end)

df_sistema = df_sistema[['Values_Code', 'Values_Name', 'Values_Type', 'Values_Disp']]
df = pd.merge(df_generacion,df_sistema,left_on=['Values_code'],right_on=['Values_Code'],how='left')
df = df.query('Values_Disp == "DESPACHADO CENTRALMENTE"')

# HYDRO
df_hidraulica = df.query('Values_Type == "HIDRAULICA"')
save_by_plant(df_hidraulica).to_excel("hydro_gen.xlsx")

# TERMO
df_termica = df.query('Values_Type == "TERMICA"')
save_by_plant(df_termica).to_excel("termo_gen.xlsx")

# DEMAND
df_demanda = api_object.request_data(
                    coleccion="DemaSIN",                 
                    metrica="Sistema",                     
                    start_date=start,       
                    end_date=end,          
                    )

df_demanda["Date"] = pd.to_datetime(df_demanda['Date'])
df_demanda.set_index('Date', inplace=True)
df_demanda["Value [GWh]"] = df_demanda["Value"] / 1e6
df_demanda.drop(columns=["Value", "Id"], inplace=True)
df_demanda.to_excel("demanda.xlsx")
