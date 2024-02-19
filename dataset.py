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























# df = api_object.get_collections()
# for metricid, metircname in zip(df["MetricId"], df["MetricName"]):
#     print(metricid, metircname)

# df_generacion = api_object.request_data(
#                                     'Gene',
#                                     'Recurso',
#                                     start,
#                                 end) 

# df_sistema = api_object.request_data(
#                                 'ListadoRecursos',
#                                 'Sistema', 
#                                 start,
#                                 end) 
# print(df_sistema.columns)
# df_sistema = df_sistema.drop(columns=['Date', "Id"])

# df = pd.merge(df_generacion,df_sistema,left_on=['Values_code'],right_on=['Values_Code'],how='left')
# df_hidraulica = df.query('Values_Type == "HIDRAULICA"')
# df_hidraulica = df_hidraulica.query('Values_Disp == "DESPACHADO CENTRALMENTE"')
# # df_hidraulica = df_hidraulica.query('Values_RecType == "NORMAL"')
# # df_hidraulica = df_hidraulica.query('Values_EnerSource == "AGUA"')
# # df_hidraulica = df_hidraulica.query('Values_State == "OPERACION"')
# # columns_to_sum = [f'Values_Hour{str(i).zfill(2)}' for i in range(1, 25)]
# # df_hidraulica = df.groupby('Date')[columns_to_sum].sum()

# # # Optionally, reset the index if you want 'date' back as a column
# # df_hidraulica.reset_index(inplace=True)
# df_generacion_diaria = df_generacion.sum(axis=1, skipna=True, numeric_only=True)
# df_hidraulica.to_excel("generation.xlsx")



# df_demareal = api_object.request_data(
#                                 'DemaReal',
#                                 'Sistema', 
#                                 start,
#                                 end) 

# df_demareal.to_excel("demand.xlsx")





# # Request data from the API
# df_variable = api_object.request_data(
#     "AporCaudal",  # MetricId field
#     "Rio",  # Entity field
#     dt.date(2000, 1, 1),  # Start date for the query
#     dt.date(2025, 1, 1),  # End date for the query
# )

# Pivot the DataFrame
# pivot_df = df_variable.pivot(index="Date", columns="Name", values="Value")

# # Remplace empty values by zero
# pivot_df.fillna(0.0, inplace=True)


# df_variable = api_object.request_data(
#     "Gene",
#     "Sistema",
#     dt.date(2024, 1, 7),
#     dt.date(2024, 3, 8)
# )
# df_variable.to_excel("dataset.xlsx")