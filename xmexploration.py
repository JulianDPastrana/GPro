#!/home/usuario/Documents/Gpro/gpvenv/bin/python3
# Correction line 148 pydataxm.py file:
# data = pd.concat([data, temporal_data], ignore_index=True)
from pydataxm import *
import datetime as dt
import pandas as pd

# Create a pydataxm.ReadDB instance
api_object = pydataxm.ReadDB()

# Request data from the API
df_variable = api_object.request_data(
    "AporCaudal",  # MetricId field
    "Rio",  # Entity field
    dt.date(2000, 1, 1),  # Start date for the query
    dt.date(2025, 1, 1),  # End date for the query
)

# Pivot the DataFrame
pivot_df = df_variable.pivot(index="Date", columns="Name", values="Value")

# Remplace empty values by zero
pivot_df.fillna(0, inplace=True)

# Save the pivoted DataFrame to Excel
pivot_df.to_excel("streamflow_dataset.xlsx")
