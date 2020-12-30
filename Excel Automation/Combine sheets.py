import pandas as pd
import os
os.chdir(r'C:\Github\Python-general-scripts\Excel Automation')

df = pd.read_excel('combinesheets.xlsx', sheet_name=None, usecols='A:C')
data = pd.DataFrame()

for key, val in df.items():
   data = pd.concat([data, val])

data.shape
data.columns
data = data.dropna(how = 'all')

data = data.iloc[:,:-1]

data.rename(columns = {'Unnamed: 1':'地市', 'Unnamed: 2':'GDP'}, inplace = True)

data.to_excel('combine GDP.xlsx', index = False)