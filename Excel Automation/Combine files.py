import pandas as pd
import os
import re

os.chdir(r'C:\Users\qunch\Documents\Tencent Files\393398415\FileRecv\89月份'
df = pd.DataFrame()

for f in os.listdir():
    data = pd.read_table(f, sep = '|', error_bad_lines=False, engine='python', header = None)
    # 
    data['类型'] = re.split('[_.]', f)[-2]
    df = pd.concat([df, data])


df.shape
df.to_excel('Combine File.xlsx', index = False)