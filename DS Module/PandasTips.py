import pandas as pd

# 读数

path = r'C:\亚信\2020年11月5G经分会\20201028V2\5G投诉汇总数据20201028V4.xlsx'
path = r'C:\Users\qunch\Documents\Tencent Files\393398415\FileRecv\89月份\tmp_PRD_COMPLAINT_202008_5gdengwang.txt'
df = pd.read_table(path, header = None, sep = '|', error_bad_lines = False, engine = 'python')
df = pd.read_table(path, header = None, usecols = [1],sep = '|', error_bad_lines = False,engine = 'python')

df.head()

df = pd.read_excel(path, header=1,usecols = 'A:G')
df.head()


#######

people = {
    "first": ["Corey", 'Jane', 'John'], 
    "last": ["Schafer", 'Doe', 'Doe'], 
    "email": ["CoreyMSchafer@gmail.com", 'JaneDoe@email.com', 'JohnDoe@email.com']
}

df = pd.DataFrame(people)
df.set_index('email', inplace = True)
df.index

df2 = df.groupby(['省份','帐期']).agg({'通话时长':np.average, '投诉单ID': pd.Series.nunique})

# merge 链接，类似SQL join
df3 = df.merge(df2, how = 'left', on = ['省份', '帐期'])

# concat 拼接，类似sql union
# Best to Combine two DataFrame objects with identical columns.
# Combine DataFrame objects with overlapping columns and return everything. Columns outside the intersection will be filled with NaN values
df1 = pd.DataFrame([['a', 1], ['b', 2]], columns=['letter', 'number'])
df2 = pd.DataFrame([['c', 3], ['d', 4]], columns=['letter', 'number'])
df3 = pd.DataFrame([['c', 3, 'cat'], ['d', 4, 'dog']], columns=['letter', 'number', 'animal'])

pd.concat([df1, df2], ignore_index=True)
pd.concat([df1, df3], sort=False)
pd.concat([df1, df3], join="inner") # keep same columns

%%time
df4 = df3[(df3['avg time'>500]) | (df3['distinct id']>500)]