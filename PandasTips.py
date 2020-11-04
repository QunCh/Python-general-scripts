import pandas as pd


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