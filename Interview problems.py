## Generator

def test():
    for i in range(4):
        yield i

def add(x,y):
    return x+y

g = test()
for n in [1,10]:
    print(n)
    print(next(g))
    g = (add(n,i) for i in g)
    print(next(g))
list(g)
n=10
g = (add(n,i) for i in g)


# Pandas
df2 = df.groupby(['省份','帐期']).agg({'通话时长':np.average, '投诉单ID': pd.Series.nunique})

df4 = df3[(df3['time']<x) & ((df3['avg time']>500) | (df3['distinct id']>500))]

pd.pivot_table(df3, index=['省份','帐期'], columns=['原因'], values = 'distinct id', aggfunc=np.sum)