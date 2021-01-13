import datetime

d = datetime.date(1993,10,11)

tday = datetime.date.today()
tdelta = datetime.timedelta(days=7)
tday + tdelta
tday - d

t = datetime.datetime(1993,10,11, 13,30,54)

print(t + tdelta)


import pytz
t = datetime.datetime(1993,10,11, 13,30,54, tzinfo=pytz.UTC)
t = datetime.datetime.now()
print(dt_now)

print(t.isoformat())

t.strftime('%Y-%m-%d %H:%M:%S')

dt_str = '2020-09-28 22:29:44'
dt = datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
dt

df = pd.read_csv('mydata.csv', parse_date = ['Date'], date_parser = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
df.set_index('Date')

df['2019'] # df.loc['2019]
df['2019-01' : '2020-02']
df['2019-01' : '2020-02']['price'].mean()

# Check for daily data instead of hourly
df['High'].resample('D').max()   # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects

# resample by week
df.resample('W').agg({'Close' : 'mean', 'High' : 'max', 'Low' : 'min', 'Volume' : 'sum'})