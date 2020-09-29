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