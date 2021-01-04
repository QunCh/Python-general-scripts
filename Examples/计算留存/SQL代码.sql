--日活计算
select substr(ts, 1, 10) as dt, count(distinct uid)
from t_login
group by substr(ts, 1, 10)

--次日留存
SELECT substr(a.ts, 1, 10) as dt, 
count(distinct a.uid), count(distinct b.uid), 
concat(round((count(distinct b.uid) / count(distinct a.uid)) * 100, 2) ,'%') as 1_day_remain
from t_login a 
left join t_login b 
on a.uid = b.uid 
and date_add(substr(a.ts, 1, 10), INTERVAL 1 day) = substr(b.ts, 1, 10)
group by substr(a.ts, 1, 10)

--多日留存
select substr(a.ts, 1, 10) as dt, count(distinct a.uid), 
count(distinct if(datediff(substr(b.ts, 1, 10), substr(a.ts, 1, 10))=1, b.uid, null)) as 1_day_remain_uid,
count(distinct if(datediff(substr(b.ts, 1, 10), substr(a.ts, 1, 10))=6, b.uid, null)) as 7_day_remain_uid,
count(distinct if(datediff(substr(b.ts, 1, 10), substr(a.ts, 1, 10))=13, b.uid, null)) as 14_day_remain_uid
from t_login a
left join t_login b 
on a.uid = b.uid
group by 
substr(a.ts, 1, 10)