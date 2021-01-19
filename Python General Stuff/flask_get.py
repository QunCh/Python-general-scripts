## get
import requests
base = 'http://127.0.0.1:7000/?inputdata=5.1,3.5,1.4,0.2'
response = requests.get(base)
answer = response.json()
print('预测结果',answer)

##post

import requests,json
data = {
    "id":1,
    "name":'lily',
    "age":11,
    "birthplace":'san',
    "grade":123
}
url = 'http://127.0.0.1:8888/add/student/'
r = requests.post(url,data=json.dumps(data))
print(r.json())
r