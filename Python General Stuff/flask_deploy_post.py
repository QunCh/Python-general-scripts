
from flask import Flask, request, jsonify
import json
 
app = Flask(__name__)
app.debug = True
 
@app.route('/add/student/',methods=['post'])
def add_stu():
    if not request.data:
        return 'fail'
    student = request.data.decode('utf-8')
    #获取到POST过来的数据，因为我这里传过来的数据需要转换一下编码。根据晶具体情况而定
    #request_id = student["id"]
    return request.data
    #student_json = json.loads(request_id)
    #把区获取到的数据转为JSON格式。
    #return jsonify(student_json)
    #返回JSON数据。
 
if __name__ == '__main__':
    #127.0.0.1是本地的意思，如果要部署到服务器需要安装世界的ip地址进行修改
    app.run(host='127.0.0.1',port=8888)