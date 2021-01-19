
import numpy as np
from flask import Flask
from flask import request
from flask import jsonify
# from sklearn.externals import joblib
#导入模型
# model = joblib.load('model.pickle')

# https://blog.csdn.net/hejp_123/article/details/85260668
 
#temp =  [5.1,3.5,1.4,0.2]
#temp = np.array(temp).reshape((1, -1))
#ouputdata = model.predict(temp)	
##获取预测分类结果
#print('分类结果是：',ouputdata[0])
 
app = Flask(__name__)
 
@app.route('/',methods=['POST','GET'])
def output_data():
    text=request.args.get('inputdata')
    if text:
        temp =  np.array([float(x) for x in text.split(',')])
        # temp = np.array(temp).reshape((1, -1))
        ouputdata = temp + 1	
        return jsonify(str(ouputdata[0]))
    return "hello w"
if __name__ == '__main__':
    #app.config['JSON_AS_ASCII'] = False
    app.run(debug=True,host='127.0.0.1',port=7000)  # 127.0.0.1 #指的是本地ip
    
print('运行结束')