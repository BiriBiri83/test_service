from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import pickle

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from flask import Flask
from flask import request
import requests
from flask import jsonify

import os
import json
from ast import literal_eval
import traceback

application = Flask(__name__)


model = pickle.load(open("./models/tfidf2.pickle", "rb"))


# тестовый вывод
@application.route("/")  
def hello():
    #resp = {'message':"Hello World!"}
    resp = {'message':"Hello World!"}
    response = jsonify(resp)
    
    return response

# предикт категории
#{"user_message":"example123rfssg gsfgfd"}
@application.route("/categoryPrediction" , methods=['GET', 'POST'])  
def registration():
    resp = {'message':'',
           'category': -1
           }
    try:
        getData = request.get_data()
        json_params = json.loads(getData) 
    
        
        #напишите прогноз и верните его в ответе в параметре 'prediction'

        #category = model.predict(vec.transform([json_params['user_message']]).toarray()).tolist()
        if len(json_params['user_message']) == 0:
            resp['category'] = -1
            resp['message'] = json_params['user_message']
        else:      
            textlist = []
            textlist.append(json_params['user_message'])
            category = model.predict_proba(textlist)
            s = ''
            i = 0
            for c in category:
                for cc in c:
                    i+=1
                    s = s + f"p(c{i}) = "+ str(cc) + ";    "
            category = s[:-5]
            resp['category'] = category
            resp['message'] = json_params['user_message'] 

        
    except Exception as e: 
        print(e)
        resp['message'] = e
        resp['category'] = -1
    
    response = jsonify(resp)
    
    return response

        

if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    application.run(debug=False, port=port, host='0.0.0.0' , threaded=True)


