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


model = pickle.load(open("./models/tfidf3.pickle", "rb")) # Модель на основе TF-IDF и RandomForestClassfier


@application.route("/")  
def hello():
    resp = {'message':"Hello World!"}
    response = jsonify(resp)
    
    return response

@application.route("/categoryPrediction" , methods=['GET', 'POST'])  
def registration():
    resp = {'message':'',
           'category': -1
           }
    try:
        getData = request.get_data()
        json_params = json.loads(getData) 
    
        
        if len(json_params['user_message']) == 0:
            resp['category'] = -1
            resp['message'] = json_params['user_message']
        else:      
            textlist = []
            textlist.append(json_params['user_message'])
            category = model.predict_proba(textlist)
            s = ''
            i = 0
            catlist = ["afs", "other", "ps"]
            for c in category:
                for cc in c:
                    i+=1
                    s = s + f"p({catlist[i-1]}) = "+ str(cc) + ";    " # Для красоты вывода
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


