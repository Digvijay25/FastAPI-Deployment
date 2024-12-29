import uvicorn 
from fastapi import FastAPI 
import numpy as np
import pickle 
import pandas as pd 
from BankNotes import BankNote

app = FastAPI() 

pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.get('/{name}')
def get_name(name: str):
    return {'Welcome': f'{name}'}

@app.post('/predict')
def predict_banknote(data: BankNote):
    data = data.dict()
    variance = data['variance'] 
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']

    predictions = classifier.predict([[variance, skewness, curtosis, entropy]])

    if predictions[0] > 0.5:
        prediction = 'Fake note'    
    else:
        prediction = 'Its a Bank note'  
    return {'prediction': str(prediction)}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
