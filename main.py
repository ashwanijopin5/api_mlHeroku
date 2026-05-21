from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle 
import json

app=FastAPI()

origins=["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
    
)
class model_input(BaseModel):
    Pregnancies:int
    Glucose:int
    BloodPressure:int
    SkinThickness:int
    Insulin:int
    BMI:float
    DiabetesPedigreeFunction:float
    Age:int
diabetes_model=pickle.load(open('trained_model.sav','rb'))
model=diabetes_model['model']
scaler=diabetes_model['scaler']

@app.post('/diabeties_prediction')

def diabetes_pred(input_parameters :model_input):
      
      input_data=input_parameters.json()
      input_dictionary=json.loads(input_data)
      preg=input_dictionary['Pregnancies']
      glu=input_dictionary['Glucose']
      bp=input_dictionary['BloodPressure']
      skin=input_dictionary['SkinThickness']
      insulin=input_dictionary['Insulin']
      bmi=input_dictionary['BMI']
      dpf=input_dictionary['DiabetesPedigreeFunction']   
      age=input_dictionary['Age']

      input_list=[preg,glu,bp,skin,insulin,bmi,dpf,age]
      sclaer_dat=scaler.transform([input_list])
      prediction=model.predict(sclaer_dat)

      if(prediction[0]==0):
         return 'the person is not diabetics'
      else :
          return 'the person is diabetics'      
