from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import csv
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS from all origins (or restrict to specific origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins or set a specific list
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Load and prepare training data
training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

reduced_data = training.groupby(training['prognosis']).max()
reduced_data.index = reduced_data.index.astype(str).str.strip().str.lower()

le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

# Dictionaries
severityDictionary = {}
description_list = {}
precautionDictionary = {}

def load_auxiliary_data():
    with open('MasterData/symptom_severity.csv') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 2 and row[0].strip():
                try:
                    severityDictionary[row[0].strip().lower()] = int(row[1])
                except ValueError:
                    print(f"Invalid severity value: {row}")
            else:
                print(f"Skipping malformed severity row: {row}")

    with open('MasterData/symptom_Description.csv') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 2 and row[0].strip():
                description_list[row[0].strip().lower()] = row[1]
            else:
                print(f"Skipping malformed description row: {row}")

    with open('MasterData/symptom_precaution.csv') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 5 and row[0].strip():
                precautionDictionary[row[0].strip().lower()] = row[1:5]
            else:
                print(f"Skipping malformed precaution row: {row}")

load_auxiliary_data()

# Input schema
class SymptomInput(BaseModel):
    symptoms: List[str]
    days: int

# Helper functions
def get_prediction(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=20)
    clf2 = DecisionTreeClassifier()
    clf2.fit(X_train, y_train)

    symptom_index = {symptom: i for i, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptom_index))
    for symptom in symptoms_exp:
        input_vector[symptom_index[symptom]] = 1

    return clf2.predict([input_vector])[0]

def calculate_risk(symptoms, days):
    severity_sum = sum(severityDictionary.get(symptom, 0) for symptom in symptoms)
    risk_score = (severity_sum * days) / (len(symptoms) + 1)
    return risk_score

@app.post("/predict")
def predict_disease(data: SymptomInput):
    user_symptoms = [symptom.strip().lower().replace(" ", "_") for symptom in data.symptoms]
    unknown = [sym for sym in user_symptoms if sym not in x.columns]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown symptoms: {', '.join(unknown)}")

    input_vector = [0] * len(cols)
    for symptom in user_symptoms:
        input_vector[x.columns.get_loc(symptom)] = 1

    prediction_index = clf.predict([input_vector])[0]
    prediction = le.inverse_transform([prediction_index])[0].strip().lower()

    second_prediction = get_prediction(user_symptoms).strip().lower()

    risk = calculate_risk(user_symptoms, data.days)
    doctor_required = risk > 13

    precautions = precautionDictionary.get(prediction, [])
    description = description_list.get(prediction, "No description available.")

    return {
        "primary_diagnosis": prediction,
        "secondary_diagnosis": second_prediction,
        "description": description,
        "precautions": precautions,
        "risk_score": risk,
        "consult_doctor": doctor_required
    }
