from fastapi import FastAPI
from pydantic import BaseModel
import ollama
import os

class ModelDetails(BaseModel):
    modelName: str
    fileName: str
    handlingMissingValueStrategy: str
    encodingCategoricalDataMethod: str
    normalizationTechnique: str
    mlModelName: str
    projectId: str
    userId: str

app = FastAPI()

@app.get("/")
async def root():
    return { "message" : "Hello world" }

@app.post("/generate")
async def generate(prompt: str):
    response = ollama.chat(model="llama3.2", messages=[{
        "role": "user",
        "content": prompt
    }])
    # return { "response": response["message"]["content"]}
    return response

# todo: develop a end point that create a python file. and creating a ml model according to details it going to receive 

@app.post("/create-ml-model")
async def create_ml_model(details: ModelDetails):
    file_content = f"""# Auto-generated ML pipeline for {details.modelName}

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import joblib

# Dummy data loading (replace with your actual data path)
data = pd.read_csv('your_dataset.csv')

# Handle missing values
imputer = SimpleImputer(strategy="{details.handlingMissingValueStrategy}")
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Encode categorical data
if "{details.encodingCategoricalDataMethod}" == "onehot":
    data_encoded = pd.get_dummies(data_imputed)
else:
    for col in data_imputed.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data_imputed[col] = le.fit_transform(data_imputed[col])
    data_encoded = data_imputed

# Normalize data
scaler = {"MinMaxScaler()" if details.normalizationTechnique == "minmax" else "StandardScaler()"}
data_scaled = pd.DataFrame(scaler.fit_transform(data_encoded), columns=data_encoded.columns)

# Define target and features
X = data_scaled.drop('target', axis=1)  # Replace 'target' with your actual target column
y = data_scaled['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create ML model
model = {"RandomForestClassifier()" if details.mlModelName == "RandomForestClassifier" else "SVC()" if details.mlModelName == "SVC" else "LogisticRegression()"}

# Train the model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, '{details.modelName}.pkl')
"""

    # Write to file
    file_path = f"./{details.fileName}"
    with open(file_path, "w") as f:
        f.write(file_content)

    return { "message": f"File '{details.fileName}' created successfully." }
    