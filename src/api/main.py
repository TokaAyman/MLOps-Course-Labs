from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import mlflow.pyfunc
from model import predict  # relative import within src/api
from prometheus_fastapi_instrumentator import Instrumentator
import joblib

model = joblib.load("best_model.pkl")
transformer = joblib.load("column_transformer.pkl")

app = FastAPI()
# Instrumentation setup
Instrumentator().instrument(app).expose(app)

# Define input schema
class CustomerInput(BaseModel):
    CreditScore: float
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
instrumentator = Instrumentator().instrument(app).expose(app)
@app.get("/")
def root():
    return {"message": "Hello from FastAPI"}
@app.get("/")
def home():
    return {"message": "Welcome to the Bank Churn Prediction API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict_endpoint(request: CustomerInput):
    data_dict = request.dict()
    df = pd.DataFrame([data_dict])

    # 1) Preprocess
    X = transformer.transform(df)

    # 2) Predict
    pred = model.predict(X)[0]

    return {"prediction": int(pred)}



