# save_model.py

import mlflow.pyfunc
import mlflow.sklearn
import joblib

# Load your trained model
model = joblib.load("best_model.pkl") 
# Save the model in MLflow format to a folder named 'model'
mlflow.sklearn.save_model(sk_model=model, path="model")