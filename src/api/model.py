import mlflow.pyfunc
import pandas as pd
import logging
import joblib


logger = logging.getLogger(__name__)

# Load the best model from MLflow model registry (production stage)
MODEL_URI = "models:/Churn Prediction Model/Production"

def load_model():
    logger.info(f"Loading model from {MODEL_URI}")
    model = mlflow.pyfunc.load_model(MODEL_URI)
    logger.info("Model loaded successfully")
    return model

model = load_model()


logger = logging.getLogger(__name__)

def predict(input_data: dict):
    try:
        # Load model and transformer
        model = joblib.load("best_model.pkl")
        transformer = joblib.load("column_transformer.pkl")

        logger.info(f"Received input data for prediction: {input_data}")

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Apply the same transformation as during training
        transformed_input = transformer.transform(input_df)

        # Make prediction
        prediction = model.predict(transformed_input)

        return {"prediction": int(prediction[0])}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise e
