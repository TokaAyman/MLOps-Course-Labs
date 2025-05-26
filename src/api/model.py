import pandas as pd
import logging
import joblib
import os

logger = logging.getLogger(__name__)

# Paths relative to the current file (src/api/model.py)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_PATH = "best_model.pkl"

TRANSFORMER_PATH = os.path.join(BASE_DIR, "column_transformer.pkl")

def load_model():
    logger.info(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully")
    return model

def load_transformer():
    logger.info(f"Loading transformer from {TRANSFORMER_PATH}")
    transformer = joblib.load(TRANSFORMER_PATH)
    logger.info("Transformer loaded successfully")
    return transformer

# Load model and transformer once to reuse
model = load_model()
transformer = load_transformer()

def predict(input_data: dict):
    try:
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
