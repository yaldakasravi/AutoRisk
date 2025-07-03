import joblib
import pandas as pd

def load_model(model_path):
    return joblib.load(model_path)

def make_prediction(model, input_data):
    df = pd.DataFrame([input_data])
    return model.predict(df)[0]
