from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from trainer import XGBPricePredictor
import joblib
from sklearn.model_selection import train_test_split

app = FastAPI(
    title="XGBRegression Playground",
    description="A backend for training and predicting with XGBoost",
    version="1.0.0"
)

# Load dataset (for training and feature names)
try:
    data = pd.read_csv("filename.csv")
    X = data.drop(["Profit (USD)", "Flight Number", "Scheduled_Year", "Scheduled_Weekday", "Scheduled_Month"], axis=1)
    y = data["Profit (USD)"]
except FileNotFoundError:
    print("Dataset not found. Using mock data.")
    data = pd.DataFrame({
        "Profit (USD)": np.random.rand(1000),
        "Flight Number": np.random.randint(1000, 9999, 1000),
        "Scheduled_Year": np.random.randint(2020, 2025, 1000),
        "Scheduled_Weekday": np.random.randint(1, 8, 1000),
        "Scheduled_Month": np.random.randint(1, 13, 1000),
        "Feature1": np.random.rand(1000),
        "Feature2": np.random.rand(1000)
    })
    X = data.drop(["Profit (USD)", "Flight Number", "Scheduled_Year", "Scheduled_Weekday", "Scheduled_Month"], axis=1)
    y = data["Profit (USD)"]

# Load the pre-trained model
try:
    pretrained_model = joblib.load("xgb_price_predictor_optimized.pkl")
    print("Pre-trained model loaded successfully.")
except FileNotFoundError:
    print("Pre-trained model not found. Please train and save the model first.")
    pretrained_model = None

class Hyperparameters(BaseModel):
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 5
    min_child_weight: float = 1.0
    gamma: float = 0.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    test_size: float = 0.2
    validation_size: float = 0.2

class PredictionInput(BaseModel):
    features: list[float]  # List of feature values matching X.columns

@app.get("/")
async def root():
    return {"message": "Welcome to the XGBRegression Playground API!"}

@app.post("/train")
async def train_model(hparams: Hyperparameters):
    params = {k: v for k, v in hparams.dict().items() if k not in ["test_size", "validation_size"]}
    test_size = hparams.test_size
    validation_size = hparams.validation_size

    if not (0 < test_size < 1):
        return {"error": "test_size must be between 0 and 1"}
    if not (0 < validation_size < 1):
        return {"error": "validation_size must be between 0 and 1"}
    if test_size + (1 - test_size) * validation_size >= 1:
        return {"error": "Combined test_size and validation_size leave no data for training"}

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=validation_size, random_state=42)

    trainer = XGBPricePredictor(params=params)
    trainer.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=False)
    
    test_metrics = trainer.evaluate(X_test, y_test)
    training_history_plots = trainer.plot_training_history()
    feature_importance_plot = trainer.plot_feature_importance(feature_names=X.columns)
    predictions_plot = trainer.plot_predictions(X_test, y_test)
    residuals_scatter_plots, residuals_dist_plots = trainer.get_residuals_plot(X_test, y_test)
    
    return {
        "test_metrics": test_metrics,
        "training_history_plots": training_history_plots,
        "feature_importance_plot": feature_importance_plot,
        "predictions_plot": predictions_plot,
        "residuals_plots": residuals_scatter_plots,
        "residuals_distribution_plots": residuals_dist_plots,
        "data_split": {
            "train_size": len(X_train),
            "validation_size": len(X_val),
            "test_size": len(X_test)
        }
    }

@app.post("/predict")
async def predict(input_data: PredictionInput):
    if pretrained_model is None:
        return {"error": "Pre-trained model not loaded. Please ensure 'xgb_price_predictor.pkl' exists."}
    
    # Validate input length matches feature count
    if len(input_data.features) != len(X.columns):
        return {"error": f"Expected {len(X.columns)} features, got {len(input_data.features)}"}
    
    # Convert input to numpy array and predict
    features = np.array([input_data.features])
    prediction = pretrained_model.predict(features)
    
    return {"prediction": float(prediction[0])}