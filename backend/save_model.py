from trainer import XGBPricePredictor
from sklearn.model_selection import train_test_split    
import pandas as pd
import joblib

data = pd.read_csv("filename.csv")

X = data.drop(["Profit (USD)", "Flight Number", "Scheduled_Year", "Scheduled_Weekday", "Scheduled_Month"], axis = 1)
y = data["Profit (USD)"]
    
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
# Initialize and train model with default parameters
xgb_model = XGBPricePredictor()
xgb_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    
# Evaluate model
train_metrics = xgb_model.evaluate(X_train, y_train)
test_metrics = xgb_model.evaluate(X_test, y_test)
    
print("Training metrics:")
for metric, value in train_metrics.items():
    print(f"{metric}: {value:.4f}")
    
print("\nTest metrics:")
for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1]
}

joblib.dump(xgb_model, 'xgb_price_predictor_optimized.pkl')
print("Optimized model saved as 'xgb_price_predictor_optimized.pkl'")
    
# Uncomment to run hyperparameter tuning (can be time-consuming)
#best_params = xgb_model.tune_hyperparameters(X_train, y_train, param_grid, cv=3)