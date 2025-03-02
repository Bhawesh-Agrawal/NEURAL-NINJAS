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

param_grid = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 7,
    'min_child_weight': 1,
    'gamma': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'random_state': 42
}
    
# Initialize and train model with default parameters
xgb_model = XGBPricePredictor(params=param_grid)
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



joblib.dump(xgb_model, 'xgb_price_predictor_optimized.pkl')
print("Optimized model saved as 'xgb_price_predictor_optimized.pkl'")
    
# Uncomment to run hyperparameter tuning (can be time-consuming)
#best_params = xgb_model.tune_hyperparameters(X_train, y_train, param_grid, cv=3)