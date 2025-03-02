import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import joblib
import warnings
warnings.filterwarnings('ignore')
import base64
import io

class XGBPricePredictor:
    def __init__(self, params=None):
        """
        Initialize the XGBRegressor model with default or custom parameters.
        
        Parameters:
        -----------
        params : dict, optional
            Dictionary of parameters to initialize the XGBRegressor model.
        """
        # Default parameters if none provided
        self.default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_child_weight': 1,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'random_state': 42
        }
        
        self.params = params if params is not None else self.default_params
        self.model = XGBRegressor(**self.params)
        self.history = {'train': {}, 'val': {}}
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True, early_stopping_rounds=None):
        """
        Fit the model to the training data and track metrics per epoch.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target values
        X_val : array-like, optional
            Validation features
        y_val : array-like, optional
            Validation target values
        verbose : bool, default=True
            Whether to print progress information
        early_stopping_rounds : int, optional
            Activates early stopping if validation score doesn't improve for this many rounds
        
        Returns:
        --------
        self : object
            Fitted model
        """
        # Set metrics directly in the model parameters
        self.model.set_params(eval_metric=['rmse', 'mae'])
        
        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        eval_metric_names = ['rmse', 'mae']
        eval_names = ['train']
        
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            eval_names.append('val')
        
        # For scikit-learn API, use different approach based on your XGBoost version
        try:
            # For newer versions try this approach first
            if early_stopping_rounds is not None:
                self.model.set_params(early_stopping_rounds=early_stopping_rounds)
            
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=verbose
            )
        except TypeError:
            # For older versions, try the direct parameters approach
            kwargs = {
                'eval_set': eval_set,
                'verbose': verbose
            }
            if early_stopping_rounds is not None:
                kwargs['early_stopping_rounds'] = early_stopping_rounds
            
            try:
                self.model.fit(X_train, y_train, **kwargs)
            except TypeError:
                # Final fallback - just do a basic fit
                print("Warning: Could not use evaluation sets with this XGBoost version. Using basic fit.")
                self.model.fit(X_train, y_train)
                
        # Try to extract results if available
        try:
            eval_results = self.model.evals_result()
            
            # Convert to our history format
            for i, name in enumerate(eval_names):
                eval_key = f"validation_{i}" if i > 0 else "validation_0"
                if eval_key in eval_results:
                    for metric, values in eval_results[eval_key].items():
                        if name not in self.history:
                            self.history[name] = {}
                        if metric not in self.history[name]:
                            self.history[name][metric] = []
                        self.history[name][metric] = values
        except:
            # If evals_result() is not available, create basic metrics manually
            print("Warning: Could not retrieve evaluation history. Metrics tracking may be limited.")
            y_pred_train = self.predict(X_train)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            mae_train = mean_absolute_error(y_train, y_pred_train)
            
            self.history['train']['rmse'] = [rmse_train]
            self.history['train']['mae'] = [mae_train]
            
            if X_val is not None and y_val is not None:
                y_pred_val = self.predict(X_val)
                rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
                mae_val = mean_absolute_error(y_val, y_pred_val)
                
                self.history['val'] = {}
                self.history['val']['rmse'] = [rmse_val]
                self.history['val']['mae'] = [mae_val]
        
        return self
    
    def predict(self, X):
        """
        Make predictions with the trained model.
        
        Parameters:
        -----------
        X : array-like
            Features for prediction
        
        Returns:
        --------
        array-like
            Predicted values
        """
        return self.model.predict(X)
    
    def evaluate(self, X, y_true):
        """
        Evaluate the model performance on given data.
        
        Parameters:
        -----------
        X : array-like
            Features for prediction
        y_true : array-like
            True target values
        
        Returns:
        --------
        dict
            Dictionary containing various evaluation metrics
        """
        y_pred = self.predict(X)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100  # Convert to percentage
        }
        
        return metrics
    
    def tune_hyperparameters(self, X_train, y_train, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1):
        """
        Perform grid search to find the best hyperparameters.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target values
        param_grid : dict
            Dictionary with parameters names as keys and lists of parameter settings to try
        cv : int, default=5
            Number of cross-validation folds
        scoring : str, default='neg_mean_squared_error'
            Scoring metric for evaluation
        verbose : int, default=1
            Controls verbosity of the grid search
        
        Returns:
        --------
        dict
            Best parameters found
        """
        grid_search = GridSearchCV(
            XGBRegressor(),
            param_grid,
            cv=cv,
            scoring=scoring,
            verbose=verbose,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        self.params = grid_search.best_params_
        self.model = XGBRegressor(**self.params)
        
        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_
    
    def _plot_to_base64(self, fig):
        """Helper method to convert matplotlib figure to base64 string"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        buf.close()
        return f"data:image/png;base64,{base64_str}"
    
    def plot_training_history(self, figsize=(12, 5)):
        """
        Plot the training and validation metrics over epochs and return as base64 string.
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 5)
            Figure size
            
        Returns:
        --------
        str
            Base64 encoded string of the plot image
        """
        if not self.history['train']:
            print("No training history available. Please train the model first.")
            return None
            
        metrics = list(self.history['train'].keys())
        n_metrics = len(metrics)
        
        if n_metrics == 0:
            print("No metrics found in training history.")
            return None
        
        if n_metrics == 1:
            fig, ax = plt.subplots(figsize=figsize)
            axes = [ax]
        else:
            fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
            if n_metrics == 2:
                axes = list(axes)
        
        for i, metric in enumerate(metrics):
            ax = axes[i] if n_metrics > 1 else axes[0]
            
            ax.plot(self.history['train'][metric], label=f'Training {metric}')
            if 'val' in self.history and metric in self.history['val']:
                ax.plot(self.history['val'][metric], label=f'Validation {metric}')
            
            ax.set_xlabel('Epochs')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} over epochs')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        return self._plot_to_base64(fig)
    
    def plot_feature_importance(self, feature_names=None, figsize=(10, 6), top_n=None):
        """
        Plot feature importance of the trained model and return as base64 string.
        
        Parameters:
        -----------
        feature_names : list, optional
            Names of features
        figsize : tuple, default=(10, 6)
            Figure size
        top_n : int, optional
            Number of top features to show
            
        Returns:
        --------
        str
            Base64 encoded string of the plot image
        """
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        if top_n is not None:
            importance_df = importance_df.head(top_n)
        
        fig = plt.figure(figsize=figsize)
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance')
        
        return self._plot_to_base64(fig)
    
    def plot_predictions(self, X, y_true, figsize=(10, 6)):
        """
        Plot actual vs predicted values and return as base64 string.
        
        Parameters:
        -----------
        X : array-like
            Features for prediction
        y_true : array-like
            True target values
        figsize : tuple, default=(10, 6)
            Figure size
            
        Returns:
        --------
        str
            Base64 encoded string of the plot image
        """
        y_pred = self.predict(X)
        
        fig = plt.figure(figsize=figsize)
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.grid(True, alpha=0.3)
        
        return self._plot_to_base64(fig)
    
    def get_residuals_plot(self, X, y_true, figsize=(10, 6)):
        """
        Plot residuals and their distribution, return as base64 strings.
        
        Parameters:
        -----------
        X : array-like
            Features for prediction
        y_true : array-like
            True target values
        figsize : tuple, default=(10, 6)
            Figure size
            
        Returns:
        --------
        tuple
            Two base64 encoded strings (residuals scatter plot, residuals distribution)
        """
        y_pred = self.predict(X)
        residuals = y_true - y_pred
        
        # First plot: Residuals scatter
        fig1 = plt.figure(figsize=figsize)
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        residuals_scatter = self._plot_to_base64(fig1)
        
        # Second plot: Residuals distribution
        fig2 = plt.figure(figsize=figsize)
        sns.histplot(residuals, kde=True)
        plt.xlabel('Residuals')
        plt.title('Residual Distribution')
        plt.grid(True, alpha=0.3)
        residuals_dist = self._plot_to_base64(fig2)
        
        return residuals_scatter, residuals_dist
