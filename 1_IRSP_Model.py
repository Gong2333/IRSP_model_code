import numpy as np
import pandas as pd
import joblib  # Used for model saving and loading
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class IRSPModel:
    def __init__(self, n_estimators=500, random_state=5, max_depth=7, min_samples_split=5, min_samples_leaf=2, max_leaf_nodes=25):
        """
        Initialize the RandomForestRegressor model.
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=5,
            max_depth=max_depth,
            # min_samples_split=min_samples_split,
            # min_samples_leaf=min_samples_leaf,
            # max_leaf_nodes=max_leaf_nodes
        )
        self.is_trained = False  # Flag indicating whether the model has been trained
        self.X_train = None  # Training data
        self.y_train = None  # Training labels

    def train(self, X_train, y_train):
        """
        Train the model and store training data.
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.X_train = X_train  # Store training data
        self.y_train = y_train  # Store training labels
        print("Model training completed! Training data stored.")

    def predict(self, X=None):
        """
        Make predictions. If X=None, use stored training data for prediction.
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Please call the train() method first!")

        if X is None:
            print("No X provided, using training data for prediction by default.")
            return self.model.predict(self.X_train)

        return self.model.predict(X)

    def evaluate(self, X_test=None, y_test=None):
        """
        Evaluate model performance and return R², NRMSE, and MAE.
        If X_test and y_test are not provided, use training data for evaluation.
        """
        if X_test is None or y_test is None:
            print("No test data provided, using training data for evaluation by default.")
            X_test, y_test = self.X_train, self.y_train

        y_pred = self.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        nrmse = rmse / (max(y_test) - min(y_test))  # Normalized Root Mean Squared Error (NRMSE)
        mae = mean_absolute_error(y_test, y_pred)

        print(f"Model Evaluation Results:")
        print(f"R² Score: {r2:.3f}")
        print(f"NRMSE: {nrmse:.3f}")
        print(f"MAE: {mae:.3f}")

        return r2, nrmse, mae

    def save_model(self, filename="random_forest_model.pkl"):
        """
        Save the model and training data to a local file.
        """
        save_data = {
            "model": self.model,
            "X_train": self.X_train,
            "y_train": self.y_train
        }
        joblib.dump(save_data, filename)
        print(f"Model and training data have been saved as {filename}")

    def load_model(self, filename="random_forest_model.pkl"):
        """
        Load the model from a local file and restore training data.
        """
        load_data = joblib.load(filename)
        self.model = load_data["model"]
        self.X_train = load_data["X_train"]
        self.y_train = load_data["y_train"]
        self.is_trained = True
        print(f"Model has been loaded from {filename}! Training data restored.")

    def plot_results(self, y_true=None, y_pred=None):
        """
        Plot a scatter plot with a trend line and confidence interval, displaying R², NRMSE, and MAE.
        If y_true and y_pred are not provided, use training data for visualization.
        """
        if y_true is None or y_pred is None:
            print("No data provided, using training data for visualization by default.")
            y_true = self.y_train
            y_pred = self.predict(self.X_train)

        fig, ax = plt.subplots(figsize=(6, 6))

        # Convert to NumPy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Plot scatter plot
        ax.scatter(y_true, y_pred, label="Data", alpha=0.6, edgecolor='k', s=80)

        # Perform linear regression fitting
        sorted_idx = np.argsort(y_true)
        X_sorted = y_true[sorted_idx]
        y_sorted = y_pred[sorted_idx]

        X_with_const = sm.add_constant(X_sorted)
        model = sm.OLS(y_sorted, X_with_const)
        results = model.fit()

        # Compute 95% confidence interval
        pred = results.get_prediction(X_with_const)
        pred_summary = pred.summary_frame(alpha=0.05)
        pred_mean = pred.predicted_mean

        # Plot trend line
        ax.plot(X_sorted, pred_mean, color="red", label="Trend Line", linewidth=2)

        # Plot confidence interval
        ax.fill_between(X_sorted, pred_summary['mean_ci_lower'], pred_summary['mean_ci_upper'],
                        color='red', alpha=0.2, label="95% Confidence Interval")

        # Compute error metrics
        r2, nrmse, mae = self.evaluate(y_true, y_pred)

        # Add diagonal reference line
        ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'b--', label="Ideal Fit")

        # Set title to display R², NRMSE, and MAE
        ax.set_title(f"$R^2 = {r2:.2f}$ | $NRMSE = {nrmse:.3f}$ | $MAE = {mae:.3f}$",
                     fontdict={'family': 'Arial', 'size': 16})

        # Set axis labels
        ax.set_xlabel("Actual Values", fontdict={'family': 'Arial', 'size': 14})
        ax.set_ylabel("Predicted Values", fontdict={'family': 'Arial', 'size': 14})

        ax.legend(fontsize=12)
        plt.show()
