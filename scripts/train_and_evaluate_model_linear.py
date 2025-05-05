import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_and_prepare_data(path="data/df_processed.csv"):
    df = pd.read_csv(path)
    
    features = [
        "avg_temp_C", "max_temp_C", "min_temp_C", "total_precip_mm",
        "avg_precip_mm", "hot_days", "dry_days", "flooding_days"
    ]
    
    X = df[features]
    X = pd.concat([X, pd.get_dummies(df["state"], drop_first=True)], axis=1)
    y = df["yield"]

    return X, y

def train_linear_model(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mae_list, rmse_list, r2_list = [], [], []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_val_scaled)
        mae_list.append(mean_absolute_error(y_val, y_pred))
        rmse_list.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        r2_list.append(r2_score(y_val, y_pred))

        last_y_val = y_val
        last_y_pred = y_pred

    print(f"Linear Regression Results:")
    print(f"Average MAE:  {np.mean(mae_list):.2f}")
    print(f"Average RMSE: {np.mean(rmse_list):.2f}")
    print(f"Average R²:   {np.mean(r2_list):.3f}")

    os.makedirs("outputs/figures", exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(last_y_val, last_y_pred, alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel("Actual Yield")
    plt.ylabel("Predicted Yield")
    plt.title("Linear Regression: Actual vs Predicted Yield")
    plt.tight_layout()
    plt.savefig("outputs/figures/yield_predictions_linear.png")
    plt.close()

if __name__ == "__main__":
    X, y = load_and_prepare_data()
    train_linear_model(X, y)