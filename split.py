import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Load preprocessed dataset
df = pd.read_csv("dataset_preprocessed.csv")
X = df.drop(columns=["Average_User_Rating"])
y = df["Average_User_Rating"]
feature_names = X.columns

# Setup 10-fold cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Define evaluation scorers
mae_scorer = make_scorer(mean_absolute_error)
rmse_scorer = make_scorer(mean_squared_error)

# Define models with basic anti-overfitting hyperparameter settings
models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_leaf=4, random_state=42
    ),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=10, min_samples_leaf=4, random_state=42
    ),
    "XGBoost": XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}

# Evaluation loop
results = {}
for name, model in models.items():
    r2_scores = cross_val_score(model, X, y, cv=kfold, scoring="r2")
    mae_scores = cross_val_score(model, X, y, cv=kfold, scoring=mae_scorer)
    rmse_scores = cross_val_score(model, X, y, cv=kfold, scoring=rmse_scorer)

    results[name] = {
        "R² (mean)": np.mean(r2_scores),
        "MAE (mean)": np.mean(mae_scores),
        "RMSE (mean)": np.mean(np.sqrt(rmse_scores))  # RMSE = sqrt(MSE)
    }

# Print results
print("=== Evaluasi K-Fold Cross Validation ===")
for model, metrics in sorted(results.items(), key=lambda x: x[1]["R² (mean)"], reverse=True):
    print(f"\n{model}")
    print(f"R²   : {metrics['R² (mean)']:.4f}")
    print(f"MAE  : {metrics['MAE (mean)']:.4f}")
    print(f"RMSE : {metrics['RMSE (mean)']:.4f}")

# Feature Importance Analysis (Random Forest & XGBoost)
print("\n=== Feature Importance ===")
for name in ["Random Forest", "XGBoost"]:
    model = models[name]
    model.fit(X, y)  # fit full data untuk melihat feature importance
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    print(f"\nTop 10 Feature Importance - {name}:")
    print(feature_importance_df.head(10).to_string(index=False))
