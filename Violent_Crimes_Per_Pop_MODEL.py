import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

train_df = pd.read_csv("new_train.csv")
test_df = pd.read_csv("new_test_for_participants.csv")

X = train_df.drop(columns=["ViolentCrimesPerPop", "ID"])
y = train_df["ViolentCrimesPerPop"]

test_ids = test_df["ID"]
X_test_final = test_df.drop(columns=["ID"])

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, random_state=48
)

model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=1500,
    learning_rate=0.02,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
)

model.fit(
    X_train.astype(float), y_train,
    eval_set=[(X_valid.astype(float), y_valid)],
    verbose=False
)

y_pred = model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
r2 = r2_score(y_valid, y_pred)
print(mean_squared_error(y_valid, y_pred))
print(f"Validation RMSE: {rmse:.4f}")
print(f"Validation R²: {r2:.4f}")

test_predictions = model.predict(X_test_final)

output = pd.DataFrame({
    "ID": test_ids,
    "PredictedViolentCrimesPerPop": test_predictions
})

output.to_csv("xgboost_predictions.csv", index=False)
print("Predictions saved to xgboost_predictions.csv")