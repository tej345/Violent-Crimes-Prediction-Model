import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
import contextlib, os

# Load data
train = pd.read_csv("new_train.csv")
test = pd.read_csv("new_test_for_participants.csv")

X = train.drop(columns=["ID", "ViolentCrimesPerPop"]).copy()
y = train["ViolentCrimesPerPop"]
X_test = test.drop(columns=["ID"]).copy()

# ================= Feature Engineering =================
# Youth population percentage: younger people may influence crime rates
X['youth_pct'] = X['agePct12t29'] + X['agePct16t24']
X_test['youth_pct'] = X_test['agePct12t29'] + X_test['agePct16t24']

# Youth-to-elderly ratio: higher youth vs elderly may impact social dynamics
X['youth_to_elderly_ratio'] = X['youth_pct'] / (X['agePct65up'] + 1e-5)
X_test['youth_to_elderly_ratio'] = X_test['youth_pct'] / (X_test['agePct65up'] + 1e-5)

# Income per capita: captures wealth distribution
X['income_per_capita'] = X['medIncome'] / (X['population'] + 1e-5)
X_test['income_per_capita'] = X_test['medIncome'] / (X_test['population'] + 1e-5)

# Poverty ratio: higher poverty might correlate with crime
X['poverty_ratio'] = X['PctPopUnderPov'] / (X['population'] + 1e-5)
X_test['poverty_ratio'] = X_test['PctPopUnderPov'] / (X_test['population'] + 1e-5)

# Income-to-rent ratio: financial stress indicator
X['income_to_rent_ratio'] = X['medIncome'] / (X['MedRent'] + 1e-5)
X_test['income_to_rent_ratio'] = X_test['medIncome'] / (X_test['MedRent'] + 1e-5)

# Single parent households percentage: social stability indicator
X['single_parent_pct'] = 100 - X['PctKids2Par']
X_test['single_parent_pct'] = 100 - X_test['PctKids2Par']

# Divorce ratio: another social stability factor
X['divorce_ratio'] = X['TotalPctDiv'] / (X['PctFam2Par'] + 1e-5)
X_test['divorce_ratio'] = X_test['TotalPctDiv'] / (X_test['PctFam2Par'] + 1e-5)

# Avg people per room: overcrowding may affect crime
X['avg_people_per_room'] = X['PersPerOccupHous'] / (X['MedNumBR'] + 1e-5)
X_test['avg_people_per_room'] = X_test['PersPerOccupHous'] / (X_test['MedNumBR'] + 1e-5)

# Youth-unemployment interaction: high youth unemployment might increase crime
X['youth_unemp_interact'] = X['youth_pct'] * X['PctUnemployed']
X_test['youth_unemp_interact'] = X_test['youth_pct'] * X_test['PctUnemployed']

# Urban-poverty interaction: urban areas with poverty could see more crime
X['urban_poverty_interact'] = X['pctUrban'] * X['PctPopUnderPov']
X_test['urban_poverty_interact'] = X_test['pctUrban'] * X_test['PctPopUnderPov']

# Immigration-language barrier: communication difficulties may relate to crime statistics
X['immig_lang_barrier'] = X['PctImmigRecent'] * X['PctNotSpeakEnglWell']
X_test['immig_lang_barrier'] = X_test['PctImmigRecent'] * X_test['PctNotSpeakEnglWell']

# Log-transform skewed features: reduce impact of outliers
for col in ['medIncome', 'perCapInc', 'NumUnderPov', 'MedRent']:
    if col in X.columns:
        X[f'log_{col}'] = np.log1p(X[col])
        X_test[f'log_{col}'] = np.log1p(X_test[col])

# ================= Split Data =================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ================= Train Models =================
# XGBoost: gradient boosting model
xgb_model = XGBRegressor(objective="reg:squarederror", n_estimators=5000, max_depth=3, learning_rate=0.01,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)
y_val_pred_xgb = xgb_model.predict(X_val)
print("XGBoost Validation MSE:", mean_squared_error(y_val, y_val_pred_xgb))

# Random Forest: ensemble of decision trees
rf_model = RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_split=2, min_samples_leaf=1,
                                 random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_val_pred_rf = rf_model.predict(X_val)
print("Random Forest Validation MSE:", mean_squared_error(y_val, y_val_pred_rf))

# LightGBM: faster gradient boosting, suppress verbose output
lgb_model = lgb.LGBMRegressor(n_estimators=5000, learning_rate=0.01, max_depth=-1, num_leaves=64,
                              subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                              random_state=42, n_jobs=-1, force_col_wise=True)
with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
    lgb_model.fit(X_train, y_train)
y_val_pred_lgb = lgb_model.predict(X_val)
print("LightGBM Validation MSE:", mean_squared_error(y_val, y_val_pred_lgb))

# ================= Ensemble =================
# Combine models with linear regression to find optimal weights
stacked_preds = np.column_stack((y_val_pred_rf, y_val_pred_xgb, y_val_pred_lgb))
lr = LinearRegression(fit_intercept=False)
lr.fit(stacked_preds, y_val)
weights = lr.coef_
print("Ensemble Weights:", weights)
y_val_pred_ensemble = stacked_preds @ weights
print("Ensemble Validation MSE:", mean_squared_error(y_val, y_val_pred_ensemble))

# ================= Train on full data =================
xgb_model.fit(X, y)
rf_model.fit(X, y)
with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
    lgb_model.fit(X, y)

# ================= Predict on Test Data =================
test_preds_matrix = np.column_stack((
    rf_model.predict(X_test),
    xgb_model.predict(X_test),
    lgb_model.predict(X_test)
))
test_preds = test_preds_matrix @ weights

# ================= Submission =================
submission = pd.DataFrame({"ID": test["ID"], "ViolentCrimesPerPop": test_preds})
submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")
print(submission.head())
