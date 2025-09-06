Violent Crimes Prediction Model

This project predicts the violent crimes per population for communities using socio-economic, demographic, and housing data. The dataset includes features like income, poverty, family stability, urbanization, and age distributions.

Key highlights of this project:

Feature Engineering: Derived meaningful features such as youth-to-elderly ratio, income-to-rent ratio, interaction terms between unemployment and youth, urbanization and poverty, and immigrant language barriers. Skewed features are log-transformed for better model performance.

Models Used:

XGBoost Regressor for gradient-boosted tree predictions.

Random Forest Regressor for ensemble-based predictions.

LightGBM Regressor with optimized parameters for faster, efficient training.

Ensemble Learning: Combined predictions from multiple models using linear regression-based weighting to minimize validation MSE.

Evaluation: Models are evaluated using Mean Squared Error (MSE) on a validation split, with final predictions submitted for Kaggle evaluation.

This project demonstrates a full workflow from data preprocessing and feature engineering to model training, ensembling, and prediction for a real-world regression problem.
