from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = None
    XGBRegressor = None

def get_baseline_classifiers():
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(),
    }
    if XGBClassifier:
        models["XGBoost"] = XGBClassifier()
    return models

def get_baseline_regressors():
    models = {
        "RandomForestRegressor": RandomForestRegressor(),
    }
    if XGBRegressor:
        models["XGBoostRegressor"] = XGBRegressor()
    return models
