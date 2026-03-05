from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix, mean_absolute_error, mean_squared_error
import numpy as np

def classification_metrics(y_true, y_pred):
    return {
        "roc_auc": roc_auc_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

def regression_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }
