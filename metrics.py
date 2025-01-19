from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix, auc
import numpy as np
from helpers import unpack_data


def get_roc_data(data, model_index, label_prefix="Model"):

    training_data = unpack_data(data, model_index)
    true_labels = training_data["true_labels"]
    model = training_data["model"]
    predictions = training_data["predictions"]
    training_instances = training_data["training_instances"]

    fpr, tpr, _ = roc_curve(true_labels, predictions)
    model_auc = auc(fpr, tpr)

    label = f"{label_prefix} {training_instances} instances (AUC = {model_auc:.2f})"

    return {"fpr": fpr, "tpr": tpr, "auc": model_auc, "label": label}


def get_pr_data(data, model_index, label_prefix="Model"):

    training_data = unpack_data(data, model_index)
    true_labels = training_data["true_labels"]
    model = training_data["model"]
    predictions = training_data["predictions"]
    training_instances = training_data["training_instances"]

    precision, recall, _ = precision_recall_curve(true_labels, predictions)
    pr_auc = auc(recall, precision)

    label = f"{label_prefix} {training_instances} instances (AUC = {pr_auc:.2f})"

    return {"precision": precision, "recall": recall, "auc": pr_auc, "label": label}


def calculate_g_means(y_pred, y_true, threshold=0.5):

    y_pred_binary = (np.array(y_pred) >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    g_means = np.sqrt(sensitivity * specificity)

    return g_means, {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
