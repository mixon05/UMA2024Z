import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def plot_model_metrics(json_file):
    # Wczytaj dane z pliku JSON
    with open(json_file, "r") as f:
        data = json.load(f)

    y_true = np.array(data["true labels"])
    y_pred = np.array(data["model"]["predictions"])

    # Obliczenia dla ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # Obliczenia dla Precision-Recall
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)

    # Oblicz G-Mean
    gmeans = np.sqrt(tpr * (1 - fpr))
    optimal_gmean = np.max(gmeans)
    optimal_threshold = thresholds[np.argmax(gmeans)]

    # Wykres ROC
    plt.figure("ROC Curve", figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # Wykres Precision-Recall
    plt.figure("PR Curve", figsize=(8, 6))
    plt.plot(recall, precision, color="blue", lw=2, label=f"PR AUC = {pr_auc:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)

    plt.show()

    # Wypisz G-Mean
    print(f"\nOptymalny G-Mean: {optimal_gmean:.4f}")
    print(f"Próg decyzyjny dla optymalnego G-Mean: {optimal_threshold:.4f}")


# Przykład użycia
if __name__ == "__main__":
    plot_model_metrics("nn_passive_learning_IR256.json")
