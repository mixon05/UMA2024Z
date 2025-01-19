import json
import matplotlib.pyplot as plt
from metrics import get_roc_data, get_pr_data, calculate_g_means


def plot_comparison_roc_pr_gmean(active_file, passive_file, index):

    # Wczytanie danych z plików JSON
    with open(active_file, "r", encoding="utf-8") as json_file:
        active = json.load(json_file)

    with open(passive_file, "r", encoding="utf-8") as json_file:
        passive = json.load(json_file)

    # Pobranie modelu dla danego indeksu
    model_active = active["models"][index]
    model_passive = passive["models"][index]

    # Wykres ROC
    fig1, ax1 = plt.subplots(figsize=(7, 5))

    # Plot ROC dla modelu aktywnego
    roc_data_active = get_roc_data(active, index)
    ax1.plot(
        roc_data_active["fpr"],
        roc_data_active["tpr"],
        label=f"Active: {roc_data_active['label']}",
    )

    # Plot ROC dla modelu pasywnego
    roc_data_passive = get_roc_data(passive, index)
    ax1.plot(
        roc_data_passive["fpr"],
        roc_data_passive["tpr"],
        label=f"Passive: {roc_data_passive['label']}",
    )

    ax1.plot([0, 1], [0, 1], "k--")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curves")
    ax1.legend()
    ax1.grid(True)

    # Wykres Precision-Recall
    fig2, ax2 = plt.subplots(figsize=(7, 5))

    # Plot Precision-Recall dla modelu aktywnego
    pr_data_active = get_pr_data(active, index)
    ax2.plot(
        pr_data_active["recall"],
        pr_data_active["precision"],
        label=f"Active: {pr_data_active['label']}",
    )

    # Plot Precision-Recall dla modelu pasywnego
    pr_data_passive = get_pr_data(passive, index)
    ax2.plot(
        pr_data_passive["recall"],
        pr_data_passive["precision"],
        label=f"Passive: {pr_data_passive['label']}",
    )

    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curves")
    ax2.legend()
    ax2.grid(True)

    # Dostosowanie układu
    plt.tight_layout()
    plt.show()

    # Obliczanie G-Mean
    gmean_active = calculate_g_means(model_active["predictions"], active["true labels"])
    gmean_passive = calculate_g_means(
        model_passive["predictions"], passive["true labels"]
    )

    # Wyświetlanie G-Mean
    print("Active Learning: ", gmean_active)

    print("Passive Learning: ", gmean_passive)
