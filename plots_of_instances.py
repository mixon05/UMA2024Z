import json
import matplotlib.pyplot as plt
from metrics import get_roc_data, get_pr_data, calculate_g_means


def process_file(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    instances = []
    roc_aucs = []
    pr_aucs = []
    gmeans = []

    for idx, model in enumerate(data["models"]):
        try:
            n_instances = model["training instances"]
            roc_data = get_roc_data(data, idx)
            pr_data = get_pr_data(data, idx)
            gmean, _ = calculate_g_means(model["predictions"], data["true labels"])

            instances.append(n_instances)
            roc_aucs.append(roc_data["auc"])
            pr_aucs.append(pr_data["auc"])
            gmeans.append(gmean)

        except Exception as e:
            print(f"Błąd w pliku {json_file}, model {idx}: {str(e)}")
            continue

    sorted_data = sorted(zip(instances, roc_aucs, pr_aucs, gmeans))
    return (
        [x[0] for x in sorted_data],
        [x[1] for x in sorted_data],
        [x[2] for x in sorted_data],
        [x[3] for x in sorted_data],
    )


def plot_comparison(active_file, passive_file):
    active_instances, active_roc, active_pr, active_gmean = process_file(active_file)
    passive_instances, passive_roc, passive_pr, passive_gmean = process_file(
        passive_file
    )

    active_style = {
        "marker": "o",
        "linestyle": "-",
        "linewidth": 2,
        "markersize": 1,
        "color": "#1f77b4",
    }
    passive_style = {
        "marker": "s",
        "linestyle": "-",
        "linewidth": 2,
        "markersize": 1,
        "color": "#ff7f0e",
    }

    plt.figure("ROC AUC Comparison", figsize=(10, 6))
    plt.plot(active_instances, active_roc, label="Active Learning", **active_style)
    plt.plot(passive_instances, passive_roc, label="Passive Learning", **passive_style)
    plt.xlabel("Number of Training Instances")
    plt.ylabel("ROC AUC")
    plt.title("ROC AUC Comparison: Active vs Passive Learning")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.figure("PR AUC Comparison", figsize=(10, 6))
    plt.plot(active_instances, active_pr, label="Active Learning", **active_style)
    plt.plot(passive_instances, passive_pr, label="Passive Learning", **passive_style)
    plt.xlabel("Number of Training Instances")
    plt.ylabel("PR AUC")
    plt.title("PR AUC Comparison: Active vs Passive Learning")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.figure("G-Mean Comparison", figsize=(10, 6))
    plt.plot(active_instances, active_gmean, label="Active Learning", **active_style)
    plt.plot(
        passive_instances, passive_gmean, label="Passive Learning", **passive_style
    )
    plt.xlabel("Number of Training Instances")
    plt.ylabel("G-Mean")
    plt.title("G-Mean Comparison: Active vs Passive Learning")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    active_file = "one_positive_class/rbf_active_learning_IR4.json"
    passive_file = "one_positive_class/rbf_passive_learning_IR4.json"
    plot_comparison(active_file, passive_file)
