import matplotlib.pyplot as plt

def plot_data_split():
    datasets = ["CODEX (p16)", "Xenium (ds score)"]
    colors = {"Positive": "tab:red", "Negative": "tab:blue"}

    fig, ax = plt.subplots(figsize=(7, 3))

    # Percentages
    codex_pos, codex_neg = 1, 99
    xenium_pos, xenium_neg = 1, 99

    # Plot CODEX
    ax.barh(0, codex_pos, color=colors["Positive"], label="Positive")
    ax.barh(0, codex_neg, left=codex_pos, color=colors["Negative"], label="Negative")

    # Plot Xenium
    ax.barh(1, xenium_pos, color=colors["Positive"])
    ax.barh(1, xenium_neg, left=xenium_pos, color=colors["Negative"])

    # Y ticks
    ax.set_yticks([0, 1])
    ax.set_yticklabels(datasets)

    # Labels & limits
    ax.set_xlabel("Percentage of cells")
    ax.set_xlim(0, 100)

    # Legend
    ax.legend(loc="upper right", bbox_to_anchor=(1.12, 1))

    plt.title("Data Split Strategy: CODEX vs Xenium")
    plt.tight_layout()
    plt.show()

plot_data_split()
