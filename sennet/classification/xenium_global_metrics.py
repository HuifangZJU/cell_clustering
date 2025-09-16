import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    f1_score, precision_score, recall_score
)
from scipy.stats import pearsonr, spearmanr

def analyze_and_plot(csv_file, k_percent=5, threshold=0.9):
    # Load CSV
    df = pd.read_csv(csv_file, sep=None, engine="python")

    ds = df["ds"].values
    pred = df["pred"].values


    # ----- Binary labels for ROC/PR -----
    n = len(ds)
    k = int(np.ceil(n * k_percent / 100))
    cutoff = np.sort(ds)[-k]
    y_true = (ds >= cutoff).astype(int)

    # ----- ROC -----
    fpr, tpr, _ = roc_curve(y_true, pred)
    auc = roc_auc_score(y_true, pred)

    # ----- Precision-Recall -----
    prec, rec, _ = precision_recall_curve(y_true, pred)
    ap = average_precision_score(y_true, pred)

    # ----- Normalize scales -----
    scaler = MinMaxScaler()
    ds_norm = scaler.fit_transform(ds.reshape(-1, 1)).flatten()
    pred_norm = scaler.fit_transform(pred.reshape(-1, 1)).flatten()

    # ----- Correlation (normalized) -----
    pearson_corr, _ = pearsonr(ds_norm, pred_norm)
    spearman_corr, _ = spearmanr(ds_norm, pred_norm)

    # ----- Confusion matrix -----
    y_pred = (pred >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    f1 = f1_score(y_true, y_pred)
    prec_bin = precision_score(y_true, y_pred)
    rec_bin = recall_score(y_true, y_pred)

    # ----- Plot -----
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # ROC
    axs[0,0].plot(fpr, tpr, label=f"AUC={auc:.3f}")
    axs[0,0].plot([0, 1], [0, 1], "k--")
    axs[0,0].set_title(f"ROC (top {k_percent}% ds as positive)")
    axs[0,0].set_xlabel("False Positive Rate")
    axs[0,0].set_ylabel("True Positive Rate")
    axs[0,0].legend()

    # PR
    axs[0,1].plot(rec, prec, label=f"AP={ap:.3f}")
    axs[0,1].set_title("Precision–Recall Curve")
    axs[0,1].set_xlabel("Recall")
    axs[0,1].set_ylabel("Precision")
    axs[0,1].legend()

    # Scatter + correlation
    axs[1, 0].scatter(ds_norm, pred_norm, s=0.1,alpha=0.5)
    axs[1, 0].set_title(f"Normalized DS vs Prediction\nPearson={pearson_corr:.3f}, Spearman={spearman_corr:.3f}")
    axs[1, 0].set_xlabel("DS score (normalized)")
    axs[1, 0].set_ylabel("Model prediction (normalized)")

    # Normalized Confusion matrix with counts
    axs[1,1].imshow(cm_norm, interpolation="nearest", cmap="Blues")
    axs[1,1].set_title(
        f"Confusion Matrix (thr={threshold})\nF1={f1:.3f}, P={prec_bin:.3f}, R={rec_bin:.3f}"
    )
    classes = ["SNC","Non-snc"]
    axs[1,1].set_xticks(np.arange(len(classes)))
    axs[1,1].set_yticks(np.arange(len(classes)))
    axs[1,1].set_xticklabels(classes)
    axs[1,1].set_yticklabels(classes)
    axs[1,1].set_ylabel("True label")
    axs[1,1].set_xlabel("Predicted label")

    # Annotate each cell
    fmt = ".2f"
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axs[1,1].text(
                j, i,
                f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black"
            )

    plt.tight_layout()
    plt.show()

    return {
        "AUC": auc,
        "AP": ap,
        "Pearson": pearson_corr,
        "Spearman": spearman_corr,
        "F1": f1,
        "Precision": prec_bin,
        "Recall": rec_bin,
        "ConfusionMatrix": cm,
        "ConfusionMatrixNormalized": cm_norm
    }
# auc = analyze_and_plot("/media/huifang/data/sennet/codex/cell_images/results/top10-models_predictions.csv", k_percent=10,threshold=0.99)
auc = analyze_and_plot("/media/huifang/data/sennet/codex/cell_images/results/top1-models-double-neg_predictions.csv", k_percent=1,threshold=0.9)
# auc = analyze_and_plot("/media/huifang/data/sennet/codex/cell_images/results/top1-models-triple-neg_predictions.csv", k_percent=1,threshold=0.9)
print("AUC =", auc)