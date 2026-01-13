import umap
import torch
import seaborn as sns
import matplotlib.gridspec as gridspec
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, average_precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

def reliability_diagram_multi(models_data, dataset_name, n_bins=10, save_name=None):

    """
    models_data = {
    "classical": (y_true, y_prob),
    "dv": (y_true, y_prob),
    "cv": (y_true, y_prob)
    }
    """
    
    plt.figure(figsize=(8,6))
    plt.plot([0,1], [0,1], '--', color='gray', label="Perfect Calibration")

    for name, (y_true, y_prob) in models_data.items():
        #ensure numpy arrays
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        
        if y_prob.ndim > 1 and y_prob.shape[1] > 1:
            conf = np.max(y_prob, axis=1)
            pred = np.argmax(y_prob, axis=1)
        else:
            conf = y_prob
            pred = (y_prob >= 0.5).astype(int)

        bins = np.linspace(0, 1, n_bins + 1)
        bin_ids = np.digitize(conf, bins) - 1

        accs, confs = [], []
        for b in range(n_bins):
            idx = np.where(bin_ids == b)[0]
            if len(idx) == 0:
                continue
            accs.append((y_true[idx] == pred[idx]).mean())
            confs.append(conf[idx].mean())

        plt.plot(confs, accs, marker="o", label=name)

    plt.xlabel("Mean confidence", fontsize=26)
    plt.ylabel("Accuracy", fontsize=26)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(True)

    if save_name:
        plt.savefig(f"/home/dalopezm/quantum-studies/quantum-cv/results/{save_name}_{dataset_name}", dpi=300)

    plt.show()

def visualize_umap(X, y):
    import umap
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(X)

    plt.figure(figsize=(6,6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap="plasma", alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.grid(True)
    plt.show()

def show_images_batch(x_batch, y_batch, n_cols=4):
    n_samples = x_batch.size(0)
    n_rows = (n_samples + n_cols -1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))

    if x_batch.ndim == 2:
        x_batch = x_batch.view(n_samples, 1, 4, 1)

    axes = axes.flatten() if n_samples > 1 else [axes]

    for i in range(n_samples):
        image = x_batch[i].squeeze().cpu().numpy()
        label = y_batch[i].item()
        axes[i].imshow(image, cmap='gray', aspect="auto")
        axes[i].set_title(f"Label: {label}")
        axes[i].axis("off")

    for i in range(n_samples, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def data_variance_plot(encoder):
    plt.plot(np.arange(1, len(encoder.pca.explained_variance_ratio_) + 1),
             np.cumsum(encoder.pca.explained_variance_ratio_))
    plt.xlabel("N Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid(True)
    plt.show()

def feature_heatmap(encoded_data):
    plt.figure(figsize=(12, 2))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        sns.heatmap(encoded_data[i].unsqueeze(0), cmap="plasma", cbar=False)
        plt.title(f"Sample {i}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def plot_metric_with_std(metric_name, train_folds, val_folds, epochs, classical=False, save_name=None):
    
    train_array = np.array(train_folds[metric_name])
    val_array = np.array(val_folds[metric_name])

    train_mean = np.mean(train_array, axis=0)
    train_std = np.std(train_array, axis=0)
    val_mean = np.mean(val_array, axis=0)
    val_std = np.std(val_array, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_mean, label="Train", color="purple", marker="o")
    plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.2, color="purple")

    plt.plot(epochs, val_mean, label="Validation", color="orange", marker="^")
    plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.2, color="orange")

    plt.xlabel("Epochs", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel(metric_name.capitalize(), fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_name:

        plt.savefig("/home/dalopezm/quantum-studies/quantum-cv/results/" + save_name, dpi=300)

        # if classical:
        #     plt.savefig("/home/dalopezm/quantum-studies/quantum-cv/results/classical-" + save_name, dpi=300)
        # else:
        #     plt.savefig("/home/dalopezm/quantum-studies/quantum-cv/results/quantum-" + save_name, dpi=300)
    plt.show()

    return {
        "metric": metric_name,
        "train_name": float(train_mean[-1]),
        "train_std": float(train_std[-1]),
        "val_mean": float(val_mean[-1]),
        "val_std": float(val_std[-1]),
    }




def plot_confusion_matrix(test_labels, test_preds, label_names, model_name, binary=False):
    cm = confusion_matrix(test_labels, test_preds)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    annot = np.empty_like(cm).astype(str)
    n_classes = cm.shape[0]

    for i in range(n_classes):
        for j in range(n_classes):
            count = cm[i, j]
            percent = cm_normalized[i, j]
            annot[i, j] = f"{percent:.2f}%\n({count})"

    # plt.figure(figsize=(8,8))
    plt.figure(figsize=(14,14))

    heatmap_kwargs = dict(
        annot=annot,
        fmt="",
        cmap="plasma",
        vmin=0, vmax=100,
        xticklabels=[f"Predicted {lbl}" for lbl in label_names],
        yticklabels=[f"Actual {lbl}" for lbl in label_names],
        annot_kws={"size": 48 if binary else 14}
        # annot_kws={"size": 24 if binary else 14}
    )

    # sns.heatmap(cm_normalized, annot=annot, fmt="", cmap="plasma",
    # # sns.heatmap(cm_normalized, annot=annot, fmt="", cmap="jet",
    #            vmin=0, vmax=100,
    #            # xticklabels=[f"Predicted {lbl}" for lbl in labels],
    #            # yticklabels=[f"Actual {lbl}" for lbl in labels],
    #            xticklabels=[f"Predicted {lbl}" for lbl in label_names],
    #            yticklabels=[f"Actual {lbl}" for lbl in label_names],
    #            # annot_kws={"size":14}
    #            annot_kws={"size":24 if binary else 14}
    #            )


    ax = sns.heatmap(cm_normalized, **heatmap_kwargs)

    colorbar = ax.collections[0].colorbar
    # colorbar.ax.tick_params(labelsize=18)
    # colorbar.set_label('Accuracy (%)', fontsize=20)
    
    # plt.ylabel("Ground Truth", fontsize=20)
    # plt.xlabel("Prediction", fontsize=20)
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)

    colorbar.ax.tick_params(labelsize=24)
    colorbar.set_label('Accuracy (%)', fontsize=26)
    
    plt.ylabel("Ground Truth", fontsize=26)
    plt.xlabel("Prediction", fontsize=26)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    
    plt.tight_layout()

    # plt.savefig(f"/home/dalopezm/quantum-studies/quantum-cv/results/quantum-{model_name}-cm", dpi=300)
    plt.savefig(f"/home/dalopezm/quantum-studies/quantum-cv/results/{model_name}-cm", dpi=300)
    plt.show()

def plot_auroc(test_labels, test_probs, model_name, binary=True):
    all_probs_np = np.array(test_probs)
    all_labels_np = np.array(test_labels)

    fig, ax = plt.subplots(figsize=(8, 6))

    if binary:
        fpr, tpr, thresholds = roc_curve(all_labels_np, all_probs_np[:,1])
        roc_auc = auc(fpr, tpr)
        
        # print("AUROC:", roc_auc)    
        
        # ax.plot(fpr, tpr, color="blue", lw=1.75, label=f"AUROC={roc_auc:.2f}")
        ax.plot(fpr, tpr, color="purple", lw=1.75, label=f"AUROC={roc_auc:.2f}")
        ax.plot([0,1], [0,1], linestyle="--", color="gray", label="Chance", lw=1.5)
        
        
        for i in [0, len(thresholds) //2, len(thresholds)-1]:
            ax.annotate(f"{thresholds[i]:.2f}",
                        (fpr[i], tpr[i]),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha="center",
                        # fontsize=9,
                        fontsize=12,
                        color="black",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5)
                       )

    else:
        n_classes = all_probs_np.shape[1]
        y_bin = label_binarize(all_labels_np, classes=np.arange(n_classes))

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:,i], all_probs_np[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=1.5, label=f"Class {i} (AUROC={roc_auc:.2f})")

        ax.plot([0,1], [0,1], linestyle="--", color="gray", lw=1.5, label="Chance")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    # ax.set_xlabel("False Positive Rate", fontsize=16)
    # ax.set_ylabel("True Positive Rate", fontsize=16)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # ax.legend(loc="lower right", fontsize=12)

    
    ax.set_xlabel("False Positive Rate", fontsize=20)
    ax.set_ylabel("True Positive Rate", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.legend(loc="lower right", fontsize=16)


    ax.set_xlabel("False Positive Rate", fontsize=24)
    ax.set_ylabel("True Positive Rate", fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    ax.legend(loc="lower right", fontsize=16)
    
    
    ax.grid(True, linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    
    
    plt.savefig(f"/home/dalopezm/quantum-studies/quantum-cv/results/{model_name}-auroc", dpi=300)
    plt.show()

def plot_prec(test_labels, test_probs, model_name, binary=True):
    y_true = np.array(test_labels)
    y_scores = np.array(test_probs)

    fig, ax = plt.subplots(figsize=(8, 6))

    if binary:
        precision, recall, _ = precision_recall_curve(y_true, y_scores[:, 1])
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, lw=1.75, color="green", label=f"AP={pr_auc:.2f}")

    else:
        n_classes = y_scores.shape[1]
        y_bin = label_binarize(y_true, classes=np.arange(n_classes))

        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_bin[:, i], y_scores[:, i])
            pr_auc = auc(recall, precision)
            ax.plot(recall, precision, lw=1.5, label=f"Class {i} (AP={pr_auc:.2f})")

    # Styling
    # ax.set_xlabel("Recall", fontsize=20)
    # ax.set_ylabel("Precision", fontsize=20)
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)


    ax.set_xlabel("Recall", fontsize=24)
    ax.set_ylabel("Precision", fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="lower left", fontsize=16)
    
    plt.tight_layout()
    plt.savefig(f"/home/dalopezm/quantum-studies/quantum-cv/results/{model_name}-precision_recall_curve", dpi=300)
    plt.show()
    

def plot_reconstructed_heatmaps_with_predictions(samples, cam, preds, probs, labels, encoder, class_names=None):
    n_samples = samples.shape[0]
    # n_rows, n_cols = 4, 4
    n_rows = (n_samples + 3) // 4
    n_cols = 8
    
    assert n_samples <= n_rows * n_cols, "Too many samples"

    # fig, axes = plt.subplots(n_rows, n_cols, figsize=(14,12))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 2.5* n_rows))

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n_samples):
        row = i// 4
        col_base = (i % 4) * 2
    
    # for i in range(n_samples):
    #     ax = axes[i // n_cols, i % n_cols]

        reconstructed = encoder.pca.inverse_transform(samples[i].detach().cpu().numpy().reshape(1, -1))
        reconstructed = torch.tensor(reconstructed).view(1, 1, 28, 28)

        heatmap_cam = cam[i].cpu().numpy().reshape(1, -1)
        heatmap_image = encoder.pca.inverse_transform(heatmap_cam)
        heatmap_image = torch.tensor(heatmap_image).view(1, 1, 28, 28)

        img = reconstructed[0, 0].numpy()
        heat = heatmap_image[0, 0].numpy()

        # ax.imshow(img, cmap="gray")
        # ax.imshow(heat, cmap="jet", alpha=0.45)

        pred_label = class_names[preds[i]] if class_names else str(preds[i].item())
        true_label = class_names[labels[i]] if class_names else str(labels[i].item())
        confidence = f"{probs[i][preds[i]]*100:.1f}%"

        ax1 = axes[row, col_base]
        ax1.imshow(img, cmap="gray")
        ax1.set_title(f"GT: {true_label}", fontsize=14)
        ax1.axis("off")

        ax2 = axes[row, col_base + 1]
        ax2.imshow(img, cmap="gray")
        ax2.imshow(heat, cmap="jet", alpha=0.45)
        # ax2.imshow(heat, cmap="plasma", alpha=0.45)
        ax2.set_title(f"Pred: {pred_label} ({confidence})", fontsize=14)
        ax2.axis("off")

        # ax.set_title(f"GT: {true_label} | Pred: {pred_label} ({confidence})", fontsize=10)
        # ax.axis("off")

    plt.tight_layout()
    plt.show()

def plot_noise_comparison(cv_metric, dv_metric, classic_metric, noise_std, choice_metric, name=None, data_name=None):
    
    plt.figure(figsize=(8,5))
    
    plt.plot(noise_std, cv_metric, marker="s", label="CV", color="purple")
    plt.plot(noise_std, dv_metric, marker="o", label="DV", color="orange")
    plt.plot(noise_std, classic_metric, marker="^", label="Classic", color="yellow")
    plt.xlabel("Gaussian Noise STD", fontsize=24)
    plt.ylabel(choice_metric, fontsize=24)
    plt.yticks(fontsize=22)
    plt.xticks(fontsize=22)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"/home/dalopezm/quantum-studies/quantum-cv/results/{data_name}_noise_{name}_performance_curve", dpi=300)
    plt.show()