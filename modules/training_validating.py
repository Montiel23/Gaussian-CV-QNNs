import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from torch.utils.data import DataLoader, Subset
from torch.utils.data import TensorDataset

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
import json
import pickle


def train_v2(model, dataloader, optimizer, loss_fn, device, epoch=None, fold=None):
    desc = f"Training | Fold {fold} | Epoch {epoch}" if fold is not None else f"Training | Epoch {epoch}"
    progress_bar = tqdm(dataloader, desc=desc, leave=True)
    model.train()

    total_loss = 0
    total_samples = 0
    correct = 0
    all_labels = []
    all_preds = []

    for x_batch, y_batch in progress_bar:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        y_batch = y_batch.view(-1).long()
        # print(outputs)
        # print("shape of output:", outputs.shape)
        # print("shape of y_batch:", y_batch.shape)
        loss = loss_fn(outputs, y_batch)
        loss.backward()

        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)
        probs = torch.softmax(outputs, dim=1)
        pred = outputs.argmax(dim=1)
        correct += (pred == y_batch).sum().item()
        total_samples += x_batch.size(0)

        progress_bar.set_postfix({
            "loss": f"{total_loss/total_samples:.4f}",
            "acc": f"{correct/total_samples:.4f}",
        })

        all_preds.extend(pred.cpu().detach().numpy())
        all_labels.extend(y_batch.cpu().detach().numpy())

    avg_loss = total_loss / total_samples
    avg_acc = correct / total_samples
    avg_precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    avg_recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    avg_f1 = f1_score(all_labels, all_preds, average="weighted")

    metrics = classification_report(
        all_labels,
        all_preds,
        output_dict = True,
        zero_division=0
    )

    return avg_loss, avg_acc, avg_precision, avg_recall, avg_f1, metrics


def evaluate_v2(model, dataloader, optimizer, loss_fn, device, epoch=None, fold=None):
    model.eval()
    desc = f"Training | Fold {fold} | Epoch {epoch}" if fold is not None else f"Training | Epoch {epoch}"
    progress_bar = tqdm(dataloader, desc=desc, leave=True)
    
    total_loss = 0
    correct = 0
    total_samples = 0
    all_labels = []
    all_preds = []
    all_probs = []

        
    for x_batch, y_batch in progress_bar:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        y_batch = y_batch.view(-1).long()
        loss = loss_fn(outputs, y_batch)

        total_loss += loss.item() * x_batch.size(0)
        probs = torch.softmax(outputs, dim=1)
        pred = outputs.argmax(dim=1)
        correct += (pred == y_batch).sum().item()
        total_samples += x_batch.size(0)

        progress_bar.set_postfix({
            "loss": f"{total_loss/total_samples:.4f}",
            "acc": f"{(correct/total_samples):.4f}"
        })

        all_probs.extend(probs.cpu().detach().numpy())
        all_preds.extend(pred.cpu().detach().numpy())
        all_labels.extend(y_batch.cpu().detach().numpy())


    avg_loss = total_loss / total_samples
    avg_acc = correct / total_samples
    avg_precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    avg_recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    avg_f1 = recall_score(all_labels, all_preds, average="weighted")

    metrics = classification_report(
        all_labels,
        all_preds,
        output_dict = True,
        zero_division=0
    )

    return avg_loss, avg_acc, avg_precision, avg_recall, avg_f1, metrics, all_preds, all_probs, all_labels

def run_kfold_training(model_class, dataset, optimizer, criterion, n_classes, model_save_name=False, device="cpu", classical=False, k=5, num_epochs=10, seed=42):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    labels = np.array([label for _, label in dataset])
    # labels = np.array([label.item() if torch.is_tensor(label) else label for _, label in dataset])

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n Fold {fold+1}/{k}")

        all_train_metrics = {"acc": [], "loss": [], "prec": [], "rec": [], "f1": []}
        all_val_metrics = {"acc": [], "loss": [], "prec": [], "rec": [], "f1": []}
        
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=True)

        if classical:
            # model = ClassicalModel().to(device)
            model = model_class(n_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            # optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

        # elif model_save_name:
        #     # model = model_class(n_qumodes=4, n_classes=2).to(device)
        #     # model = model_class(4, 2).to(device)
        #     model = model_class(4, n_classes).to(device)
        #     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        else:
            # model = QuantumWrapper().to(device)
            # model = model_class(n_qumodes=4, n_classes=2, hidden_dim=16).to(device)
            model = model_class(4, n_classes=n_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            # optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
            

        for epoch in range(num_epochs):
            tqdm.write(f" \nEpoch {epoch+1}/{num_epochs} (fold {fold+1})")
            # train_loss, train_acc, train_prec, train_rec, train_f1, _ = train_v2(model_class, train_loader, optimizer, criterion, device)
            train_loss, train_acc, train_prec, train_rec, train_f1, _ = train_v2(model, train_loader, optimizer, criterion, device, epoch=epoch+1, fold=fold+1)
            # val_loss, val_acc, val_prec, val_rec, val_f1, _, _, _, _ = evaluate_v2(model_class, val_loader, criterion, device)
            val_loss, val_acc, val_prec, val_rec, val_f1, _, _, _, _ = evaluate_v2(model, val_loader, optimizer, criterion, device, epoch=epoch+1, fold=fold+1)



            all_train_metrics["acc"].append(train_acc)
            all_train_metrics["loss"].append(train_loss)
            all_train_metrics["prec"].append(train_prec)
            all_train_metrics["rec"].append(train_rec)
            all_train_metrics["f1"].append(train_f1)

            all_val_metrics["acc"].append(val_acc)
            all_val_metrics["loss"].append(val_loss)
            all_val_metrics["prec"].append(val_prec)
            all_val_metrics["rec"].append(val_rec)
            all_val_metrics["f1"].append(val_f1)

            tqdm.write(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Train rec: {train_rec:.4f} | Train prec: {train_prec:.4f} | Train f1: {train_f1:.4f}")
            tqdm.write(f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f} | Val rec: {val_rec:.4f} | Val prec: {val_prec:.4f} | Val f1: {val_f1:.4f}")

        # if classical:
            # with open(f"/home/dalopezm/quantum-studies/quantum-cv/model_checkpoints/classical_fold_{fold+1}_metrics.pkl", "wb") as f:
            # with open(f"/home/dalopezm/quantum-studies/quantum-cv/model_checkpoints/classical-{model_save_name}_fold_{fold+1}_metrics.pkl", "wb") as f:
                # pickle.dump((all_train_metrics, all_val_metrics), f)
    
            # torch.save(model_class.state_dict(), f"/home/dalopezm/quantum-studies/quantum-cv/model_checkpoints/classical_fold_{fold+1}_best.pth")
            # torch.save(model.state_dict(), f"/home/dalopezm/quantum-studies/quantum-cv/model_checkpoints/classical-{model_save_name}_fold_{fold+1}_best.pth")

        # else:
        with open(f"/home/dalopezm/quantum-studies/quantum-cv/model_checkpoints/{model_save_name}_fold_{fold+1}_metrics.pkl", "wb") as f:
            pickle.dump((all_train_metrics, all_val_metrics), f)
    
            # torch.save(model_class.state_dict(), f"/home/dalopezm/quantum-studies/quantum-cv/model_checkpoints/cv_fold_{fold+1}_best.pth")
        torch.save(model.state_dict(), f"/home/dalopezm/quantum-studies/quantum-cv/model_checkpoints/{model_save_name}_fold_{fold+1}_best.pth")



def noise_robustness_validation(model, loader, noise_std, noise_metrics, model_type, optimizer, criterion, device):
    f1_scores = []

    full_samples = []
    full_labels = []

    for x_batch, y_batch in loader:
        full_samples.append(x_batch)
        full_labels.append(y_batch)

    full_samples = torch.cat(full_samples)
    full_labels = torch.cat(full_labels)

    for std in noise_std:
        noisy_samples = (full_samples + torch.randn_like(full_samples) * std).to(device)
        full_labels = full_labels.to(device)
        print(f"[STD: {std:.2f}] Clean mean: {full_samples.mean().item():.4f}, Noisy mean: {noisy_samples.mean().item():.4f}")
        noisy_dataset = TensorDataset(noisy_samples, full_labels)
        noisy_loader = DataLoader(noisy_dataset, batch_size=32, shuffle=False)
        
        _, acc, prec, rec, f1, _, _, _, _ = evaluate_v2(model,
                                                   noisy_loader,
                                                    optimizer, 
                                                  criterion,
                                                  device)
    
    
        # _, dv_acc, dv_prec, dv_rec, dv_f1, _, _, _, _ = evaluate_v2(test_dv_model,
        #                                        noisy_loader,
        #                                       criterion,
        #                                       device)
    
    
    
        # _, classical_acc, classical_prec, classical_rec, classical_f1, _, _, _, _ = evaluate_v2(test_classical_model,
        #                                            noisy_loader,
        #                                           criterion,
        #                                           device)
    
    
        noise_metrics["acc"].append(acc)
        noise_metrics["rec"].append(rec)
        noise_metrics["prec"].append(prec)
        noise_metrics["f1"].append(f1)
        
    # with open(f"/home/dalopezm/quantum-studies/quantum-cv/results/cv_noise_metrics.json", "w") as f:
    with open(f"/home/dalopezm/quantum-studies/quantum-cv/results/{model_type}_noise_metrics.json", "w") as f:
        json.dump(noise_metrics, f, indent=4)

    return noise_metrics
        
