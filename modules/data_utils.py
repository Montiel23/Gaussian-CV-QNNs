from medmnist import PneumoniaMNIST, OrganAMNIST, BreastMNIST
import medmnist
from medmnist import INFO

from mpl_toolkits.mplot3d import Axes3D



import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

import numpy as np

import pickle
import json
import joblib

def load_data(data_flag, batch_size=32, train_fraction=0.2, val_fraction=1.0, test_fraction=1.0):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    
    if data_flag == "pneumoniamnist":
        train_dataset = PneumoniaMNIST(split="train", transform=transform, download=True)
        val_dataset = PneumoniaMNIST(split="val", transform=transform, download=True)
        test_dataset = PneumoniaMNIST(split="test", transform=transform, download=True)

    elif data_flag == "organamnist":
        train_dataset = OrganAMNIST(split="train", transform=transform, download=True)
        val_dataset = OrganAMNIST(split="val", transform=transform, download=True)
        test_dataset = OrganAMNIST(split="test", transform=transform, download=True)

    elif data_flag == "breastmnist":
        train_dataset = BreastMNIST(split="train", transform=transform, download=True)
        val_dataset = BreastMNIST(split="val", transform=transform, download=True)
        test_dataset = BreastMNIST(split="test", transform=transform, download=True)

    else:
        raise NotImplementedError(f"Dataset {data_flag} not supported")

    if train_fraction < 1.0:
        n_train = int(len(train_dataset) * train_fraction)
        indices = np.random.choice(len(train_dataset), n_train, replace=False)
        train_dataset = Subset(train_dataset, indices)

    if val_fraction < 1.0:
        n_val = int(len(val_dataset) * val_fraction)
        indices = np.random.choice(len(val_dataset), n_val, replace=False)
        val_dataset = Subset(val_dataset, indices)

    if test_fraction < 1.0:
        n_test = int(len(test_dataset) * test_fraction)
        indices = np.random.choice(len(test_dataset), n_test, replace=False)
        test_dataset = Subset(test_dataset, indices)
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, train_dataset, val_loader, val_dataset, test_loader, test_dataset



# def load_data(batch_size=32, train_fraction=0.2, val_fraction=1.0, test_fraction=1.0):
#     data_flag = "pneumoniamnist"
#     info = INFO[data_flag]
#     transform = transforms.Compose([
#         transforms.Resize((28, 28)),
#         transforms.ToTensor()
#     ])

#     train_dataset = PneumoniaMNIST(split='train', transform=transform, download=True)
#     val_dataset = PneumoniaMNIST(split='val', transform=transform, download=True)
#     test_dataset = PneumoniaMNIST(split='test', transform=transform, download=True)


#     if train_fraction < 1.0:
#         n_train = int(len(train_dataset) * train_fraction)
#         indices = np.random.choice(len(train_dataset), n_train, replace=False)
#         train_dataset = Subset(train_dataset, indices)

#     if val_fraction < 1.0:
#         n_val = int(len(val_dataset) * val_fraction)
#         indices = np.random.choice(len(val_dataset), n_val, replace=False)
#         val_dataset = Subset(val_dataset, indices)

#     if test_fraction < 1.0:
#         n_test = int(len(test_dataset) * test_fraction)
#         indices = np.random.choice(len(test_dataset), n_test, replace=False)
#         test_dataset = Subset(test_dataset, indices)
    
    
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     return train_loader, train_dataset, val_loader, val_dataset, test_loader, test_dataset


def extract_xy_from_loader(dataloader):
    x_all = []
    y_all = []

    for x_batch, y_batch in dataloader:
        x_all.append(x_batch.view(x_batch.size(0), -1))
        y_all.append(y_batch)

    x = torch.cat(x_all, dim=0)
    y = torch.cat(y_all, dim=0)

    return x, y

def data_scaler(train_encoded_data, val_encoded_data, test_encoded_data, save_database_name=False):
    scaler = StandardScaler()
    train_scaled_data = scaler.fit_transform(train_encoded_data)

    val_scaled_data = scaler.transform(val_encoded_data)
    test_scaled_data = scaler.transform(test_encoded_data)

    if save_database_name:
        joblib.dump(scaler, f"/home/dalopezm/quantum-studies/quantum-cv/model_checkpoints/pca_scaler_{save_database_name}.pkl")

    train_tensor_data = torch.tensor(train_scaled_data, dtype=torch.float32)
    val_tensor_data = torch.tensor(val_scaled_data, dtype=torch.float32)
    test_tensor_data = torch.tensor(test_scaled_data, dtype=torch.float32)
    
    return train_tensor_data, val_tensor_data, test_tensor_data


def dataset_dataloaders(x_tensor, y_tensor, train=True):
    dataset = TensorDataset(x_tensor, y_tensor)
    if train:
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

    else:
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

    return dataset, loader

def sample_batch(dataloader, n_samples=8, device='cpu'):
    for x, y in dataloader:
        x = x.view(x.size(0), -1).to(device)
        y = y.to(device)

        if n_samples <= x.size(0):
            return x[:n_samples], y[:n_samples]
        else:
            raise ValueError(f"Batch size {x.size(0)} is smaller than requested {n_samples}")
            

class PCAEncoder:
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.fitted = False

    def fit(self, dataloader):
        all_images = []
        for x, _ in dataloader:
            x = x.view(x.size(0), -1)
            all_images.append(x)
        X = torch.cat(all_images, dim=0).numpy()
        self.pca.fit(X)
        self.fitted = True
        
    def transform(self, dataloader):
        if not self.fitted:
            raise RuntimeError("PCA Encoder mus be fit before calling transform.")

        all_transformed = []
        all_labels = []
        for x, y in dataloader:
            x = x.view(x.size(0), -1).numpy()
            x_transformed = self.pca.transform(x)
            all_transformed.append(torch.tensor(x_transformed, dtype=torch.float32))
            all_labels.append(y)

        x_all = torch.cat(all_transformed, dim=0)
        y_all = torch.cat(all_labels, dim=0)
        return x_all, y_all
