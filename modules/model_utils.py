import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

import pennylane as qml
from pennylane import numpy as np
from pennylane import draw_mpl
from pennylane.qnn import TorchLayer
import pickle
import json

from tqdm import tqdm

# def dv_qcnn(inputs, weights, n_qubits=4):
#     # Encode classical data
#     for i in range(n_qubits):
#         qml.RY(inputs[i], wires=i)

#     # Variational layers (2 repetitions)
#     for l in range(weights.shape[0]):
#         for i in range(n_qubits):
#             # Use all 4 parameters per qubit per layer
#             qml.RY(weights[l, i, 0], wires=i)
#             qml.RZ(weights[l, i, 1], wires=i)
#             qml.RY(weights[l, i, 2], wires=i)
#             qml.RZ(weights[l, i, 3], wires=i)

#         # Add lightweight entanglement between neighboring qubits
#         for j in range(n_qubits - 1):
#             qml.CNOT(wires=[j, j + 1])

#     # Expectation values as outputs
#     return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


def dv_qcnn(inputs, weights, n_qubits=4):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    for i in range(n_qubits):
        qml.RZ(weights[0, i, 0], wires=i)
        qml.RY(weights[0, i, 1], wires=i)

    qml.CNOT(wires=[0, 1])

    for i in range(n_qubits):
        qml.RZ(weights[0, i, 0], wires=i)
        qml.RY(weights[0, i, 1], wires=i)

    qml.CNOT(wires=[0, 1])

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

def get_dv_qcnn_qnode(n_qubits, dev_name="default.qubit"):
    dev = qml.device(dev_name, wires=n_qubits, shots=None)
    # return qml.QNode(lambda inputs, weights: dv_qcnn(inputs, weights, n_qubits), dev, interface="torch", diff_method="parameter-shift")
    return qml.QNode(lambda inputs, weights: dv_qcnn(inputs, weights, n_qubits), dev, interface="torch", diff_method="backprop")


def cv_qcnn(inputs, weights, n_qumodes=4):
    for i in range(n_qumodes):
        qml.Displacement(inputs[i], 0.0, wires=i)

    for l in range(weights.shape[0]):
        for i in range(n_qumodes):
            qml.Rotation(weights[l, i, 0], wires=i)
            qml.Squeezing(weights[l, i, 1], 0.0, wires=i)
        qml.Beamsplitter(weights[l, 0, 2], weights[l, 0, 3], wires=[0, 1])

    return [qml.expval(qml.X(wires=i)) for i in range(n_qumodes)]

def get_cv_qcnn_qnode(n_qumodes, depth, dev_name="default.gaussian"):
    dev = qml.device(dev_name, wires=n_qumodes, shots=None)
    return qml.QNode(lambda inputs, weights: cv_qcnn(inputs, weights, n_qumodes), dev, interface="torch", diff_method="parameter-shift")
    # return qml.QNode(lambda inputs, weights: cv_qcnn(inputs, weights, n_qumodes), dev, interface="torch", diff_method="backprop")


class DVQuantumLinear(nn.Module):
    def __init__(self, n_qubits=4, n_classes=2):
        super().__init__()

        weight_shapes = {"weights": (2, n_qubits, 4)}

        qnode = get_dv_qcnn_qnode(n_qubits)

        self.quantum = TorchLayer(qnode, weight_shapes)

        self.head = nn.Sequential(
            nn.Linear(n_qubits, n_classes),
        )

    def forward(self, x):
        quantum_outs = torch.stack([self.quantum(sample) for sample in x])
        return self.head(quantum_outs)


class QuantumLinear(nn.Module):
    def __init__(self, n_qumodes=4, n_classes=2, depth=2):
        super().__init__()

        self.n_qumodes = n_qumodes
        self.n_classes = n_classes
        self.depth = depth

        # weight_shapes = {"weights": (2, n_qumodes, 4)}
        weight_shapes = {"weights": (depth, n_qumodes, 4)}
        # weight_shapes = {"weights": (depth, n_qumodes, n_qumodes)}

        # self.n_qumodes = n_qumodes
        # self.n_classes = n_classes 
        # self.depth = depth

        # qnode = get_cv_qcnn_qnode(n_qumodes)
        qnode = get_cv_qcnn_qnode(n_qumodes, depth, dev_name="default.gaussian")

        self.quantum = TorchLayer(qnode, weight_shapes)

        self.head = nn.Sequential(
            nn.Linear(n_qumodes, n_classes),
            # nn.Softmax(dim=1) if n_classes > 2 else nn.Sigmoid()
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        quantum_outs = torch.stack([self.quantum(sample) for sample in x])
        return self.head(quantum_outs)


class ClassicalModel(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(4,4),
            nn.ReLU(),
            nn.Linear(4, n_classes)
        )

    def forward(self, x):
        return self.head(x)


def gradcam_model(model, input_tensor, quantum=False, class_idx=None):
    model.eval()

    input_tensor.requires_grad = True

    if quantum:

        quantum_features = torch.stack([model.quantum(sample) for sample in input_tensor])
        quantum_features.retain_grad()
    
        outputs = model.head(quantum_features)
        probs = f.softmax(outputs, dim=1)
    
        if class_idx is None:
            class_idx = outputs.argmax(dim=1)
    
        scores = outputs[:, class_idx] if len(class_idx.shape) == 1 else outputs[0, class_idx]
        scores.sum().backward()
    
        gradients = quantum_features.grad
        activations = quantum_features

    else:
        activations = []
        gradients = []
        def forward_hook(module, input, output):
            activations.append(output.detach())

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0].detach())

        target_layer = model.head[0]
        handle_fwd = target_layer.register_forward_hook(forward_hook)
        handle_bwd = target_layer.register_full_backward_hook(backward_hook)

        outputs = model(input_tensor)

        if class_idx is None:
            class_idx = outputs.argmax(dim=1)

        scores = outputs[range(len(outputs)), class_idx]
        scores.sum().backward()

        handle_fwd.remove()
        handle_bwd.remove()

        gradients = activations[0]
        activations = gradients[0]
        probs = f.softmax(outputs, dim=1)
        

    weights = gradients.mean(dim=1, keepdim=True)
    cam = weights * activations
    # cam = cam.sum(dim=1)

    # return cam.detach().cpu(), outputs.detach().cpu(), class_idx.detach().cpu()
    return cam.detach().cpu(), probs.detach().cpu(), class_idx.detach().cpu()


def evaluate_calibration(model, loader, device, n_bins=15):
    model.eval()
    all_logits = []
    all_y = []

    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        all_logits.append(logits.detach().cpu())
        all_y.append(yb.cpu())

    logits = torch.cat(all_logits)
    y_true = torch.cat(all_y).numpy()
    y_prob = torch.softmax(logits, dim=1).numpy()

    return {
        "ece": compute_ece(y_true, y_prob, n_bins),
        "brier": brier_score(y_true, y_prob)
    }


def brier_score(y_true, y_prob):
    #binary
    if y_prob.ndim == 1 or y_prob.shape[1] == 1:
        return np.mean((y_prob - y_true)**2)

    #multiclass
    n_classes = y_prob.shape[1]
    y_true_onehot = np.eye(n_classes)[y_true]
    return np.mean(np.sum((y_prob - y_true_onehot)**2, axis=1))

def compute_ece(y_true, y_prob, n_bins=10):
    #ensure numpy arrays
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    if y_prob.ndim > 1 and y_prob.shape[1] > 1:
        confidences = np.max(y_prob, axis=1)
        preds = np.argmax(y_prob, axis=1)

    else:
        confidences = y_prob
        preds = (y_prob >= 0.5).astype(int)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(confidences, bins) - 1

    ece = 0.0
    total = len(y_true)

    for b in range(n_bins):
        idx = np.where(bin_ids == b)[0]
        if len(idx) == 0:
            continue

        acc = np.mean(y_true[idx] == preds[idx])
        conf = np.mean(confidences[idx])
        ece += (len(idx) / total) * abs(acc - conf)

    return ece