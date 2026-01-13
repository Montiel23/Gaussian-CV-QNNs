# Continuous-variable Gaussian Quantum Neural Networks for Biomedical Image Classification

This repository contains the code accompanying the paper:

Gaussian Continuous-Variable Quantum Neural Networks for MedMNIST Classification

Daniel Alejandro Lopez, Oscar Montiel, Oscar Castillo, Miguel Lopez-Montiel
Institute of Physics - Quantum Science and Technology / arXiv: 	
https://doi.org/10.48550/arXiv.2511.02051

## Scope

This work implements **Gaussian continuous-variable quantum neural networks**
simulated using the covariance-matrix formalism.
No non-Gaussian gates or fault-tolerant quantum advantage claims are made.

## Repository structure
- .ipynb # main experiments files
- modules/ # python scripts for functions and code structure
- model_checkpoints/ # model and data checkpoints
- results/ # results, figures, logs from experiments

Generated figures and results are not tracked to keep the repository lightweight.

## Environment
- Python 3.10+
- PennyLane 0.29.1
- PyTorch, numpy, scipy, scikit-learn

## Reproducibility
Every notebook is tailored towards one of the evaluated datasets from MedMNIST, running with the functions from modules
1. Run breastmnist, organamnist, pneumoniamnist notebooks.
2. evaluate hypothesis testing using the saved logs and results using hypothesis-testing file.

## Computational notes
CV models are simulated using Gaussian backends from PennyLane.
Parameter-shift is used for gradient evaluation for both DV and CV models.

## License
This project is licensed unser the MIT license.

## Citation
If you use this code, please cite the accompanying paper.
