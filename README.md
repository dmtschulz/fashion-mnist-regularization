# FashionMNIST Regularization (Deep Learning 1 - Homework)

This is a small homework project focused on regularization techniques for fully connected neural networks, using a reduced-size FashionMNIST dataset to deliberately induce overfitting.

## 🧠 What’s Inside

- Simple Feedforward Neural Net (FFNN) trained on 200 samples
- Regularization via:
  - Early Stopping (based on validation loss)
  - Dropout (p = 0.8)
- Visualized training/validation losses
- Test accuracy comparison for both methods

## 📊 Results

Both early stopping and dropout helped reduce overfitting and slightly improved test performance compared to the baseline.

Plots and saved models included.

## 🛠 Tech Stack

- PyTorch
- Torchvision
- Matplotlib

## 📁 Structure

- `fashion_mnist_regularization.ipynb` — main notebook
- `src/` — utility functions
- `plots/` — training loss graphs
- `saved_models/` — best models

---

Just homework — but done cleanly.
