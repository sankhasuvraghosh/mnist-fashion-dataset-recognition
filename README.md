# Fashion-MNIST Image Classification
### Convolutional Neural Network · PyTorch · 10-Class Grayscale Clothing Recognition

A deep learning model that classifies grayscale fashion product images across 10 clothing categories — T-shirts, trousers, pullovers, dresses, coats, sandals, shirts, sneakers, bags, and ankle boots — trained on the Fashion-MNIST benchmark dataset.

Built with **PyTorch**, this project demonstrates a production-style CNN pipeline with convolutional feature extraction, batch normalization, dropout regularization, and adaptive optimization.

> Fashion-MNIST is a more challenging replacement for handwritten digit recognition, using **28×28 grayscale apparel images** that require richer feature learning than MNIST.

---

## Demo

```bash
Enter the index of the Fashion-MNIST test image (0–9999): 782

Actual label:    Sneaker
Model prediction: Sneaker ✓
```

[Sample prediction]
<img width="402" height="103" alt="image" src="https://github.com/user-attachments/assets/3e5f3cd5-bb5f-4884-8ce4-fcf88b9e7ef7" />


---

## Model Architecture

```plaintext
Input (1×28×28)
    │
    ▼
Conv Block 1 → Conv2d(1→32)   + BatchNorm + ReLU + MaxPool   → 32×14×14
Conv Block 2 → Conv2d(32→64)  + BatchNorm + ReLU + MaxPool   → 64×7×7
Conv Block 3 → Conv2d(64→128) + BatchNorm + ReLU             → 128×7×7
    │
    ▼
Flatten → Linear(6272→256) + ReLU + Dropout(0.5)
        → Linear(256→128) + ReLU + Dropout(0.3)
        → Linear(128→10)
    │
    ▼
Output (10 classes)
```

| Component         | Detail                             |
|------------------|-----------------------------------|
| Conv blocks      | 3 (channels: 1→32→64→128)         |
| Regularization   | BatchNorm + Dropout (0.5, 0.3)    |
| Optimizer        | Adam (lr=0.001)                   |
| Loss function    | CrossEntropyLoss                  |
| Batch size       | 64                                |
| Epochs           | 15–25                             |

---

## Results

| Metric          | Value        |
|-----------------|-------------|
| Test Accuracy   | **~91–94%** |
| Dataset         | Fashion-MNIST |
| Training images | 60,000 |
| Test images     | 10,000 |
| Image size      | 28×28 Grayscale |

> Replace with your actual trained accuracy.

---

## Data Preprocessing

The dataset is normalized to improve convergence during training.

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

---

## Project Structure

```plaintext
fashion-mnist-classification/
│
├── fashion_mnist_classifier.py   # Main training + evaluation script
├── fashion_mnist_model.pth       # Saved trained weights
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip
- GPU recommended (CUDA-compatible)

---

### Installation

```bash
# Clone repository
git clone https://github.com/sankhasuvraghosh/fashion-mnist-classification.git
cd fashion-mnist-classification

# Install dependencies
pip install -r requirements.txt

# Run training
python fashion_mnist_classifier.py
```

The Fashion-MNIST dataset will be downloaded automatically on first run.

---

## How It Works

### 1. Data Loading
Fashion-MNIST is loaded via `torchvision.datasets.FashionMNIST`.

- **Training set:** 60,000 images
- **Test set:** 10,000 images

---

### 2. Feature Extraction
Three convolutional blocks progressively detect:

- edges
- texture patterns
- shape structures
- high-level clothing features

---

### 3. Classification

The flattened feature vector passes through fully connected layers with dropout regularization for robust prediction.

---

### 4. Training

The network trains using:

- **Adam optimizer**
- **CrossEntropy loss**
- Mini-batch gradient descent
- Automatic GPU acceleration if available

---

### 5. Evaluation

Accuracy is computed on unseen test data.

---

### 6. Inference

Input a test image index to visualize:

- Actual class
- Model prediction
- Correct/incorrect classification

---

## Classes

| Label | Class       | Label | Class       |
|------|-------------|------|-------------|
| 0 | T-shirt/top | 5 | Sandal |
| 1 | Trouser | 6 | Shirt |
| 2 | Pullover | 7 | Sneaker |
| 3 | Dress | 8 | Bag |
| 4 | Coat | 9 | Ankle boot |

---

## Technologies

| Tool | Purpose |
|------|---------|
| PyTorch | Model building and training |
| Torchvision | Dataset loading |
| Matplotlib | Visualization |
| CUDA (optional) | GPU acceleration |

---

## Requirements

```txt
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
```

---

## Why Fashion-MNIST?

Unlike handwritten digits, Fashion-MNIST introduces:

- Similar-looking classes (shirt vs t-shirt)
- Complex textures
- Higher feature ambiguity
- More realistic computer vision challenges

This makes it a strong  benchmark for learning CNNs.

---

## Author

**Sankha Suvra Ghosh**  
GitHub: https://github.com/sankhasuvraghosh

---
