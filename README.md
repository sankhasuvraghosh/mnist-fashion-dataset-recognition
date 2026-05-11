This project implements a Convolutional Neural Network (CNN) for image classification on the Fashion MNIST dataset — a benchmark of 70,000 grayscale 28×28 images across 10 clothing categories (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot). The model uses two convolutional blocks with batch normalization and max pooling, followed by a fully connected head with dropout regularization, trained with the Adam optimizer. Achieves ~93–94% test accuracy.
A deep learning image classification project built with PyTorch using a CNN trained on the Fashion-MNIST dataset.
The model classifies grayscale fashion images into 10 categories such as T-shirts, trousers, dresses, shoes, bags, and coats using convolutional layers, batch normalization, max pooling, dropout regularization, and Adam optimization.

Features:
CNN-based image classification
Batch normalization for stable training
Max pooling for feature extraction
Dropout to reduce overfitting
Learning rate scheduling
Accuracy evaluation on test set
Single-image prediction visualization
Model saving/loading support

Classes:
T-shirt/top
Trouser
Pullover
Dress
Coat
Sandal
Shirt
Sneaker
Bag
Ankle boot

Tech Stack:-
Python
PyTorch
Torchvision
Matplotlib
NumPy
Output

Trained model predicts fashion item categories with high classification accuracy and visualizes predictions for test images.
