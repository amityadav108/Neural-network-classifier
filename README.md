# Neural-network-classifier
# Neural Network for Image Recognition of Letters A, B, C
# Project Overview
This project a part of modeule 11- Assignment, where we implement a feedforward neural network from scratch using only NumPy (no machine learning libraries like TensorFlow or PyTorch). The goal is to classify simple 5x6 pixel images of the letters A, B, and C.

# Dataset Description
We manually create small binary image representations of:
   - Letter A
   - Letter B
   - Letter C
Each image has 30 pixels (5 rows × 6 columns), and values are either 1 (white) or 0 (black).
# Example
Letter A = 
0 0 1 1 0 0
0 1 0 0 1 0
1 1 1 1 1 1
1 0 0 0 0 1
1 0 0 0 0 1

The labels are:
A → [1,0,0]
B → [0,1,0]
C → [0,0,1]

# Neural Network Architecture
   - Input Layer: 30 neurons (each for one pixel)
   - Hidden Layer: 5 neurons
   - Output Layer: 3 neurons (for A, B, or C)
We use the sigmoid activation function for both layers.

# Implementation Steps
1. Data visualization : Use matplotlib to plot and check each letter's shape
2. Data preparation: Convert data and labels into NumPy arrays for easier processing.
3. Weight Initialization: Use random values for the weights between input-hidden and hidden-output layers.
4. Feedforward Function: Calculates outputs using matrix multiplications and activations functions.
5. Loss function: Use Mean Squared Error(MSE)
6.Backpropagation: Calculates gradients to adjust weights using the chain rule and updates them using a learning rate (alpha).
7. Training Loop:
   Run multipleRun multiple epochs where:
    - Forward pass is done.
    - Error is calculated.
    - Weights are updated.
    - Accuracy and loss are tracked.

8.Predection Function:
Based on the trained model, identifies whether the image is of A,B or C

# Training Results
    - 100 epochs of training.
    - Accuracy and loss are plotted using Matplotlib.
    - Accuracy improves as the loss reduces.

# Result
   - The model successfully predicts:
   - A image → "Image is of letter A"
   - B image → "Image is of letter B"
   - C image → "Image is of letter C"
Each prediction is shown along with the image using plt.imshow().

# Visual Output
   - Accuracy vs Epochs: Shows improvement over time.
   - Loss vs Epochs: Decreases as the model learns.

# Key Learnings
    - How a basic neural network works without any external ML libraries.
    - Importance of activation functions, loss, and backpropagation.
    - Visualizing accuracy and loss during training.
    - Hands-on experience with manual weight updates and predictions.

# To Run the Code
    - Make sure you have Python and NumPy installed.
    - Copy and run the script in a Python environment.
    - Watch the model train and then predict the letters.
