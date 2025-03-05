# Logistic Regression for Sonar/SIDO Data Classification
 
**Date:** 05.03.2025

## Introduction

This report investigates the application of a logistic regression model to classify sonar signals as either rocks or mines. The significance of this model lies in its capacity to extract meaningful insights from sonar data, which is essential for underwater exploration and mining operations.

## Description of the Model

The analysis employs logistic regression, a statistical method designed for binary classification. This model estimates the probability that a given input belongs to a specific category. The logistic function, also known as the sigmoid function, is utilized to transform predicted values into probabilities. In the subsequent sections, we will explore how varying the learning rate affects the model's performance.

## Description of the Problem

The primary challenge involves classifying sonar signals, represented as features within a dataset, into two distinct categories: rocks (denoted as 'R') and mines (denoted as 'M'). The dataset comprises various features derived from sonar readings, and our objective is to accurately predict the class of each signal based on these features.

## Explanation of the Chosen Solutions

To address this classification problem, we implemented the following steps:

- **Data Preprocessing:** The dataset was loaded, and the target variable was encoded in binary format (1 for rocks and 0 for mines).
- **Feature Selection:** For simplicity and clarity, we selected only the first two features for visualization purposes.
- **Model Training:** We utilized gradient descent to optimize the weights and bias of the logistic regression model over 1000 epochs.
- **Prediction:** The trained model was employed to predict the classes of the input data, and the accuracy of these predictions was calculated.

## Graphs and Figures

The following figures illustrate the performance of the model with different learning rates:

### Learning Rate = 0.01
![Learning Rate = 0.01](https://git.service.rjd/RJD/FirstSteps/src/branch/master/images/image01.jpg)

### Learning Rate = 0.1
![Learning Rate = 0.1](https://git.service.rjd/RJD/FirstSteps/src/branch/master/images/image001.jpg)

### Learning Rate = 0.7
![Learning Rate = 0.7](https://git.service.rjd/RJD/FirstSteps/src/branch/master/images/image07.jpg)

- The image contains three plots labeled (a), (b), and (c), each illustrating the loss function over epochs for different learning rates in a machine learning context. 
  - Plot (a) shows the loss function with a learning rate of 0.01, indicating a gradual decrease in loss as the number of epochs increases, suggesting the model is learning slowly and steadily. 
  - Plot (b) represents the loss function with a learning rate of 0.1, which also shows a steady decline in loss over the epochs, but at a faster rate compared to the first plot. 
  - Finally, plot (c) presents the loss function with a learning rate of 0.7, where the loss decreases rapidly at first, but may exhibit instability as it approaches lower loss values, characteristic of a higher learning rate potentially leading to overshooting optimal values. 
- Each plot effectively demonstrates how different learning rates can significantly impact the convergence and behavior of the loss function during training.

## Overview of the Code and Mathematical Formulas

In this work, a logistic regression algorithm was implemented using the Python programming language and the NumPy library. The code includes several key components that facilitate the execution of the logistic regression model for classifying sonar data.

### Data Loading and Preparation

First, the data is loaded from a CSV file using the Pandas library. The data consists of a feature matrix \(X\) and a target variable vector \(y\). The target variable is encoded in binary format, where 'R' (rocks) is represented as 1 and 'M' (mines) as 0. For visualization and simplification purposes, only the first two features are used.

### Sigmoid Function

The primary mathematical function used in the model is the sigmoid function, which transforms linear combinations of input data into probabilities. The sigmoid function is defined as:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

where \(z\) is the linear combination of input data, calculated as \(z = X \cdot w + b\), with \(w\) being the weight vector and \(b\) the bias.

### Loss Function

To evaluate the model's performance, a loss function is used, which in this case is the logistic loss function (or cross-entropy):

\[
L(y_{\text{true}}, y_{\text{pred}}) = -\frac{1}{m} \sum_{i=1}^{m} \left( y_{\text{true}} \log(y_{\text{pred}}) + (1 - y_{\text{true}}) \log(1 - y_{\text{pred}}) \right)
\]

where \(m\) is the number of examples in the training set, \(y_{\text{true}}\) are the true class labels, and \(y_{\text{pred}}\) are the predicted probabilities.

### Gradient Descent

To optimize the weights and bias, the gradient descent method is employed. At each iteration, the weights and bias are updated using the following formulas:

\[
w = w - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} (y_{\text{pred}} - y) \cdot X
\]
\[
b = b - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} (y_{\text{pred}} - y)
\]

where \(\alpha\) is the learning rate, \(X\) is the feature matrix, and \(y\) is the vector of true labels.

### Prediction and Accuracy Evaluation

After training the model, predictions are made based on the learned weights and bias. The accuracy of the model is calculated as the proportion of correct predictions:

\[
\text{Accuracy} = \frac{1}{m} \sum_{i=1}^{m} \mathbb{I}(y_{\text{pred}} = y_{\text{true}})
\]

where \(\mathbb{I}\) is an indicator function that equals 1 if the prediction matches the true value and 0 otherwise.

### Loss Function Visualization

Finally, to analyze the training process, the loss function over epochs is visualized, allowing for an assessment of the model's learning efficiency.

Thus, the code implements the main stages of logistic regression, including data preparation, model training, prediction, and result visualization, which helps to understand the model's behavior depending on the chosen learning rate.

## Conclusion

The analysis of the loss functions over various epochs with different learning rates indicates distinct behaviors in the model's convergence.

- **Learning Rate = 0.01:** The loss decreases slowly, suggesting that the learning process might be too conservative, leading to longer training times.
- **Learning Rate = 0.1:** This configuration shows a more favorable decrease in loss, indicating a balanced learning rate that promotes effective convergence.
- **Learning Rate = 0.7:** Here, the loss drops rapidly initially but appears to plateau quickly, possibly indicating overfitting or instability in the learning process.

In summary, the choice of learning rate significantly influences the model's training efficiency and performance. A balanced learning rate, such as 0.1, is recommended for achieving optimal convergence.
