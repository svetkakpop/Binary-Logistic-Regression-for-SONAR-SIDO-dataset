# Logistic Regression for Sonar/SIDO Data Classification


## Introduction
Machine learning has a wide range of applications across various fields, demonstrating its versatility and importance. Notable examples include:

- **Facial Recognition:** Used in security systems for real-time identification, such as in airports and smartphones, and in social media platforms like Facebook for tagging friends in photos.
- **Product Recommendations:** E-commerce giants like Amazon utilize algorithms to analyze user behavior and preferences, significantly increasing sales and customer satisfaction.
- **Fraud Detection:** Banks employ machine learning to analyze transaction patterns and flag unusual activities, helping to prevent financial losses and protect customers from identity theft.
- **Chatbots:** Companies like Zendesk use chatbots to provide instant responses to customer inquiries, improving service efficiency and user experience.
- **Healthcare Diagnostics:** Machine learning algorithms analyze medical images, such as X-rays and MRIs, to assist radiologists in identifying conditions like tumors or fractures, enhancing diagnostic accuracy and speed.

These applications highlight the transformative impact of machine learning across different sectors. For more detailed examples and sources, you can explore the following links: [Machine Learning Examples, Applications & Use Cases](https://www.ibm.com/think/topics/machine-learning-use-cases) and [30 Machine Learning Examples and Applications to Know](https://builtin.com/artificial-intelligence/machine-learning-examples-applications).

One of the key challenges in this field is classification—a method that involves assigning objects to specific categories based on their characteristics. This report will focus on binary classification, which entails dividing objects into two categories. For numerical calculations and experiments, two datasets were utilized: the sonar dataset, which classifies signals as either "rocks" or "mines," and the SIDO dataset, which contains values of 0 and 1 representing the effectiveness of compounds tested against the HIV virus (0 indicates ineffectiveness and 1 indicates effectiveness). The significance of the sonar model lies in its ability to extract meaningful insights essential for underwater exploration and mining operations. The SIDO dataset is crucial for evaluating the effectiveness of antiviral compounds.

## Description of the Model
Logistic regression is a statistical method used for binary classification, separating data into two categories. The logistic function, also known as the sigmoid function, estimates the probability that a given input belongs to one of these categories. 

Logistic regression calculates a value based on the input data using a linear combination of features (weights and feature values). This value is then transformed into a probability, indicating how likely it is that the object belongs to the positive class (e.g., "yes" or "1"). Thus, logistic regression helps us understand which of the two categories (positive or negative) a given object is more likely to belong to.

### Model Components
Let’s define the components of our model:

- **Input Vector:** Let $$\mathbf{x}$$ be the input vector, where 
  $$\mathbf{x} = [x_1, x_2, \ldots, x_n]$$
  - $$\mathbf{x}$$ — это вектор входных данных, представляющий собой набор признаков (фич), используемых для классификации.
  - $$n$$ — количество признаков.

- **Output:** Let $$y$$ be the output label, which can take values 0 or 1 for binary classification.
  - $$y$$ — это метка выходных данных, которая указывает на класс, к которому принадлежит объект (0 или 1).

- **Weights:** Let $$\mathbf{w}$$ be the weight vector associated with the input features, where 
  $$\mathbf{w} = [w_1, w_2, \ldots, w_n]$$
  - $$\mathbf{w}$$ — это вектор весов, который определяет важность каждого признака в процессе классификации.

- **Bias:** Let $$b$$ be the bias term.
  - $$b$$ — это смещение, которое позволяет модели лучше подстраиваться под данные.

In summary, the model takes an input vector $$\mathbf{x}$$ of dimension $$n$$, applies the weights $$\mathbf{w}$$ of the same dimension, adds the bias $$b$$, and produces a scalar output $$y$$.

## Prediction and the Sigmoid Function
The logistic function is defined as:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

where $$z$$ is the linear combination of input data, calculated as 

$$
z = \mathbf{X} \cdot \mathbf{w} + b
$$ 

- $$\sigma(z)$$ — это сигмоидная функция, которая преобразует линейную комбинацию входных данных в вероятность.
- $$z$$ — это линейная комбинация входных данных, которая рассчитывается как скалярное произведение вектора входных данных и вектора весов, с добавлением смещения.

To obtain the predicted labels $$y_{\text{pred}}$$, we apply a threshold to the output of the logistic function. Specifically, if the predicted probability $$\sigma(z)$$ is greater than or equal to 0.5, we classify the instance as 1 (positive class); otherwise, we classify it as 0 (negative class):

$$
y_{\text{pred}} = 
\begin{cases} 
1 & \text{if } \sigma(z) \geq 0.5 \\
0 & \text{if } \sigma(z) < 0.5 
\end{cases}
$$

The accuracy of the model is calculated as the proportion of correct predictions:

$$
\text{Accuracy} = \frac{1}{m} \sum_{i=1}^{m} \mathbb{I}(y_{\text{pred}}^{(i)} = y_{\text{true}}^{(i)})
$$

where $$\mathbb{I}$$ is an indicator function that equals 1 if the prediction matches the true value and 0 otherwise, and $$m$$ is the total number of predictions made.

## Maximum Likelihood Estimation
To estimate the parameters $$\mathbf{w}$$ and $$b$$, we use the method of maximum likelihood. The likelihood function for our binary classification problem can be defined as:

$$
L(\mathbf{w}, b) = \prod_{i=1}^{m} P(y^{(i)} | \mathbf{x}^{(i)}; \mathbf{w}, b)
$$

For each sample $$\mathbf{x}^{(i)}$$, there are two possible outcomes for the label $$y^{(i)}$$: it can either be 0 or 1. Thus, we can express the probabilities as follows:
- The probability of the outcome being 1 is given by $$p(y^{(i)} = 1 | \mathbf{x}^{(i)}, \mathbf{w})$$.
- The probability of the outcome being 0 is given by $$p(y^{(i)} = 0 | \mathbf{x}^{(i)}, \mathbf{w})$$.

These probabilities are related by the equation:

$$
p(y^{(i)} = 0 | \mathbf{x}^{(i)}, \mathbf{w}) + p(y^{(i)} = 1 | \mathbf{x}^{(i)}, \mathbf{w}) = 1
$$

If we denote $$p(y^{(i)} = 1 | \mathbf{x}^{(i)}, \mathbf{w}) = \sigma(z^{(i)})$$, we can express the probability of the outcome being 0 as:

$$
p(y^{(i)} = 0 | \mathbf{x}^{(i)}, \mathbf{w}) = 1 - \sigma(z^{(i)}) = \sigma(-z^{(i)})
$$

Given these insights, we can now express the likelihood function for binary classification as:

$$
L(\mathbf{w}, b) = \prod_{i=1}^{m} \sigma(z^{(i)})^{y^{(i)}} (1 - \sigma(z^{(i)}))^{(1 - y^{(i)})}
$$

Substituting the probabilities we defined earlier, this can be rewritten as:

$$
L(\mathbf{w}, b) = \prod_{i=1}^{m} \sigma(z^{(i)})^{y^{(i)}} (1 - \sigma(z^{(i)}))^{1 - y^{(i)}}
$$

For better overview of the function, we take the logarithm of the likelihood function:

$$
\ell(\mathbf{w}, b) = \sum_{i=1}^{m} \left( y^{(i)} \log(\sigma(z^{(i)})) + (1 - y^{(i)}) \log(1 - \sigma(z^{(i)})) \right)
$$

Maximizing the log-likelihood is equivalent to minimizing the negative log-likelihood, which is often referred to as the loss function. Thus, our classification task can be framed as:

$$
\min_{\mathbf{w}, b} -\ell(\mathbf{w}, b)
$$

This transformation from a product to a sum is achieved through logarithm properties, making the optimization process more manageable.

## Loss Function Analysis
The error function derived from the log-likelihood captures how well our model predicts the binary outcomes. A lower value of this function indicates better model performance, as it signifies that the predicted probabilities are closer to the actual labels.

The loss function, which we aim to minimize, reflects the model's ability to correctly classify the input data. If the model predicts a probability close to 1 for a positive class and close to 0 for a negative class, the loss will be low. Conversely, if the model makes incorrect predictions, the loss will increase, guiding the optimization process to adjust the weights and bias accordingly.

### Gradient Descent for Minimization
To optimize the weights and bias in logistic regression, we employ the gradient descent method. This iterative optimization algorithm updates the model parameters to minimize the loss function, which quantifies the difference between the predicted and actual outcomes.

At each iteration, the weights and bias are updated using the following formulas:

$$
\mathbf{w} \leftarrow \mathbf{w} - \alpha \cdot \nabla_{\mathbf{w}} J(\mathbf{w}, b)
$$
$$
b \leftarrow b - \alpha \cdot \nabla_{b} J(\mathbf{w}, b)
$$

where:
- $$\alpha$$ is the learning rate, a hyperparameter that controls the step size during optimization.
- $$J(\mathbf{w}, b)$$ is the loss function.

### Gradient Calculation
To find the gradient, we need to compute the partial derivatives of the loss function with respect to the weights and bias. The gradient can be derived as follows:

1. **Gradient with respect to weights:**  
   The partial derivative of the loss function with respect to the weights is given by:

   $$
   \nabla_{\mathbf{w}} J(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{pred}}^{(i)} - y^{(i)}) \cdot \mathbf{x}^{(i)}
   $$

2. **Gradient with respect to bias:**  
   The partial derivative of the loss function with respect to the bias is given by: 

   $$
   \nabla_{b} J(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{pred}}^{(i)} - y^{(i)})
   $$

### Learning Rate Considerations
The learning rate $$\alpha$$ is a critical hyperparameter in the gradient descent algorithm. If the learning rate is too small, convergence will be slow, requiring many iterations to reach the minimum. Conversely, if the learning rate is too large, the algorithm may overshoot the minimum, leading to divergence. It is essential to choose an appropriate learning rate to ensure efficient convergence.

## Problem Description
The primary challenge involves classifying sonar signals and molecular descriptors into two distinct categories. Specifically, we focus on two datasets:

1. **Sonar Dataset:**
   - **Objective:** Classify sonar signals into two categories: rocks (denoted as 'R') and mines (denoted as 'M').
   - **Number of Objects:** The dataset contains 208 samples.
   - **Number of Features:** Each sample is represented by 60 features derived from sonar readings.
   - **Target Labels:** The target variable is encoded in binary format, where 'R' is represented as 1 (effective) and 'M' as 0 (ineffective).

2. **SIDO Dataset:**
   - **Objective:** Classify compounds based on their effectiveness against the HIV virus.
   - **Number of Objects:** The dataset includes 1000 samples.
   - **Number of Features:** Each sample is represented by 10 molecular descriptors.
   - **Target Labels:** The target variable is already in binary format, where 1 indicates effectiveness (effective) and 0 indicates ineffectiveness (ineffective).

The datasets comprise various features derived from sonar readings and molecular properties. Our objective is to accurately predict the class of each signal or compound based on these features. In the case of having a value of -1 in our dataset, it was automatically changed to 0, as -1 was considered a negative result, indicating ineffectiveness.

## Workflow Overview
To address this classification problem, we implemented the following steps:

- **Data Preprocessing:** The dataset was loaded, and the target variable was encoded in binary format (1 for rocks and 0 for mines). For the SIDO dataset, the target variable was already in binary format (1 for effective and 0 for ineffective). The dataset was split into training and testing sets, with 60% for training.

- **Model Training:** The gradient descent method was used to train the logistic regression model by updating the weights and bias to minimize the loss function. Experiments were conducted with five different learning rates: 0.01, 0.08, 0.25, 0.5, and 0.8. The model was trained over multiple epochs, with the number of epochs varying from 100 to 10,000 to assess its impact on the loss function and accuracy. During training, information about the loss function and accuracy was recorded at each epoch.

- **Prediction:** The trained model was employed to predict the classes of the input data. The accuracy of these predictions was calculated by comparing the predicted labels with the actual labels from the test dataset.

## Graphs and Analysis
Let's examine different graphs based on parameters that affect the loss function and accuracy of the results: learning rate and number of epochs.

### Epochs 100
![Epochs 100](https://git.service.rjd/RJD/FirstSteps/raw/branch/master/images/epochs100.jpg)

With a very low learning rate of 0.01, the model struggles with numerous lost predictions and achieves poor accuracy, typically around 55-65%. A higher learning rate, such as 0.8, shows minimal loss but faces initial accuracy challenges.

### Epochs 1000
![Epochs 1000](https://git.service.rjd/RJD/FirstSteps/raw/branch/master/images/epjchs1000.jpg)

At 1000 epochs, a low learning rate of 0.01 still results in challenges with lost predictions, but the accuracy improves slightly to 60-67%. A higher learning rate like 0.8 maintains stability but struggles with accuracy in the middle stages. The optimal performance is achieved with learning rates of 0.25 or 0.5, which balance convergence speed and accuracy, reaching 77-84%.

### Epochs 10k
![Epochs 10.000](https://git.service.rjd/RJD/FirstSteps/raw/branch/master/images/epochs10k.jpg)

At 10,000 epochs, a very low learning rate of 0.01 still faces issues but achieves a higher accuracy of around 80% by the end. A higher learning rate like 0.8 shows minimal loss and reaches an accuracy of about 95%, despite some challenges. The most effective results are observed with a learning rate of around 0.5, offering a well-balanced performance with accuracy stabilizing around 93%.

While these observations provide insights into the model's performance at different stages, it is important to note that it may be premature to draw definitive conclusions. The loss functions for all learning rates have not yet reached their minimum, indicating that the model is still in the training process. Although a higher learning rate may accelerate convergence, the optimal learning rate should be determined after further training and analysis.

## Analysis
The analysis across different epochs reveals that the selection of the learning rate significantly impacts model performance. Lower learning rates (e.g., 0.01) result in slower convergence and lower accuracy, particularly in the early stages, though they improve over time. Higher learning rates (e.g., 0.8) demonstrate faster initial progress but may encounter challenges with accuracy. A learning rate in the range of 0.25 to 0.5 consistently offers the best balance between convergence speed and accuracy, making it the optimal choice for stable and reliable model performance across various training durations.

Additionally, the number of epochs plays a crucial role; the results indicate that training for 1000 to 3000 epochs provides a good balance for achieving consistent and stable results without overfitting or underfitting. Further experimentation with different learning rates and epoch counts will help refine the model's performance and ensure robust predictions.

## Conclusion
In this report, we explored the application of logistic regression for binary classification tasks using two distinct datasets: the sonar dataset and the SIDO dataset. Through the implementation of the gradient descent optimization algorithm, we effectively trained the model to classify sonar signals as either "rocks" or "mines" and to evaluate the effectiveness of compounds against the HIV virus.

The analysis of the model's performance revealed several key insights:

1. **Impact of Learning Rate:** The choice of learning rate significantly influences the convergence speed and accuracy of the model. Lower learning rates (e.g., 0.01) resulted in slower convergence and lower accuracy, particularly in the initial epochs. Conversely, higher learning rates (e.g., 0.8) demonstrated faster initial progress but faced challenges in maintaining accuracy throughout training.

2. **Optimal Learning Rate:** A learning rate in the range of 0.25 to 0.5 consistently provided the best balance between convergence speed and accuracy. This range allowed the model to achieve stable and reliable performance across various training durations.

3. **Effect of Epochs:** The number of epochs also played a crucial role in model performance. Results indicated that training for 1000 to 3000 epochs yielded consistent and stable results without leading to overfitting or underfitting. The model's accuracy improved significantly with increased epochs, particularly when paired with an optimal learning rate.
