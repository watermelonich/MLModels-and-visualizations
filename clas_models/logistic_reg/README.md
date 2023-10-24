## Logistic Regression
Logistic Regression is a binary classification algorithm that is widely used in machine learning. This README explains the math and processes behind logistic regression using a simple Python code example.

# Math and Processes
Logistic Regression is a classification algorithm that uses the logistic or sigmoid function to model the probability that a given input belongs to a particular class. Here's a breakdown of the code and the key concepts involved:

# Sigmoid Function:

The sigmoid function is used to squash the linear combination of features into a range between 0 and 1, representing probabilities.
Constructor:

The constructor of the LogisticRegression class allows you to set hyperparameters such as the learning rate (lr) and the number of iterations (n_iters).

# Training (fit) Method
This method is used to train the logistic regression model.
Initialize the model's weights and bias to zero.
In each iteration, compute linear predictions by taking the dot product of input features and model weights, then apply the sigmoid function.
Calculate the gradient of the cost function with respect to the weights and bias.
Update the weights and bias using gradient descent to minimize the cost function.
Prediction (predict) Method:

Given a new set of input features, this method computes linear predictions, applies the sigmoid function, and converts the probabilities to class labels (0 or 1) using a threshold of 0.5.