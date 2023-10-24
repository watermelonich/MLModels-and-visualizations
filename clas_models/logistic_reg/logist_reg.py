import numpy as np

# Define the sigmoid function
def sigmoid(x):
    """
    Calculate the sigmoid function.

    Args:
        x (float): Input value.

    Returns:
        float: Sigmoid output in the range (0, 1).
    """
    return 1 / (1 + np.exp(-x))

class LogisticRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        """
        Initialize the logistic regression model.

        Args:
            lr (float): Learning rate for gradient descent.
            n_iters (int): Number of iterations for training.
        """
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        """
        Train the logistic regression model.

        Args:
            x (numpy array): Input features.
            y (numpy array): Target labels (0 or 1).

        This method trains the model by updating its weights and bias using
        gradient descent to minimize the log loss.
        """
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(x, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1 / n_samples) * np.dot(x.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, x):
        """
        Make predictions with the trained model.

        Args:
            x (numpy array): Input features for prediction.

        Returns:
            list: List of binary class predictions (0 or 1).
        """
        linear_pred = np.dot(x, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)

        # Convert probability to binary class (0 or 1)
        class_pred = [0 if y <= 0 else 1 for y in y_pred]

        return class_pred
