import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset (update the path to your dataset)
# For example: df = pd.read_csv('path/to/Nairobi_Office_Price_Ex.csv')
df = pd.read_csv('Nairobi Office Price Ex.csv')  # Load your dataset

# Extracting features and target variable
X = df['SIZE'].values  # Feature (office size)
y = df['PRICE'].values  # Target (office price)


# Mean Squared Error Function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Gradient Descent Function
def gradient_descent(X, y, m, c, learning_rate, epochs):
    n = len(y)  # Number of observations
    errors = []  # To store the error at each epoch

    for epoch in range(epochs):
        # Predictions
        y_pred = m * X + c

        # Calculate the error
        error = mean_squared_error(y, y_pred)
        errors.append(error)

        # Calculate gradients
        dm = (-2 / n) * np.dot(X, (y - y_pred))  # Gradient for m
        dc = (-2 / n) * np.sum(y - y_pred)  # Gradient for c

        # Update weights
        m -= learning_rate * dm
        c -= learning_rate * dc

        # Print the error for the current epoch
        print(f'Epoch {epoch + 1}/{epochs}, MSE: {error:.4f}')

    return m, c, errors


# Initialize parameters
np.random.seed(42)  # For reproducibility
m = np.random.rand()  # Random slope
c = np.random.rand()  # Random y-intercept
learning_rate = 0.01  # Learning rate
epochs = 10  # Number of epochs

# Train the model
m, c, errors = gradient_descent(X, y, m, c, learning_rate, epochs)

# Plot the line of best fit
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, m * X + c, color='red', label='Line of Best Fit')
plt.xlabel('Office Size (sq. ft.)')
plt.ylabel('Office Price')
plt.title('Linear Regression: Office Size vs. Price')
plt.legend()
plt.show()

# Predict the office price for 100 sq. ft.
size = 100
predicted_price = m * size + c
print(f'Predicted office price for {size} sq. ft.: ${predicted_price:.2f}')
