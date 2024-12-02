#Python Code for CSI Prediction with LSTM
#Below is a simplified example code in Python using TensorFlow/Keras to predict CSI. In practice, CSI data would be collected from actual or simulated MIMO channels.

# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate simulated CSI data
np.random.seed(0)
data_length = 1000
timesteps = 10
csi_data = np.sin(np.linspace(0, 20, data_length)) + 0.5 * np.random.randn(data_length)

# Prepare dataset for LSTM
X, y = [], []
for i in range(len(csi_data) - timesteps):
    X.append(csi_data[i:i+timesteps])
    y.append(csi_data[i+timesteps])
X = np.array(X)
y = np.array(y)

# Reshape for LSTM input
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(timesteps, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X, y, epochs=10, verbose=1, batch_size=32)  # Added batch_size for better training performance

# Predict on the dataset
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print(f'Mean Squared Error: {mse:.4f}')

# Plotting the prediction
plt.figure(figsize=(10, 6))
plt.plot(y, label='Actual CSI', color='blue', alpha=0.7)
plt.plot(y_pred, label='Predicted CSI', linestyle='--', color='red', alpha=0.7)
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('CSI Value')
plt.title('CSI Prediction using LSTM')
plt.grid(True)
plt.show()

#Explanation of Python Code
#1.Data Preparation: Generates synthetic CSI data and reshapes it to fit the LSTM model. timesteps is set to 10, meaning the model looks at the previous 10 samples to predict the next one.
#2.Model Definition: An LSTM layer with 50 units is created, followed by a dense output layer for prediction.
#3.Training: The model is trained to minimize the Mean Squared Error between predicted and actual CSI values.
#4.Prediction and Evaluation: Predictions are made on the dataset, and Mean Squared Error is calculated. A plot compares actual and predicted values.
