import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load your data
data = pd.read_csv("solar_weather_data.csv", parse_dates=['timestamp'], index_col='timestamp')
data = data[['solar_power_kwh', 'temperature', 'humidity', 'irradiance']]  # adjust based on your CSV

# Fill missing values
data.fillna(method='ffill', inplace=True)

# Normalize features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length][0])  # predicting solar_power_kwh
    return np.array(X), np.array(y)

seq_length = 24  # 24 hours
X, y = create_sequences(scaled_data, seq_length)

# Split into train/test
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM Model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(32))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Predict and plot
predictions = model.predict(X_test)
predicted = scaler.inverse_transform(np.hstack((predictions, np.zeros((len(predictions), scaled_data.shape[1] - 1)))))[:,0]
actual = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test), scaled_data.shape[1] - 1)))))[:,0]

plt.plot(actual, label='Actual')
plt.plot(predicted, label='Predicted')
plt.legend()
plt.title("Solar Energy Forecasting")
plt.xlabel("Time Step")
plt.ylabel("Solar Power (kWh)")
plt.show()
