import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime

st.title("Stock Market Analysis with GRU")

# User input for stock selection
stock = st.selectbox('Select stock to analyze', ['AMZN', 'IBM', 'MSFT'], index=0)
year = st.number_input('Enter year for prediction', min_value=2010, max_value=datetime.now().year, value=datetime.now().year, step=1)

# Load data
@st.cache
def load_data(stock):
    start_date = '2010-01-01'
    end_date = f'{year}-12-31'
    data = yf.download(stock, start=start_date, end=end_date)['Adj Close']
    return data

data = load_data(stock)
st.subheader('Raw Data')
st.write(data.head())

# Data statistics
st.subheader('Data Statistics')
st.write(data.describe())

# Plot the stock data
st.subheader('Stock Data Over Time')
st.line_chart(data)

# Data preprocessing
scaler = MinMaxScaler()

# Scale the data
scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

# Convert the data into sequences
def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :])
        y.append(data[i + time_step, :])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_sequences(scaled_data, time_step)

# Split the data into training and testing sets
split_ratio = 0.8
split = int(len(X) * split_ratio)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build the GRU model
model = Sequential()
model.add(GRU(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))
model.add(GRU(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the model
with st.spinner('Training the model...'):
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Plot training and validation loss
st.subheader('Model Training Loss')
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
st.pyplot(plt)

# Predicting on the test set
y_pred = model.predict(X_test)

# Inverse scaling to get actual prices
y_test_scaled = scaler.inverse_transform(y_test)
y_pred_scaled = scaler.inverse_transform(y_pred)

# Evaluate the model
mse = mean_squared_error(y_test_scaled, y_pred_scaled)
mae = mean_absolute_error(y_test_scaled, y_pred_scaled)
r2 = r2_score(y_test_scaled, y_pred_scaled)

st.subheader('Model Performance')
st.write(f'Mean Squared Error: {mse}')
st.write(f'Mean Absolute Error: {mae}')
st.write(f'R-squared: {r2}')

# Plotting the actual vs. predicted prices
st.subheader(f'{stock} Stock Price Prediction')
plt.figure(figsize=(12, 6))
plt.plot(y_test_scaled, label=f'Actual {stock} Price')
plt.plot(y_pred_scaled, label=f'Predicted {stock} Price')
plt.title(f'{stock} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(plt)

# Predicting stock price for the user-specified year
@st.cache
def predict_price_for_year(stock, year):
    future_date = f'{year}-12-31'
    future_data = yf.download(stock, start='2010-01-01', end=future_date)['Adj Close']
    future_scaled_data = scaler.transform(future_data.values.reshape(-1, 1))
    
    # Generate sequences for future data
    X_future, _ = create_sequences(future_scaled_data, time_step)
    
    if len(X_future) > 0:
        future_prediction = model.predict(X_future)
        future_pred_scaled = scaler.inverse_transform(future_prediction)
        return future_pred_scaled[-1][0]  # Return the last prediction
    else:
        return None

future_price = predict_price_for_year(stock, year)

st.subheader(f'Predicted Stock Price for {stock} in {year}')
if future_price is not None:
    st.write(f'The predicted stock price of {stock} for the end of {year} is ${future_price:.2f}')
else:
    st.write('Insufficient data to make prediction.')
