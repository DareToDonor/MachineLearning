import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Function to build and train the model
def build_and_train_model(X_train, y_train, X_test, y_test):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    return model, history

# Function to make predictions
def make_predictions(model, scaler_X, scaler_y, input_data):
    input_data_scaled = scaler_X.transform(input_data)
    predictions_scaled = model.predict(input_data_scaled)
    predictions = scaler_y.inverse_transform(predictions_scaled)
    return predictions

# Read data from file or other data source
data = pd.read_csv('blood_demand_surabaya_data.csv', parse_dates=['Date'])

# Extract time features from the date column
data['hari'] = data['Date'].dt.day
data['bulan'] = data['Date'].dt.month
data['tahun'] = data['Date'].dt.year

# Separate features (X) and target (y) for each blood type
X_A = data[['hari', 'bulan', 'tahun']]
y_A = data['A']

X_O = data[['hari', 'bulan', 'tahun']]
y_O = data['O']

X_B = data[['hari', 'bulan', 'tahun']]
y_B = data['B']

X_AB = data[['hari', 'bulan', 'tahun']]
y_AB = data['AB']

# Normalization for each blood type
scaler_X_A = MinMaxScaler()
scaler_y_A = MinMaxScaler()

X_A_scaled = scaler_X_A.fit_transform(X_A)
y_A_scaled = scaler_y_A.fit_transform(y_A.values.reshape(-1, 1))

scaler_X_O = MinMaxScaler()
scaler_y_O = MinMaxScaler()

X_O_scaled = scaler_X_O.fit_transform(X_O)
y_O_scaled = scaler_y_O.fit_transform(y_O.values.reshape(-1, 1))

scaler_X_B = MinMaxScaler()
scaler_y_B = MinMaxScaler()

X_B_scaled = scaler_X_B.fit_transform(X_B)
y_B_scaled = scaler_y_B.fit_transform(y_B.values.reshape(-1, 1))

scaler_X_AB = MinMaxScaler()
scaler_y_AB = MinMaxScaler()

X_AB_scaled = scaler_X_AB.fit_transform(X_AB)
y_AB_scaled = scaler_y_AB.fit_transform(y_AB.values.reshape(-1, 1))

# Build and train models for each blood type
model_A, history_A = build_and_train_model(X_A_scaled, y_A_scaled, X_A_scaled, y_A_scaled)  # Use entire data for simplicity
model_O, history_O = build_and_train_model(X_O_scaled, y_O_scaled, X_O_scaled, y_O_scaled)
model_B, history_B = build_and_train_model(X_B_scaled, y_B_scaled, X_B_scaled, y_B_scaled)
model_AB, history_AB = build_and_train_model(X_AB_scaled, y_AB_scaled, X_AB_scaled, y_AB_scaled)

# Predictions for the next two years (2024)
tahun_berikutnya = 2024
tanggal_2024 = pd.date_range(start=f'{tahun_berikutnya}-01-01', end=f'{tahun_berikutnya+1}-12-31', freq='D')
data_2024 = pd.DataFrame({'hari': tanggal_2024.day, 'bulan': tanggal_2024.month, 'tahun': tanggal_2024.year})

predictions_A_2024 = make_predictions(model_A, scaler_X_A, scaler_y_A, data_2024)
predictions_O_2024 = make_predictions(model_O, scaler_X_O, scaler_y_O, data_2024)
predictions_B_2024 = make_predictions(model_B, scaler_X_B, scaler_y_B, data_2024)
predictions_AB_2024 = make_predictions(model_AB, scaler_X_AB, scaler_y_AB, data_2024)

# Add prediction results to the DataFrame
data_2024['permintaan_A_prediksi'] = predictions_A_2024.flatten()
data_2024['permintaan_O_prediksi'] = predictions_O_2024.flatten()
data_2024['permintaan_B_prediksi'] = predictions_B_2024.flatten()
data_2024['permintaan_AB_prediksi'] = predictions_AB_2024.flatten()

# Predictions for 2025
tanggal_2025 = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')
data_2025 = pd.DataFrame({'hari': tanggal_2025.day, 'bulan': tanggal_2025.month, 'tahun': tanggal_2025.year})

predictions_A_2025 = make_predictions(model_A, scaler_X_A, scaler_y_A, data_2025)
predictions_O_2025 = make_predictions(model_O, scaler_X_O, scaler_y_O, data_2025)
predictions_B_2025 = make_predictions(model_B, scaler_X_B, scaler_y_B, data_2025)
predictions_AB_2025 = make_predictions(model_AB, scaler_X_AB, scaler_y_AB, data_2025)

# Add prediction results to the DataFrame
data_2025['permintaan_A_prediksi'] = predictions_A_2025.flatten()
data_2025['permintaan_O_prediksi'] = predictions_O_2025.flatten()
data_2025['permintaan_B_prediksi'] = predictions_B_2025.flatten()
data_2025['permintaan_AB_prediksi'] = predictions_AB_2025.flatten()

# Display the table of prediction results
print(data_2024)
print(data_2025)

# Combine 'hari', 'bulan', dan 'tahun' menjadi satu kolom 'tanggal'
data_2024['tanggal'] = pd.to_datetime(data_2024[['tahun', 'bulan', 'hari']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d')

# Hapus kolom 'hari', 'bulan', dan 'tahun' yang sudah tidak diperlukan
data_2024 = data_2024.drop(['hari', 'bulan', 'tahun'], axis=1)

# Set kolom 'tanggal' sebagai indeks
data_2024.set_index('tanggal', inplace=True)

# Tampilkan hasil prediksi untuk tahun 2024
print(data_2024)

data_2024.index = data_2024.index.to_period("M")
data_bulanan = data_2024.groupby(data_2024.index).sum().reset_index()
data_bulanan.index = data_bulanan['tanggal'].dt.to_timestamp()
data_bulanan = data_bulanan.drop('tanggal', axis=1)

print(data_bulanan)gi