from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from chatbot import clean_up_sentence, predict_class, get_response, intents
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta


app = Flask(__name__)

# Get the current date
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

model_A = load_model('model_A.h5')
model_AB = load_model('model_AB.h5')
model_B = load_model('model_B.h5')
model_O = load_model('model_O.h5')
scaler_X_A = MinMaxScaler()
scaler_y_A = MinMaxScaler()

scaler_X_O = MinMaxScaler()
scaler_y_O = MinMaxScaler()

scaler_X_B = MinMaxScaler()
scaler_y_B = MinMaxScaler()

scaler_X_AB = MinMaxScaler()
scaler_y_AB = MinMaxScaler()

X_A_scaled = scaler_X_A.fit_transform(X_A)
y_A_scaled = scaler_y_A.fit_transform(y_A.values.reshape(-1, 1))
X_O_scaled = scaler_X_O.fit_transform(X_O)
y_O_scaled = scaler_y_O.fit_transform(y_O.values.reshape(-1, 1))
X_B_scaled = scaler_X_B.fit_transform(X_B)
y_B_scaled = scaler_y_B.fit_transform(y_B.values.reshape(-1, 1))
X_AB_scaled = scaler_X_AB.fit_transform(X_AB)
y_AB_scaled = scaler_y_AB.fit_transform(y_AB.values.reshape(-1, 1))

def make_predictions(model, scaler_X, scaler_y, input_data):
    input_data_scaled = scaler_X.transform(input_data)
    predictions_scaled = model.predict(input_data_scaled)
    predictions = scaler_y.inverse_transform(predictions_scaled)
    return predictions



@app.route('/predict', methods=['GET'])
def predictGet():
    try:
        start_date = datetime.now().replace(day=1).date()

        # Get the last day of the next 12 months
        end_date = (start_date + pd.DateOffset(months=12) - pd.DateOffset(days=1)).date()

        # Generate a range of dates from start_date to end_date
        tanggal = pd.date_range(start=start_date, end=end_date, freq='D')
        data_tanggal = pd.DataFrame({'hari': tanggal.day, 'bulan': tanggal.month, 'tahun': tanggal.year})

        predictions_A = make_predictions(model_A, scaler_X_A, scaler_y_A, data_tanggal)
        predictions_O = make_predictions(model_O, scaler_X_O, scaler_y_O, data_tanggal)
        predictions_B = make_predictions(model_B, scaler_X_B, scaler_y_B, data_tanggal)
        predictions_AB = make_predictions(model_AB, scaler_X_AB, scaler_y_AB, data_tanggal)

        data_tanggal['prediksi_permintaan_A'] = predictions_A.flatten()
        data_tanggal['prediksi_permintaan_O'] = predictions_O.flatten()
        data_tanggal['prediksi_permintaan_B'] = predictions_B.flatten()
        data_tanggal['prediksi_permintaan_AB'] = predictions_AB.flatten()

        data_tanggal['tanggal'] = pd.to_datetime(data_tanggal[['tahun', 'bulan', 'hari']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d')
        # Drop 'hari', 'bulan', dan 'tahun' columns
        data_tanggal = data_tanggal.drop(['hari', 'bulan', 'tahun'], axis=1)

        # Set 'tanggal' column as the index
        data_tanggal.set_index('tanggal', inplace=True)

        # Group by month and calculate the sum for each month
        data_tanggal.index = data_tanggal.index.to_period("M")
        data_bulanan = data_tanggal.groupby(data_tanggal.index).sum().reset_index().round(0)
        data_bulanan.index = data_bulanan['tanggal'].dt.to_timestamp()

        # Drop the 'tanggal' column
        data_bulanan = data_bulanan.drop('tanggal', axis=1)
        # Convert DataFrame to dictionary with "date" field
        data_bulanan['month'] = data_bulanan.index.strftime('%B')
        data_bulanan['year'] = data_bulanan.index.strftime('%Y')
        data_bulanan.reset_index(drop=True, inplace=True)
        data_prediksi_dict = data_bulanan.astype({'prediksi_permintaan_A': 'int', 
                                                  'prediksi_permintaan_AB': 'int', 
                                                  'prediksi_permintaan_B': 'int', 
                                                  'prediksi_permintaan_O': 'int'}).to_dict(orient='records')

        # Return the JSON response
        return jsonify({"status": "success",
        "message": "response succcess",
        "data": data_prediksi_dict
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/')
def index():
    return jsonify({"response": "Welcome"})

@app.route('/chat', methods=['POST'])
def chat():
    try :
        data = request.get_json()
        message = data['message']

        if message.lower() == 'exit':
            return jsonify({"response": "Goodbye!"})

        ints = predict_class(message)
        res = get_response(ints, intents)

        return jsonify({
        "status": "success",
        "message": "response succcess",
        "data": res,
        }), 200
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return jsonify({
            "message": "An error occurred",
            "status": "error",
            "error": str(e),
        }), 500  


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=8080)
