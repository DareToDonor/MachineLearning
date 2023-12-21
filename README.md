# Blood Demand Prediction with Neural Network
This repository contains Python code for predicting blood demand using neural networks. The code utilizes TensorFlow and scikit-learn libraries for building and training regression models.

## Project Overview
The goal of this project is to predict the demand for blood based on historical data. The dataset includes information such as the date, day, month, year, and blood type categories (A, O, B, AB). 
The project involves preprocessing data, scaling, building separate neural network models for each blood type, making predictions, and analyzing the results.

## How to use
1. Clone this repository.
2. Install the dependencies by running the command `pip install -r requirements.txt`.
3. Run the application with the command `python prediksi.py`.

## Ways of working
### 1.Preprocessing Data:
Using pandas to read data from CSV files.
Removed the 'Location' column.
Create additional columns 'day', 'month', and 'year' from the 'Date' column.

### 2.Data Scaling:
Uses Min-Max Scaling to convert input and output values ​​to a range between 0 and 1.

### 3.Building Models:
Create and train a neural network model for each blood type (A, O, B, AB).
Uses one input layer with 32 neurons and ReLU activation function.
Using one output layer without an activation function (linear) for regression problems.
Using Adam optimizer and mean squared error as loss function.

### 4.Saving Models:
Save the trained model in the files model_A.h5, model_O.h5, model_B.h5, and model_AB.h5.

### 5.Making Predictions:
Make predictions for 2024 and 2025 using pre-trained models.
Converts normalized predictions back to the original scale.

### 6.Data Manipulation:
Organize prediction results into DataFrames for 2024 and 2025.
Change the date format and set the DataFrame index for further analysis.
Group monthly data and calculate the total number of blood requests each month.
