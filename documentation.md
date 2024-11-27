# Stock Price and Return Viewer Documentation

## Overview
The **Stock Price and Return Viewer** is a Flask-based web application that allows users to view historical stock prices, analyze returns, and predict the next day's stock price. It utilizes data from Yahoo Finance, visualizes it using `matplotlib`, and employs a Random Forest model for stock price predictions.

## Key Features
- **View Historical Prices**: Users can view historical closing prices for selected timeframes.
- **Cumulative Returns Analysis**: Provides visualization of cumulative returns, including moving averages.
- **Price Prediction**: Uses a machine learning model to predict the next day's closing price, providing Mean Squared Error (MSE) as a measure of accuracy.

## System Requirements
- **Python**: Version 3.7 or later.
- **Libraries**: Install the following packages:
  - `Flask`
  - `yfinance`
  - `matplotlib`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `joblib`

## Installation Guide
### Step 1: Clone or Download the Project
Clone the repository or download the project files to your local machine.

### Step 2: Install Dependencies
Navigate to the project folder and install the required Python libraries using pip:
```bash
pip install flask yfinance matplotlib pandas numpy scikit-learn joblib
```

### Step 3: Run the Application
Start the Flask application using the following command:
```bash
python app.py
```
Open your browser and go to `http://127.0.0.1:5000` to access the application.

## Usage Instructions
### User Interface
1. **Stock Ticker**: Enter the stock ticker symbol (e.g., `MSFT`, `AAPL`).
2. **Timeframe Selection**: Select a timeframe (e.g., `1 Day`, `1 Month`, `1 Year`).
3. **Mode Selection**:
   - **Price**: View historical closing prices.
   - **Returns**: View cumulative returns for the selected timeframe.
   - **Tomorrow's Prediction**: Predict tomorrow's closing price. This mode is disabled for `1 Day` timeframe as intraday predictions aren't applicable.
4. **View Chart**: Click the button to generate the chart.

### Available Modes
- **Price**: Plots the historical closing prices for the chosen timeframe.
- **Returns**: Plots cumulative returns with short-term and long-term moving averages.
- **Tomorrow's Prediction**: Uses a Random Forest regression model to predict the next day's price and displays the prediction along with the model's MSE.

## Technical Information
### Model Training
- The **Random Forest Regressor** is used to predict stock prices, trained using the following features:
  - Returns
  - 5-day and 20-day Simple Moving Averages (SMA)
  - 10-day Exponential Moving Average (EMA)
  - **Relative Strength Index (RSI)**
  - **Moving Average Convergence Divergence (MACD)**
- **MSE Calculation**: Mean Squared Error (MSE) is computed to evaluate the prediction accuracy.
- **Model Saving**: Models are saved with the ticker symbol to allow future reuse.

### Dynamic Model Management
- The application automatically retrains models if outdated models are detected (e.g., models without an associated MSE).
- Models and their respective MSE are saved using `joblib` for easy loading and evaluation.

## File Structure
- **app.py**: Main Flask application script.
- **templates/index.html**: HTML template for the application interface.
- **README.md**: Project documentation (included).
- **static/**: Contains any additional static resources (currently empty).

## Error Handling
- Invalid stock tickers or unavailable data are handled gracefully with error messages.
- The **Tomorrow's Prediction** mode is disabled for the `1 Day` timeframe, preventing misleading predictions.

## Frequently Asked Questions
### 1. What should I do if no data is displayed?
   Ensure that the stock ticker is valid and that Yahoo Finance has data for the selected timeframe.

### 2. Why is the "Tomorrow's Prediction" mode unavailable?
   The prediction is disabled for `1 Day` timeframe to avoid inaccurate predictions based on intraday data.

### 3. How accurate is the model?
   The model's **Mean Squared Error (MSE)** is displayed along with the prediction, giving users an indication of the prediction accuracy.

## Future Improvements
- Add additional machine learning models for prediction comparison.
- Improve data visualizations using interactive charting libraries such as Plotly.
- Implement user authentication for saving and accessing user preferences.

## License
This project is available under the MIT License.

