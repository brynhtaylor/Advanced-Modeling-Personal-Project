# Stock Price and Return Viewer

This is a Flask-based web application that allows users to view historical stock prices, returns, and predictions for tomorrow's stock price using a Random Forest regression model. The app fetches stock data using the Yahoo Finance API and generates visualizations for different timeframes.

## Features
- **View Historical Prices**: Display historical stock prices for selected timeframes.
- **Analyze Returns**: Calculate and display cumulative returns for the selected timeframe.
- **Predict Tomorrow's Price**: Use a trained Random Forest model to predict tomorrow's stock price and display the Mean Squared Error (MSE) for the model.

## How It Works
1. **Data Fetching**: The app uses `yfinance` to fetch historical stock data based on the selected ticker and timeframe.
2. **Visualization**: Historical prices and returns are plotted using `matplotlib` and rendered in the browser.
3. **Prediction**: A Random Forest regression model is trained on historical data with technical indicators (e.g., moving averages, RSI, MACD) as features. The model predicts tomorrow's stock price and displays its accuracy using MSE.

## Requirements
- Python 3.7+
- Required Python libraries:
  - `Flask`
  - `yfinance`
  - `matplotlib`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `joblib`

## Setup
1. Clone the repository or download the files.
2. Install the required Python libraries:
   ```bash
   pip install flask yfinance matplotlib pandas numpy scikit-learn joblib
   ```
3. Run the application:
   ```bash
   python app.py
   ```
4. Open your browser and navigate to `http://127.0.0.1:5000`.

## Usage
1. Enter a stock ticker (e.g., `MSFT`, `AAPL`) in the "Stock Ticker" input field.
2. Select a timeframe from the dropdown menu (e.g., `1 Day`, `1 Month`, `1 Year`).
3. Choose a mode:
   - **Price**: View historical stock prices.
   - **Returns**: View cumulative returns for the selected timeframe.
   - **Tomorrow's Prediction**: Predict tomorrow's stock price. *(This mode is disabled for the 1-day timeframe.)*
4. Click the "View Chart" button to generate the visualization.

## File Structure
```
.
├── app.py          # Main Python script for the Flask application
├── templates
│   └── index.html  # HTML template for the web interface
├── static          # Static files (if any)
├── README.md       # Documentation file
```

## Technical Details
### Model Training
- The Random Forest regression model is trained on historical stock prices using the following features:
  - Returns
  - 5-day and 20-day Simple Moving Averages (SMA)
  - 10-day Exponential Moving Average (EMA)
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
- The model and its Mean Squared Error (MSE) are saved using `joblib`.

### Data Visualization
- **Price Mode**: Plots the closing prices for the selected timeframe.
- **Returns Mode**: Plots cumulative returns with short-term and long-term moving averages.
- **Prediction Mode**: Plots historical prices and the predicted price for tomorrow.

### Dynamic Prediction
- If an outdated model (without MSE) is detected, it is automatically retrained to include MSE.

## Notes
- The "Tomorrow's Prediction" mode is disabled for the 1-day timeframe because predictions based on intraday data are not meaningful.
- The app gracefully handles errors (e.g., invalid stock tickers or missing data).

## Future Improvements
- Add more machine learning models for comparison.
- Implement user authentication for saving preferences.
- Enhance visualizations with interactive charts using libraries like Plotly.

## License
This project is open-source and available under the MIT License.