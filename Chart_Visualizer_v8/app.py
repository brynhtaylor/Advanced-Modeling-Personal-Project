from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64
from datetime import datetime
from datetime import timedelta
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import joblib  # For saving/loading the model
import os

app = Flask(__name__)

def prepare_features(data):
    # Create technical indicators
    data['Returns'] = data['Close'].pct_change()
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['RSI'] = calculate_rsi(data['Close'])
    data['MACD'] = calculate_macd(data['Close'])
    
    data.dropna(inplace=True)  # Remove rows with NaN values from rolling calculations
    
    # Features and target
    features = ['Returns', 'SMA_5', 'SMA_20', 'EMA_10', 'RSI', 'MACD']
    X = data[features]
    y = data['Close'].shift(-1)  # Predict next day's price
    
    # Remove rows where target is NaN
    valid_data = X.join(y).dropna()
    X = valid_data[features]
    y = valid_data['Close']
    
    return X, y

def train_model(data, ticker):
    X, y = prepare_features(data)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate performance
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error for {ticker}: {mse}")
    
    # Save the model and MSE as a tuple
    model_filename = f"{ticker}_random_forest_model.pkl"
    joblib.dump((model, mse), model_filename)
    return model_filename

def calculate_rsi(series, window=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd - macd_signal

# Function to get stock data
def get_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    
    # Determine how much extra data to pull
    if period == "1mo":
        extended_period = "3mo"
    elif period == "3mo":
        extended_period = "6mo"
    elif period == "6mo":
        extended_period = "1y"
    elif period == "1y":
        extended_period = "2y"
    elif period == "3y":
        extended_period = "5y"
    else:
        extended_period = "10y"

    # Fetch data with extended period for complete moving averages
    if period == "1d":
        # Fetch intraday data with 1-minute interval
        data = stock.history(period="1d", interval="1m")
    else:
        # Fetch data for extended periods
        data = stock.history(period=extended_period)
    
    # Ensure data contains 'Close' column
    if 'Close' not in data.columns or data.empty:
        raise ValueError(f"Data for ticker {ticker} does not contain 'Close' prices or is empty.")
    
    return data

# Function to calculate returns
def calculate_returns(data, frequency="D"):
    # Calculate daily returns based on the 'Close' price
    daily_returns = data['Close'].pct_change()
    return daily_returns

# Function to generate plot
def generate_plot(price_data=None, cumulative_returns_data=None, mode="price", period="1y", prediction=None):
    plt.figure(figsize=(10, 6))

    # Plot based on the mode (either "price" or "returns")
    if mode == "returns" and cumulative_returns_data is not None:
        # Ensure `cumulative_returns_data` has a DatetimeIndex; convert if necessary
        if not isinstance(cumulative_returns_data.index, pd.DatetimeIndex):
            raise ValueError("The `cumulative_returns_data` DataFrame must have a DatetimeIndex.")

        # Convert cumulative_returns_data index to timezone-naive if necessary
        if isinstance(cumulative_returns_data.index, pd.DatetimeIndex) and cumulative_returns_data.index.tz is not None:
            cumulative_returns_data.index = cumulative_returns_data.index.tz_localize(None)

        # Plot cumulative returns
        plt.plot(cumulative_returns_data.index, cumulative_returns_data['Cumulative Returns'] * 100, label="Cumulative Returns (%)", color="blue")
        
        # Determine short-term and long-term moving average window sizes based on the selected period
        if period == "1d":
            short_window = 12  # 12 intervals (approximately 1 hour)
            long_window = 48  # 48 intervals (approximately 4 hours)
        elif period in ["1mo", "3mo"]:
            short_window = 5  # ~1 week
            long_window = 15  # ~3 weeks
        elif period in ["6mo", "1y"]:
            short_window = 30  # 1 month
            long_window = 90  # 3 months
        elif period in ["3y", "5y"]:
            short_window = 90  # 3 months
            long_window = 180  # 6 months
        else:
            short_window = 180  # 6 months
            long_window = 365  # 1 year

        # Calculate moving averages based on the cumulative returns data
        short_term_mavg = cumulative_returns_data['Cumulative Returns'].rolling(window=short_window).mean() * 100
        long_term_mavg = cumulative_returns_data['Cumulative Returns'].rolling(window=long_window).mean() * 100

        # Plot the moving averages
        plt.plot(short_term_mavg.index, short_term_mavg, color="orange", label=f"{short_window}-{'Intervals (~1 hour)' if period == '1d' else 'Day'} Moving Avg (Short Term)")
        plt.plot(long_term_mavg.index, long_term_mavg, color="red", label=f"{long_window}-{'Intervals (~4 hours)' if period == '1d' else 'Day'} Moving Avg (Long Term)")

        # Calculate mean and standard deviation based on cumulative returns
        cumulative_mean = cumulative_returns_data['Cumulative Returns'].mean() * 100
        cumulative_std = cumulative_returns_data['Cumulative Returns'].std() * 100

        # Mean and standard deviation reference lines based on cumulative returns
        plt.axhline(cumulative_mean, color="green", linestyle="--", label="Mean Cumulative Return")
        plt.axhline(cumulative_mean + cumulative_std, color="gray", linestyle="--", label="+1 Std Dev")
        plt.axhline(cumulative_mean - cumulative_std, color="gray", linestyle="--", label="-1 Std Dev")

        plt.ylabel("Cumulative Returns (%)")

    elif mode == "price" and price_data is not None:
        # Ensure `price_data` has a DatetimeIndex; convert if necessary
        if not isinstance(price_data.index, pd.DatetimeIndex):
            raise ValueError("The `price_data` DataFrame must have a DatetimeIndex.")

        # Convert to timezone-naive DatetimeIndex if necessary
        if isinstance(price_data.index, pd.DatetimeIndex):
            price_data.index = price_data.index.tz_localize(None)

        # Plotting price data directly if mode is "price"
        plt.plot(price_data.index, price_data['Close'], label="Closing Price", color="blue")
        plt.ylabel("Price ($)")

    # Plot the predicted price if available
    if prediction is not None and mode == "tomorrow":
        # Plot historical data
        plt.plot(price_data.index, price_data['Close'], label="Historical Price", color="blue")
        # Add predicted price for tomorrow
        tomorrow_date = price_data.index[-1] + timedelta(days=1)
        plt.plot([price_data.index[-1], tomorrow_date], [price_data['Close'].iloc[-1], prediction], linestyle="--", color="purple", label=f"Predicted Price for Tomorrow: ${prediction:.2f}")

    plt.xlabel("Date" if period != "1d" else "Time")
    plt.legend()
    plt.grid()

    # Adjust x-axis labels based on the period selected
    if period == "1d":
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 1)))  # Label every hour
        plt.xticks(rotation=45)
    elif period in ["1mo", "3mo"]:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))  # Label every week
    elif period in ["6mo", "1y"]:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Label every month
    elif period in ["3y", "5y"]:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # Label every 6 months
    else:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())

    plt.gcf().autofmt_xdate()

    # Save the plot as a base64-encoded image
    img = BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    plt.clf()
    plt.close()
    
    return plot_url

# Function to predict tomorrow's price
def predict_tomorrow_price(data, ticker):
    model_filename = f"{ticker}_random_forest_model.pkl"
    
    try:
        # Try loading the model and MSE
        model, mse = joblib.load(model_filename)
    except ValueError:
        # If the model does not contain MSE, retrain it
        print(f"Old model format detected for {ticker}. Retraining the model...")
        train_model(data, ticker)
        model, mse = joblib.load(model_filename)  # Reload the newly saved model with MSE
    
    # Prepare features for the last row
    X, _ = prepare_features(data)
    last_row = X.iloc[[-1]]  # Select the last row with feature names
    
    # Predict next day's price
    predicted_price = model.predict(last_row)[0]
    return predicted_price, mse

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        ticker = request.form.get("ticker")
        period = request.form.get("period")
        mode = request.form.get("mode")
        
        # Check if the model exists for the ticker
        model_filename = f"{ticker}_random_forest_model.pkl"
        if not os.path.exists(model_filename):
            print(f"No model found for {ticker}. Training a new model.")
            data = get_stock_data(ticker, period="5y")  # Use a longer period for training
            train_model(data, ticker)

        # Fetch stock data
        data = get_stock_data(ticker, period)
        
        # Ensure that data has a datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Expected `data` to have a DatetimeIndex.")

        # Convert data index to timezone-naive if necessary
        if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        # Calculate start date based on the user-requested period
        if period == "1mo":
            start_date = pd.Timestamp.now() - pd.DateOffset(months=1)
        elif period == "3mo":
            start_date = pd.Timestamp.now() - pd.DateOffset(months=3)
        elif period == "6mo":
            start_date = pd.Timestamp.now() - pd.DateOffset(months=6)
        elif period == "1y":
            start_date = pd.Timestamp.now() - pd.DateOffset(years=1)
        elif period == "3y":
            start_date = pd.Timestamp.now() - pd.DateOffset(years=3)
        elif period == "5y":
            start_date = pd.Timestamp.now() - pd.DateOffset(years=5)
        else:
            start_date = data.index.min()  # Default to the earliest available date

        # Convert start_date to timezone-naive if necessary
        start_date = start_date.tz_localize(None) if getattr(start_date, 'tz', None) else start_date

        # Filter data to the requested timeframe
        filtered_data = data[data.index >= start_date]

        # Determine whether to show prices, returns, or predict tomorrow's price
        if mode == "price":
            plot_url = generate_plot(price_data=filtered_data, mode="price", period=period)
        elif mode == "returns":
            # Calculate daily returns
            returns = data['Close'].pct_change()
            returns = returns.dropna()  # Drop NaN values resulting from pct_change()

            # Calculate cumulative returns
            cumulative_returns = (1 + returns).cumprod() - 1
            
            # Create a DataFrame for cumulative returns without changing index lengths manually
            cumulative_returns_df = pd.DataFrame({'Cumulative Returns': cumulative_returns})

            # Generate the plot
            plot_url = generate_plot(cumulative_returns_data=cumulative_returns_df, mode="returns", period=period)
        elif mode == "tomorrow":
            predicted_price, mse = predict_tomorrow_price(data, ticker)
            plot_url = generate_plot(price_data=data, mode="tomorrow", period=period, prediction=predicted_price)
            
            return jsonify({
                "plot_url": plot_url,
                "predicted_price": predicted_price,
                "mse": mse  # Include the MSE in the response
            })
        
        return jsonify({"plot_url": plot_url})
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)