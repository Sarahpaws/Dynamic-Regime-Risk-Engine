import yfinance as yf
import pandas as pd

def load_data(ticker="SPY", period="1y"):
    df = yf.download(ticker, period=period)

    # keep only Close price (important for risk models)
    df = df[['Close']]

    # compute daily returns
    df['returns'] = df['Close'].pct_change()

    # remove first NaN row
    df = df.dropna()

    return df


if __name__ == "__main__":
    data = load_data("SPY", "1y")
    print(data.head())