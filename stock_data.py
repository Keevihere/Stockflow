import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def format_indian_ticker(ticker):
    """
    Format ticker for Indian markets (NSE/BSE)
    """
    ticker = ticker.upper().strip()
    if not ticker.endswith('.NS') and not ticker.endswith('.BO'):
        return f"{ticker}.NS"
    return ticker


def format_us_ticker(ticker):
    """
    Format ticker for US markets
    """
    return ticker.upper().strip()


def get_stock_data(ticker, market='US', period='1y'):
    """
    Fetch stock data from yfinance
    
    Args:
        ticker: Stock ticker symbol
        market: 'US' or 'India'
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
    
    Returns:
        DataFrame with stock data
    """
    try:
        if market.upper() == 'INDIA':
            formatted_ticker = format_indian_ticker(ticker)
        else:
            formatted_ticker = format_us_ticker(ticker)
        
        stock = yf.Ticker(formatted_ticker)
        data = stock.history(period=period)
        
        if data.empty:
            return None
        
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


def get_current_price(ticker, market='US'):
    """
    Get current stock price
    """
    try:
        if market.upper() == 'INDIA':
            formatted_ticker = format_indian_ticker(ticker)
        else:
            formatted_ticker = format_us_ticker(ticker)
        
        stock = yf.Ticker(formatted_ticker)
        info = stock.info
        
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        if current_price is None:
            data = stock.history(period='1d')
            if not data.empty:
                current_price = data['Close'].iloc[-1]
        
        return current_price
    except Exception as e:
        print(f"Error fetching current price for {ticker}: {e}")
        return None


def get_stock_info(ticker, market='US'):
    """
    Get detailed stock information
    """
    try:
        if market.upper() == 'INDIA':
            formatted_ticker = format_indian_ticker(ticker)
        else:
            formatted_ticker = format_us_ticker(ticker)
        
        stock = yf.Ticker(formatted_ticker)
        info = stock.info
        
        return {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'marketCap': info.get('marketCap', 0),
            'currency': info.get('currency', 'USD' if market == 'US' else 'INR'),
            'exchange': info.get('exchange', 'N/A')
        }
    except Exception as e:
        print(f"Error fetching info for {ticker}: {e}")
        return {
            'name': ticker,
            'sector': 'N/A',
            'industry': 'N/A',
            'marketCap': 0,
            'currency': 'USD' if market == 'US' else 'INR',
            'exchange': 'N/A'
        }


def calculate_returns(data):
    """
    Calculate daily returns from price data
    """
    if data is None or data.empty:
        return None
    
    returns = data['Close'].pct_change().dropna()
    return returns


def calculate_log_returns(data):
    """
    Calculate log returns from price data
    """
    if data is None or data.empty:
        return None
    
    log_returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
    return log_returns


def calculate_metrics(data):
    """
    Calculate key metrics for stock data
    """
    if data is None or data.empty:
        return None
    
    returns = calculate_returns(data)
    
    if returns is None or returns.empty:
        return None
    
    metrics = {
        'mean_return': returns.mean(),
        'volatility': returns.std(),
        'annualized_return': returns.mean() * 252,
        'annualized_volatility': returns.std() * np.sqrt(252),
        'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0,
        'max_price': data['Close'].max(),
        'min_price': data['Close'].min(),
        'current_price': data['Close'].iloc[-1],
        'price_change_pct': ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
    }
    
    return metrics


def get_multiple_stocks_data(tickers, market='US', period='1y'):
    """
    Fetch data for multiple stocks
    
    Returns:
        Dictionary with ticker as key and data as value
    """
    stocks_data = {}
    
    for ticker in tickers:
        data = get_stock_data(ticker, market, period)
        if data is not None:
            stocks_data[ticker] = data
    
    return stocks_data


def get_market_indices(market='US'):
    """
    Get major market indices
    """
    if market.upper() == 'INDIA':
        indices = {
            'NIFTY 50': '^NSEI',
            'SENSEX': '^BSESN',
            'NIFTY Bank': '^NSEBANK'
        }
    else:
        indices = {
            'S&P 500': '^GSPC',
            'Dow Jones': '^DJI',
            'NASDAQ': '^IXIC'
        }
    
    indices_data = {}
    for name, ticker in indices.items():
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period='5d')
            if not data.empty:
                current = data['Close'].iloc[-1]
                previous = data['Close'].iloc[0]
                change = ((current - previous) / previous) * 100
                indices_data[name] = {
                    'value': current,
                    'change': change
                }
        except:
            continue
    
    return indices_data
