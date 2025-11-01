import numpy as np
import pandas as pd
from scipy.optimize import minimize
from stock_data import calculate_returns


def calculate_portfolio_returns(weights, returns):
    """
    Calculate expected portfolio return
    """
    return np.sum(returns.mean() * weights) * 252


def calculate_portfolio_volatility(weights, returns):
    """
    Calculate portfolio volatility (standard deviation)
    """
    cov_matrix = returns.cov() * 252
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return np.sqrt(portfolio_variance)


def calculate_sharpe_ratio(weights, returns, risk_free_rate=0.02):
    """
    Calculate Sharpe ratio for portfolio
    """
    portfolio_return = calculate_portfolio_returns(weights, returns)
    portfolio_volatility = calculate_portfolio_volatility(weights, returns)
    
    if portfolio_volatility == 0:
        return 0
    
    sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility
    return sharpe


def negative_sharpe(weights, returns, risk_free_rate=0.02):
    """
    Negative Sharpe ratio for minimization
    """
    return -calculate_sharpe_ratio(weights, returns, risk_free_rate)


def optimize_portfolio(stocks_data, risk_free_rate=0.02, target_return=None):
    """
    Optimize portfolio using Modern Portfolio Theory
    
    Args:
        stocks_data: Dictionary of {ticker: DataFrame} with stock data
        risk_free_rate: Risk-free rate (default 2%)
        target_return: Target return for optimization (if None, maximize Sharpe ratio)
    
    Returns:
        Dictionary with optimized weights and metrics
    """
    if not stocks_data or len(stocks_data) < 2:
        return None
    
    tickers = list(stocks_data.keys())
    
    returns_data = {}
    for ticker, data in stocks_data.items():
        returns = calculate_returns(data)
        if returns is not None and not returns.empty:
            returns_data[ticker] = returns
    
    if len(returns_data) < 2:
        return None
    
    returns_df = pd.DataFrame(returns_data)
    returns_df = returns_df.dropna()
    
    if returns_df.empty:
        return None
    
    num_assets = len(returns_df.columns)
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.array([1/num_assets] * num_assets)
    
    if target_return is not None:
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: calculate_portfolio_returns(x, returns_df) - target_return}
        ]
        
        def portfolio_volatility_objective(weights):
            return calculate_portfolio_volatility(weights, returns_df)
        
        result = minimize(
            portfolio_volatility_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
    else:
        result = minimize(
            negative_sharpe,
            initial_weights,
            args=(returns_df, risk_free_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
    
    if not result.success:
        return None
    
    optimal_weights = result.x
    
    portfolio_return = calculate_portfolio_returns(optimal_weights, returns_df)
    portfolio_volatility = calculate_portfolio_volatility(optimal_weights, returns_df)
    sharpe = calculate_sharpe_ratio(optimal_weights, returns_df, risk_free_rate)
    
    weights_dict = {ticker: float(weight) for ticker, weight in zip(returns_df.columns, optimal_weights)}
    
    return {
        'weights': weights_dict,
        'expected_return': portfolio_return,
        'volatility': portfolio_volatility,
        'sharpe_ratio': sharpe,
        'tickers': list(returns_df.columns)
    }


def calculate_efficient_frontier(stocks_data, num_portfolios=50, risk_free_rate=0.02):
    """
    Calculate efficient frontier
    
    Returns:
        List of portfolios with different risk-return profiles
    """
    if not stocks_data or len(stocks_data) < 2:
        return None
    
    returns_data = {}
    for ticker, data in stocks_data.items():
        returns = calculate_returns(data)
        if returns is not None and not returns.empty:
            returns_data[ticker] = returns
    
    if len(returns_data) < 2:
        return None
    
    returns_df = pd.DataFrame(returns_data)
    returns_df = returns_df.dropna()
    
    if returns_df.empty:
        return None
    
    min_return = returns_df.mean().min() * 252
    max_return = returns_df.mean().max() * 252
    
    target_returns = np.linspace(min_return, max_return, num_portfolios)
    
    efficient_portfolios = []
    
    for target_return in target_returns:
        result = optimize_portfolio(stocks_data, risk_free_rate, target_return)
        if result is not None:
            efficient_portfolios.append({
                'return': result['expected_return'],
                'volatility': result['volatility'],
                'sharpe': result['sharpe_ratio']
            })
    
    return efficient_portfolios


def calculate_portfolio_value(weights, current_prices, initial_investment=10000):
    """
    Calculate current portfolio value given weights and prices
    """
    portfolio_allocation = {}
    total_value = 0
    
    for ticker, weight in weights.items():
        allocation = initial_investment * weight
        shares = allocation / current_prices.get(ticker, 1)
        current_value = shares * current_prices.get(ticker, 1)
        
        portfolio_allocation[ticker] = {
            'weight': weight,
            'allocation': allocation,
            'shares': shares,
            'current_value': current_value,
            'current_price': current_prices.get(ticker, 0)
        }
        
        total_value += current_value
    
    return {
        'total_value': total_value,
        'initial_investment': initial_investment,
        'profit_loss': total_value - initial_investment,
        'profit_loss_pct': ((total_value - initial_investment) / initial_investment) * 100,
        'allocations': portfolio_allocation
    }


def equal_weight_portfolio(tickers):
    """
    Create equal-weighted portfolio
    """
    num_assets = len(tickers)
    weight = 1.0 / num_assets
    return {ticker: weight for ticker in tickers}


def calculate_correlation_matrix(stocks_data):
    """
    Calculate correlation matrix for stocks
    """
    returns_data = {}
    for ticker, data in stocks_data.items():
        returns = calculate_returns(data)
        if returns is not None and not returns.empty:
            returns_data[ticker] = returns
    
    if len(returns_data) < 2:
        return None
    
    returns_df = pd.DataFrame(returns_data)
    returns_df = returns_df.dropna()
    
    if returns_df.empty:
        return None
    
    return returns_df.corr()
