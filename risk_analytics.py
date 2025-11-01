import numpy as np
import pandas as pd
from scipy import stats
from stock_data import calculate_returns


def calculate_var(returns, confidence_level=0.95, method='historical'):
    """
    Calculate Value at Risk (VaR)
    
    Args:
        returns: Series or array of returns
        confidence_level: Confidence level (default 95%)
        method: 'historical', 'parametric', or 'monte_carlo'
    
    Returns:
        VaR value (positive number representing potential loss)
    """
    if returns is None or len(returns) == 0:
        return None
    
    if method == 'historical':
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return abs(var)
    
    elif method == 'parametric':
        mean = np.mean(returns)
        std = np.std(returns)
        var = stats.norm.ppf(1 - confidence_level, mean, std)
        return abs(var)
    
    elif method == 'monte_carlo':
        mean = np.mean(returns)
        std = np.std(returns)
        simulations = np.random.normal(mean, std, 10000)
        var = np.percentile(simulations, (1 - confidence_level) * 100)
        return abs(var)
    
    return None


def calculate_cvar(returns, confidence_level=0.95):
    """
    Calculate Conditional Value at Risk (CVaR / Expected Shortfall)
    
    CVaR is the expected loss given that the loss exceeds VaR
    """
    if returns is None or len(returns) == 0:
        return None
    
    var = calculate_var(returns, confidence_level, method='historical')
    
    threshold = np.percentile(returns, (1 - confidence_level) * 100)
    cvar = returns[returns <= threshold].mean()
    
    return abs(cvar)


def calculate_portfolio_var(weights, returns_df, confidence_level=0.95, investment=10000):
    """
    Calculate portfolio VaR
    
    Args:
        weights: Dictionary of {ticker: weight}
        returns_df: DataFrame with returns for each ticker
        confidence_level: Confidence level
        investment: Portfolio value
    
    Returns:
        Dictionary with VaR metrics
    """
    if returns_df.empty or not weights:
        return None
    
    weights_array = np.array([weights.get(ticker, 0) for ticker in returns_df.columns])
    
    portfolio_returns = (returns_df * weights_array).sum(axis=1)
    
    var_1d = calculate_var(portfolio_returns, confidence_level, method='historical')
    cvar_1d = calculate_cvar(portfolio_returns, confidence_level)
    
    var_annual = var_1d * np.sqrt(252)
    cvar_annual = cvar_1d * np.sqrt(252)
    
    var_dollar_1d = var_1d * investment
    cvar_dollar_1d = cvar_1d * investment
    
    return {
        'var_1d_pct': var_1d * 100,
        'cvar_1d_pct': cvar_1d * 100,
        'var_annual_pct': var_annual * 100,
        'cvar_annual_pct': cvar_annual * 100,
        'var_dollar_1d': var_dollar_1d,
        'cvar_dollar_1d': cvar_dollar_1d,
        'confidence_level': confidence_level * 100,
        'portfolio_returns_mean': portfolio_returns.mean() * 100,
        'portfolio_returns_std': portfolio_returns.std() * 100
    }


def calculate_drawdown(stock_data):
    """
    Calculate maximum drawdown
    """
    prices = stock_data['Close']
    
    running_max = prices.expanding().max()
    drawdown = (prices - running_max) / running_max
    
    max_drawdown = drawdown.min()
    
    max_dd_date = drawdown.idxmin()
    
    return {
        'max_drawdown': abs(max_drawdown) * 100,
        'max_drawdown_date': max_dd_date,
        'current_drawdown': abs(drawdown.iloc[-1]) * 100,
        'drawdown_series': drawdown
    }


def calculate_beta(stock_returns, market_returns):
    """
    Calculate beta (systematic risk measure)
    """
    if len(stock_returns) != len(market_returns):
        min_len = min(len(stock_returns), len(market_returns))
        stock_returns = stock_returns[-min_len:]
        market_returns = market_returns[-min_len:]
    
    covariance = np.cov(stock_returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    
    if market_variance == 0:
        return None
    
    beta = covariance / market_variance
    return beta


def risk_adjusted_metrics(returns, risk_free_rate=0.02):
    """
    Calculate comprehensive risk-adjusted performance metrics
    """
    if returns is None or len(returns) == 0:
        return None
    
    annual_return = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
    
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino_ratio = (annual_return - risk_free_rate) / downside_std if downside_std != 0 and len(downside_returns) > 0 else 0
    
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = abs(drawdown.min())
    
    calmar_ratio = annual_return / max_drawdown if max_drawdown != 0 else 0
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown * 100,
        'annual_return': annual_return * 100,
        'annual_volatility': annual_volatility * 100,
        'downside_deviation': downside_std * 100
    }


def stress_test_portfolio(weights, returns_df, scenarios):
    """
    Perform stress testing on portfolio
    
    Args:
        weights: Portfolio weights
        returns_df: Historical returns
        scenarios: List of stress scenarios (e.g., market crash percentages)
    
    Returns:
        Dictionary with stress test results
    """
    weights_array = np.array([weights.get(ticker, 0) for ticker in returns_df.columns])
    portfolio_returns = (returns_df * weights_array).sum(axis=1)
    
    results = {}
    
    for scenario in scenarios:
        scenario_name = scenario['name']
        shock = scenario['shock']
        
        stressed_return = portfolio_returns.mean() + shock
        
        results[scenario_name] = {
            'shock': shock * 100,
            'expected_return': stressed_return * 100,
            'probability': scenario.get('probability', 'N/A')
        }
    
    return results


def portfolio_risk_decomposition(weights, returns_df):
    """
    Decompose portfolio risk by individual holdings
    """
    if returns_df.empty or not weights:
        return None
    
    weights_array = np.array([weights.get(ticker, 0) for ticker in returns_df.columns])
    
    cov_matrix = returns_df.cov() * 252
    
    portfolio_variance = np.dot(weights_array.T, np.dot(cov_matrix, weights_array))
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    marginal_contrib = np.dot(cov_matrix, weights_array) / portfolio_volatility
    
    risk_contrib = weights_array * marginal_contrib
    
    risk_decomp = {}
    for i, ticker in enumerate(returns_df.columns):
        risk_decomp[ticker] = {
            'weight': weights.get(ticker, 0) * 100,
            'risk_contribution': risk_contrib[i] * 100,
            'risk_contribution_pct': (risk_contrib[i] / portfolio_volatility) * 100 if portfolio_volatility != 0 else 0
        }
    
    return risk_decomp
