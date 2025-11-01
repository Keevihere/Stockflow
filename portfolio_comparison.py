import pandas as pd
import numpy as np
from portfolio_optimizer import optimize_portfolio
from stock_data import calculate_returns
from risk_analytics import risk_adjusted_metrics, calculate_portfolio_var
from backtesting import backtest_portfolio


def create_strategy_portfolio(stocks_data, strategy_type, **kwargs):
    """
    Create a portfolio based on strategy type
    
    Strategy types:
    - 'equal_weight': Equal allocation to all stocks
    - 'market_cap_weight': Weight by market capitalization
    - 'minimum_variance': Minimize portfolio variance
    - 'maximum_sharpe': Maximize Sharpe ratio (default MPT optimization)
    - 'risk_parity': Equal risk contribution from each asset
    """
    tickers = list(stocks_data.keys())
    
    if strategy_type == 'equal_weight':
        weights = {ticker: 1.0 / len(tickers) for ticker in tickers}
        return weights
    
    elif strategy_type == 'market_cap_weight':
        market_caps = kwargs.get('market_caps', {})
        total_cap = sum(market_caps.values())
        if total_cap > 0:
            weights = {ticker: market_caps.get(ticker, 0) / total_cap for ticker in tickers}
        else:
            weights = {ticker: 1.0 / len(tickers) for ticker in tickers}
        return weights
    
    elif strategy_type == 'minimum_variance':
        optimized = optimize_portfolio(stocks_data, target_return=None)
        if optimized:
            return optimized['weights']
        return {ticker: 1.0 / len(tickers) for ticker in tickers}
    
    elif strategy_type == 'maximum_sharpe':
        optimized = optimize_portfolio(stocks_data)
        if optimized:
            return optimized['weights']
        return {ticker: 1.0 / len(tickers) for ticker in tickers}
    
    elif strategy_type == 'risk_parity':
        weights = calculate_risk_parity_weights(stocks_data)
        return weights
    
    return {ticker: 1.0 / len(tickers) for ticker in tickers}


def calculate_risk_parity_weights(stocks_data, max_iterations=100):
    """
    Calculate risk parity portfolio weights (equal risk contribution)
    """
    returns_data = {}
    for ticker, data in stocks_data.items():
        returns = calculate_returns(data)
        if returns is not None and not returns.empty:
            returns_data[ticker] = returns
    
    if len(returns_data) < 2:
        tickers = list(stocks_data.keys())
        return {ticker: 1.0 / len(tickers) for ticker in tickers}
    
    returns_df = pd.DataFrame(returns_data).dropna()
    
    if returns_df.empty:
        tickers = list(stocks_data.keys())
        return {ticker: 1.0 / len(tickers) for ticker in tickers}
    
    cov_matrix = returns_df.cov().values
    n_assets = len(returns_df.columns)
    
    weights = np.array([1.0 / n_assets] * n_assets)
    
    for _ in range(max_iterations):
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
        risk_contrib = weights * marginal_contrib
        
        target_risk = portfolio_vol / n_assets
        
        weights = weights * target_risk / risk_contrib
        weights = weights / weights.sum()
    
    return {ticker: float(weight) for ticker, weight in zip(returns_df.columns, weights)}


def compare_portfolios(stocks_data, strategies, initial_investment=10000):
    """
    Compare multiple portfolio strategies
    
    Args:
        stocks_data: Stock data dictionary
        strategies: List of strategy dictionaries with 'name' and 'type'
        initial_investment: Starting capital
    
    Returns:
        Comprehensive comparison DataFrame
    """
    results = []
    
    for strategy in strategies:
        strategy_name = strategy['name']
        strategy_type = strategy['type']
        strategy_kwargs = strategy.get('kwargs', {})
        
        weights = create_strategy_portfolio(stocks_data, strategy_type, **strategy_kwargs)
        
        returns_data = {}
        for ticker, data in stocks_data.items():
            returns = calculate_returns(data)
            if returns is not None and not returns.empty:
                returns_data[ticker] = returns
        
        if returns_data:
            returns_df = pd.DataFrame(returns_data).dropna()
            weights_array = np.array([weights.get(ticker, 0) for ticker in returns_df.columns])
            portfolio_returns = (returns_df * weights_array).sum(axis=1)
            
            metrics = risk_adjusted_metrics(portfolio_returns)
            
            var_metrics = calculate_portfolio_var(
                weights,
                returns_df,
                confidence_level=0.95,
                investment=initial_investment
            )
            
            backtest = backtest_portfolio(
                stocks_data,
                strategy=strategy_type if strategy_type != 'maximum_sharpe' else 'optimized',
                rebalance_frequency='quarterly',
                initial_investment=initial_investment
            )
            
            result = {
                'Strategy': strategy_name,
                'Annual Return (%)': metrics['annual_return'] if metrics else 0,
                'Annual Volatility (%)': metrics['annual_volatility'] if metrics else 0,
                'Sharpe Ratio': metrics['sharpe_ratio'] if metrics else 0,
                'Sortino Ratio': metrics['sortino_ratio'] if metrics else 0,
                'Max Drawdown (%)': metrics['max_drawdown'] if metrics else 0,
                'VaR 95% (1-day %)': var_metrics['var_1d_pct'] if var_metrics else 0,
                'CVaR 95% (1-day %)': var_metrics['cvar_1d_pct'] if var_metrics else 0,
                'Backtest Total Return (%)': backtest['total_return_pct'] if backtest else 0,
                'Backtest Final Value': backtest['final_value'] if backtest else initial_investment
            }
            
            results.append(result)
    
    return pd.DataFrame(results)


def rank_strategies(comparison_df):
    """
    Rank strategies based on multiple criteria
    """
    if comparison_df.empty:
        return None
    
    ranking_criteria = {
        'Annual Return (%)': 'high',
        'Sharpe Ratio': 'high',
        'Sortino Ratio': 'high',
        'Max Drawdown (%)': 'low',
        'Annual Volatility (%)': 'low'
    }
    
    ranks = pd.DataFrame(index=comparison_df.index)
    
    for criterion, direction in ranking_criteria.items():
        if criterion in comparison_df.columns:
            if direction == 'high':
                ranks[criterion] = comparison_df[criterion].rank(ascending=False)
            else:
                ranks[criterion] = comparison_df[criterion].rank(ascending=True)
    
    ranks['Average Rank'] = ranks.mean(axis=1)
    
    comparison_df['Overall Rank'] = ranks['Average Rank'].rank()
    
    return comparison_df.sort_values('Overall Rank')


def generate_strategy_recommendation(comparison_df, risk_tolerance='moderate'):
    """
    Recommend best strategy based on risk tolerance
    
    Args:
        comparison_df: Comparison DataFrame
        risk_tolerance: 'conservative', 'moderate', or 'aggressive'
    """
    if comparison_df.empty:
        return None
    
    if risk_tolerance == 'conservative':
        best_idx = comparison_df['Max Drawdown (%)'].idxmin()
        criteria = 'lowest drawdown'
    elif risk_tolerance == 'aggressive':
        best_idx = comparison_df['Annual Return (%)'].idxmax()
        criteria = 'highest return'
    else:
        best_idx = comparison_df['Sharpe Ratio'].idxmax()
        criteria = 'best risk-adjusted return (Sharpe ratio)'
    
    recommended = comparison_df.loc[best_idx]
    
    return {
        'recommended_strategy': recommended['Strategy'],
        'selection_criteria': criteria,
        'expected_annual_return': recommended['Annual Return (%)'],
        'expected_volatility': recommended['Annual Volatility (%)'],
        'sharpe_ratio': recommended['Sharpe Ratio'],
        'max_drawdown': recommended['Max Drawdown (%)']
    }


def portfolio_similarity_analysis(weights1, weights2):
    """
    Calculate similarity between two portfolios
    """
    all_tickers = set(weights1.keys()) | set(weights2.keys())
    
    vec1 = np.array([weights1.get(ticker, 0) for ticker in all_tickers])
    vec2 = np.array([weights2.get(ticker, 0) for ticker in all_tickers])
    
    cosine_similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    euclidean_distance = np.linalg.norm(vec1 - vec2)
    
    overlap = sum(min(weights1.get(ticker, 0), weights2.get(ticker, 0)) for ticker in all_tickers)
    
    return {
        'cosine_similarity': cosine_similarity,
        'euclidean_distance': euclidean_distance,
        'overlap_coefficient': overlap,
        'similarity_pct': cosine_similarity * 100
    }
