import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from portfolio_optimizer import optimize_portfolio, calculate_portfolio_returns, calculate_portfolio_volatility
from stock_data import calculate_returns


def backtest_portfolio(stocks_data, strategy='equal_weight', rebalance_frequency='quarterly', initial_investment=10000):
    """
    Backtest a portfolio strategy
    
    Args:
        stocks_data: Dictionary of {ticker: DataFrame} with historical data
        strategy: 'equal_weight', 'optimized', or 'buy_and_hold'
        rebalance_frequency: 'monthly', 'quarterly', 'annually', or 'never'
        initial_investment: Starting portfolio value
    
    Returns:
        Dictionary with backtest results
    """
    if not stocks_data or len(stocks_data) < 1:
        return None
    
    tickers = list(stocks_data.keys())
    
    all_dates = sorted(set().union(*[set(data.index) for data in stocks_data.values()]))
    
    if len(all_dates) < 30:
        return None
    
    prices_df = pd.DataFrame({ticker: data['Close'] for ticker, data in stocks_data.items()})
    prices_df = prices_df.fillna(method='ffill').fillna(method='bfill')
    
    returns_df = prices_df.pct_change().dropna()
    
    if strategy == 'equal_weight':
        weights = {ticker: 1.0 / len(tickers) for ticker in tickers}
    elif strategy == 'buy_and_hold':
        weights = {ticker: 1.0 / len(tickers) for ticker in tickers}
        rebalance_frequency = 'never'
    else:
        initial_data = {ticker: data[:len(data)//2] for ticker, data in stocks_data.items()}
        optimized = optimize_portfolio(initial_data)
        if optimized:
            weights = optimized['weights']
        else:
            weights = {ticker: 1.0 / len(tickers) for ticker in tickers}
    
    rebalance_days = get_rebalance_dates(all_dates, rebalance_frequency)
    
    portfolio_values = [initial_investment]
    portfolio_weights_history = [weights.copy()]
    dates = [all_dates[0]]
    
    current_value = initial_investment
    shares = {}
    
    for ticker in tickers:
        shares[ticker] = (current_value * weights.get(ticker, 0)) / prices_df[ticker].iloc[0]
    
    for i in range(1, len(prices_df)):
        current_date = prices_df.index[i]
        
        current_value = sum(shares[ticker] * prices_df[ticker].iloc[i] for ticker in tickers)
        
        if current_date in rebalance_days and i < len(prices_df) - 1:
            if strategy == 'optimized':
                lookback_data = {ticker: stocks_data[ticker][:i] for ticker in tickers}
                optimized = optimize_portfolio(lookback_data)
                if optimized:
                    weights = optimized['weights']
            
            for ticker in tickers:
                shares[ticker] = (current_value * weights.get(ticker, 0)) / prices_df[ticker].iloc[i]
            
            portfolio_weights_history.append(weights.copy())
        
        portfolio_values.append(current_value)
        dates.append(current_date)
    
    portfolio_series = pd.Series(portfolio_values, index=dates)
    portfolio_returns = portfolio_series.pct_change().dropna()
    
    total_return = ((portfolio_values[-1] - initial_investment) / initial_investment) * 100
    
    annual_return = (portfolio_returns.mean() * 252) * 100
    annual_volatility = (portfolio_returns.std() * np.sqrt(252)) * 100
    sharpe_ratio = (annual_return - 2) / annual_volatility if annual_volatility != 0 else 0
    
    cumulative_returns = (1 + portfolio_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = abs(drawdown.min()) * 100
    
    winning_days = len(portfolio_returns[portfolio_returns > 0])
    total_days = len(portfolio_returns)
    win_rate = (winning_days / total_days) * 100 if total_days > 0 else 0
    
    return {
        'strategy': strategy,
        'rebalance_frequency': rebalance_frequency,
        'initial_investment': initial_investment,
        'final_value': portfolio_values[-1],
        'total_return_pct': total_return,
        'annual_return_pct': annual_return,
        'annual_volatility_pct': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_pct': max_drawdown,
        'win_rate_pct': win_rate,
        'num_rebalances': len(rebalance_days),
        'portfolio_values': portfolio_values,
        'dates': dates,
        'final_weights': weights
    }


def get_rebalance_dates(dates, frequency):
    """
    Get rebalancing dates based on frequency
    """
    if frequency == 'never':
        return []
    
    rebalance_dates = []
    dates_list = pd.to_datetime(dates)
    
    if frequency == 'monthly':
        period = 30
    elif frequency == 'quarterly':
        period = 90
    elif frequency == 'annually':
        period = 365
    else:
        return []
    
    last_rebalance = dates_list[0]
    
    for date in dates_list:
        if (date - last_rebalance).days >= period:
            rebalance_dates.append(date)
            last_rebalance = date
    
    return rebalance_dates


def compare_strategies(stocks_data, initial_investment=10000):
    """
    Compare different portfolio strategies
    """
    strategies = [
        {'name': 'Equal Weight', 'strategy': 'equal_weight', 'rebalance': 'quarterly'},
        {'name': 'Buy and Hold', 'strategy': 'buy_and_hold', 'rebalance': 'never'},
        {'name': 'Optimized (Quarterly)', 'strategy': 'optimized', 'rebalance': 'quarterly'},
        {'name': 'Optimized (Annual)', 'strategy': 'optimized', 'rebalance': 'annually'}
    ]
    
    results = {}
    
    for strat in strategies:
        backtest = backtest_portfolio(
            stocks_data,
            strategy=strat['strategy'],
            rebalance_frequency=strat['rebalance'],
            initial_investment=initial_investment
        )
        
        if backtest:
            results[strat['name']] = backtest
    
    return results


def calculate_benchmark_comparison(portfolio_returns, benchmark_returns):
    """
    Compare portfolio performance against a benchmark
    """
    if len(portfolio_returns) != len(benchmark_returns):
        min_len = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns[-min_len:]
        benchmark_returns = benchmark_returns[-min_len:]
    
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    benchmark_cumulative = (1 + benchmark_returns).cumprod()
    
    portfolio_total_return = (portfolio_cumulative.iloc[-1] - 1) * 100
    benchmark_total_return = (benchmark_cumulative.iloc[-1] - 1) * 100
    
    alpha = portfolio_total_return - benchmark_total_return
    
    tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(252) * 100
    
    information_ratio = alpha / tracking_error if tracking_error != 0 else 0
    
    return {
        'portfolio_return': portfolio_total_return,
        'benchmark_return': benchmark_total_return,
        'alpha': alpha,
        'tracking_error': tracking_error,
        'information_ratio': information_ratio,
        'outperformance': alpha > 0
    }


def monte_carlo_simulation(stocks_data, weights, days=252, simulations=1000, initial_investment=10000):
    """
    Monte Carlo simulation for portfolio forecasting
    """
    tickers = list(stocks_data.keys())
    
    returns_data = {}
    for ticker, data in stocks_data.items():
        returns = calculate_returns(data)
        if returns is not None and not returns.empty:
            returns_data[ticker] = returns
    
    returns_df = pd.DataFrame(returns_data).dropna()
    
    if returns_df.empty:
        return None
    
    mean_returns = returns_df.mean().values
    cov_matrix = returns_df.cov().values
    
    weights_array = np.array([weights.get(ticker, 0) for ticker in returns_df.columns])
    
    simulation_results = np.zeros((simulations, days))
    
    for i in range(simulations):
        portfolio_values = [initial_investment]
        
        for day in range(days):
            random_returns = np.random.multivariate_normal(mean_returns, cov_matrix)
            
            portfolio_return = np.dot(weights_array, random_returns)
            
            new_value = portfolio_values[-1] * (1 + portfolio_return)
            portfolio_values.append(new_value)
        
        simulation_results[i, :] = portfolio_values[1:]
    
    percentiles = {
        '5th': np.percentile(simulation_results, 5, axis=0),
        '50th': np.percentile(simulation_results, 50, axis=0),
        '95th': np.percentile(simulation_results, 95, axis=0)
    }
    
    expected_value = simulation_results[:, -1].mean()
    worst_case = simulation_results[:, -1].min()
    best_case = simulation_results[:, -1].max()
    
    return {
        'simulations': simulations,
        'days': days,
        'percentiles': percentiles,
        'expected_final_value': expected_value,
        'worst_case': worst_case,
        'best_case': best_case,
        'probability_profit': (simulation_results[:, -1] > initial_investment).sum() / simulations * 100
    }
