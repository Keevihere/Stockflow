import pandas as pd
import numpy as np
from portfolio_optimizer import optimize_portfolio, calculate_portfolio_returns
from stock_data import calculate_returns, get_current_price
from risk_analytics import calculate_portfolio_var


def analyze_portfolio_drift(current_weights, target_weights, threshold=0.05):
    """
    Analyze how much portfolio has drifted from target allocation
    
    Args:
        current_weights: Current portfolio weights
        target_weights: Target portfolio weights
        threshold: Drift threshold to trigger rebalancing (default 5%)
    
    Returns:
        Dictionary with drift analysis
    """
    drifts = {}
    total_drift = 0
    needs_rebalancing = False
    
    all_tickers = set(current_weights.keys()) | set(target_weights.keys())
    
    for ticker in all_tickers:
        current = current_weights.get(ticker, 0)
        target = target_weights.get(ticker, 0)
        drift = abs(current - target)
        
        drifts[ticker] = {
            'current_weight': current * 100,
            'target_weight': target * 100,
            'drift': drift * 100,
            'needs_adjustment': drift > threshold
        }
        
        total_drift += drift
        
        if drift > threshold:
            needs_rebalancing = True
    
    return {
        'drifts': drifts,
        'total_drift': total_drift * 100,
        'needs_rebalancing': needs_rebalancing,
        'threshold': threshold * 100
    }


def calculate_current_weights(holdings, current_prices):
    """
    Calculate current portfolio weights based on holdings and prices
    
    Args:
        holdings: Dictionary of {ticker: shares}
        current_prices: Dictionary of {ticker: current_price}
    
    Returns:
        Dictionary of current weights
    """
    total_value = sum(holdings.get(ticker, 0) * current_prices.get(ticker, 0) 
                     for ticker in holdings.keys())
    
    if total_value == 0:
        return {}
    
    weights = {}
    for ticker in holdings.keys():
        value = holdings[ticker] * current_prices.get(ticker, 0)
        weights[ticker] = value / total_value
    
    return weights


def generate_rebalancing_trades(current_weights, target_weights, portfolio_value, current_prices):
    """
    Generate specific buy/sell orders to rebalance portfolio
    
    Args:
        current_weights: Current allocation
        target_weights: Desired allocation
        portfolio_value: Total portfolio value
        current_prices: Current stock prices
    
    Returns:
        List of trades to execute
    """
    trades = []
    
    all_tickers = set(current_weights.keys()) | set(target_weights.keys())
    
    for ticker in all_tickers:
        current_weight = current_weights.get(ticker, 0)
        target_weight = target_weights.get(ticker, 0)
        
        current_value = current_weight * portfolio_value
        target_value = target_weight * portfolio_value
        
        diff_value = target_value - current_value
        
        if abs(diff_value) > 10:
            price = current_prices.get(ticker, 0)
            if price > 0:
                shares = diff_value / price
                
                trades.append({
                    'ticker': ticker,
                    'action': 'BUY' if shares > 0 else 'SELL',
                    'shares': abs(shares),
                    'price': price,
                    'value': abs(diff_value),
                    'current_weight_pct': current_weight * 100,
                    'target_weight_pct': target_weight * 100
                })
    
    trades.sort(key=lambda x: x['value'], reverse=True)
    
    return trades


def tax_aware_rebalancing(current_weights, target_weights, holdings_age, long_term_threshold=365):
    """
    Generate tax-aware rebalancing recommendations
    
    Args:
        current_weights: Current weights
        target_weights: Target weights
        holdings_age: Dictionary of {ticker: days_held}
        long_term_threshold: Days for long-term capital gains (default 365)
    
    Returns:
        Tax-optimized rebalancing plan
    """
    recommendations = []
    
    all_tickers = set(current_weights.keys()) | set(target_weights.keys())
    
    for ticker in all_tickers:
        current = current_weights.get(ticker, 0)
        target = target_weights.get(ticker, 0)
        age = holdings_age.get(ticker, 0)
        
        diff = target - current
        
        if abs(diff) > 0.01:
            tax_status = 'long_term' if age >= long_term_threshold else 'short_term'
            
            if diff < 0:
                priority = 'high' if tax_status == 'long_term' else 'low'
            else:
                priority = 'medium'
            
            recommendations.append({
                'ticker': ticker,
                'action': 'reduce' if diff < 0 else 'increase',
                'amount_pct': abs(diff) * 100,
                'tax_status': tax_status,
                'priority': priority,
                'days_held': age
            })
    
    recommendations.sort(key=lambda x: (x['priority'] == 'high', x['amount_pct']), reverse=True)
    
    return recommendations


def smart_rebalancing_triggers(stocks_data, current_weights, target_weights):
    """
    Determine if rebalancing should be triggered based on multiple criteria
    
    Returns:
        Dictionary with trigger analysis
    """
    triggers = {
        'drift_trigger': False,
        'volatility_trigger': False,
        'correlation_trigger': False,
        'recommended': False,
        'reasons': []
    }
    
    drift_analysis = analyze_portfolio_drift(current_weights, target_weights, threshold=0.05)
    if drift_analysis['needs_rebalancing']:
        triggers['drift_trigger'] = True
        triggers['reasons'].append(f"Portfolio drift of {drift_analysis['total_drift']:.1f}% exceeds 5% threshold")
    
    returns_data = {}
    for ticker, data in stocks_data.items():
        returns = calculate_returns(data)
        if returns is not None and not returns.empty:
            returns_data[ticker] = returns
    
    if returns_data:
        returns_df = pd.DataFrame(returns_data).dropna()
        
        for ticker in returns_df.columns:
            recent_volatility = returns_df[ticker].tail(30).std()
            historical_volatility = returns_df[ticker].std()
            
            if recent_volatility > historical_volatility * 1.5:
                triggers['volatility_trigger'] = True
                triggers['reasons'].append(f"{ticker} showing elevated volatility")
                break
    
    if triggers['drift_trigger'] or triggers['volatility_trigger']:
        triggers['recommended'] = True
    
    return triggers


def periodic_rebalancing_schedule(start_date, end_date, frequency='quarterly'):
    """
    Generate rebalancing schedule
    
    Args:
        start_date: Start date
        end_date: End date
        frequency: 'monthly', 'quarterly', 'semiannual', 'annual'
    
    Returns:
        List of rebalancing dates
    """
    freq_map = {
        'monthly': 30,
        'quarterly': 90,
        'semiannual': 180,
        'annual': 365
    }
    
    period_days = freq_map.get(frequency, 90)
    
    dates = []
    current = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    while current <= end:
        dates.append(current)
        current += pd.Timedelta(days=period_days)
    
    return dates


def calculate_rebalancing_cost(trades, transaction_fee_pct=0.001):
    """
    Calculate estimated cost of rebalancing
    
    Args:
        trades: List of trade dictionaries
        transaction_fee_pct: Transaction fee as percentage (default 0.1%)
    
    Returns:
        Dictionary with cost breakdown
    """
    total_traded_value = sum(trade['value'] for trade in trades)
    transaction_costs = total_traded_value * transaction_fee_pct
    
    num_trades = len(trades)
    
    return {
        'total_traded_value': total_traded_value,
        'transaction_costs': transaction_costs,
        'num_trades': num_trades,
        'cost_percentage': (transaction_costs / total_traded_value * 100) if total_traded_value > 0 else 0
    }


def optimize_and_compare(stocks_data, current_weights, current_portfolio_value):
    """
    Optimize portfolio and compare with current allocation
    
    Returns:
        Comparison and rebalancing recommendation
    """
    optimized = optimize_portfolio(stocks_data)
    
    if not optimized:
        return None
    
    target_weights = optimized['weights']
    
    drift_analysis = analyze_portfolio_drift(current_weights, target_weights)
    
    improvement = {
        'current_sharpe': 0,
        'optimized_sharpe': optimized['sharpe_ratio'],
        'current_return': 0,
        'optimized_return': optimized['expected_return'] * 100,
        'current_volatility': 0,
        'optimized_volatility': optimized['volatility'] * 100
    }
    
    returns_data = {}
    for ticker, data in stocks_data.items():
        returns = calculate_returns(data)
        if returns is not None and not returns.empty:
            returns_data[ticker] = returns
    
    if returns_data:
        returns_df = pd.DataFrame(returns_data).dropna()
        weights_array = np.array([current_weights.get(ticker, 0) for ticker in returns_df.columns])
        current_portfolio_returns = (returns_df * weights_array).sum(axis=1)
        
        improvement['current_return'] = current_portfolio_returns.mean() * 252 * 100
        improvement['current_volatility'] = current_portfolio_returns.std() * np.sqrt(252) * 100
        improvement['current_sharpe'] = (improvement['current_return'] - 2) / improvement['current_volatility'] if improvement['current_volatility'] != 0 else 0
    
    return {
        'target_weights': target_weights,
        'drift_analysis': drift_analysis,
        'improvement_metrics': improvement,
        'recommendation': 'REBALANCE' if drift_analysis['needs_rebalancing'] else 'HOLD'
    }
