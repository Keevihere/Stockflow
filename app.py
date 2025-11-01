import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime

from stock_data import (
    get_stock_data,
    get_current_price,
    get_stock_info,
    calculate_metrics,
    get_market_indices,
    get_multiple_stocks_data,
    calculate_returns
)
from ai_helper import (
    predict_stock_price,
    analyze_portfolio,
    get_market_insights
)
from portfolio_optimizer import (
    optimize_portfolio,
    calculate_portfolio_value,
    equal_weight_portfolio,
    calculate_correlation_matrix,
    calculate_efficient_frontier
)
from advanced_predictions import (
    ensemble_prediction,
    prophet_forecast,
    arima_forecast
)
from risk_analytics import (
    calculate_portfolio_var,
    calculate_drawdown,
    risk_adjusted_metrics,
    portfolio_risk_decomposition
)
from backtesting import (
    backtest_portfolio,
    compare_strategies,
    monte_carlo_simulation
)
from rebalancing import (
    analyze_portfolio_drift,
    optimize_and_compare,
    generate_rebalancing_trades,
    calculate_current_weights
)
from portfolio_comparison import (
    compare_portfolios,
    rank_strategies,
    generate_strategy_recommendation
)
from pdf_reports import (
    create_portfolio_report,
    create_comparison_report,
    create_stock_analysis_report
)

st.set_page_config(
    page_title="Finance AI Agent",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Finance AI Agent")
st.markdown("### Real-Time Stock Analysis, AI Predictions & Portfolio Optimization")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🌍 Market Overview",
    "📊 Stock Analysis & AI Predictions",
    "💼 Portfolio Optimizer",
    "📈 Advanced Analytics",
    "🔮 Advanced Predictions",
    "⚖️ Risk Analysis",
    "🔄 Backtesting & Comparison"
])

with tab1:
    st.header("Market Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🇺🇸 US Markets")
        us_indices = get_market_indices('US')
        
        if us_indices:
            for name, data in us_indices.items():
                delta_color = "normal" if data['change'] >= 0 else "inverse"
                st.metric(
                    label=name,
                    value=f"${data['value']:,.2f}",
                    delta=f"{data['change']:.2f}%",
                    delta_color=delta_color
                )
        else:
            st.info("Loading US market data...")
    
    with col2:
        st.subheader("🇮🇳 Indian Markets")
        indian_indices = get_market_indices('India')
        
        if indian_indices:
            for name, data in indian_indices.items():
                delta_color = "normal" if data['change'] >= 0 else "inverse"
                st.metric(
                    label=name,
                    value=f"₹{data['value']:,.2f}",
                    delta=f"{data['change']:.2f}%",
                    delta_color=delta_color
                )
        else:
            st.info("Loading Indian market data...")

with tab2:
    st.header("Stock Analysis & AI Predictions")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ticker = st.text_input(
            "Enter Stock Ticker",
            placeholder="e.g., AAPL, RELIANCE, TCS",
            help="For Indian stocks: RELIANCE, TCS, INFY, etc. For US stocks: AAPL, GOOGL, MSFT, etc."
        )
    
    with col2:
        market = st.selectbox(
            "Market",
            ["US", "India"],
            help="Select the market for the stock"
        )
    
    period = st.selectbox(
        "Time Period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )
    
    if ticker:
        with st.spinner(f"Fetching data for {ticker}..."):
            stock_data = get_stock_data(ticker, market, period)
            
            if stock_data is not None and not stock_data.empty:
                current_price = get_current_price(ticker, market)
                stock_info = get_stock_info(ticker, market)
                metrics = calculate_metrics(stock_data)
                
                st.subheader(f"{stock_info['name']} ({ticker})")
                
                col1, col2, col3, col4 = st.columns(4)
                
                currency_symbol = "₹" if market == "India" else "$"
                
                with col1:
                    if current_price:
                        st.metric(
                            "Current Price",
                            f"{currency_symbol}{current_price:,.2f}"
                        )
                
                with col2:
                    if metrics:
                        st.metric(
                            "Price Change",
                            f"{metrics['price_change_pct']:.2f}%",
                            delta=f"{metrics['price_change_pct']:.2f}%"
                        )
                
                with col3:
                    if metrics:
                        st.metric(
                            "Volatility (Annual)",
                            f"{metrics['annualized_volatility']*100:.2f}%"
                        )
                
                with col4:
                    if metrics:
                        st.metric(
                            "Sharpe Ratio",
                            f"{metrics['sharpe_ratio']:.2f}"
                        )
                
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=stock_data.index,
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close'],
                    name=ticker
                ))
                
                fig.update_layout(
                    title=f"{ticker} Price History",
                    xaxis_title="Date",
                    yaxis_title=f"Price ({currency_symbol})",
                    height=500,
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                volume_fig = go.Figure()
                volume_fig.add_trace(go.Bar(
                    x=stock_data.index,
                    y=stock_data['Volume'],
                    name='Volume'
                ))
                
                volume_fig.update_layout(
                    title=f"{ticker} Trading Volume",
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    height=300
                )
                
                st.plotly_chart(volume_fig, use_container_width=True)
                
                st.divider()
                st.subheader("🤖 AI-Powered Predictions")
                
                if st.button("Generate AI Prediction", type="primary"):
                    with st.spinner("AI is analyzing the stock..."):
                        prediction = predict_stock_price(ticker, stock_data, current_price)
                        
                        if 'error' not in prediction or prediction.get('confidence', 0) > 0:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("#### Price Predictions")
                                st.metric(
                                    "7-Day Target",
                                    f"{currency_symbol}{prediction['price_target_7d']:.2f}",
                                    delta=f"{((prediction['price_target_7d'] - current_price) / current_price * 100):.2f}%"
                                )
                                st.metric(
                                    "30-Day Target",
                                    f"{currency_symbol}{prediction['price_target_30d']:.2f}",
                                    delta=f"{((prediction['price_target_30d'] - current_price) / current_price * 100):.2f}%"
                                )
                                
                                sentiment_emoji = {
                                    'bullish': '🟢',
                                    'bearish': '🔴',
                                    'neutral': '🟡'
                                }
                                
                                st.markdown(f"**Sentiment:** {sentiment_emoji.get(prediction['short_term_prediction'], '⚪')} {prediction['short_term_prediction'].upper()}")
                                st.markdown(f"**Confidence:** {prediction['confidence']*100:.1f}%")
                            
                            with col2:
                                st.markdown("#### Analysis")
                                st.write(prediction['analysis'])
                                
                                if prediction.get('key_factors'):
                                    st.markdown("**Key Factors:**")
                                    for factor in prediction['key_factors']:
                                        st.write(f"• {factor}")
                        else:
                            st.error(f"Error generating prediction: {prediction.get('analysis', 'Unknown error')}")
                
                with st.expander("📊 Detailed Metrics"):
                    if metrics and stock_info:
                        metric_data = {
                            "Metric": [
                                "Sector",
                                "Industry",
                                "Exchange",
                                "Max Price",
                                "Min Price",
                                "Annual Return",
                                "Annual Volatility",
                                "Sharpe Ratio"
                            ],
                            "Value": [
                                stock_info['sector'],
                                stock_info['industry'],
                                stock_info['exchange'],
                                f"{currency_symbol}{metrics['max_price']:.2f}",
                                f"{currency_symbol}{metrics['min_price']:.2f}",
                                f"{metrics['annualized_return']*100:.2f}%",
                                f"{metrics['annualized_volatility']*100:.2f}%",
                                f"{metrics['sharpe_ratio']:.2f}"
                            ]
                        }
                        st.table(pd.DataFrame(metric_data))
            else:
                st.error(f"Could not fetch data for {ticker}. Please check the ticker symbol and market selection.")

with tab3:
    st.header("Portfolio Optimizer")
    st.markdown("Build and optimize your investment portfolio using Modern Portfolio Theory")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        portfolio_market = st.selectbox(
            "Portfolio Market",
            ["US", "India", "Mixed"],
            key="portfolio_market"
        )
    
    with col2:
        initial_investment = st.number_input(
            "Initial Investment",
            min_value=1000,
            max_value=10000000,
            value=10000,
            step=1000
        )
    
    if portfolio_market == "Mixed":
        st.warning("⚠️ Mixed portfolios may have limited functionality due to non-overlapping trading days between US and Indian markets. For best results, use single-market portfolios.")
    
    tickers_input = st.text_area(
        "Enter Stock Tickers (one per line or comma-separated)",
        placeholder="AAPL\nGOOGL\nMSFT\n\nor\n\nAAPL, GOOGL, MSFT",
        help="For Indian stocks, use symbols like RELIANCE, TCS, INFY. For US stocks, use AAPL, GOOGL, MSFT."
    )
    
    if tickers_input:
        tickers = [t.strip().upper() for t in tickers_input.replace(',', '\n').split('\n') if t.strip()]
        
        if len(tickers) >= 2:
            if st.button("Optimize Portfolio", type="primary"):
                with st.spinner("Fetching stock data and optimizing portfolio..."):
                    if portfolio_market == "Mixed":
                        stocks_data = {}
                        for ticker_entry in tickers:
                            if ':' in ticker_entry:
                                ticker, mkt = ticker_entry.split(':')
                                data = get_stock_data(ticker.strip(), mkt.strip(), '1y')
                            else:
                                data = get_stock_data(ticker_entry, 'US', '1y')
                            
                            if data is not None:
                                stocks_data[ticker_entry] = data
                    else:
                        stocks_data = get_multiple_stocks_data(tickers, portfolio_market, '1y')
                    
                    if len(stocks_data) >= 2:
                        optimized = optimize_portfolio(stocks_data)
                        
                        if optimized:
                            st.success("✅ Portfolio optimized successfully!")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Expected Annual Return",
                                    f"{optimized['expected_return']*100:.2f}%"
                                )
                            
                            with col2:
                                st.metric(
                                    "Annual Volatility",
                                    f"{optimized['volatility']*100:.2f}%"
                                )
                            
                            with col3:
                                st.metric(
                                    "Sharpe Ratio",
                                    f"{optimized['sharpe_ratio']:.2f}"
                                )
                            
                            st.subheader("Optimal Allocation")
                            
                            weights_df = pd.DataFrame(
                                list(optimized['weights'].items()),
                                columns=['Ticker', 'Weight']
                            )
                            weights_df['Weight'] = weights_df['Weight'] * 100
                            weights_df = weights_df.sort_values('Weight', ascending=False)
                            
                            fig = px.pie(
                                weights_df,
                                values='Weight',
                                names='Ticker',
                                title='Portfolio Allocation'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.dataframe(
                                weights_df.style.format({'Weight': '{:.2f}%'}),
                                use_container_width=True
                            )
                            
                            current_prices = {}
                            for ticker in optimized['weights'].keys():
                                if portfolio_market == "Mixed":
                                    if ':' in ticker:
                                        ticker_symbol, mkt = ticker.split(':')
                                        price = get_current_price(ticker_symbol.strip(), mkt.strip())
                                    else:
                                        price = get_current_price(ticker, 'US')
                                else:
                                    price = get_current_price(ticker, portfolio_market)
                                if price:
                                    current_prices[ticker] = price
                            
                            if current_prices:
                                portfolio_value = calculate_portfolio_value(
                                    optimized['weights'],
                                    current_prices,
                                    initial_investment
                                )
                                
                                st.subheader("Investment Breakdown")
                                
                                allocation_data = []
                                currency_symbol = "₹" if portfolio_market == "India" else "$"
                                
                                for ticker, alloc in portfolio_value['allocations'].items():
                                    allocation_data.append({
                                        'Ticker': ticker,
                                        'Weight': f"{alloc['weight']*100:.2f}%",
                                        'Investment': f"{currency_symbol}{alloc['allocation']:.2f}",
                                        'Shares': f"{alloc['shares']:.4f}",
                                        'Current Price': f"{currency_symbol}{alloc['current_price']:.2f}",
                                        'Current Value': f"{currency_symbol}{alloc['current_value']:.2f}"
                                    })
                                
                                st.dataframe(
                                    pd.DataFrame(allocation_data),
                                    use_container_width=True
                                )
                            
                            correlation = calculate_correlation_matrix(stocks_data)
                            if correlation is not None:
                                st.subheader("Correlation Matrix")
                                fig = px.imshow(
                                    correlation,
                                    labels=dict(color="Correlation"),
                                    x=correlation.columns,
                                    y=correlation.columns,
                                    color_continuous_scale='RdBu_r',
                                    aspect="auto"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            st.divider()
                            st.subheader("🤖 AI Portfolio Analysis")
                            
                            if st.button("Get AI Insights", key="portfolio_ai"):
                                with st.spinner("AI is analyzing your portfolio..."):
                                    portfolio_data = {
                                        'weights': optimized['weights'],
                                        'expected_return': f"{optimized['expected_return']*100:.2f}%",
                                        'volatility': f"{optimized['volatility']*100:.2f}%",
                                        'sharpe_ratio': f"{optimized['sharpe_ratio']:.2f}"
                                    }
                                    
                                    market_conditions = {
                                        'market': portfolio_market,
                                        'tickers': list(optimized['weights'].keys())
                                    }
                                    
                                    analysis = analyze_portfolio(portfolio_data, market_conditions)
                                    
                                    if 'error' not in analysis:
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.markdown(f"**Overall Health:** {analysis['overall_health'].upper()}")
                                            st.markdown(f"**Risk Assessment:** {analysis['risk_assessment'].upper()}")
                                            st.markdown(f"**Diversification Score:** {analysis['diversification_score']}/100")
                                        
                                        with col2:
                                            st.markdown("**Summary:**")
                                            st.write(analysis['summary'])
                                        
                                        if analysis.get('strengths'):
                                            st.markdown("**💪 Strengths:**")
                                            for strength in analysis['strengths']:
                                                st.write(f"• {strength}")
                                        
                                        if analysis.get('weaknesses'):
                                            st.markdown("**⚠️ Weaknesses:**")
                                            for weakness in analysis['weaknesses']:
                                                st.write(f"• {weakness}")
                                        
                                        if analysis.get('recommendations'):
                                            st.markdown("**💡 Recommendations:**")
                                            for rec in analysis['recommendations']:
                                                st.write(f"• {rec}")
                                    else:
                                        st.error(f"Error: {analysis.get('summary', 'Unknown error')}")
                        else:
                            st.error("Could not optimize portfolio. Please ensure all stocks have sufficient historical data.")
                    else:
                        st.warning(f"Only {len(stocks_data)} stocks have valid data. Need at least 2 stocks for portfolio optimization.")
        else:
            st.info("Please enter at least 2 stock tickers to optimize a portfolio.")

with tab4:
    st.header("Advanced Analytics")
    
    st.subheader("📉 Efficient Frontier")
    st.markdown("Visualize the risk-return tradeoff for different portfolio combinations")
    
    ef_tickers_input = st.text_area(
        "Enter Stock Tickers for Efficient Frontier",
        placeholder="AAPL, GOOGL, MSFT, AMZN",
        key="ef_tickers"
    )
    
    ef_market = st.selectbox(
        "Market",
        ["US", "India"],
        key="ef_market"
    )
    
    if ef_tickers_input:
        ef_tickers = [t.strip().upper() for t in ef_tickers_input.replace(',', '\n').split('\n') if t.strip()]
        
        if len(ef_tickers) >= 2:
            if st.button("Calculate Efficient Frontier"):
                with st.spinner("Calculating efficient frontier..."):
                    stocks_data = get_multiple_stocks_data(ef_tickers, ef_market, '1y')
                    
                    if len(stocks_data) >= 2:
                        efficient_portfolios = calculate_efficient_frontier(stocks_data, num_portfolios=50)
                        
                        if efficient_portfolios:
                            ef_df = pd.DataFrame(efficient_portfolios)
                            
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=ef_df['volatility'] * 100,
                                y=ef_df['return'] * 100,
                                mode='markers',
                                marker=dict(
                                    size=10,
                                    color=ef_df['sharpe'],
                                    colorscale='Viridis',
                                    showscale=True,
                                    colorbar=dict(title="Sharpe Ratio")
                                ),
                                text=[f"Return: {r*100:.2f}%<br>Volatility: {v*100:.2f}%<br>Sharpe: {s:.2f}" 
                                      for r, v, s in zip(ef_df['return'], ef_df['volatility'], ef_df['sharpe'])],
                                hovertemplate='%{text}<extra></extra>',
                                name='Efficient Frontier'
                            ))
                            
                            fig.update_layout(
                                title='Efficient Frontier',
                                xaxis_title='Volatility (Annual %)',
                                yaxis_title='Expected Return (Annual %)',
                                height=600
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.info("Each point represents an optimized portfolio. Color indicates the Sharpe ratio (higher is better).")
                        else:
                            st.error("Could not calculate efficient frontier.")
                    else:
                        st.warning("Need at least 2 stocks with valid data.")

with tab5:
    st.header("🔮 Advanced Predictions (ARIMA & Prophet)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ap_ticker = st.text_input(
            "Enter Stock Ticker for Advanced Prediction",
            placeholder="e.g., AAPL, GOOGL, RELIANCE",
            key="ap_ticker"
        )
    
    with col2:
        ap_market = st.selectbox(
            "Market",
            ["US", "India"],
            key="ap_market"
        )
    
    if ap_ticker:
        if st.button("Generate Advanced Predictions", type="primary"):
            with st.spinner(f"Running ARIMA and Prophet models for {ap_ticker}..."):
                stock_data = get_stock_data(ap_ticker, ap_market, '1y')
                
                if stock_data is not None and len(stock_data) >= 30:
                    ensemble_pred = ensemble_prediction(stock_data, periods=30)
                    
                    if ensemble_pred and 'error' not in ensemble_pred:
                        st.success("✅ Predictions generated successfully!")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Current Price",
                                f"${ensemble_pred['current_price']:.2f}"
                            )
                        
                        with col2:
                            st.metric(
                                "7-Day Prediction (Ensemble)",
                                f"${ensemble_pred['predicted_7d']:.2f}",
                                delta=f"{ensemble_pred['price_change_7d']:.2f}%"
                            )
                        
                        with col3:
                            st.metric(
                                "30-Day Prediction (Ensemble)",
                                f"${ensemble_pred['predicted_30d']:.2f}",
                                delta=f"{ensemble_pred['price_change_30d']:.2f}%"
                            )
                        
                        st.divider()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Model Comparison")
                            comparison_data = {
                                'Model': ['Prophet', 'ARIMA', 'Ensemble'],
                                '7-Day Forecast ($)': [
                                    ensemble_pred.get('prophet_7d', 0),
                                    ensemble_pred.get('arima_7d', 0),
                                    ensemble_pred['predicted_7d']
                                ],
                                '30-Day Forecast ($)': [
                                    ensemble_pred.get('prophet_30d', 0),
                                    ensemble_pred.get('arima_30d', 0),
                                    ensemble_pred['predicted_30d']
                                ]
                            }
                            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
                        
                        with col2:
                            st.subheader("Model Details")
                            st.write(f"**Model Type:** {ensemble_pred['model_type']}")
                            st.write(f"**ARIMA Order:** {ensemble_pred.get('arima_order', 'N/A')}")
                            st.write(f"**Trend:** {ensemble_pred['trend'].upper()}")
                            st.write(f"**30-Day Change:** {ensemble_pred['price_change_30d']:.2f}%")
                    else:
                        st.error("Could not generate predictions. Please try a different stock or time period.")
                else:
                    st.error("Insufficient data for advanced predictions. Need at least 30 days of historical data.")

with tab6:
    st.header("⚖️ Risk Analysis & VaR")
    
    st.subheader("Portfolio Risk Assessment")
    
    risk_tickers_input = st.text_area(
        "Enter Stock Tickers for Risk Analysis",
        placeholder="AAPL, GOOGL, MSFT",
        key="risk_tickers"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_market = st.selectbox(
            "Market",
            ["US", "India"],
            key="risk_market"
        )
    
    with col2:
        risk_investment = st.number_input(
            "Portfolio Value",
            min_value=1000,
            max_value=10000000,
            value=10000,
            step=1000,
            key="risk_investment"
        )
    
    if risk_tickers_input:
        risk_tickers = [t.strip().upper() for t in risk_tickers_input.replace(',', '\n').split('\n') if t.strip()]
        
        if len(risk_tickers) >= 1:
            if st.button("Analyze Portfolio Risk", type="primary"):
                with st.spinner("Calculating risk metrics..."):
                    stocks_data = get_multiple_stocks_data(risk_tickers, risk_market, '1y')
                    
                    if stocks_data:
                        optimized = optimize_portfolio(stocks_data)
                        
                        if optimized:
                            weights = optimized['weights']
                            
                            returns_data = {}
                            for ticker, data in stocks_data.items():
                                returns = calculate_returns(data)
                                if returns is not None and not returns.empty:
                                    returns_data[ticker] = returns
                            
                            if returns_data:
                                returns_df = pd.DataFrame(returns_data).dropna()
                                
                                var_metrics = calculate_portfolio_var(
                                    weights,
                                    returns_df,
                                    confidence_level=0.95,
                                    investment=risk_investment
                                )
                                
                                if var_metrics:
                                    st.success("✅ Risk analysis complete!")
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        st.metric(
                                            "VaR (95%, 1-day)",
                                            f"${var_metrics['var_dollar_1d']:.2f}"
                                        )
                                    
                                    with col2:
                                        st.metric(
                                            "CVaR (95%, 1-day)",
                                            f"${var_metrics['cvar_dollar_1d']:.2f}"
                                        )
                                    
                                    with col3:
                                        st.metric(
                                            "VaR % (1-day)",
                                            f"{var_metrics['var_1d_pct']:.2f}%"
                                        )
                                    
                                    with col4:
                                        st.metric(
                                            "Annual VaR %",
                                            f"{var_metrics['var_annual_pct']:.2f}%"
                                        )
                                    
                                    st.divider()
                                    
                                    st.subheader("Risk Decomposition")
                                    risk_decomp = portfolio_risk_decomposition(weights, returns_df)
                                    
                                    if risk_decomp:
                                        decomp_data = []
                                        for ticker, risk_info in risk_decomp.items():
                                            decomp_data.append({
                                                'Ticker': ticker,
                                                'Weight (%)': risk_info['weight'],
                                                'Risk Contribution (%)': risk_info['risk_contribution'],
                                                'Risk % of Total': risk_info['risk_contribution_pct']
                                            })
                                        
                                        st.dataframe(pd.DataFrame(decomp_data), use_container_width=True)
                                    
                                    st.divider()
                                    
                                    st.subheader("📊 Risk Metrics Summary")
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write("**Value at Risk (VaR)**")
                                        st.info("VaR estimates the maximum potential loss over a given time period at a specific confidence level. A 95% VaR of $500 means there's a 5% chance of losing more than $500 in one day.")
                                    
                                    with col2:
                                        st.write("**Conditional VaR (CVaR)**")
                                        st.info("CVaR (Expected Shortfall) represents the average loss when the loss exceeds VaR. It provides a more conservative risk measure than VaR alone.")

with tab7:
    st.header("🔄 Backtesting & Strategy Comparison")
    
    st.subheader("Backtest Portfolio Strategies")
    
    bt_tickers_input = st.text_area(
        "Enter Stock Tickers for Backtesting",
        placeholder="AAPL, GOOGL, MSFT, AMZN",
        key="bt_tickers"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        bt_market = st.selectbox(
            "Market",
            ["US", "India"],
            key="bt_market"
        )
    
    with col2:
        bt_investment = st.number_input(
            "Initial Investment",
            min_value=1000,
            max_value=10000000,
            value=10000,
            step=1000,
            key="bt_investment"
        )
    
    if bt_tickers_input:
        bt_tickers = [t.strip().upper() for t in bt_tickers_input.replace(',', '\n').split('\n') if t.strip()]
        
        if len(bt_tickers) >= 2:
            if st.button("Run Backtest & Compare Strategies", type="primary"):
                with st.spinner("Running backtests... This may take a moment..."):
                    stocks_data = get_multiple_stocks_data(bt_tickers, bt_market, '2y')
                    
                    if len(stocks_data) >= 2:
                        comparison_results = compare_strategies(stocks_data, initial_investment=bt_investment)
                        
                        if comparison_results:
                            st.success("✅ Backtest complete!")
                            
                            comparison_data = []
                            for strategy_name, results in comparison_results.items():
                                comparison_data.append({
                                    'Strategy': strategy_name,
                                    'Final Value ($)': results['final_value'],
                                    'Total Return (%)': results['total_return_pct'],
                                    'Annual Return (%)': results['annual_return_pct'],
                                    'Annual Volatility (%)': results['annual_volatility_pct'],
                                    'Sharpe Ratio': results['sharpe_ratio'],
                                    'Max Drawdown (%)': results['max_drawdown_pct'],
                                    'Win Rate (%)': results['win_rate_pct'],
                                    'Num Rebalances': results['num_rebalances']
                                })
                            
                            comparison_df = pd.DataFrame(comparison_data)
                            
                            best_return_idx = comparison_df['Total Return (%)'].idxmax()
                            best_sharpe_idx = comparison_df['Sharpe Ratio'].idxmax()
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                best_return_strategy = comparison_df.loc[best_return_idx]
                                st.metric(
                                    "Best Total Return",
                                    f"{best_return_strategy['Strategy']}",
                                    f"{best_return_strategy['Total Return (%)']:.2f}%"
                                )
                            
                            with col2:
                                best_sharpe_strategy = comparison_df.loc[best_sharpe_idx]
                                st.metric(
                                    "Best Sharpe Ratio",
                                    f"{best_sharpe_strategy['Strategy']}",
                                    f"{best_sharpe_strategy['Sharpe Ratio']:.2f}"
                                )
                            
                            with col3:
                                st.metric(
                                    "Strategies Tested",
                                    f"{len(comparison_data)}"
                                )
                            
                            st.divider()
                            
                            st.subheader("Strategy Comparison Table")
                            st.dataframe(comparison_df.style.format({
                                'Final Value ($)': '${:,.2f}',
                                'Total Return (%)': '{:.2f}%',
                                'Annual Return (%)': '{:.2f}%',
                                'Annual Volatility (%)': '{:.2f}%',
                                'Sharpe Ratio': '{:.2f}',
                                'Max Drawdown (%)': '{:.2f}%',
                                'Win Rate (%)': '{:.2f}%'
                            }), use_container_width=True)
                            
                            st.divider()
                            
                            st.subheader("Performance Visualization")
                            
                            fig = go.Figure()
                            
                            for strategy_name, results in comparison_results.items():
                                fig.add_trace(go.Scatter(
                                    x=list(range(len(results['portfolio_values']))),
                                    y=results['portfolio_values'],
                                    mode='lines',
                                    name=strategy_name
                                ))
                            
                            fig.update_layout(
                                title='Portfolio Value Over Time',
                                xaxis_title='Days',
                                yaxis_title='Portfolio Value ($)',
                                height=500,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.divider()
                            
                            st.subheader("📄 Export Report")
                            if st.button("Generate PDF Report"):
                                try:
                                    report_data = {
                                        'strategies': comparison_data,
                                        'recommendation': {
                                            'recommended_strategy': best_sharpe_strategy['Strategy'],
                                            'selection_criteria': 'Best risk-adjusted return (Sharpe ratio)',
                                            'expected_annual_return': best_sharpe_strategy['Annual Return (%)'],
                                            'expected_volatility': best_sharpe_strategy['Annual Volatility (%)'],
                                            'sharpe_ratio': best_sharpe_strategy['Sharpe Ratio'],
                                            'max_drawdown': best_sharpe_strategy['Max Drawdown (%)']
                                        }
                                    }
                                    
                                    pdf_file = create_comparison_report(report_data, 'strategy_comparison.pdf')
                                    
                                    if pdf_file:
                                        with open(pdf_file, 'rb') as f:
                                            st.download_button(
                                                label="📥 Download PDF Report",
                                                data=f,
                                                file_name="strategy_comparison.pdf",
                                                mime="application/pdf"
                                            )
                                        st.success("PDF report generated successfully!")
                                except Exception as e:
                                    st.error(f"Error generating PDF: {e}")
                        else:
                            st.error("Could not complete backtest.")
                    else:
                        st.warning("Need at least 2 stocks with valid data for backtesting.")

st.sidebar.title("About")
st.sidebar.info(
    """
    **Finance AI Agent** combines real-time market data with artificial intelligence 
    to provide comprehensive stock analysis and portfolio optimization.
    
    **Features:**
    - Real-time stock prices (US & India)
    - AI-powered price predictions
    - Portfolio optimization using Modern Portfolio Theory
    - Risk analysis and correlation metrics
    - Efficient frontier calculation
    
    **Markets Supported:**
    - 🇺🇸 US Markets (NYSE, NASDAQ)
    - 🇮🇳 Indian Markets (NSE, BSE)
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
