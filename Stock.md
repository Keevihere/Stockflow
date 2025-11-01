# Finance AI Agent

## Overview

Finance AI Agent is a Streamlit-based financial analysis application that combines real-time stock market data with AI-powered predictions and portfolio optimization capabilities. The application provides market insights, technical analysis, AI-driven price predictions using GPT-5, and Modern Portfolio Theory-based portfolio optimization. It supports both US and Indian stock markets (NSE/BSE), offering a comprehensive toolkit for individual investors and traders.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
**Technology:** Streamlit web framework  
**Rationale:** Streamlit enables rapid development of data-driven applications with minimal frontend code, allowing focus on financial logic and AI integration. It provides native support for interactive visualizations and real-time data updates.

**Multi-tab Interface:**
- Market Overview: Dashboard for major market indices (US & India)
- Stock Analysis & AI Predictions: Individual stock analysis with GPT-5 AI predictions
- Portfolio Optimizer: Modern Portfolio Theory-based optimization
- Advanced Analytics: Correlation matrices and efficient frontier calculations
- Advanced Predictions: ARIMA and Prophet time series forecasting with ensemble modeling
- Risk Analysis: VaR, CVaR, and comprehensive risk decomposition
- Backtesting & Comparison: Strategy backtesting, comparison tools, and PDF reports

**Visualization Library:** Plotly (graph_objects and express)  
**Rationale:** Plotly provides interactive, publication-quality charts essential for financial data visualization, supporting candlestick charts, time series, and complex portfolio analytics.

### Backend Architecture

**Data Layer:**
- **Library:** yfinance (Yahoo Finance API wrapper)
- **Purpose:** Real-time and historical stock data retrieval
- **Market Support:** Dual-market capability for US stocks and Indian markets (NSE/BSE)
- **Data Processing:** pandas and numpy for time-series analysis and numerical computations

**AI/ML Layer:**
- **Provider:** OpenAI GPT-5 API for sentiment analysis and portfolio recommendations
- **Advanced Models:** ARIMA and Prophet for time series forecasting
- **Ensemble Approach:** Combines multiple models for robust predictions
- **Response Format:** Structured JSON responses for programmatic consumption
- **Capabilities:**
  - GPT-5: Sentiment analysis, portfolio insights, market trend analysis
  - ARIMA: Statistical time series forecasting with auto-parameter tuning
  - Prophet: Facebook's forecasting model with seasonality detection
  - Ensemble: Average of multiple models for improved accuracy
  - Short-term (7-day) and medium-term (30-day) price predictions
  - Confidence scoring and key factor identification

**Portfolio Optimization & Risk Engine:**
- **Framework:** Modern Portfolio Theory (MPT)
- **Optimization Method:** scipy.optimize for constrained optimization
- **Risk Analytics:** VaR, CVaR, drawdown analysis, risk decomposition
- **Backtesting:** Historical strategy validation with multiple rebalancing frequencies
- **Strategy Comparison:** Equal weight, buy-and-hold, optimized portfolios
- **Metrics Calculated:**
  - Expected returns (annualized)
  - Portfolio volatility and Sharpe ratio
  - Value at Risk (VaR) at 95% confidence
  - Conditional VaR (Expected Shortfall)
  - Maximum drawdown and Calmar ratio
  - Sortino ratio (downside deviation)
  - Risk contribution by holding
  - Correlation matrices and efficient frontier

**Key Architectural Decisions:**

1. **Modular Design Pattern**
   - **Decision:** Separate concerns into distinct modules (stock_data, ai_helper, portfolio_optimizer)
   - **Rationale:** Enables independent testing, easier maintenance, and clear separation between data fetching, AI analysis, and optimization logic
   - **Pros:** Better code organization, reusability, testability
   - **Cons:** Requires careful interface design between modules

2. **Dual Market Support**
   - **Decision:** Built-in ticker formatting for both US and Indian markets
   - **Rationale:** Expand addressable user base to include international investors
   - **Implementation:** Automatic suffix addition (.NS for NSE, .BO for BSE)

3. **AI-First Analysis Approach**
   - **Decision:** Integrate GPT-5 for predictive analysis rather than traditional statistical models
   - **Rationale:** Leverage advanced language models' pattern recognition on financial data
   - **Trade-off:** API dependency and cost vs. more sophisticated analysis than basic technical indicators
   - **Alternatives Considered:** Traditional ARIMA, LSTM neural networks
   - **Chosen Approach:** GPT-5 for faster implementation and natural language insights

4. **Real-Time Data Strategy**
   - **Decision:** Direct yfinance API calls without local caching
   - **Rationale:** Ensure users always see current market data
   - **Cons:** Higher latency, API rate limit exposure
   - **Future Consideration:** May need caching layer for production scale

## External Dependencies

### Financial Data Services
- **yfinance:** Primary data source for real-time and historical stock prices, market indices, and company information
- **Yahoo Finance API:** Underlying data provider (accessed via yfinance)
- **Supported Markets:** NYSE, NASDAQ, NSE (India), BSE (India)

### AI/ML Services
- **OpenAI API (GPT-5):** Financial analysis, price predictions, and market insights
- **API Key:** Required via `OPENAI_API_KEY` environment variable
- **Model:** GPT-5 (released August 7, 2025) - configured for structured JSON responses

### Python Libraries
**Data Processing:**
- pandas: Time series manipulation and financial data handling
- numpy: Numerical computations for returns, volatility calculations
- scipy: Optimization algorithms and statistical functions

**Machine Learning & Forecasting:**
- statsmodels: ARIMA time series models
- prophet: Facebook's forecasting library with seasonality
- scikit-learn: Additional ML utilities

**Visualization & Reports:**
- plotly: Interactive financial charts (candlestick, line charts, heatmaps)
- streamlit: Web application framework
- fpdf2: PDF report generation

**Financial Data:**
- yfinance: Yahoo Finance data wrapper

### Configuration Requirements
- **Environment Variables:**
  - `OPENAI_API_KEY`: Required for AI predictions and analysis features
  
- **Market Data Access:** Internet connection required for real-time data fetching from Yahoo Finance

### Data Flow
1. User inputs ticker symbols via Streamlit interface
2. stock_data module fetches historical data from yfinance
3. Financial metrics calculated locally (returns, volatility, Sharpe ratio)
4. ai_helper module sends processed data to OpenAI GPT-5 for predictions
5. portfolio_optimizer performs MPT calculations using scipy
6. Results visualized through Plotly and displayed in Streamlit