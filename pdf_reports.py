from fpdf import FPDF
from datetime import datetime
import os


class PortfolioReportPDF(FPDF):
    """Custom PDF class for portfolio reports"""
    
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Finance AI Agent - Portfolio Analysis Report', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 5, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(2)
    
    def chapter_body(self, text):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, text)
        self.ln()


def create_portfolio_report(portfolio_data, filename='portfolio_report.pdf'):
    """
    Create comprehensive PDF report for portfolio analysis
    
    Args:
        portfolio_data: Dictionary with portfolio information
        filename: Output filename
    
    Returns:
        Path to generated PDF file
    """
    pdf = PortfolioReportPDF()
    pdf.add_page()
    
    pdf.chapter_title('Portfolio Summary')
    
    summary_text = f"""
Portfolio Name: {portfolio_data.get('name', 'My Portfolio')}
Total Value: ${portfolio_data.get('total_value', 0):,.2f}
Number of Holdings: {portfolio_data.get('num_holdings', 0)}
Strategy: {portfolio_data.get('strategy', 'N/A')}
"""
    pdf.chapter_body(summary_text.strip())
    
    if 'holdings' in portfolio_data and portfolio_data['holdings']:
        pdf.chapter_title('Portfolio Holdings')
        
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(50, 8, 'Ticker', 1)
        pdf.cell(40, 8, 'Weight (%)', 1)
        pdf.cell(50, 8, 'Value ($)', 1)
        pdf.cell(40, 8, 'Return (%)', 1)
        pdf.ln()
        
        pdf.set_font('Arial', '', 10)
        for holding in portfolio_data['holdings']:
            pdf.cell(50, 7, str(holding.get('ticker', '')), 1)
            pdf.cell(40, 7, f"{holding.get('weight', 0):.2f}", 1)
            pdf.cell(50, 7, f"{holding.get('value', 0):,.2f}", 1)
            pdf.cell(40, 7, f"{holding.get('return', 0):.2f}", 1)
            pdf.ln()
        
        pdf.ln(5)
    
    if 'performance' in portfolio_data:
        pdf.chapter_title('Performance Metrics')
        
        perf = portfolio_data['performance']
        metrics_text = f"""
Annual Return: {perf.get('annual_return', 0):.2f}%
Annual Volatility: {perf.get('annual_volatility', 0):.2f}%
Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}
Sortino Ratio: {perf.get('sortino_ratio', 0):.2f}
Maximum Drawdown: {perf.get('max_drawdown', 0):.2f}%
Calmar Ratio: {perf.get('calmar_ratio', 0):.2f}
"""
        pdf.chapter_body(metrics_text.strip())
    
    if 'risk_metrics' in portfolio_data:
        pdf.chapter_title('Risk Analysis')
        
        risk = portfolio_data['risk_metrics']
        risk_text = f"""
Value at Risk (95%, 1-day): {risk.get('var_1d_pct', 0):.2f}%
Conditional VaR (95%, 1-day): {risk.get('cvar_1d_pct', 0):.2f}%
Value at Risk (95%, Annual): {risk.get('var_annual_pct', 0):.2f}%
Conditional VaR (95%, Annual): {risk.get('cvar_annual_pct', 0):.2f}%

VaR (1-day, Dollar): ${risk.get('var_dollar_1d', 0):,.2f}
CVaR (1-day, Dollar): ${risk.get('cvar_dollar_1d', 0):,.2f}
"""
        pdf.chapter_body(risk_text.strip())
    
    if 'ai_analysis' in portfolio_data:
        pdf.chapter_title('AI-Powered Analysis')
        
        ai = portfolio_data['ai_analysis']
        ai_text = f"""
Overall Health: {ai.get('overall_health', 'N/A').upper()}
Risk Assessment: {ai.get('risk_assessment', 'N/A').upper()}
Diversification Score: {ai.get('diversification_score', 0)}/100

Summary: {ai.get('summary', 'No analysis available')}
"""
        pdf.chapter_body(ai_text.strip())
        
        if ai.get('recommendations'):
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 7, 'Recommendations:', 0, 1)
            pdf.set_font('Arial', '', 10)
            for i, rec in enumerate(ai['recommendations'], 1):
                pdf.multi_cell(0, 5, f"  {i}. {rec}")
            pdf.ln()
    
    if 'predictions' in portfolio_data:
        pdf.chapter_title('Price Predictions')
        
        pred = portfolio_data['predictions']
        pred_text = f"""
Model Type: {pred.get('model_type', 'N/A')}
Current Price: ${pred.get('current_price', 0):.2f}
7-Day Prediction: ${pred.get('predicted_7d', 0):.2f}
30-Day Prediction: ${pred.get('predicted_30d', 0):.2f}
Trend: {pred.get('trend', 'N/A').upper()}
Price Change (30-day): {pred.get('price_change_30d', 0):.2f}%
"""
        pdf.chapter_body(pred_text.strip())
    
    if 'backtest' in portfolio_data:
        pdf.chapter_title('Backtest Results')
        
        bt = portfolio_data['backtest']
        bt_text = f"""
Strategy: {bt.get('strategy', 'N/A')}
Rebalance Frequency: {bt.get('rebalance_frequency', 'N/A')}
Initial Investment: ${bt.get('initial_investment', 0):,.2f}
Final Value: ${bt.get('final_value', 0):,.2f}
Total Return: {bt.get('total_return_pct', 0):.2f}%
Annual Return: {bt.get('annual_return_pct', 0):.2f}%
Annual Volatility: {bt.get('annual_volatility_pct', 0):.2f}%
Sharpe Ratio: {bt.get('sharpe_ratio', 0):.2f}
Maximum Drawdown: {bt.get('max_drawdown_pct', 0):.2f}%
Win Rate: {bt.get('win_rate_pct', 0):.2f}%
Number of Rebalances: {bt.get('num_rebalances', 0)}
"""
        pdf.chapter_body(bt_text.strip())
    
    pdf.add_page()
    pdf.chapter_title('Disclaimer')
    disclaimer_text = """
This report is generated by Finance AI Agent for informational purposes only. 
It does not constitute financial advice, investment recommendations, or an offer 
to buy or sell any securities.

Past performance is not indicative of future results. All investments carry risk, 
including the potential loss of principal. The predictions and analyses provided 
are based on historical data and AI models, which may not accurately reflect 
future market conditions.

Please consult with a qualified financial advisor before making any investment 
decisions. The creators of this software are not responsible for any financial 
losses incurred based on information provided in this report.
"""
    pdf.chapter_body(disclaimer_text.strip())
    
    try:
        pdf.output(filename)
        return filename
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return None


def create_comparison_report(comparison_data, filename='strategy_comparison.pdf'):
    """
    Create PDF report comparing multiple portfolio strategies
    """
    pdf = PortfolioReportPDF()
    pdf.add_page()
    
    pdf.chapter_title('Portfolio Strategy Comparison')
    
    if 'strategies' in comparison_data and comparison_data['strategies']:
        pdf.set_font('Arial', 'B', 9)
        
        headers = ['Strategy', 'Return %', 'Volatility %', 'Sharpe', 'Max DD %']
        col_widths = [50, 30, 30, 25, 30]
        
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 8, header, 1)
        pdf.ln()
        
        pdf.set_font('Arial', '', 9)
        for strategy in comparison_data['strategies']:
            pdf.cell(col_widths[0], 7, str(strategy.get('name', '')), 1)
            pdf.cell(col_widths[1], 7, f"{strategy.get('annual_return', 0):.2f}", 1)
            pdf.cell(col_widths[2], 7, f"{strategy.get('annual_volatility', 0):.2f}", 1)
            pdf.cell(col_widths[3], 7, f"{strategy.get('sharpe_ratio', 0):.2f}", 1)
            pdf.cell(col_widths[4], 7, f"{strategy.get('max_drawdown', 0):.2f}", 1)
            pdf.ln()
        
        pdf.ln(5)
    
    if 'recommendation' in comparison_data:
        pdf.chapter_title('Recommended Strategy')
        
        rec = comparison_data['recommendation']
        rec_text = f"""
Based on your risk tolerance and investment goals, we recommend:

Strategy: {rec.get('recommended_strategy', 'N/A')}
Selection Criteria: {rec.get('selection_criteria', 'N/A')}
Expected Annual Return: {rec.get('expected_annual_return', 0):.2f}%
Expected Volatility: {rec.get('expected_volatility', 0):.2f}%
Sharpe Ratio: {rec.get('sharpe_ratio', 0):.2f}
Maximum Drawdown: {rec.get('max_drawdown', 0):.2f}%
"""
        pdf.chapter_body(rec_text.strip())
    
    try:
        pdf.output(filename)
        return filename
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return None


def create_stock_analysis_report(stock_data, filename='stock_analysis.pdf'):
    """
    Create PDF report for individual stock analysis
    """
    pdf = PortfolioReportPDF()
    pdf.add_page()
    
    pdf.chapter_title(f"Stock Analysis: {stock_data.get('ticker', 'Unknown')}")
    
    info_text = f"""
Company Name: {stock_data.get('name', 'N/A')}
Ticker: {stock_data.get('ticker', 'N/A')}
Current Price: ${stock_data.get('current_price', 0):.2f}
Market Cap: ${stock_data.get('market_cap', 0):,.0f}
Sector: {stock_data.get('sector', 'N/A')}
Industry: {stock_data.get('industry', 'N/A')}
Exchange: {stock_data.get('exchange', 'N/A')}
"""
    pdf.chapter_body(info_text.strip())
    
    if 'metrics' in stock_data:
        pdf.chapter_title('Performance Metrics')
        
        metrics = stock_data['metrics']
        metrics_text = f"""
Price Change: {metrics.get('price_change_pct', 0):.2f}%
Annual Return: {metrics.get('annualized_return', 0)*100:.2f}%
Annual Volatility: {metrics.get('annualized_volatility', 0)*100:.2f}%
Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
52-Week High: ${metrics.get('max_price', 0):.2f}
52-Week Low: ${metrics.get('min_price', 0):.2f}
"""
        pdf.chapter_body(metrics_text.strip())
    
    if 'predictions' in stock_data:
        pdf.chapter_title('AI Predictions')
        
        pred = stock_data['predictions']
        pred_text = f"""
Model: {pred.get('model_type', 'N/A')}
7-Day Prediction: ${pred.get('predicted_7d', 0):.2f}
30-Day Prediction: ${pred.get('predicted_30d', 0):.2f}
Trend: {pred.get('trend', 'N/A').upper()}
Confidence: {pred.get('confidence', 0)*100:.1f}%
"""
        pdf.chapter_body(pred_text.strip())
        
        if pred.get('analysis'):
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 7, 'Analysis:', 0, 1)
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 5, pred['analysis'])
            pdf.ln()
    
    try:
        pdf.output(filename)
        return filename
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return None
