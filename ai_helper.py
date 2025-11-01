import json
import os
from openai import OpenAI

# the newest OpenAI model is "gpt-5" which was released August 7, 2025.
# do not change this unless explicitly requested by the user

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai = OpenAI(api_key=OPENAI_API_KEY)


def predict_stock_price(ticker, historical_data, current_price):
    """
    Use AI to analyze stock data and provide price predictions
    """
    try:
        recent_prices = historical_data['Close'].tail(30).tolist()
        price_change = ((current_price - recent_prices[0]) / recent_prices[0]) * 100
        
        prompt = f"""You are a financial analyst. Analyze the following stock data for {ticker}:
        
Current Price: ${current_price:.2f}
30-day price change: {price_change:.2f}%
Recent 30-day closing prices: {recent_prices}

Provide a JSON response with the following structure:
{{
    "short_term_prediction": "bullish/bearish/neutral",
    "price_target_7d": <predicted price in 7 days>,
    "price_target_30d": <predicted price in 30 days>,
    "confidence": <confidence score 0-1>,
    "key_factors": ["factor1", "factor2", "factor3"],
    "analysis": "<brief analysis>"
}}
"""
        
        response = openai.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert financial analyst specializing in stock market predictions. Provide data-driven analysis based on price patterns."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        return {
            "error": str(e),
            "short_term_prediction": "error",
            "price_target_7d": current_price,
            "price_target_30d": current_price,
            "confidence": 0,
            "key_factors": [],
            "analysis": f"Error generating prediction: {str(e)}"
        }


def analyze_portfolio(portfolio_data, market_conditions):
    """
    Use AI to provide portfolio analysis and recommendations
    """
    try:
        prompt = f"""You are a portfolio management expert. Analyze the following portfolio:

Portfolio Data:
{json.dumps(portfolio_data, indent=2)}

Market Conditions:
{json.dumps(market_conditions, indent=2)}

Provide a JSON response with:
{{
    "overall_health": "excellent/good/moderate/poor",
    "risk_assessment": "low/medium/high",
    "diversification_score": <0-100>,
    "recommendations": ["recommendation1", "recommendation2", "recommendation3"],
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "summary": "<brief portfolio summary>"
}}
"""
        
        response = openai.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert portfolio manager with deep knowledge of modern portfolio theory and risk management."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        return {
            "error": str(e),
            "overall_health": "unknown",
            "risk_assessment": "unknown",
            "diversification_score": 0,
            "recommendations": [],
            "strengths": [],
            "weaknesses": [],
            "summary": f"Error generating analysis: {str(e)}"
        }


def get_market_insights(tickers, market_type):
    """
    Get AI-powered market insights for given tickers
    """
    try:
        prompt = f"""Provide market insights for the following {market_type} market stocks: {', '.join(tickers)}

Provide a JSON response with:
{{
    "market_sentiment": "bullish/bearish/neutral",
    "sector_trends": {{"sector": "trend"}},
    "key_events": ["event1", "event2"],
    "outlook": "<brief market outlook>",
    "risks": ["risk1", "risk2"]
}}
"""
        
        response = openai.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a market analyst specializing in {market_type} markets. Provide general market insights."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        return {
            "error": str(e),
            "market_sentiment": "neutral",
            "sector_trends": {},
            "key_events": [],
            "outlook": f"Error generating insights: {str(e)}",
            "risks": []
        }
