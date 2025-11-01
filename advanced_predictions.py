import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')


def prepare_prophet_data(stock_data):
    """
    Prepare data for Prophet model (requires 'ds' and 'y' columns)
    """
    df = pd.DataFrame({
        'ds': stock_data.index,
        'y': stock_data['Close'].values
    })
    return df


def prophet_forecast(stock_data, periods=30):
    """
    Use Prophet for time series forecasting
    
    Args:
        stock_data: DataFrame with stock price history
        periods: Number of days to forecast
    
    Returns:
        Dictionary with predictions and model info
    """
    try:
        df = prepare_prophet_data(stock_data)
        
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        model.fit(df)
        
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        
        current_price = stock_data['Close'].iloc[-1]
        predicted_price_7d = forecast['yhat'].iloc[-23] if len(forecast) >= 23 else forecast['yhat'].iloc[-7]
        predicted_price_30d = forecast['yhat'].iloc[-1]
        
        return {
            'model_type': 'Prophet',
            'current_price': current_price,
            'predicted_7d': predicted_price_7d,
            'predicted_30d': predicted_price_30d,
            'forecast_df': predictions,
            'confidence_7d': (predicted_price_7d - forecast['yhat_lower'].iloc[-23]) / predicted_price_7d if len(forecast) >= 23 else 0.8,
            'confidence_30d': (predicted_price_30d - forecast['yhat_lower'].iloc[-1]) / predicted_price_30d,
            'trend': 'bullish' if predicted_price_30d > current_price else 'bearish'
        }
    except Exception as e:
        return {
            'error': str(e),
            'model_type': 'Prophet',
            'current_price': stock_data['Close'].iloc[-1],
            'predicted_7d': stock_data['Close'].iloc[-1],
            'predicted_30d': stock_data['Close'].iloc[-1]
        }


def arima_forecast(stock_data, order=(5, 1, 0), periods=30):
    """
    Use ARIMA for time series forecasting
    
    Args:
        stock_data: DataFrame with stock price history
        order: ARIMA order (p, d, q)
        periods: Number of days to forecast
    
    Returns:
        Dictionary with predictions and model info
    """
    try:
        prices = stock_data['Close'].values
        
        model = ARIMA(prices, order=order)
        fitted_model = model.fit()
        
        forecast = fitted_model.forecast(steps=periods)
        forecast_conf = fitted_model.get_forecast(steps=periods)
        conf_int = forecast_conf.conf_int()
        
        current_price = prices[-1]
        predicted_price_7d = forecast[6] if len(forecast) > 6 else forecast[-1]
        predicted_price_30d = forecast[-1]
        
        return {
            'model_type': 'ARIMA',
            'order': order,
            'current_price': current_price,
            'predicted_7d': predicted_price_7d,
            'predicted_30d': predicted_price_30d,
            'forecast': forecast.tolist(),
            'confidence_interval': {
                'lower': conf_int[:, 0].tolist(),
                'upper': conf_int[:, 1].tolist()
            },
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'trend': 'bullish' if predicted_price_30d > current_price else 'bearish'
        }
    except Exception as e:
        return {
            'error': str(e),
            'model_type': 'ARIMA',
            'current_price': stock_data['Close'].iloc[-1],
            'predicted_7d': stock_data['Close'].iloc[-1],
            'predicted_30d': stock_data['Close'].iloc[-1]
        }


def auto_arima_order(stock_data, max_p=5, max_q=5):
    """
    Find best ARIMA order using AIC criterion
    """
    prices = stock_data['Close'].values
    best_aic = np.inf
    best_order = (1, 1, 0)
    
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                model = ARIMA(prices, order=(p, 1, q))
                fitted = model.fit()
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, 1, q)
            except:
                continue
    
    return best_order


def ensemble_prediction(stock_data, periods=30):
    """
    Combine Prophet and ARIMA predictions for more robust forecasting
    """
    prophet_result = prophet_forecast(stock_data, periods)
    
    best_order = auto_arima_order(stock_data)
    arima_result = arima_forecast(stock_data, order=best_order, periods=periods)
    
    if 'error' not in prophet_result and 'error' not in arima_result:
        ensemble_7d = (prophet_result['predicted_7d'] + arima_result['predicted_7d']) / 2
        ensemble_30d = (prophet_result['predicted_30d'] + arima_result['predicted_30d']) / 2
        
        current_price = stock_data['Close'].iloc[-1]
        
        return {
            'model_type': 'Ensemble (Prophet + ARIMA)',
            'current_price': current_price,
            'predicted_7d': ensemble_7d,
            'predicted_30d': ensemble_30d,
            'prophet_7d': prophet_result['predicted_7d'],
            'prophet_30d': prophet_result['predicted_30d'],
            'arima_7d': arima_result['predicted_7d'],
            'arima_30d': arima_result['predicted_30d'],
            'arima_order': arima_result.get('order', 'N/A'),
            'trend': 'bullish' if ensemble_30d > current_price else 'bearish',
            'price_change_7d': ((ensemble_7d - current_price) / current_price) * 100,
            'price_change_30d': ((ensemble_30d - current_price) / current_price) * 100
        }
    elif 'error' not in prophet_result:
        current_price = stock_data['Close'].iloc[-1]
        prophet_result['price_change_7d'] = ((prophet_result['predicted_7d'] - current_price) / current_price) * 100
        prophet_result['price_change_30d'] = ((prophet_result['predicted_30d'] - current_price) / current_price) * 100
        return prophet_result
    elif 'error' not in arima_result:
        current_price = stock_data['Close'].iloc[-1]
        arima_result['price_change_7d'] = ((arima_result['predicted_7d'] - current_price) / current_price) * 100
        arima_result['price_change_30d'] = ((arima_result['predicted_30d'] - current_price) / current_price) * 100
        return arima_result
    else:
        return {
            'error': 'Both models failed',
            'current_price': stock_data['Close'].iloc[-1],
            'predicted_7d': stock_data['Close'].iloc[-1],
            'predicted_30d': stock_data['Close'].iloc[-1]
        }


def calculate_prediction_accuracy(stock_data, model_type='ensemble', lookback=30):
    """
    Calculate historical accuracy of predictions
    """
    if len(stock_data) < lookback * 2:
        return None
    
    train_data = stock_data[:-lookback]
    actual_prices = stock_data['Close'][-lookback:].values
    
    if model_type == 'prophet':
        result = prophet_forecast(train_data, periods=lookback)
    elif model_type == 'arima':
        result = arima_forecast(train_data, periods=lookback)
    else:
        result = ensemble_prediction(train_data, periods=lookback)
    
    if 'error' in result:
        return None
    
    predicted_prices = result.get('forecast', [result['predicted_30d']])
    
    if len(predicted_prices) > 0:
        min_len = min(len(predicted_prices), len(actual_prices))
        mae = np.mean(np.abs(np.array(predicted_prices[:min_len]) - actual_prices[:min_len]))
        mape = np.mean(np.abs((actual_prices[:min_len] - predicted_prices[:min_len]) / actual_prices[:min_len])) * 100
        
        return {
            'mae': mae,
            'mape': mape,
            'accuracy': 100 - mape if mape < 100 else 0
        }
    
    return None
