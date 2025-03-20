import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings
import sys
import time

# Suppress warnings
warnings.filterwarnings('ignore')

# Download required NLTK data with error handling
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    st.error(f"Error downloading NLTK data: {str(e)}")
    st.info("Some features may be limited. Please try refreshing the page.")

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Private Markets Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize sentiment analyzers with error handling
try:
    vader = SentimentIntensityAnalyzer()
except Exception as e:
    st.error(f"Error initializing sentiment analyzer: {str(e)}")
    vader = None

def create_sample_data():
    """Create sample data for demonstration"""
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'Open': np.random.normal(100, 10, len(dates)),
        'High': np.random.normal(105, 10, len(dates)),
        'Low': np.random.normal(95, 10, len(dates)),
        'Close': np.random.normal(100, 10, len(dates)),
        'Volume': np.random.normal(1000000, 100000, len(dates))
    }, index=dates)
    
    # Ensure High is highest and Low is lowest
    df['High'] = df[['Open', 'Close']].max(axis=1) + abs(np.random.normal(2, 0.5, len(dates)))
    df['Low'] = df[['Open', 'Close']].min(axis=1) - abs(np.random.normal(2, 0.5, len(dates)))
    
    return df

def get_stock_data(symbol, period="1y"):
    """Fetch stock data using yfinance with improved error handling"""
    try:
        # Add retry mechanism
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Create Ticker object
                ticker = yf.Ticker(symbol)
                
                # Fetch historical data first
                df = ticker.history(period=period, interval="1d", prepost=False)
                
                if df.empty:
                    st.error(f"No data available for {symbol}")
                    return None
                
                # Basic data validation
                if len(df) < 10:
                    st.error(f"Insufficient data for {symbol}")
                    return None
                
                # Check for missing values
                if df.isnull().any().any():
                    # Forward fill missing values
                    df = df.fillna(method='ffill')
                    # Backward fill any remaining NaN values
                    df = df.fillna(method='bfill')
                
                # Verify data quality
                if df['Close'].isnull().any():
                    st.error(f"Data quality issues detected for {symbol}")
                    return None
                
                return df
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    st.error(f"Error fetching data for {symbol}: {str(e)}")
                    return None
                    
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

def calculate_technical_indicators(df):
    """Calculate technical indicators for financial analysis"""
    try:
        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_Upper'] = bollinger.bollinger_hband()
        df['BB_Lower'] = bollinger.bollinger_lband()
        
        # Volume indicators
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        
        return df
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        return None

def analyze_sentiment(text):
    """Analyze sentiment using VADER and TextBlob"""
    try:
        if vader is None:
            return None
            
        # VADER sentiment
        vader_sentiment = vader.polarity_scores(text)
        
        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment.polarity
        
        # Calculate combined sentiment score
        combined_sentiment = (vader_sentiment['compound'] + textblob_sentiment) / 2
        
        return {
            'vader': vader_sentiment['compound'],
            'textblob': textblob_sentiment,
            'transformers': combined_sentiment  # Using combined score instead of transformers
        }
    except Exception as e:
        st.error(f"Error in sentiment analysis: {str(e)}")
        return None

def prepare_ml_data(df):
    """Prepare data for machine learning with enhanced features"""
    try:
        # Create more sophisticated features
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df['Price_Change'] = df['Close'] - df['Open']
        df['Daily_Range'] = df['High'] - df['Low']
        df['Price_Momentum'] = df['Close'].pct_change(periods=5)
        df['Volume_Momentum'] = df['Volume'].pct_change(periods=5)
        
        # Add market regime features
        df['Trend'] = df['Close'].rolling(window=20).mean() - df['Close'].rolling(window=50).mean()
        df['Volatility_Regime'] = df['Volatility'].rolling(window=20).mean()
        
        # Create target variables
        df['Next_Day_Return'] = df['Close'].shift(-1) / df['Close'] - 1
        df['Next_Day_Price'] = df['Close'].shift(-1)
        df['Price_Direction'] = (df['Next_Day_Price'] > df['Close']).astype(int)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        if len(df) < 100:
            st.warning("Insufficient data for machine learning analysis")
            return None, None, None
        
        # Select features for ML
        features = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 
            'MACD_Signal', 'BB_Upper', 'BB_Lower', 'OBV', 'Returns', 
            'Volatility', 'Price_Change', 'Daily_Range', 'Price_Momentum',
            'Volume_Momentum', 'Trend', 'Volatility_Regime'
        ]
        
        X = df[features]
        y_return = df['Next_Day_Return']
        y_direction = df['Price_Direction']
        
        return X, y_return, y_direction
    except Exception as e:
        st.error(f"Error preparing ML data: {str(e)}")
        return None, None, None

def train_ml_model(X, y_return, y_direction):
    """Train multiple ML models for different predictions"""
    try:
        # Split data
        X_train, X_test, y_return_train, y_return_test, y_direction_train, y_direction_test = train_test_split(
            X, y_return, y_direction, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train return prediction model
        return_model = RandomForestRegressor(n_estimators=100, random_state=42)
        return_model.fit(X_train_scaled, y_return_train)
        return_train_score = return_model.score(X_train_scaled, y_return_train)
        return_test_score = return_model.score(X_test_scaled, y_return_test)
        
        # Train direction prediction model
        direction_model = RandomForestRegressor(n_estimators=100, random_state=42)
        direction_model.fit(X_train_scaled, y_direction_train)
        direction_train_score = direction_model.score(X_train_scaled, y_direction_train)
        direction_test_score = direction_model.score(X_test_scaled, y_direction_test)
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': return_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'return_model': return_model,
            'direction_model': direction_model,
            'scaler': scaler,
            'return_scores': (return_train_score, return_test_score),
            'direction_scores': (direction_train_score, direction_test_score),
            'feature_importance': feature_importance
        }
    except Exception as e:
        st.error(f"Error training ML model: {str(e)}")
        return None

def main():
    st.title("📊 Private Markets Analytics Dashboard")
    
    # Sidebar
    st.sidebar.header("Settings")
    
    symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., MSFT)", "MSFT")
    period = st.sidebar.selectbox(
        "Select Time Period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
        index=3
    )
    
    # Add a refresh button
    if st.sidebar.button("Refresh Data"):
        st.experimental_rerun()
    
    # Main content
    df = get_stock_data(symbol, period)
    
    if df is not None and not df.empty:
        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        
        if df is not None:
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4 = st.tabs(["Market Analysis", "Technical Indicators", "Machine Learning", "Sentiment Analysis"])
            
            with tab1:
                st.subheader("Market Analysis")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                                        open=df['Open'],
                                                        high=df['High'],
                                                        low=df['Low'],
                                                        close=df['Close'])])
                    fig.update_layout(title=f"{symbol} Stock Price",
                                    yaxis_title="Price",
                                    xaxis_title="Date")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    try:
                        current_price = df['Close'].iloc[-1]
                        price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
                        price_change_pct = (price_change / df['Close'].iloc[-2]) * 100
                        
                        st.metric("Current Price", f"${current_price:.2f}")
                        st.metric("Price Change", f"${price_change:.2f} ({price_change_pct:.2f}%)")
                    except Exception as e:
                        st.error("Error calculating price metrics")
                    
                    # Volume analysis
                    st.subheader("Volume Analysis")
                    volume_fig = go.Figure(data=[go.Bar(x=df.index, y=df['Volume'])])
                    volume_fig.update_layout(title="Trading Volume",
                                          yaxis_title="Volume",
                                          xaxis_title="Date")
                    st.plotly_chart(volume_fig, use_container_width=True)
                
                # Market Analysis Reference Notes
                st.markdown("---")
                st.markdown("""
                ### 📚 Market Analysis Reference Notes
                
                **Candlestick Chart Interpretation:**
                - Green candles: Price closed higher than it opened
                - Red candles: Price closed lower than it opened
                - Thick bars: Opening and closing prices
                - Thin lines (wicks): Highest and lowest prices during the period
                
                **Key Price Metrics:**
                - Current Price: Latest market value
                - Price Change: Daily price movement and percentage change
                - Volume: Trading activity indicator
                
                **Volume Analysis:**
                - Higher volume often indicates stronger price movements
                - Volume spikes may signal important news or events
                - Volume trends can confirm price trends
                
                *Note: Price movements should be analyzed in conjunction with volume patterns for better understanding of market dynamics.*
                """)
            
            with tab2:
                st.subheader("Technical Indicators")
                col1, col2 = st.columns(2)
                
                with col1:
                    # RSI
                    rsi_fig = go.Figure(data=[go.Scatter(x=df.index, y=df['RSI'])])
                    rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
                    rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
                    rsi_fig.update_layout(title="Relative Strength Index (RSI)",
                                       yaxis_title="RSI",
                                       xaxis_title="Date")
                    st.plotly_chart(rsi_fig, use_container_width=True)
                    
                    # MACD
                    macd_fig = go.Figure()
                    macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD"))
                    macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name="Signal"))
                    macd_fig.update_layout(title="MACD",
                                        yaxis_title="Value",
                                        xaxis_title="Date")
                    st.plotly_chart(macd_fig, use_container_width=True)
                
                with col2:
                    # Bollinger Bands
                    bb_fig = go.Figure()
                    bb_fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price"))
                    bb_fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name="Upper BB"))
                    bb_fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name="Lower BB"))
                    bb_fig.update_layout(title="Bollinger Bands",
                                      yaxis_title="Price",
                                      xaxis_title="Date")
                    st.plotly_chart(bb_fig, use_container_width=True)
                    
                    # On-Balance Volume
                    obv_fig = go.Figure(data=[go.Scatter(x=df.index, y=df['OBV'])])
                    obv_fig.update_layout(title="On-Balance Volume (OBV)",
                                       yaxis_title="OBV",
                                       xaxis_title="Date")
                    st.plotly_chart(obv_fig, use_container_width=True)
                
                # Technical Indicators Reference Notes
                st.markdown("---")
                st.markdown("""
                ### 📚 Technical Indicators Reference Notes
                
                **RSI (Relative Strength Index)**
                - Range: 0-100
                - Overbought: >70 (potential sell signal)
                - Oversold: <30 (potential buy signal)
                - Best used for identifying potential reversal points
                - Divergence from price action may signal trend weakness
                
                **MACD (Moving Average Convergence Divergence)**
                - Shows relationship between short and long-term moving averages
                - Signal line crossovers indicate potential trend changes
                - Divergence from price action may signal trend weakness
                - Histogram shows momentum strength
                
                **Bollinger Bands**
                - Upper/Lower bands indicate volatility
                - Price touching upper band: potential overbought
                - Price touching lower band: potential oversold
                - Band width indicates market volatility
                - Price tends to stay within the bands
                
                **On-Balance Volume (OBV)**
                - Confirms price trends
                - Divergence from price may signal trend weakness
                - Volume precedes price movement
                - Rising OBV confirms uptrend
                - Falling OBV confirms downtrend
                
                *Note: Technical indicators should be used in combination with other analysis methods for better decision-making.*
                """)
            
            with tab3:
                st.subheader("Machine Learning Analysis")
                X, y_return, y_direction = prepare_ml_data(df)
                
                if X is not None and y_return is not None and y_direction is not None:
                    ml_results = train_ml_model(X, y_return, y_direction)
                    
                    if ml_results is not None:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Model Performance")
                            st.metric("Return Prediction R² Score (Training)", 
                                    f"{ml_results['return_scores'][0]:.3f}")
                            st.metric("Return Prediction R² Score (Testing)", 
                                    f"{ml_results['return_scores'][1]:.3f}")
                            st.metric("Direction Prediction Accuracy (Training)", 
                                    f"{ml_results['direction_scores'][0]:.3f}")
                            st.metric("Direction Prediction Accuracy (Testing)", 
                                    f"{ml_results['direction_scores'][1]:.3f}")
                            
                            # Feature importance
                            st.subheader("Key Price Drivers")
                            fig = plt.figure(figsize=(10, 6))
                            sns.barplot(x='importance', y='feature', 
                                      data=ml_results['feature_importance'].head(10))
                            plt.title("Top 10 Most Important Features")
                            st.pyplot(fig)
                        
                        with col2:
                            st.subheader("Price Predictions")
                            # Make predictions
                            last_data = X.iloc[-1:]
                            last_data_scaled = ml_results['scaler'].transform(last_data)
                            
                            # Return prediction
                            return_pred = ml_results['return_model'].predict(last_data_scaled)[0]
                            direction_prob = ml_results['direction_model'].predict(last_data_scaled)[0]
                            
                            current_price = df['Close'].iloc[-1]
                            predicted_price = current_price * (1 + return_pred)
                            
                            st.metric("Current Price", f"${current_price:.2f}")
                            st.metric("Predicted Next Day Price", f"${predicted_price:.2f}")
                            st.metric("Predicted Return", f"{return_pred*100:.2f}%")
                            st.metric("Probability of Price Increase", f"{direction_prob*100:.2f}%")
                            
                            # Calculate prediction confidence
                            std_dev = np.std(y_return - ml_results['return_model'].predict(ml_results['scaler'].transform(X)))
                            st.metric("Prediction Standard Deviation", f"{std_dev*100:.2f}%")
                            
                            # Risk assessment
                            st.subheader("Risk Assessment")
                            volatility = df['Volatility'].iloc[-1]
                            trend_strength = abs(df['Trend'].iloc[-1])
                            
                            st.metric("Current Volatility", f"{volatility*100:.2f}%")
                            st.metric("Trend Strength", f"{trend_strength:.2f}")
                            
                            # Risk level indicator
                            risk_level = "High" if volatility > 0.02 else "Medium" if volatility > 0.01 else "Low"
                            st.metric("Risk Level", risk_level)
                
                # Machine Learning Reference Notes
                st.markdown("---")
                st.markdown("""
                ### 📚 Machine Learning Analysis Reference Notes
                
                **Model Performance Metrics**
                - Return Prediction: Estimates next-day price change percentage
                    - Higher R² scores indicate better prediction accuracy
                    - Testing score shows real-world performance
                
                - Direction Prediction: Probability of price increase
                    - Above 0.5 suggests bullish sentiment
                    - Below 0.5 suggests bearish sentiment
                
                **Key Price Drivers**
                - Shows which factors most influence price predictions
                - Higher importance indicates stronger predictive power
                - Helps identify key market dynamics
                
                **Risk Assessment**
                - Volatility: Measures price fluctuation intensity
                - Trend Strength: Indicates current market momentum
                - Risk Level: Overall market risk assessment
                
                **Investment Insights**
                - Use predictions as one of multiple decision factors
                - Consider risk level when making investment decisions
                - Higher volatility suggests more conservative approach
                - Strong trends may indicate momentum trading opportunities
                
                *Note: Machine learning predictions are based on historical patterns and should be used in conjunction with fundamental analysis, market conditions, and personal risk tolerance. Past performance does not guarantee future results.*
                """)
            
            with tab4:
                st.subheader("Sentiment Analysis")
                text_input = st.text_area("Enter text for sentiment analysis:", 
                                        "Example: The company's latest earnings report shows strong growth potential.")
                
                if st.button("Analyze Sentiment"):
                    if text_input:
                        sentiment_results = analyze_sentiment(text_input)
                        
                        if sentiment_results is not None:
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("VADER Sentiment", 
                                         f"{sentiment_results['vader']:.2f}",
                                         delta=None)
                            
                            with col2:
                                st.metric("TextBlob Sentiment", 
                                         f"{sentiment_results['textblob']:.2f}",
                                         delta=None)
                            
                            with col3:
                                st.metric("Transformers Sentiment", 
                                         f"{sentiment_results['transformers']:.2f}",
                                         delta=None)
                    else:
                        st.warning("Please enter some text to analyze.")
                
                # Sentiment Analysis Reference Notes
                st.markdown("---")
                st.markdown("""
                ### 📚 Sentiment Analysis Reference Notes
                
                **Multiple Analysis Methods**
                - VADER: Rule-based sentiment analysis
                    - Analyzes text for positive, negative, and neutral words
                    - Good for social media and short texts
                    - Handles slang and emojis well
                
                - TextBlob: Machine learning-based analysis
                    - Uses natural language processing
                    - Better for longer, more formal texts
                    - Considers context and grammar
                
                - Combined Score: Average of multiple methods
                    - Provides more robust sentiment assessment
                    - Reduces bias from single method
                    - Better for decision-making
                
                **Score Interpretation**
                - Range: -1 (very negative) to +1 (very positive)
                - Values near 0 indicate neutral sentiment
                - Strong positive/negative values indicate clear sentiment
                - Multiple methods provide consensus
                
                *Note: Sentiment analysis is most effective when analyzing multiple sources of text and considering the context of the information. Use this tool to supplement other analysis methods.*
                """)

if __name__ == "__main__":
    main() 