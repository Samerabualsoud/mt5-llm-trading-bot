"""
Configuration Template
======================

Copy this file to config.py and fill in your actual credentials.
config.py is in .gitignore and won't be pushed to GitHub.
"""

CONFIG = {
    # MT5 Connection
    'mt5_login': 12345,  # Your MT5 account number
    'mt5_password': 'YOUR_PASSWORD_HERE',
    'mt5_server': 'YOUR_SERVER_NAME',  # e.g., 'ACYSecurities-Demo'
    
    # API Keys
    'deepseek_api_key': 'YOUR_DEEPSEEK_API_KEY',
    'openai_api_key': 'YOUR_OPENAI_API_KEY',
    
    # Trading Symbols
    'symbols': [
        # Major pairs (RSI + EMA + ADX strategy)
        'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD',
        'USDCAD', 'USDCHF', 'NZDUSD',
        
        # Cross pairs (Bollinger + Stochastic + RSI strategy)
        'EURGBP', 'EURJPY', 'GBPJPY',
        
        # Metals (RSI + MACD + ATR strategy)
        'XAUUSD',
        
        # Crypto (RSI extremes + Volume + Bollinger strategy)
        # 'BTCUSD',
    ],
    
    # Signal thresholds
    'min_technical_confidence': 50,  # Technical signal strength (0-100)
    'min_llm_confidence': 50,  # LLM approval threshold (0-100)
    
    # Timeframe
    'timeframe': 'H1',  # M1, M5, M15, M30, H1, H4, D1, W1
    
    # Risk Management
    'risk_percent': 0.002,  # 0.2% risk per trade (testing)
    # 'risk_percent': 0.005,  # 0.5% (conservative)
    # 'risk_percent': 0.01,   # 1% (moderate)
    # 'risk_percent': 0.02,   # 2% (aggressive)
    
    'max_positions': 5,
    
    # Safety limits
    'max_daily_loss': 0.05,  # -5% stop trading
    'max_daily_trades': 10,
    'check_interval': 300,  # 5 minutes
    
    # Debug
    'debug_mode': True,
}

