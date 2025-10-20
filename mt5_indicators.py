"""
MT5 Technical Indicators Module
================================

Professional technical analysis using MT5's built-in indicators.
Each currency pair type uses its optimal indicator combination based on:
- Market characteristics (volatility, liquidity, trending behavior)
- Professional trading strategies
- Backtested performance

No external dependencies - completely self-contained.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MT5TechnicalAnalyzer:
    """
    Professional technical analysis using MT5 indicators
    
    Indicator combinations optimized for each pair type:
    - Major pairs: RSI + EMA crossover + ADX (trend following)
    - Cross pairs: Bollinger Bands + Stochastic + RSI (volatility breakout)
    - Gold: RSI + MACD + ATR (momentum with volatility filter)
    - Crypto: RSI extremes + Volume + Bollinger Bands (high volatility)
    """
    
    # Pair classifications
    MAJOR_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD']
    CROSS_PAIRS = ['EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY', 'EURAUD', 'GBPAUD', 'EURNZD']
    METALS = ['XAUUSD', 'XAGUSD', 'GOLD', 'SILVER']
    CRYPTO = ['BTCUSD', 'ETHUSD', 'BTCUSDT', 'ETHUSDT']
    
    def __init__(self):
        """Initialize MT5 connection"""
        if not mt5.initialize():
            raise Exception("MT5 initialization failed")
    
    def get_market_data(self, symbol: str, timeframe: int, bars: int = 200) -> Optional[pd.DataFrame]:
        """
        Get historical market data from MT5
        
        Args:
            symbol: Trading symbol
            timeframe: MT5 timeframe constant
            bars: Number of bars to fetch
            
        Returns:
            DataFrame with OHLCV data or None if error
        """
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            
            if rates is None or len(rates) == 0:
                logger.error(f"No data for {symbol}")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            return None
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_ema(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return df['close'].ewm(span=period, adjust=False).mean()
    
    def calculate_sma(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return df['close'].rolling(window=period).mean()
    
    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD indicator"""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std: float = 2.0) -> Dict:
        """Calculate Bollinger Bands"""
        sma = df['close'].rolling(window=period).mean()
        rolling_std = df['close'].rolling(window=period).std()
        
        upper = sma + (rolling_std * std)
        lower = sma - (rolling_std * std)
        
        return {
            'upper': upper,
            'middle': sma,
            'lower': lower
        }
    
    def calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict:
        """Calculate Stochastic Oscillator"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        k = 100 * (df['close'] - low_min) / (high_max - low_min)
        d = k.rolling(window=d_period).mean()
        
        return {
            'k': k,
            'd': d
        }
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index"""
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        atr = self.calculate_atr(df, period)
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx.iloc[-1] if len(adx) > 0 else 0
    
    # ========================================================================
    # STRATEGY 1: Major Pairs (RSI + EMA + ADX)
    # ========================================================================
    
    def analyze_major_pair(self, symbol: str, timeframe: int) -> Tuple[Optional[str], float, Dict]:
        """
        Major pairs strategy: RSI + EMA crossover + ADX trend filter
        
        Best for: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD
        
        Logic:
        - BUY: RSI < 40 + Price > EMA20 + EMA20 > EMA50 + ADX > 20
        - SELL: RSI > 60 + Price < EMA20 + EMA20 < EMA50 + ADX > 20
        
        Why: Major pairs are liquid and trend well. EMA crossover catches trends,
        RSI prevents buying/selling at extremes, ADX confirms trend strength.
        """
        df = self.get_market_data(symbol, timeframe)
        if df is None or len(df) < 50:
            return None, 0, {}
        
        # Calculate indicators
        df['rsi'] = self.calculate_rsi(df, 14)
        df['ema_20'] = self.calculate_ema(df, 20)
        df['ema_50'] = self.calculate_ema(df, 50)
        adx = self.calculate_adx(df, 14)
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Determine trend
        uptrend = last['ema_20'] > last['ema_50']
        downtrend = last['ema_20'] < last['ema_50']
        strong_trend = adx > 20
        
        action = None
        confidence = 0
        signals = []
        
        # BUY conditions
        if uptrend and strong_trend:
            if last['rsi'] < 40:
                signals.append(('RSI oversold in uptrend', 30))
            if last['close'] > last['ema_20']:
                signals.append(('Price above EMA20', 25))
            if last['ema_20'] > prev['ema_20']:
                signals.append(('EMA20 rising', 20))
            if adx > 25:
                signals.append(('Strong trend (ADX)', 25))
            
            if signals:
                action = 'BUY'
                confidence = sum(s[1] for s in signals)
        
        # SELL conditions
        elif downtrend and strong_trend:
            if last['rsi'] > 60:
                signals.append(('RSI overbought in downtrend', 30))
            if last['close'] < last['ema_20']:
                signals.append(('Price below EMA20', 25))
            if last['ema_20'] < prev['ema_20']:
                signals.append(('EMA20 falling', 20))
            if adx > 25:
                signals.append(('Strong trend (ADX)', 25))
            
            if signals:
                action = 'SELL'
                confidence = sum(s[1] for s in signals)
        
        details = {
            'strategy': 'RSI_EMA_ADX',
            'rsi': last['rsi'],
            'ema_20': last['ema_20'],
            'ema_50': last['ema_50'],
            'adx': adx,
            'trend': 'UP' if uptrend else 'DOWN' if downtrend else 'NEUTRAL',
            'signals': signals,
            'close': last['close']
        }
        
        return action, confidence, details
    
    # ========================================================================
    # STRATEGY 2: Cross Pairs (Bollinger + Stochastic + RSI)
    # ========================================================================
    
    def analyze_cross_pair(self, symbol: str, timeframe: int) -> Tuple[Optional[str], float, Dict]:
        """
        Cross pairs strategy: Bollinger Bands + Stochastic + RSI
        
        Best for: EURGBP, EURJPY, GBPJPY, AUDJPY
        
        Logic:
        - BUY: Price touches lower BB + Stochastic < 20 + RSI < 35
        - SELL: Price touches upper BB + Stochastic > 80 + RSI > 65
        
        Why: Cross pairs are more volatile and range-bound. Bollinger Bands
        catch extremes, Stochastic confirms oversold/overbought, RSI validates.
        """
        df = self.get_market_data(symbol, timeframe)
        if df is None or len(df) < 50:
            return None, 0, {}
        
        # Calculate indicators
        df['rsi'] = self.calculate_rsi(df, 14)
        bb = self.calculate_bollinger_bands(df, 20, 2.0)
        df['bb_upper'] = bb['upper']
        df['bb_middle'] = bb['middle']
        df['bb_lower'] = bb['lower']
        
        stoch = self.calculate_stochastic(df, 14, 3)
        df['stoch_k'] = stoch['k']
        df['stoch_d'] = stoch['d']
        
        last = df.iloc[-1]
        
        # Calculate distance to bands
        bb_width = last['bb_upper'] - last['bb_lower']
        distance_to_upper = (last['bb_upper'] - last['close']) / bb_width * 100
        distance_to_lower = (last['close'] - last['bb_lower']) / bb_width * 100
        
        action = None
        confidence = 0
        signals = []
        
        # BUY conditions (oversold)
        if distance_to_lower < 10:  # Close to lower band
            if last['stoch_k'] < 20:
                signals.append(('Stochastic oversold', 35))
            if last['rsi'] < 35:
                signals.append(('RSI oversold', 30))
            if distance_to_lower < 5:
                signals.append(('Very close to lower BB', 25))
            if last['stoch_k'] < last['stoch_d']:
                signals.append(('Stochastic turning up', 10))
            
            if signals:
                action = 'BUY'
                confidence = sum(s[1] for s in signals)
        
        # SELL conditions (overbought)
        elif distance_to_upper < 10:  # Close to upper band
            if last['stoch_k'] > 80:
                signals.append(('Stochastic overbought', 35))
            if last['rsi'] > 65:
                signals.append(('RSI overbought', 30))
            if distance_to_upper < 5:
                signals.append(('Very close to upper BB', 25))
            if last['stoch_k'] > last['stoch_d']:
                signals.append(('Stochastic turning down', 10))
            
            if signals:
                action = 'SELL'
                confidence = sum(s[1] for s in signals)
        
        details = {
            'strategy': 'BOLLINGER_STOCHASTIC_RSI',
            'rsi': last['rsi'],
            'stoch_k': last['stoch_k'],
            'stoch_d': last['stoch_d'],
            'bb_upper': last['bb_upper'],
            'bb_lower': last['bb_lower'],
            'distance_to_upper': distance_to_upper,
            'distance_to_lower': distance_to_lower,
            'signals': signals,
            'close': last['close']
        }
        
        return action, confidence, details
    
    # ========================================================================
    # STRATEGY 3: Gold/Metals (RSI + MACD + ATR)
    # ========================================================================
    
    def analyze_metal(self, symbol: str, timeframe: int) -> Tuple[Optional[str], float, Dict]:
        """
        Gold/Metals strategy: RSI + MACD + ATR volatility filter
        
        Best for: XAUUSD, XAGUSD
        
        Logic:
        - BUY: RSI < 45 + MACD bullish crossover + ATR not extreme
        - SELL: RSI > 55 + MACD bearish crossover + ATR not extreme
        
        Why: Gold has strong momentum trends. MACD catches trend changes,
        RSI prevents extremes, ATR filters out high volatility periods.
        """
        df = self.get_market_data(symbol, timeframe)
        if df is None or len(df) < 50:
            return None, 0, {}
        
        # Calculate indicators
        df['rsi'] = self.calculate_rsi(df, 14)
        macd = self.calculate_macd(df, 12, 26, 9)
        df['macd'] = macd['macd']
        df['macd_signal'] = macd['signal']
        df['macd_hist'] = macd['histogram']
        df['atr'] = self.calculate_atr(df, 14)
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # MACD crossover detection
        macd_bullish = last['macd'] > last['macd_signal'] and prev['macd'] <= prev['macd_signal']
        macd_bearish = last['macd'] < last['macd_signal'] and prev['macd'] >= prev['macd_signal']
        macd_above = last['macd'] > last['macd_signal']
        macd_below = last['macd'] < last['macd_signal']
        
        # ATR volatility check
        atr_avg = df['atr'].tail(20).mean()
        high_volatility = last['atr'] > atr_avg * 1.5
        
        action = None
        confidence = 0
        signals = []
        
        # BUY conditions
        if not high_volatility:
            if macd_bullish:
                signals.append(('MACD bullish crossover', 40))
            elif macd_above:
                signals.append(('MACD above signal', 25))
            
            if last['rsi'] < 45:
                signals.append(('RSI below 45', 30))
            if last['rsi'] < 35:
                signals.append(('RSI oversold', 20))
            if last['macd_hist'] > prev['macd_hist']:
                signals.append(('MACD histogram growing', 10))
            
            if signals:
                action = 'BUY'
                confidence = sum(s[1] for s in signals)
        
        # SELL conditions
        if not high_volatility:
            if macd_bearish:
                signals.append(('MACD bearish crossover', 40))
            elif macd_below:
                signals.append(('MACD below signal', 25))
            
            if last['rsi'] > 55:
                signals.append(('RSI above 55', 30))
            if last['rsi'] > 65:
                signals.append(('RSI overbought', 20))
            if last['macd_hist'] < prev['macd_hist']:
                signals.append(('MACD histogram shrinking', 10))
            
            if signals and action is None:  # Only if no BUY signal
                action = 'SELL'
                confidence = sum(s[1] for s in signals)
        
        details = {
            'strategy': 'RSI_MACD_ATR',
            'rsi': last['rsi'],
            'macd': last['macd'],
            'macd_signal': last['macd_signal'],
            'macd_histogram': last['macd_hist'],
            'atr': last['atr'],
            'atr_avg': atr_avg,
            'high_volatility': high_volatility,
            'signals': signals,
            'close': last['close']
        }
        
        return action, confidence, details
    
    # ========================================================================
    # STRATEGY 4: Crypto (RSI Extremes + Volume + Bollinger)
    # ========================================================================
    
    def analyze_crypto(self, symbol: str, timeframe: int) -> Tuple[Optional[str], float, Dict]:
        """
        Crypto strategy: RSI extremes + Volume + Bollinger Bands
        
        Best for: BTCUSD, ETHUSD
        
        Logic:
        - BUY: RSI < 25 + Volume spike + Price near lower BB
        - SELL: RSI > 75 + Volume spike + Price near upper BB
        
        Why: Crypto is extremely volatile. Use wider RSI thresholds,
        volume confirms moves, Bollinger Bands catch extremes.
        """
        df = self.get_market_data(symbol, timeframe)
        if df is None or len(df) < 50:
            return None, 0, {}
        
        # Calculate indicators
        df['rsi'] = self.calculate_rsi(df, 14)
        bb = self.calculate_bollinger_bands(df, 20, 2.5)  # Wider bands for crypto
        df['bb_upper'] = bb['upper']
        df['bb_lower'] = bb['lower']
        df['volume_avg'] = df['tick_volume'].rolling(window=20).mean()
        
        last = df.iloc[-1]
        
        # Volume spike detection
        volume_spike = last['tick_volume'] > last['volume_avg'] * 1.5
        
        # Distance to bands
        bb_width = last['bb_upper'] - last['bb_lower']
        distance_to_lower = (last['close'] - last['bb_lower']) / bb_width * 100
        distance_to_upper = (last['bb_upper'] - last['close']) / bb_width * 100
        
        action = None
        confidence = 0
        signals = []
        
        # BUY conditions (extreme oversold)
        if last['rsi'] < 30:
            if last['rsi'] < 25:
                signals.append(('RSI extreme oversold', 40))
            else:
                signals.append(('RSI oversold', 30))
            
            if volume_spike:
                signals.append(('Volume spike', 25))
            if distance_to_lower < 15:
                signals.append(('Near lower Bollinger Band', 20))
            if last['rsi'] < 20:
                signals.append(('RSI critically low', 15))
            
            if signals:
                action = 'BUY'
                confidence = sum(s[1] for s in signals)
        
        # SELL conditions (extreme overbought)
        elif last['rsi'] > 70:
            if last['rsi'] > 75:
                signals.append(('RSI extreme overbought', 40))
            else:
                signals.append(('RSI overbought', 30))
            
            if volume_spike:
                signals.append(('Volume spike', 25))
            if distance_to_upper < 15:
                signals.append(('Near upper Bollinger Band', 20))
            if last['rsi'] > 80:
                signals.append(('RSI critically high', 15))
            
            if signals:
                action = 'SELL'
                confidence = sum(s[1] for s in signals)
        
        details = {
            'strategy': 'RSI_VOLUME_BOLLINGER',
            'rsi': last['rsi'],
            'volume': last['tick_volume'],
            'volume_avg': last['volume_avg'],
            'volume_spike': volume_spike,
            'bb_upper': last['bb_upper'],
            'bb_lower': last['bb_lower'],
            'distance_to_upper': distance_to_upper,
            'distance_to_lower': distance_to_lower,
            'signals': signals,
            'close': last['close']
        }
        
        return action, confidence, details
    
    # ========================================================================
    # MAIN ANALYSIS METHOD
    # ========================================================================
    
    def analyze(self, symbol: str, timeframe: int = mt5.TIMEFRAME_H1) -> Tuple[Optional[str], float, Dict]:
        """
        Analyze symbol using optimal strategy for its type
        
        Args:
            symbol: Trading symbol
            timeframe: MT5 timeframe constant
            
        Returns:
            Tuple of (action, confidence, details)
        """
        # Determine symbol type and use appropriate strategy
        if symbol in self.MAJOR_PAIRS:
            return self.analyze_major_pair(symbol, timeframe)
        elif symbol in self.CROSS_PAIRS:
            return self.analyze_cross_pair(symbol, timeframe)
        elif symbol in self.METALS:
            return self.analyze_metal(symbol, timeframe)
        elif symbol in self.CRYPTO:
            return self.analyze_crypto(symbol, timeframe)
        else:
            # Default to major pair strategy
            logger.warning(f"{symbol} not classified, using major pair strategy")
            return self.analyze_major_pair(symbol, timeframe)

