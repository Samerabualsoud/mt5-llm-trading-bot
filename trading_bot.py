"""
MT5 + LLM Trading Bot
=====================

Professional trading bot using:
1. MT5 built-in indicators (optimal mix for each pair type)
2. LLM strategic approval (DeepSeek + OpenAI)
3. Automated execution with risk management

No external dependencies - completely self-contained and reliable.
"""

import MetaTrader5 as mt5
from datetime import datetime
import time
import logging
from typing import Dict
import json
import sys

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from mt5_indicators import MT5TechnicalAnalyzer
from llm_agents import LLMConsensusEngine

# Import configuration
try:
    from config import CONFIG
except ImportError:
    print("ERROR: config.py not found!")
    print("Please copy config_template.py to config.py and fill in your credentials.")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MT5TradingBot:
    """Trading bot with MT5 indicators + LLM approval"""
    
    # Timeframe mapping
    TIMEFRAME_MAP = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
        'W1': mt5.TIMEFRAME_W1,
    }
    
    def __init__(self, config: Dict):
        self.config = config
        self.start_balance = 0
        self.daily_trades = 0
        self.trade_history = []
        
        # Initialize components
        self.analyzer = MT5TechnicalAnalyzer()
        self.llm_engine = LLMConsensusEngine(
            config['deepseek_api_key'],
            config['openai_api_key']
        )
        
        # Initialize MT5
        if not mt5.initialize():
            raise Exception("MT5 initialization failed")
        
        if not mt5.login(config['mt5_login'], password=config['mt5_password'], server=config['mt5_server']):
            raise Exception(f"MT5 login failed: {mt5.last_error()}")
        
        account_info = mt5.account_info()
        self.start_balance = account_info.balance
        
        logger.info("=" * 80)
        logger.info("MT5 + LLM TRADING BOT - PROFESSIONAL EDITION")
        logger.info("=" * 80)
        logger.info(f"Account: {account_info.login}")
        logger.info(f"Balance: ${account_info.balance:,.2f}")
        logger.info(f"Leverage: 1:{account_info.leverage}")
        logger.info(f"Symbols: {len(config['symbols'])}")
        logger.info(f"Timeframe: {config.get('timeframe', 'H1')}")
        logger.info(f"Min Technical Confidence: {config.get('min_technical_confidence', 50)}%")
        logger.info(f"Min LLM Confidence: {config.get('min_llm_confidence', 50)}%")
        logger.info(f"Min Margin Level: {config.get('min_margin_level', 800)}%")
        logger.info(f"Risk per Trade: {config.get('risk_percent', 0.01) * 100}%")
        logger.info("=" * 80)
    
    def scan_markets(self):
        """Scan all symbols using MT5 technical analysis"""
        account_info = mt5.account_info()
        positions = mt5.positions_get()
        open_positions_count = len(positions) if positions else 0
        
        margin_level = account_info.margin_level if account_info.margin > 0 else float('inf')
        margin_display = f"{margin_level:.2f}%" if margin_level != float('inf') else "N/A (no positions)"
        
        logger.info("\n" + "=" * 80)
        logger.info(f"SCANNING MARKETS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Balance: ${account_info.balance:,.2f} | Equity: ${account_info.equity:,.2f}")
        logger.info(f"Margin Level: {margin_display} | Open Positions: {open_positions_count} | Daily Trades: {self.daily_trades}")
        logger.info("=" * 80)
        
        # Safety checks - Margin level instead of position count
        margin_level = account_info.margin_level if account_info.margin > 0 else float('inf')
        min_margin_level = self.config.get('min_margin_level', 800)  # Default 800%
        
        if margin_level < min_margin_level and account_info.margin > 0:
            logger.info(f"Margin level too low ({margin_level:.2f}% < {min_margin_level}%)")
            logger.info(f"Current: Equity ${account_info.equity:,.2f} | Margin ${account_info.margin:,.2f}")
            return
        
        if self.daily_trades >= self.config['max_daily_trades']:
            logger.info(f"Max daily trades reached ({self.config['max_daily_trades']})")
            return
        
        daily_pnl_pct = ((account_info.balance - self.start_balance) / self.start_balance) * 100
        if daily_pnl_pct <= -self.config['max_daily_loss'] * 100:
            logger.info(f"Daily loss limit reached ({daily_pnl_pct:.2f}%)")
            return
        
        # Get timeframe
        timeframe_str = self.config.get('timeframe', 'H1')
        timeframe = self.TIMEFRAME_MAP.get(timeframe_str, mt5.TIMEFRAME_H1)
        
        # Scan each symbol
        signals_found = 0
        signals_approved = 0
        
        for symbol in self.config['symbols']:
            # Skip if already have position
            if positions and any(pos.symbol == symbol for pos in positions):
                continue
            
            # Analyze with MT5 indicators
            action, confidence, details = self.analyzer.analyze(symbol, timeframe)
            
            # Log analysis in debug mode
            if self.config.get('debug_mode') and details:
                logger.info(f"\n[{details.get('strategy', 'UNKNOWN')}] {symbol}:")
                if action:
                    logger.info(f"  Signal: {action} ({confidence:.1f}%)")
                    if 'signals' in details:
                        for reason, score in details['signals']:
                            logger.info(f"    - {reason} ({score} points)")
                else:
                    logger.info(f"  No signal")
                
                # Log key indicators
                if 'rsi' in details:
                    logger.info(f"  RSI: {details['rsi']:.2f}")
                if 'macd' in details:
                    logger.info(f"  MACD: {details['macd']:.5f}")
                if 'trend' in details:
                    logger.info(f"  Trend: {details['trend']}")
            
            if action is None:
                continue
            
            # Check if confidence meets minimum
            if confidence < self.config.get('min_technical_confidence', 50):
                logger.info(f"{symbol}: Signal too weak ({confidence:.1f}% < {self.config.get('min_technical_confidence', 50)}%)")
                continue
            
            signals_found += 1
            logger.info(f"\n>>> {symbol}: {action} signal ({confidence:.1f}%)")
            logger.info(f"    Strategy: {details.get('strategy', 'N/A')}")
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                continue
            
            # Get symbol-specific parameters
            params = self.get_symbol_parameters(symbol)
            
            # Prepare signal for LLM
            signal_data = {
                'symbol': symbol,
                'action': action,
                'confidence': confidence,
                'strategy': details.get('strategy', 'N/A'),
                'timeframe': timeframe_str,
                
                # Technical indicators
                'rsi': details.get('rsi'),
                'macd': details.get('macd'),
                'stochastic_k': details.get('stoch_k'),
                'ema_20': details.get('ema_20'),
                'adx': details.get('adx'),
                
                # Signal details
                'signals': details.get('signals', []),
                
                # Risk parameters
                'balance': account_info.balance,
                'risk_percent': self.config['risk_percent'],
                'lot_size': self.calculate_position_size(account_info.balance, symbol, params['stop_loss_pips']),
                'stop_loss': params['stop_loss_pips'],
                'take_profit': params['take_profit_pips'],
                'risk_reward': params['risk_reward'],
                'spread': (tick.ask - tick.bid) / mt5.symbol_info(symbol).point / 10,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
            
            # Get LLM consensus
            logger.info("Sending signal to LLM agents for approval...")
            approved, consensus = self.llm_engine.get_consensus(
                signal_data,
                self.config.get('min_llm_confidence', 50)
            )
            
            if approved:
                signals_approved += 1
                logger.info(f"[LLM APPROVED] {symbol} {action}")
                self.execute_trade(symbol, action, tick, confidence, consensus, params, details)
            else:
                logger.info(f"[LLM REJECTED] {symbol} {action}")
        
        logger.info("\n" + "=" * 80)
        logger.info(f"SCAN COMPLETE: Technical Signals {signals_found} | LLM Approved {signals_approved}")
        logger.info("=" * 80)
    
    def get_symbol_parameters(self, symbol: str) -> Dict:
        """Get symbol-specific trading parameters based on timeframe"""
        timeframe = self.config.get('timeframe', 'H1').upper()
        
        # Major pairs - Conservative Scalping
        if symbol in ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD']:
            if timeframe in ['M1', 'M5']:
                return {'stop_loss_pips': 20, 'take_profit_pips': 40, 'risk_reward': 2.0}
            elif timeframe in ['M15', 'M30']:
                return {'stop_loss_pips': 25, 'take_profit_pips': 50, 'risk_reward': 2.0}
            else:  # H1, H4, D1
                return {'stop_loss_pips': 30, 'take_profit_pips': 60, 'risk_reward': 2.0}
        
        # Cross pairs - Conservative Scalping
        elif symbol in ['EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY', 'EURAUD']:
            if timeframe in ['M1', 'M5']:
                return {'stop_loss_pips': 30, 'take_profit_pips': 60, 'risk_reward': 2.0}
            elif timeframe in ['M15', 'M30']:
                return {'stop_loss_pips': 35, 'take_profit_pips': 70, 'risk_reward': 2.0}
            else:
                return {'stop_loss_pips': 40, 'take_profit_pips': 80, 'risk_reward': 2.0}
        
        # Gold - Conservative Scalping
        elif symbol in ['XAUUSD', 'GOLD']:
            if timeframe in ['M1', 'M5']:
                return {'stop_loss_pips': 30, 'take_profit_pips': 60, 'risk_reward': 2.0}
            elif timeframe in ['M15', 'M30']:
                return {'stop_loss_pips': 40, 'take_profit_pips': 80, 'risk_reward': 2.0}
            else:
                return {'stop_loss_pips': 50, 'take_profit_pips': 100, 'risk_reward': 2.0}
        
        # Crypto - Conservative Scalping
        elif symbol in ['BTCUSD', 'ETHUSD']:
            if timeframe in ['M1', 'M5']:
                return {'stop_loss_pips': 80, 'take_profit_pips': 160, 'risk_reward': 2.0}
            else:
                return {'stop_loss_pips': 100, 'take_profit_pips': 200, 'risk_reward': 2.0}
        
        # Default - Conservative Scalping
        else:
            if timeframe in ['M1', 'M5']:
                return {'stop_loss_pips': 20, 'take_profit_pips': 40, 'risk_reward': 2.0}
            else:
                return {'stop_loss_pips': 30, 'take_profit_pips': 60, 'risk_reward': 2.0}
    
    def calculate_position_size(self, balance: float, symbol: str, stop_loss_pips: int) -> float:
        """Calculate position size based on risk percentage"""
        risk_amount = balance * self.config['risk_percent']
        symbol_info = mt5.symbol_info(symbol)
        
        if symbol_info is None:
            return 0.01
        
        pip_value = symbol_info.trade_tick_value
        if 'JPY' in symbol:
            pip_value *= 100
        
        lots = risk_amount / (stop_loss_pips * pip_value)
        lots = round(lots / symbol_info.volume_step) * symbol_info.volume_step
        lots = max(symbol_info.volume_min, min(lots, symbol_info.volume_max))
        
        return lots
    
    def execute_trade(self, symbol: str, action: str, tick, technical_confidence: float,
                     llm_consensus: Dict, params: Dict, details: Dict):
        """Execute approved trade"""
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Cannot get symbol info for {symbol}")
            return
        
        price = tick.ask if action == 'BUY' else tick.bid
        
        # Calculate pip size based on symbol type
        if 'JPY' in symbol:
            pip_size = 0.01  # JPY pairs: 1 pip = 0.01
        elif 'XAU' in symbol or 'GOLD' in symbol:
            pip_size = 0.10  # Gold: 1 pip = 0.10
        else:
            pip_size = 0.0001  # Most forex pairs: 1 pip = 0.0001
        
        # Calculate SL/TP
        if action == 'BUY':
            sl = price - (params['stop_loss_pips'] * pip_size)
            tp = price + (params['take_profit_pips'] * pip_size)
            order_type = mt5.ORDER_TYPE_BUY
        else:
            sl = price + (params['stop_loss_pips'] * pip_size)
            tp = price - (params['take_profit_pips'] * pip_size)
            order_type = mt5.ORDER_TYPE_SELL
        
        lots = self.calculate_position_size(mt5.account_info().balance, symbol, params['stop_loss_pips'])
        
        # Prepare order
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lots,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 234000,
            "comment": f"MT5_LLM_{llm_consensus['avg_confidence']:.0f}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode} - {result.comment}")
            return
        
        self.daily_trades += 1
        
        # Log trade
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': action,
            'lots': lots,
            'price': price,
            'sl': sl,
            'tp': tp,
            'strategy': details.get('strategy', 'N/A'),
            'technical_confidence': technical_confidence,
            'llm_confidence': llm_consensus['avg_confidence'],
            'deepseek_decision': llm_consensus['deepseek'],
            'openai_decision': llm_consensus['openai'],
        }
        
        self.trade_history.append(trade_record)
        self.save_trade_history()
        
        logger.info("\n" + "=" * 80)
        logger.info("[TRADE EXECUTED]")
        logger.info(f"Symbol: {symbol} | Action: {action}")
        logger.info(f"Strategy: {details.get('strategy', 'N/A')}")
        logger.info(f"Lots: {lots:.2f} | Price: {price:.5f}")
        logger.info(f"SL: {sl:.5f} | TP: {tp:.5f}")
        logger.info(f"Technical: {technical_confidence:.1f}%")
        logger.info(f"LLM Consensus: {llm_consensus['avg_confidence']:.1f}%")
        logger.info("=" * 80)
    
    def save_trade_history(self):
        """Save trade history to file"""
        with open('trade_history.json', 'w') as f:
            json.dump(self.trade_history, f, indent=2)
    
    def run(self):
        """Main bot loop"""
        logger.info("\n>>> Bot started - MT5 indicators + LLM analysis active")
        logger.info(f"Technical threshold: {self.config.get('min_technical_confidence', 50)}%")
        logger.info(f"LLM threshold: {self.config.get('min_llm_confidence', 50)}%")
        logger.info(f"Scan interval: {self.config['check_interval']} seconds\n")
        
        try:
            while True:
                self.scan_markets()
                logger.info(f"\n>>> Next scan in {self.config['check_interval']} seconds...\n")
                time.sleep(self.config['check_interval'])
        except KeyboardInterrupt:
            logger.info("\n>>> Bot stopped by user")
        finally:
            mt5.shutdown()


if __name__ == "__main__":
    bot = MT5TradingBot(CONFIG)
    bot.run()

