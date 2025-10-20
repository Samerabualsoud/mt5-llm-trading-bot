"""
LLM Consensus Engine
====================

Uses DeepSeek and OpenAI to validate trading signals.
Both LLMs must approve for a trade to execute.
"""

import requests
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class LLMConsensusEngine:
    """Get consensus from multiple LLM agents"""
    
    def __init__(self, deepseek_key: str, openai_key: str):
        self.deepseek_key = deepseek_key
        self.openai_key = openai_key
        self.deepseek_url = "https://api.deepseek.com/v1/chat/completions"
        self.openai_url = "https://api.openai.com/v1/chat/completions"
    
    def call_deepseek(self, prompt: str) -> Dict:
        """Call DeepSeek API"""
        try:
            response = requests.post(
                self.deepseek_url,
                headers={"Authorization": f"Bearer {self.deepseek_key}"},
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                logger.error(f"DeepSeek API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"DeepSeek error: {e}")
            return None
    
    def call_openai(self, prompt: str) -> Dict:
        """Call OpenAI API"""
        try:
            response = requests.post(
                self.openai_url,
                headers={"Authorization": f"Bearer {self.openai_key}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                logger.error(f"OpenAI API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return None
    
    def parse_llm_response(self, response: str) -> Tuple[str, int, str]:
        """Parse LLM response to extract decision, confidence, and reasoning"""
        if not response:
            return 'REJECT', 0, 'No response from LLM'
        
        response_lower = response.lower()
        
        # Extract decision
        if 'approve' in response_lower or 'execute' in response_lower or 'yes' in response_lower:
            decision = 'APPROVE'
        else:
            decision = 'REJECT'
        
        # Extract confidence (look for percentage)
        confidence = 50  # default
        import re
        conf_match = re.search(r'(\d+)%', response)
        if conf_match:
            confidence = int(conf_match.group(1))
        
        # Extract reasoning (first sentence or paragraph)
        reasoning = response[:200] + '...' if len(response) > 200 else response
        
        return decision, confidence, reasoning
    
    def create_prompt(self, signal_data: Dict) -> str:
        """Create prompt for LLM analysis"""
        prompt = f"""You are a professional forex trading analyst. Analyze this trading signal and decide whether to APPROVE or REJECT it.

SIGNAL DETAILS:
Symbol: {signal_data['symbol']}
Action: {signal_data['action']}
Strategy: {signal_data.get('strategy', 'N/A')}
Technical Confidence: {signal_data['confidence']:.1f}%

TECHNICAL INDICATORS:
{self._format_indicators(signal_data)}

RISK PARAMETERS:
Account Balance: ${signal_data['balance']:,.2f}
Risk Per Trade: {signal_data['risk_percent']*100:.1f}%
Position Size: {signal_data['lot_size']:.2f} lots
Stop Loss: {signal_data['stop_loss']} pips
Take Profit: {signal_data['take_profit']} pips
Risk/Reward: {signal_data['risk_reward']}:1

MARKET CONDITIONS:
Spread: {signal_data.get('spread', 'N/A')} pips
Timestamp: {signal_data['timestamp']}

ANALYSIS REQUIRED:
1. Evaluate the technical setup quality
2. Assess risk/reward ratio
3. Consider market conditions
4. Check if timing is appropriate
5. Decide: APPROVE or REJECT

Respond in this format:
Decision: [APPROVE/REJECT]
Confidence: [0-100]%
Reasoning: [Your analysis in 2-3 sentences]
"""
        return prompt
    
    def _format_indicators(self, signal_data: Dict) -> str:
        """Format indicator data for prompt"""
        lines = []
        
        # Common indicators
        if 'rsi' in signal_data:
            lines.append(f"RSI: {signal_data['rsi']:.2f}")
        if 'macd' in signal_data:
            lines.append(f"MACD: {signal_data['macd']:.5f}")
        if 'stochastic_k' in signal_data:
            lines.append(f"Stochastic K: {signal_data['stochastic_k']:.2f}")
        if 'ema_20' in signal_data:
            lines.append(f"EMA20: {signal_data['ema_20']:.5f}")
        if 'adx' in signal_data:
            lines.append(f"ADX: {signal_data['adx']:.2f}")
        
        # Signal details
        if 'signals' in signal_data and signal_data['signals']:
            lines.append("\nSignal Reasons:")
            for reason, score in signal_data['signals']:
                lines.append(f"  - {reason} ({score} points)")
        
        return '\n'.join(lines) if lines else 'No detailed indicators available'
    
    def get_consensus(self, signal_data: Dict, min_confidence: int = 50) -> Tuple[bool, Dict]:
        """
        Get consensus from both LLMs
        
        Returns:
            Tuple of (approved, consensus_data)
        """
        prompt = self.create_prompt(signal_data)
        
        logger.info(f"\nLLM CONSENSUS ANALYSIS: {signal_data['symbol']} {signal_data['action']}")
        logger.info("=" * 70)
        
        # Get DeepSeek analysis
        deepseek_response = self.call_deepseek(prompt)
        deepseek_decision, deepseek_conf, deepseek_reasoning = self.parse_llm_response(deepseek_response)
        
        logger.info(f"[DeepSeek Analyst] Decision: {deepseek_decision} | Confidence: {deepseek_conf}%")
        
        # Get OpenAI analysis
        openai_response = self.call_openai(prompt)
        openai_decision, openai_conf, openai_reasoning = self.parse_llm_response(openai_response)
        
        logger.info(f"[OpenAI Strategist] Decision: {openai_decision} | Confidence: {openai_conf}%")
        logger.info("")
        
        # Calculate consensus
        avg_confidence = (deepseek_conf + openai_conf) / 2
        
        # Both must approve
        both_approve = (deepseek_decision == 'APPROVE' and openai_decision == 'APPROVE')
        
        # Or one strong approve (>75% confidence)
        one_strong_approve = (
            (deepseek_decision == 'APPROVE' and deepseek_conf >= 75) or
            (openai_decision == 'APPROVE' and openai_conf >= 75)
        )
        
        approved = (both_approve or one_strong_approve) and avg_confidence >= min_confidence
        
        consensus_data = {
            'deepseek': {
                'decision': deepseek_decision,
                'confidence': deepseek_conf,
                'reasoning': deepseek_reasoning
            },
            'openai': {
                'decision': openai_decision,
                'confidence': openai_conf,
                'reasoning': openai_reasoning
            },
            'avg_confidence': avg_confidence,
            'approved': approved
        }
        
        status = "[APPROVED]" if approved else "[REJECTED]"
        logger.info(f"{status} Consensus: {avg_confidence:.1f}%")
        logger.info("=" * 70)
        
        return approved, consensus_data

