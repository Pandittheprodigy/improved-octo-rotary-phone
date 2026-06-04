
# ============================================================
# FILE 2: utils/data_fetcher.py — Enhanced with all metrics for the new UI
# ============================================================

data_fetcher_code = '''"""
Data Fetching & Presentation Layer
Real-time market data, technical indicators, and news aggregation.
Enhanced for the professional dashboard UI.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


class StockDataFetcher:
    """Professional-grade data extraction and cleaning engine."""
    
    def __init__(self):
        self.cache = {}
    
    def get_stock_data(self, ticker: str, period: str = "1y") -> Dict:
        """Fetch comprehensive stock data with technical indicators."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            info = stock.info
            
            if hist.empty:
                return {"error": f"No data found for {ticker}"}
            
            # Calculate technical indicators
            hist = self._calculate_indicators(hist)
            
            # Price change calculations
            current_price = round(hist["Close"].iloc[-1], 2)
            prev_close = round(hist["Close"].iloc[-2], 2) if len(hist) > 1 else current_price
            price_change = round(current_price - prev_close, 2)
            price_change_pct = round((price_change / prev_close) * 100, 2) if prev_close else 0
            
            # Market cap formatting
            raw_market_cap = info.get("marketCap")
            market_cap_str = self._format_market_cap(raw_market_cap)
            
            # Key metrics
            metrics = {
                "ticker": ticker.upper(),
                "company_name": info.get("longName", ticker.upper()),
                "current_price": current_price,
                "previous_close": prev_close,
                "price_change": price_change,
                "price_change_percent": price_change_pct,
                "day_high": round(hist["High"].iloc[-1], 2),
                "day_low": round(hist["Low"].iloc[-1], 2),
                "volume": int(hist["Volume"].iloc[-1]),
                "avg_volume_20d": int(hist["Volume"].tail(20).mean()),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
                "market_cap": raw_market_cap,
                "market_cap_formatted": market_cap_str,
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "pb_ratio": info.get("priceToBook"),
                "peg_ratio": info.get("pegRatio"),
                "dividend_yield": info.get("dividendYield"),
                "beta": info.get("beta"),
                "eps": info.get("trailingEps"),
                "revenue_growth": info.get("revenueGrowth"),
                "profit_margins": info.get("profitMargins"),
                "debt_to_equity": info.get("debtToEquity"),
                "return_on_equity": info.get("returnOnEquity"),
                "return_on_assets": info.get("returnOnAssets"),
                "current_ratio": info.get("currentRatio"),
                "quick_ratio": info.get("quickRatio"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "business_summary": info.get("longBusinessSummary", "")[:500],
                "employees": info.get("fullTimeEmployees"),
                "website": info.get("website"),
            }
            
            # Technical snapshot
            latest = hist.iloc[-1]
            technical = {
                "sma_20": round(latest.get("SMA_20", 0), 2),
                "sma_50": round(latest.get("SMA_50", 0), 2),
                "sma_200": round(latest.get("SMA_200", 0), 2),
                "ema_12": round(latest.get("EMA_12", 0), 2),
                "ema_26": round(latest.get("EMA_26", 0), 2),
                "rsi_14": round(latest.get("RSI_14", 0), 2),
                "macd": round(latest.get("MACD", 0), 4),
                "macd_signal": round(latest.get("MACD_Signal", 0), 4),
                "macd_hist": round(latest.get("MACD_Hist", 0), 4),
                "bb_upper": round(latest.get("BB_Upper", 0), 2),
                "bb_lower": round(latest.get("BB_Lower", 0), 2),
                "bb_middle": round(latest.get("BB_Middle", 0), 2),
                "atr_14": round(latest.get("ATR_14", 0), 2),
                "stoch_k": round(latest.get("Stoch_K", 0), 2),
                "stoch_d": round(latest.get("Stoch_D", 0), 2),
                "adx_14": round(latest.get("ADX_14", 0), 2),
                "volume_sma_20": round(latest.get("Volume_SMA_20", 0), 0),
                "price_vs_sma20_pct": round((latest["Close"] / latest.get("SMA_20", latest["Close"]) - 1) * 100, 2),
                "price_vs_sma50_pct": round((latest["Close"] / latest.get("SMA_50", latest["Close"]) - 1) * 100, 2),
                "price_vs_sma200_pct": round((latest["Close"] / latest.get("SMA_200", latest["Close"]) - 1) * 100, 2),
            }
            
            # Determine trend signal
            trend = "NEUTRAL"
            if latest["Close"] > latest.get("SMA_50", 0) > latest.get("SMA_200", 0):
                trend = "BULLISH"
            elif latest["Close"] < latest.get("SMA_50", 0) < latest.get("SMA_200", 0):
                trend = "BEARISH"
            
            rsi_signal = "NEUTRAL"
            if technical["rsi_14"] > 70:
                rsi_signal = "OVERBOUGHT"
            elif technical["rsi_14"] < 30:
                rsi_signal = "OVERSOLD"
            
            macd_signal = "BULLISH" if technical["macd_hist"] > 0 else "BEARISH"
            
            volume_trend = "ABOVE_AVG" if metrics["volume"] > metrics["avg_volume_20d"] * 1.2 else "BELOW_AVG" if metrics["volume"] < metrics["avg_volume_20d"] * 0.8 else "NORMAL"
            
            technical_indicators = {
                "trend": trend,
                "rsi_14": technical["rsi_14"],
                "rsi_signal": rsi_signal,
                "macd_line": technical["macd"],
                "macd_signal_line": technical["macd_signal"],
                "macd_histogram": technical["macd_hist"],
                "volume_trend": volume_trend,
            }
            
            # Recent price action (last 30 days)
            recent = hist.tail(30)
            price_action = {
                "30d_high": round(recent["High"].max(), 2),
                "30d_low": round(recent["Low"].min(), 2),
                "30d_return_pct": round((recent["Close"].iloc[-1] / recent["Close"].iloc[0] - 1) * 100, 2),
                "volatility_30d": round(recent["Close"].pct_change().std() * np.sqrt(252) * 100, 2),
                "consecutive_up_days": self._count_consecutive(recent["Close"].values, "up"),
                "consecutive_down_days": self._count_consecutive(recent["Close"].values, "down"),
            }
            
            return {
                "metrics": metrics,
                "technical": technical,
                "technical_indicators": technical_indicators,
                "price_action": price_action,
                "history": hist,
                "info": info,
                "current_price": current_price,
                "price_change": price_change,
                "price_change_percent": price_change_pct,
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": raw_market_cap,
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "price_to_book": info.get("priceToBook"),
                "peg_ratio": info.get("pegRatio"),
                "return_on_equity": info.get("returnOnEquity"),
                "return_on_assets": info.get("returnOnAssets"),
                "profit_margins": info.get("profitMargins"),
                "revenue_growth": info.get("revenueGrowth"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "quick_ratio": info.get("quickRatio"),
                "beta": info.get("beta"),
                "dividend_yield": info.get("dividendYield"),
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _format_market_cap(self, market_cap) -> str:
        """Format market cap into readable string."""
        if not market_cap:
            return "N/A"
        if market_cap >= 1e12:
            return f"${market_cap/1e12:.2f}T"
        elif market_cap >= 1e9:
            return f"${market_cap/1e9:.2f}B"
        elif market_cap >= 1e6:
            return f"${market_cap/1e6:.2f}M"
        else:
            return f"${market_cap:,.0f}"
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators."""
        df = df.copy()
        
        # Simple Moving Averages
        df["SMA_20"] = df["Close"].rolling(20).mean()
        df["SMA_50"] = df["Close"].rolling(50).mean()
        df["SMA_200"] = df["Close"].rolling(200).mean()
        
        # Exponential Moving Averages
        df["EMA_12"] = df["Close"].ewm(span=12).mean()
        df["EMA_26"] = df["Close"].ewm(span=26).mean()
        
        # RSI
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["RSI_14"] = 100 - (100 / (1 + rs))
        
        # MACD
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
        
        # Bollinger Bands
        df["BB_Middle"] = df["Close"].rolling(20).mean()
        bb_std = df["Close"].rolling(20).std()
        df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
        df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)
        
        # ATR
        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - df["Close"].shift())
        low_close = np.abs(df["Low"] - df["Close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR_14"] = tr.rolling(14).mean()
        
        # Stochastic
        low_14 = df["Low"].rolling(14).min()
        high_14 = df["High"].rolling(14).max()
        df["Stoch_K"] = 100 * ((df["Close"] - low_14) / (high_14 - low_14))
        df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()
        
        # Volume SMA
        df["Volume_SMA_20"] = df["Volume"].rolling(20).mean()
        
        # ADX (simplified)
        plus_dm = df["High"].diff()
        minus_dm = df["Low"].diff().abs()
        df["ADX_14"] = (plus_dm.rolling(14).mean() + minus_dm.rolling(14).mean()) / 2
        
        return df
    
    def _count_consecutive(self, prices: np.ndarray, direction: str) -> int:
        """Count consecutive up/down days."""
        if len(prices) < 2:
            return 0
        changes = np.diff(prices)
        if direction == "up":
            mask = changes > 0
        else:
            mask = changes < 0
        
        count = 0
        for i in range(len(mask) - 1, -1, -1):
            if mask[i]:
                count += 1
            else:
                break
        return count
    
    def get_news_summary(self, ticker: str) -> List[Dict]:
        """Fetch recent news for a ticker."""
        try:
            stock = yf.Ticker(ticker)
            news = stock.news[:10] if stock.news else []
            cleaned = []
            for item in news:
                cleaned.append({
                    "title": item.get("title", ""),
                    "publisher": item.get("publisher", ""),
                    "published": datetime.fromtimestamp(item.get("content", {}).get("pubDate", 0)).strftime("%Y-%m-%d %H:%M") if item.get("content") else "",
                    "summary": item.get("summary", "")[:300],
                    "url": item.get("link", ""),
                })
            return cleaned
        except:
            return []
    
    def format_for_agents(self, data: Dict) -> str:
        """Format data into a clean, presentable string for agent consumption."""
        if "error" in data:
            return f"DATA ERROR: {data['error']}"
        
        m = data["metrics"]
        t = data["technical"]
        p = data["price_action"]
        
        return f"""
=== MARKET DATA DASHBOARD: {m['ticker']} ===
Company: {m['company_name']} | Sector: {m['sector']} | Industry: {m['industry']}

--- PRICE & VOLUME ---
Current Price: ${m['current_price']} | Previous Close: ${m['previous_close']}
Change: ${m['price_change']} ({m['price_change_percent']}%)
Day Range: ${m['day_low']} - ${m['day_high']}
Volume: {m['volume']:,} | 20D Avg Volume: {m['avg_volume_20d']:,}
52W Range: ${m['52w_low']} - ${m['52w_high']}

--- TECHNICAL INDICATORS ---
SMA 20: ${t['sma_20']} | SMA 50: ${t['sma_50']} | SMA 200: ${t['sma_200']}
Price vs SMA20: {t['price_vs_sma20_pct']}% | vs SMA50: {t['price_vs_sma50_pct']}% | vs SMA200: {t['price_vs_sma200_pct']}%
RSI(14): {t['rsi_14']} | MACD: {t['macd']} | Signal: {t['macd_signal']}
BB Upper: ${t['bb_upper']} | Middle: ${t['bb_middle']} | Lower: ${t['bb_lower']}
ATR(14): ${t['atr_14']} | Stoch %K: {t['stoch_k']} | Stoch %D: {t['stoch_d']}
ADX(14): {t['adx_14']}

--- PRICE ACTION (30D) ---
30D High: ${p['30d_high']} | 30D Low: ${p['30d_low']}
30D Return: {p['30d_return_pct']}% | Annualized Volatility: {p['volatility_30d']}%
Consecutive Up Days: {p['consecutive_up_days']} | Consecutive Down Days: {p['consecutive_down_days']}

--- FUNDAMENTAL SNAPSHOT ---
Market Cap: {m['market_cap_formatted']}
P/E: {m['pe_ratio']} | Forward P/E: {m['forward_pe']} | P/B: {m['pb_ratio']} | PEG: {m['peg_ratio']}
EPS: ${m['eps']} | Revenue Growth: {m['revenue_growth']}
Profit Margin: {m['profit_margins']} | ROE: {m['return_on_equity']} | ROA: {m['return_on_assets']}
Debt/Equity: {m['debt_to_equity']} | Beta: {m['beta']}
Dividend Yield: {m['dividend_yield']}

--- BUSINESS CONTEXT ---
{m['business_summary'][:400]}...
"""


class DataPresentation:
    """Clean data presentation utilities for Streamlit."""
    
    @staticmethod
    def color_metric(value, threshold_up=0, threshold_down=0, fmt=".2f"):
        """Return colored HTML for a metric."""
        if value is None:
            return "N/A"
        try:
            v = float(value)
            color = "#22c55e" if v > threshold_up else "#ef4444" if v < threshold_down else "#f1f5f9"
            return f'<span style="color:{color};font-weight:bold">{v:{fmt}}</span>'
        except:
            return str(value)
    
    @staticmethod
    def format_trade_card(trade_data: Dict) -> str:
        """Format a trade recommendation into a professional card."""
        direction = trade_data.get("direction", "HOLD")
        color = "#22c55e" if direction == "LONG" else "#ef4444" if direction == "SHORT" else "#f59e0b"
        
        return f"""
<div style="border:2px solid {color};border-radius:12px;padding:20px;background:linear-gradient(135deg,#1a1a2e,#16213e);margin:10px 0">
    <h2 style="color:{color};margin:0">{direction} — {trade_data.get('conviction', 0)}% CONVICTION</h2>
    <hr style="border-color:{color};opacity:0.3">
    <p><strong>Entry Zone:</strong> {trade_data.get('entry_zone', 'N/A')}</p>
    <p><strong>Stop Loss:</strong> <span style="color:#ef4444">{trade_data.get('stop_loss', 'N/A')}</span></p>
    <p><strong>Target 1 (50%):</strong> <span style="color:#22c55e">{trade_data.get('target_1', 'N/A')}</span> | R:R = {trade_data.get('rr_1', 'N/A')}</p>
    <p><strong>Target 2 (30%):</strong> <span style="color:#22c55e">{trade_data.get('target_2', 'N/A')}</span> | R:R = {trade_data.get('rr_2', 'N/A')}</p>
    <p><strong>Target 3 (20% trail):</strong> <span style="color:#22c55e">{trade_data.get('target_3', 'N/A')}</span></p>
    <p><strong>Position Sizing:</strong> {trade_data.get('position_size', 'N/A')}</p>
    <p><strong>Time Horizon:</strong> {trade_data.get('time_horizon', 'N/A')}</p>
    <p><strong>Invalidation:</strong> {trade_data.get('invalidation', 'N/A')}</p>
</div>
"""
'''

with open(f"{output_dir}/utils/data_fetcher.py", "w") as f:
    f.write(data_fetcher_code)

# __init__.py for utils
with open(f"{output_dir}/utils/__init__.py", "w") as f:
    f.write("""from .llm_factory import get_agent_llm, get_all_agent_status, validate_all_keys, NVIDIA_AGENT_KEY_MAP
from .data_fetcher import StockDataFetcher, DataPresentation
""")

print("✅ utils/data_fetcher.py and utils/__init__.py written")
