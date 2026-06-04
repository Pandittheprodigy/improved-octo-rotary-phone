"""
Data Fetching Utilities for Master Trading Crew
Handles stock data, technical indicators, fundamental data, and news
"""
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import feedparser
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from bs4 import BeautifulSoup
import os


class StockDataFetcher:
    """Fetches comprehensive stock market data"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_stock_data(self, symbol: str, period: str = "6mo") -> Dict[str, Any]:
        """Fetch comprehensive stock data including price, technicals, and info"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            info = ticker.info
            
            if hist.empty:
                return {"error": f"No data found for symbol: {symbol}"}
            
            # Calculate technical indicators
            tech_indicators = self._calculate_technical_indicators(hist)
            
            return {
                "symbol": symbol.upper(),
                "company_name": info.get("longName", symbol),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "current_price": round(hist['Close'].iloc[-1], 2),
                "previous_close": round(hist['Close'].iloc[-2], 2) if len(hist) > 1 else None,
                "price_change": round(hist['Close'].iloc[-1] - hist['Close'].iloc[-2], 2) if len(hist) > 1 else 0,
                "price_change_percent": round(
                    ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100), 2
                ) if len(hist) > 1 else 0,
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                "market_cap": info.get("marketCap"),
                "volume": int(hist['Volume'].iloc[-1]),
                "avg_volume": int(hist['Volume'].mean()),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "debt_to_equity": info.get("debtToEquity"),
                "return_on_equity": info.get("returnOnEquity"),
                "return_on_assets": info.get("returnOnAssets"),
                "profit_margins": info.get("profitMargins"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "free_cashflow": info.get("freeCashflow"),
                "operating_cashflow": info.get("operatingCashflow"),
                "total_cash": info.get("totalCash"),
                "total_debt": info.get("totalDebt"),
                "current_ratio": info.get("currentRatio"),
                "quick_ratio": info.get("quickRatio"),
                "beta": info.get("beta"),
                "dividend_yield": info.get("dividendYield"),
                "eps": info.get("trailingEps"),
                "revenue_per_share": info.get("revenuePerShare"),
                "analyst_rating": info.get("recommendationKey"),
                "target_high": info.get("targetHighPrice"),
                "target_low": info.get("targetLowPrice"),
                "target_mean": info.get("targetMeanPrice"),
                "target_median": info.get("targetMedianPrice"),
                "number_of_analysts": info.get("numberOfAnalystOpinions"),
                "short_ratio": info.get("shortRatio"),
                "short_percent_of_float": info.get("shortPercentOfFloat"),
                "technical_indicators": tech_indicators,
                "historical_data": hist.tail(30).to_dict(),
                "price_history_5d": hist['Close'].tail(5).tolist(),
                "price_history_20d": hist['Close'].tail(20).tolist(),
                "volume_history_5d": hist['Volume'].tail(5).tolist(),
                "high_5d": hist['High'].tail(5).tolist(),
                "low_5d": hist['Low'].tail(5).tolist(),
            }
        except Exception as e:
            return {"error": f"Error fetching data for {symbol}: {str(e)}"}
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive technical indicators"""
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # Moving Averages
        sma_20 = close.rolling(window=20).mean()
        sma_50 = close.rolling(window=50).mean()
        sma_200 = close.rolling(window=200).mean()
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        
        # MACD
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_middle = sma_20
        bb_std = close.rolling(window=20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        
        # Stochastic Oscillator
        lowest_low = low.rolling(window=14).min()
        highest_high = high.rolling(window=14).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=3).mean()
        
        # ATR (Average True Range)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        
        # OBV (On Balance Volume)
        obv = (np.sign(close.diff()) * volume).cumsum()
        
        # VWAP
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        # Momentum
        momentum_10 = close - close.shift(10)
        momentum_14 = close - close.shift(14)
        
        # CCI (Commodity Channel Index)
        tp = (high + low + close) / 3
        cci = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std())
        
        # Williams %R
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        # ADX
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        atr_14 = tr.rolling(window=14).mean()
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr_14)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=14).mean()
        
        return {
            "sma_20": round(sma_20.iloc[-1], 2) if not pd.isna(sma_20.iloc[-1]) else None,
            "sma_50": round(sma_50.iloc[-1], 2) if not pd.isna(sma_50.iloc[-1]) else None,
            "sma_200": round(sma_200.iloc[-1], 2) if not pd.isna(sma_200.iloc[-1]) else None,
            "ema_12": round(ema_12.iloc[-1], 2),
            "ema_26": round(ema_26.iloc[-1], 2),
            "macd_line": round(macd_line.iloc[-1], 4),
            "macd_signal": round(signal_line.iloc[-1], 4),
            "macd_histogram": round(histogram.iloc[-1], 4),
            "rsi_14": round(rsi.iloc[-1], 2),
            "rsi_signal": "Overbought" if rsi.iloc[-1] > 70 else "Oversold" if rsi.iloc[-1] < 30 else "Neutral",
            "bb_upper": round(bb_upper.iloc[-1], 2),
            "bb_middle": round(bb_middle.iloc[-1], 2),
            "bb_lower": round(bb_lower.iloc[-1], 2),
            "bb_position": round((close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]), 2),
            "stoch_k": round(k_percent.iloc[-1], 2),
            "stoch_d": round(d_percent.iloc[-1], 2),
            "atr_14": round(atr.iloc[-1], 2),
            "obv": int(obv.iloc[-1]),
            "vwap": round(vwap.iloc[-1], 2),
            "momentum_10": round(momentum_10.iloc[-1], 2),
            "momentum_14": round(momentum_14.iloc[-1], 2),
            "cci_20": round(cci.iloc[-1], 2),
            "williams_r": round(williams_r.iloc[-1], 2),
            "adx": round(adx.iloc[-1], 2) if not pd.isna(adx.iloc[-1]) else None,
            "plus_di": round(plus_di.iloc[-1], 2) if not pd.isna(plus_di.iloc[-1]) else None,
            "minus_di": round(minus_di.iloc[-1], 2) if not pd.isna(minus_di.iloc[-1]) else None,
            "trend": "Bullish" if close.iloc[-1] > sma_50.iloc[-1] > sma_200.iloc[-1] else 
                     "Bearish" if close.iloc[-1] < sma_50.iloc[-1] < sma_200.iloc[-1] else "Mixed",
            "volume_trend": "Above Average" if volume.iloc[-1] > volume.mean() * 1.5 else 
                           "Below Average" if volume.iloc[-1] < volume.mean() * 0.5 else "Normal",
        }
    
    def get_sector_performance(self, sector: str) -> Dict[str, Any]:
        """Get sector ETF performance for relative analysis"""
        sector_etfs = {
            "technology": "XLK",
            "healthcare": "XLV",
            "financial": "XLF",
            "consumer_discretionary": "XLY",
            "consumer_staples": "XLP",
            "industrial": "XLI",
            "energy": "XLE",
            "utilities": "XLU",
            "real_estate": "XLRE",
            "materials": "XLB",
            "communication": "XLC"
        }
        
        try:
            etf_symbol = sector_etfs.get(sector.lower(), "SPY")
            etf = yf.Ticker(etf_symbol)
            hist = etf.history(period="1mo")
            
            return {
                "sector_etf": etf_symbol,
                "sector_return_1m": round(
                    (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100, 2
                ),
                "sector_avg_volume": int(hist['Volume'].mean()),
            }
        except:
            return {}
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get broad market summary"""
        indices = {
            "S&P 500": "^GSPC",
            "Dow Jones": "^DJI",
            "NASDAQ": "^IXIC",
            "Russell 2000": "^RUT",
            "VIX": "^VIX"
        }
        
        summary = {}
        for name, symbol in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                if not hist.empty:
                    summary[name] = {
                        "current": round(hist['Close'].iloc[-1], 2),
                        "change": round(
                            (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100, 2
                        ),
                    }
            except:
                continue
        
        return summary


class NewsDataFetcher:
    """Fetches news and sentiment-related data"""
    
    def __init__(self):
        self.newsapi_key = os.getenv("NEWSAPI_KEY", "")
    
    def get_stock_news(self, symbol: str, company_name: str = "") -> List[Dict[str, Any]]:
        """Fetch recent news for a stock"""
        news_items = []
        
        # Try Yahoo Finance news first (most reliable)
        try:
            ticker = yf.Ticker(symbol)
            yf_news = ticker.news
            
            if yf_news:
                for item in yf_news[:10]:
                    news_items.append({
                        "title": item.get("title", ""),
                        "publisher": item.get("publisher", ""),
                        "published": datetime.fromtimestamp(
                            item.get("published", 0)
                        ).strftime("%Y-%m-%d %H:%M") if item.get("published") else "",
                        "link": item.get("link", ""),
                        "summary": item.get("summary", ""),
                        "source": "Yahoo Finance"
                    })
        except Exception as e:
            print(f"Yahoo Finance news error: {e}")
        
        # Try RSS feeds as backup
        try:
            rss_feeds = [
                f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US",
            ]
            
            for feed_url in rss_feeds:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:5]:
                    news_items.append({
                        "title": entry.get("title", ""),
                        "publisher": feed.feed.get("title", "RSS"),
                        "published": entry.get("published", ""),
                        "link": entry.get("link", ""),
                        "summary": entry.get("summary", ""),
                        "source": "RSS Feed"
                    })
        except Exception as e:
            print(f"RSS news error: {e}")
        
        return news_items
    
    def get_global_news(self) -> List[Dict[str, Any]]:
        """Fetch global macro news"""
        global_topics = [
            "Federal Reserve interest rates",
            "inflation economic outlook",
            "geopolitical tensions markets",
            "global trade tariffs"
        ]
        
        all_news = []
        
        # Use RSS for global financial news
        rss_sources = [
            ("https://www.reutersagency.com/feed/?taxonomy=markets&post_type=reuters-best", "Reuters Markets"),
            ("https://feeds.bbci.co.uk/news/business/rss.xml", "BBC Business"),
        ]
        
        for url, source in rss_sources:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:3]:
                    all_news.append({
                        "title": entry.get("title", ""),
                        "source": source,
                        "published": entry.get("published", ""),
                        "link": entry.get("link", ""),
                        "summary": entry.get("summary", "")[:200],
                    })
            except:
                continue
        
        return all_news
    
    def get_earnings_calendar(self, symbol: str) -> Dict[str, Any]:
        """Get upcoming earnings data"""
        try:
            ticker = yf.Ticker(symbol)
            calendar = ticker.calendar
            
            if calendar is not None and not calendar.empty:
                return {
                    "earnings_date": calendar.index[0].strftime("%Y-%m-%d") if hasattr(calendar.index[0], 'strftime') else str(calendar.index[0]),
                    "eps_estimate": calendar.iloc[0].get("Earnings Estimate", "N/A") if len(calendar.columns) > 0 else "N/A",
                }
            return {}
        except:
            return {}


class DataPresentation:
    """Cleans and formats data for analysis"""
    
    @staticmethod
    def format_stock_report(data: Dict[str, Any]) -> str:
        """Create a formatted text report from stock data"""
        if "error" in data:
            return f"Error: {data['error']}"
        
        report = f"""
=== {data['company_name']} ({data['symbol']}) - COMPREHENSIVE DATA REPORT ===

--- PRICE INFORMATION ---
Current Price: ${data['current_price']}
Previous Close: ${data['previous_close']}
Daily Change: ${data['price_change']} ({data['price_change_percent']}%)
52-Week Range: ${data['fifty_two_week_low']} - ${data['fifty_two_week_high']}

--- FUNDAMENTAL METRICS ---
Market Cap: {DataPresentation._format_number(data.get('market_cap'))}
P/E Ratio (TTM): {data.get('pe_ratio', 'N/A')}
Forward P/E: {data.get('forward_pe', 'N/A')}
PEG Ratio: {data.get('peg_ratio', 'N/A')}
Price-to-Book: {data.get('price_to_book', 'N/A')}
Debt-to-Equity: {data.get('debt_to_equity', 'N/A')}
Return on Equity: {DataPresentation._format_percent(data.get('return_on_equity'))}
Return on Assets: {DataPresentation._format_percent(data.get('return_on_assets'))}
Profit Margins: {DataPresentation._format_percent(data.get('profit_margins'))}
Revenue Growth: {DataPresentation._format_percent(data.get('revenue_growth'))}
Earnings Growth: {DataPresentation._format_percent(data.get('earnings_growth'))}
EPS (TTM): {data.get('eps', 'N/A')}
Beta: {data.get('beta', 'N/A')}

--- TECHNICAL INDICATORS ---
Trend: {data['technical_indicators'].get('trend', 'N/A')}
RSI (14): {data['technical_indicators'].get('rsi_14', 'N/A')} ({data['technical_indicators'].get('rsi_signal', 'N/A')})
MACD: {data['technical_indicators'].get('macd_line', 'N/A')}
MACD Signal: {data['technical_indicators'].get('macd_signal', 'N/A')}
MACD Histogram: {data['technical_indicators'].get('macd_histogram', 'N/A')}
SMA 20: {data['technical_indicators'].get('sma_20', 'N/A')}
SMA 50: {data['technical_indicators'].get('sma_50', 'N/A')}
SMA 200: {data['technical_indicators'].get('sma_200', 'N/A')}
Bollinger Upper: {data['technical_indicators'].get('bb_upper', 'N/A')}
Bollinger Lower: {data['technical_indicators'].get('bb_lower', 'N/A')}
BB Position: {data['technical_indicators'].get('bb_position', 'N/A')}
ATR (14): {data['technical_indicators'].get('atr_14', 'N/A')}
ADX: {data['technical_indicators'].get('adx', 'N/A')}
Stochastic %K: {data['technical_indicators'].get('stoch_k', 'N/A')}
Stochastic %D: {data['technical_indicators'].get('stoch_d', 'N/A')}
CCI (20): {data['technical_indicators'].get('cci_20', 'N/A')}
Williams %R: {data['technical_indicators'].get('williams_r', 'N/A')}
Volume Trend: {data['technical_indicators'].get('volume_trend', 'N/A')}
VWAP: {data['technical_indicators'].get('vwap', 'N/A')}

--- ANALYST ESTIMATES ---
Consensus Rating: {data.get('analyst_rating', 'N/A')}
Target Price (High): ${data.get('target_high', 'N/A')}
Target Price (Low): ${data.get('target_low', 'N/A')}
Target Price (Mean): ${data.get('target_mean', 'N/A')}
Target Price (Median): ${data.get('target_median', 'N/A')}
Number of Analysts: {data.get('number_of_analysts', 'N/A')}

--- RECENT PRICE HISTORY (Last 5 Days) ---
{data.get('price_history_5d', [])}

--- RECENT VOLUME (Last 5 Days) ---
{data.get('volume_history_5d', [])}
"""
        return report
    
    @staticmethod
    def format_news_report(news_items: List[Dict[str, Any]]) -> str:
        """Format news items for analysis"""
        if not news_items:
            return "No recent news available."
        
        report = "=== RECENT NEWS & DEVELOPMENTS ===\n\n"
        for i, item in enumerate(news_items[:15], 1):
            report += f"--- Article {i} ---\n"
            report += f"Title: {item.get('title', 'N/A')}\n"
            report += f"Source: {item.get('publisher', item.get('source', 'N/A'))}\n"
            report += f"Published: {item.get('published', 'N/A')}\n"
            report += f"Summary: {item.get('summary', 'N/A')[:300]}\n\n"
        
        return report
    
    @staticmethod
    def format_global_news(news_items: List[Dict[str, Any]]) -> str:
        """Format global macro news"""
        if not news_items:
            return "No global news available."
        
        report = "=== GLOBAL MACRO & MARKET NEWS ===\n\n"
        for i, item in enumerate(news_items[:10], 1):
            report += f"--- Global Update {i} ---\n"
            report += f"Title: {item.get('title', 'N/A')}\n"
            report += f"Source: {item.get('source', 'N/A')}\n"
            report += f"Published: {item.get('published', 'N/A')}\n"
            report += f"Summary: {item.get('summary', 'N/A')[:250]}\n\n"
        
        return report
    
    @staticmethod
    def _format_number(value) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, (int, float)):
            if value >= 1e12:
                return f"${value/1e12:.2f}T"
            elif value >= 1e9:
                return f"${value/1e9:.2f}B"
            elif value >= 1e6:
                return f"${value/1e6:.2f}M"
            else:
                return f"${value:,.2f}"
        return str(value)
    
    @staticmethod
    def _format_percent(value) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, (int, float)):
            return f"{value*100:.2f}%" if abs(value) < 1 else f"{value:.2f}%"
        return str(value)


# Convenience functions
def fetch_all_data(symbol: str) -> Dict[str, Any]:
    """Fetch all data for a symbol in one call"""
    stock_fetcher = StockDataFetcher()
    news_fetcher = NewsDataFetcher()
    
    # Fetch all data
    stock_data = stock_fetcher.get_stock_data(symbol)
    stock_news = news_fetcher.get_stock_news(symbol, stock_data.get("company_name", ""))
    global_news = news_fetcher.get_global_news()
    market_summary = stock_fetcher.get_market_summary()
    earnings = news_fetcher.get_earnings_calendar(symbol)
    
    # Format for agents
    formatted_stock = DataPresentation.format_stock_report(stock_data)
    formatted_news = DataPresentation.format_news_report(stock_news)
    formatted_global = DataPresentation.format_global_news(global_news)
    
    return {
        "raw_data": stock_data,
        "stock_report": formatted_stock,
        "news_report": formatted_news,
        "global_news_report": formatted_global,
        "market_summary": market_summary,
        "earnings_data": earnings,
        "news_items": stock_news,
        "global_news_items": global_news,
    }