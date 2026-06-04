"""
Master Trading Crew — Streamlit Command Center
Professional-grade trading analysis powered by AI multi-agent system
Each agent uses its own dedicated NVIDIA NIM API key.
"""
import streamlit as st
import os
import sys
import json
import re
import traceback
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crew.crew import run_master_crew
from utils.data_fetcher import StockDataFetcher, DataPresentation
from utils.llm_factory import get_all_agent_status, validate_all_keys, NVIDIA_AGENT_KEY_MAP
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import yfinance as yf


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Master Trading Crew",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — Professional Dark Theme
# ============================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #1a1f3a 50%, #0d1117 100%);
    }
    .main-title {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff, #7c3aed, #f59e0b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    .trade-card {
        background: linear-gradient(135deg, rgba(16, 30, 20, 0.95), rgba(10, 20, 15, 0.98));
        border: 2px solid rgba(34, 197, 94, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 12px 40px rgba(34, 197, 94, 0.1);
    }
    .trade-card-short {
        background: linear-gradient(135deg, rgba(30, 15, 15, 0.95), rgba(20, 10, 10, 0.98));
        border: 2px solid rgba(239, 68, 68, 0.3);
        box-shadow: 0 12px 40px rgba(239, 68, 68, 0.1);
    }
    .trade-card-neutral {
        background: linear-gradient(135deg, rgba(30, 25, 10, 0.95), rgba(20, 15, 5, 0.98));
        border: 2px solid rgba(245, 158, 11, 0.3);
        box-shadow: 0 12px 40px rgba(245, 158, 11, 0.1);
    }
    .agent-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        border-left: 4px solid #6366f1;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(99, 102, 241, 0.3);
    }
    .agent-name {
        font-weight: 700;
        color: #818cf8;
        font-size: 1.1rem;
    }
    .price-display {
        font-size: 3rem;
        font-weight: 800;
        color: #f8fafc;
    }
    .price-up { color: #22c55e; }
    .price-down { color: #ef4444; }
    .level-entry {
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.4);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .level-stop {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .level-target {
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.4);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .conviction-high {
        background: linear-gradient(90deg, rgba(34, 197, 94, 0.2), rgba(34, 197, 94, 0.4));
        border: 1px solid rgba(34, 197, 94, 0.5);
        border-radius: 12px;
        padding: 1rem 2rem;
        text-align: center;
    }
    .conviction-medium {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.2), rgba(245, 158, 11, 0.4));
        border: 1px solid rgba(245, 158, 11, 0.5);
        border-radius: 12px;
        padding: 1rem 2rem;
        text-align: center;
    }
    .conviction-low {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.2), rgba(239, 68, 68, 0.4));
        border: 1px solid rgba(239, 68, 68, 0.5);
        border-radius: 12px;
        padding: 1rem 2rem;
        text-align: center;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #7c3aed);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
    }
    .stTextInput > div > div {
        background: rgba(30, 41, 59, 0.8);
        border: 2px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        color: #f1f5f9;
    }
    .stProgress > div > div {
        background: linear-gradient(90deg, #6366f1, #22c55e);
        border-radius: 10px;
    }
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 10px;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    .api-ok { color: #22c55e; font-weight: bold; }
    .api-missing { color: #ef4444; font-weight: bold; }
    .api-key-display {
        font-family: monospace;
        font-size: 0.75rem;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_trade_details(output_text: str) -> dict:
    """Extract structured trade details from crew output"""
    details = {
        "direction": "NEUTRAL",
        "conviction": 0,
        "entry": None,
        "stop": None,
        "target1": None,
        "target2": None,
        "target3": None,
        "position_size": "",
        "duration": "",
        "rationale": "",
    }
    
    if not output_text:
        return details
    
    text = str(output_text)
    
    # Direction
    text_upper = text.upper()
    if "LONG" in text_upper and "SHORT" not in text_upper:
        details["direction"] = "LONG"
    elif "SHORT" in text_upper:
        details["direction"] = "SHORT"
    elif "BUY" in text_upper:
        details["direction"] = "LONG"
    elif "SELL" in text_upper and "SHORT" in text_upper:
        details["direction"] = "SHORT"
    
    # Conviction
    conviction_match = re.search(r'(\d+)%\s*(?:CONVICTION|conviction|confidence)', text, re.IGNORECASE)
    if conviction_match:
        details["conviction"] = int(conviction_match.group(1))
    
    # Entry
    entry_match = re.search(r'(?:Entry|Primary Entry|ENTRY)[:\s]+\$?([\d.,]+)', text, re.IGNORECASE)
    if entry_match:
        details["entry"] = float(entry_match.group(1).replace(',', ''))
    
    # Stop
    stop_match = re.search(r'(?:Stop|Stop Loss|STOP)[:\s]+\$?([\d.,]+)', text, re.IGNORECASE)
    if stop_match:
        details["stop"] = float(stop_match.group(1).replace(',', ''))
    
    # Targets
    targets = re.findall(r'Target\s*\d*[:\s]+\$?([\d.,]+)', text, re.IGNORECASE)
    if len(targets) >= 1:
        details["target1"] = float(targets[0].replace(',', ''))
    if len(targets) >= 2:
        details["target2"] = float(targets[1].replace(',', ''))
    if len(targets) >= 3:
        details["target3"] = float(targets[2].replace(',', ''))
    
    # Duration
    duration_match = re.search(r'(?:Duration|Hold Duration|Expected.*Duration)[:\s]+([^\n]+)', text, re.IGNORECASE)
    if duration_match:
        details["duration"] = duration_match.group(1).strip()
    
    # Rationale
    rationale_match = re.search(r'(?:SYNTHESIS SUMMARY|synthesis summary)[:\s]+([^#]+)', text, re.IGNORECASE | re.DOTALL)
    if rationale_match:
        details["rationale"] = rationale_match.group(1).strip()[:500]
    
    return details


def render_price_chart(symbol: str):
    """Render interactive price chart with indicators"""
    try:
        hist = yf.Ticker(symbol).history(period="3mo")
        
        if hist.empty:
            st.warning("No chart data available")
            return
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=('Price & Moving Averages', 'Volume', 'RSI')
        )
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name='Price',
            increasing_line_color='#22c55e',
            decreasing_line_color='#ef4444',
        ), row=1, col=1)
        
        # Moving averages
        hist['SMA20'] = hist['Close'].rolling(20).mean()
        hist['SMA50'] = hist['Close'].rolling(50).mean()
        
        fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA20'], name='SMA 20', line=dict(color='#3b82f6', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA50'], name='SMA 50', line=dict(color='#f59e0b', width=1)), row=1, col=1)
        
        # Volume
        colors = ['#22c55e' if hist['Close'].iloc[i] >= hist['Open'].iloc[i] else '#ef4444' for i in range(len(hist))]
        fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Volume', marker_color=colors), row=2, col=1)
        
        # RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        fig.add_trace(go.Scatter(x=hist.index, y=rsi, name='RSI', line=dict(color='#8b5cf6', width=1.5)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
        
        fig.update_layout(
            title=f"{symbol} - Technical Chart",
            plot_bgcolor='rgba(15, 23, 42, 0.8)',
            paper_bgcolor='rgba(15, 23, 42, 0)',
            font_color='#f1f5f9',
            xaxis_rangeslider_visible=False,
            height=700,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Chart error: {e}")


def render_trade_dashboard(results: dict):
    """Render the comprehensive trade recommendation dashboard"""
    
    output = results.get("crew_output", "")
    raw_data = results.get("raw_data", {})
    full_data = raw_data.get("full_data", {}) if isinstance(raw_data, dict) else {}
    
    trade_details = extract_trade_details(output)
    
    symbol = results.get("ticker", "")
    company = full_data.get("company_name", symbol) if isinstance(full_data, dict) else symbol
    current_price = full_data.get("current_price", 0) if isinstance(full_data, dict) else 0
    
    # Header
    st.markdown(f"<h1 class='main-title'>{symbol} - {company}</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Master Trading Crew Analysis Complete</p>", unsafe_allow_html=True)
    
    # Price Display
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        price_change = full_data.get("price_change", 0) if isinstance(full_data, dict) else 0
        price_change_pct = full_data.get("price_change_percent", 0) if isinstance(full_data, dict) else 0
        price_class = "price-up" if price_change >= 0 else "price-down"
        sign = "+" if price_change >= 0 else ""
        st.markdown(f"""
            <div class='metric-card'>
                <div style='color: #94a3b8; font-size: 0.9rem;'>Current Price</div>
                <div class='price-display'>${current_price}</div>
                <div class='{price_class}' style='font-size: 1.2rem; font-weight: 600;'>
                    {sign}${price_change} ({sign}{price_change_pct}%)
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        sector = full_data.get("sector", "N/A") if isinstance(full_data, dict) else "N/A"
        industry = full_data.get("industry", "N/A") if isinstance(full_data, dict) else "N/A"
        st.markdown(f"""
            <div class='metric-card'>
                <div style='color: #94a3b8; font-size: 0.85rem;'>Sector</div>
                <div style='color: #f1f5f9; font-weight: 600; font-size: 1rem;'>{sector}</div>
                <div style='color: #94a3b8; font-size: 0.8rem; margin-top: 0.5rem;'>{industry}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        market_cap = full_data.get("market_cap", 0) if isinstance(full_data, dict) else 0
        if market_cap:
            if market_cap >= 1e12:
                mc_str = f"${market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                mc_str = f"${market_cap/1e9:.2f}B"
            elif market_cap >= 1e6:
                mc_str = f"${market_cap/1e6:.2f}M"
            else:
                mc_str = f"${market_cap:,.0f}"
        else:
            mc_str = "N/A"
        
        st.markdown(f"""
            <div class='metric-card'>
                <div style='color: #94a3b8; font-size: 0.85rem;'>Market Cap</div>
                <div style='color: #f1f5f9; font-weight: 600; font-size: 1.2rem;'>{mc_str}</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Trade Recommendation Card
    st.markdown("---")
    st.markdown("<h2 class='section-header'>🎯 Master Trade Recommendation</h2>", unsafe_allow_html=True)
    
    direction = trade_details["direction"]
    conviction = trade_details["conviction"]
    
    if direction == "LONG":
        card_class = "trade-card"
        direction_emoji = "🟢"
        direction_color = "#22c55e"
    elif direction == "SHORT":
        card_class = "trade-card trade-card-short"
        direction_emoji = "🔴"
        direction_color = "#ef4444"
    else:
        card_class = "trade-card trade-card-neutral"
        direction_emoji = "⚪"
        direction_color = "#94a3b8"
    
    if conviction >= 75:
        conv_class = "conviction-high"
        conv_color = "#22c55e"
    elif conviction >= 50:
        conv_class = "conviction-medium"
        conv_color = "#f59e0b"
    else:
        conv_class = "conviction-low"
        conv_color = "#ef4444"
    
    st.markdown(f"""
        <div class='{card_class}'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;'>
                <div>
                    <div style='font-size: 2.5rem; font-weight: 800; color: {direction_color};'>
                        {direction_emoji} {direction}
                    </div>
                    <div style='color: #94a3b8; margin-top: 0.3rem;'>Trade Direction</div>
                </div>
                <div class='{conv_class}'>
                    <div style='font-size: 2rem; font-weight: 800; color: {conv_color};'>{conviction}%</div>
                    <div style='color: #94a3b8; font-size: 0.9rem;'>Conviction Score</div>
                </div>
            </div>
    """, unsafe_allow_html=True)
    
    # Trade Levels
    if trade_details["entry"] and trade_details["stop"]:
        c1, c2, c3, c4, c5 = st.columns(5)
        
        with c1:
            st.markdown(f"""
                <div class='level-entry'>
                    <div style='color: #94a3b8; font-size: 0.8rem;'>ENTRY</div>
                    <div style='color: #22c55e; font-size: 1.5rem; font-weight: 700;'>${trade_details['entry']:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with c2:
            st.markdown(f"""
                <div class='level-stop'>
                    <div style='color: #94a3b8; font-size: 0.8rem;'>STOP LOSS</div>
                    <div style='color: #ef4444; font-size: 1.5rem; font-weight: 700;'>${trade_details['stop']:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
        
        if trade_details["target1"]:
            with c3:
                st.markdown(f"""
                    <div class='level-target'>
                        <div style='color: #94a3b8; font-size: 0.8rem;'>TARGET 1</div>
                        <div style='color: #3b82f6; font-size: 1.5rem; font-weight: 700;'>${trade_details['target1']:.2f}</div>
                        <div style='color: #94a3b8; font-size: 0.7rem;'>50% Position</div>
                    </div>
                """, unsafe_allow_html=True)
        
        if trade_details["target2"]:
            with c4:
                st.markdown(f"""
                    <div class='level-target'>
                        <div style='color: #94a3b8; font-size: 0.8rem;'>TARGET 2</div>
                        <div style='color: #3b82f6; font-size: 1.5rem; font-weight: 700;'>${trade_details['target2']:.2f}</div>
                        <div style='color: #94a3b8; font-size: 0.7rem;'>30% Position</div>
                    </div>
                """, unsafe_allow_html=True)
        
        if trade_details["target3"]:
            with c5:
                st.markdown(f"""
                    <div class='level-target'>
                        <div style='color: #94a3b8; font-size: 0.8rem;'>TARGET 3</div>
                        <div style='color: #3b82f6; font-size: 1.5rem; font-weight: 700;'>${trade_details['target3']:.2f}</div>
                        <div style='color: #94a3b8; font-size: 0.7rem;'>Trail 20%</div>
                    </div>
                """, unsafe_allow_html=True)
        
        # Risk Metrics
        if trade_details["entry"] and trade_details["stop"] and trade_details["target1"]:
            risk = abs(trade_details["entry"] - trade_details["stop"])
            reward1 = abs(trade_details["target1"] - trade_details["entry"])
            rr1 = reward1 / risk if risk > 0 else 0
            
            st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            
            with m1:
                st.markdown(f"""
                    <div class='metric-card' style='text-align: center;'>
                        <div style='color: #94a3b8; font-size: 0.85rem;'>Risk/Reward (T1)</div>
                        <div style='color: #f59e0b; font-size: 1.8rem; font-weight: 700;'>1:{rr1:.1f}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with m2:
                risk_pct = (risk / trade_details["entry"] * 100) if trade_details["entry"] else 0
                st.markdown(f"""
                    <div class='metric-card' style='text-align: center;'>
                        <div style='color: #94a3b8; font-size: 0.85rem;'>Risk per Share</div>
                        <div style='color: #ef4444; font-size: 1.8rem; font-weight: 700;'>${risk:.2f} ({risk_pct:.1f}%)</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with m3:
                duration = trade_details.get("duration", "")
                st.markdown(f"""
                    <div class='metric-card' style='text-align: center;'>
                        <div style='color: #94a3b8; font-size: 0.85rem;'>Expected Duration</div>
                        <div style='color: #818cf8; font-size: 1.5rem; font-weight: 700;'>{duration or 'N/A'}</div>
                    </div>
                """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Tabs for detailed analysis
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Technical Chart", "📋 Full Analysis", "📰 Key Metrics", "🤖 Agent Reports"])
    
    with tab1:
        render_price_chart(symbol)
    
    with tab2:
        st.markdown("### Complete Crew Analysis")
        st.text_area("", value=str(output), height=600, disabled=True, label_visibility="collapsed")
    
    with tab3:
        render_key_metrics(full_data)
    
    with tab4:
        render_agent_reports(results.get("agent_outputs", {}))


def render_key_metrics(data: dict):
    """Render key financial metrics in a grid"""
    if not isinstance(data, dict):
        st.warning("No metrics data available")
        return
    
    col1, col2, col3 = st.columns(3)
    
    metrics_groups = {
        "Valuation": [
            ("P/E Ratio (TTM)", data.get("pe_ratio")),
            ("Forward P/E", data.get("forward_pe")),
            ("PEG Ratio", data.get("peg_ratio")),
            ("Price-to-Book", data.get("price_to_book")),
        ],
        "Profitability": [
            ("ROE", data.get("return_on_equity")),
            ("ROA", data.get("return_on_assets")),
            ("Profit Margin", data.get("profit_margins")),
            ("Revenue Growth", data.get("revenue_growth")),
        ],
        "Financial Health": [
            ("Debt-to-Equity", data.get("debt_to_equity")),
            ("Current Ratio", data.get("current_ratio")),
            ("Quick Ratio", data.get("quick_ratio")),
            ("Beta", data.get("beta")),
        ],
    }
    
    for col, (group_name, metrics) in zip([col1, col2, col3], metrics_groups.items()):
        with col:
            st.markdown(f"<h4 style='color: #818cf8;'>{group_name}</h4>", unsafe_allow_html=True)
            for label, value in metrics:
                if value is not None:
                    if isinstance(value, float):
                        formatted = f"{value:.2f}"
                    else:
                        formatted = str(value)
                else:
                    formatted = "N/A"
                
                st.markdown(f"""
                    <div style='display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.1);'>
                        <span style='color: #94a3b8;'>{label}</span>
                        <span style='color: #f1f5f9; font-weight: 600;'>{formatted}</span>
                    </div>
                """, unsafe_allow_html=True)
    
    # Technical Indicators
    st.markdown("---")
    st.markdown("<h4 style='color: #818cf8;'>Technical Indicators</h4>", unsafe_allow_html=True)
    
    tech = data.get("technical_indicators", {})
    if tech:
        c1, c2, c3, c4 = st.columns(4)
        
        indicators = [
            ("Trend", tech.get("trend"), c1),
            ("RSI (14)", f"{tech.get('rsi_14', 'N/A')} ({tech.get('rsi_signal', '')})", c2),
            ("MACD", tech.get("macd_line"), c3),
            ("Volume", tech.get("volume_trend"), c4),
        ]
        
        for label, value, col in indicators:
            with col:
                val_str = f"{value:.4f}" if isinstance(value, float) else str(value or "N/A")
                st.markdown(f"""
                    <div class='metric-card' style='text-align: center;'>
                        <div style='color: #94a3b8; font-size: 0.8rem;'>{label}</div>
                        <div style='color: #f1f5f9; font-weight: 700; font-size: 1.2rem;'>{val_str}</div>
                    </div>
                """, unsafe_allow_html=True)


def render_agent_reports(agent_outputs: dict):
    """Render individual agent analysis reports"""
    if not agent_outputs:
        st.info("No individual agent reports available")
        return
    
    agent_icons = {
        "Chief Data Scientist": "🔬",
        "Senior Global Macro": "🌍",
        "Chief Technical Strategist": "📈",
        "Senior Valuation": "💰",
        "Market Sentiment": "🧠",
        "Senior Equity Research": "🔬",
        "Chief Investment Officer": "👑",
    }
    
    for role, output in agent_outputs.items():
        icon = "🤖"
        for key, val in agent_icons.items():
            if key in role:
                icon = val
                break
        
        with st.expander(f"{icon} {role}", expanded=False):
            st.markdown(output)


# ============================================================
# SIDEBAR — API KEY MANAGEMENT
# ============================================================

def render_sidebar():
    """Render the application sidebar with per-agent API key management"""
    
    st.sidebar.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <div style='font-size: 2rem;'>📈</div>
            <div style='font-size: 1.3rem; font-weight: 700; color: #f1f5f9;'>Master Trading Crew</div>
            <div style='font-size: 0.85rem; color: #94a3b8; margin-top: 0.5rem;'>AI-Powered Trading Intelligence</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # API Key Management Section
    st.sidebar.markdown("<h4 style='color: #818cf8;'>🔑 API Configuration</h4>", unsafe_allow_html=True)
    
    use_secrets = st.sidebar.checkbox("Use .streamlit/secrets.toml (recommended)", value=True, key="use_secrets")
    
    if not use_secrets:
        st.sidebar.markdown("<p style='color: #94a3b8; font-size: 0.8rem;'>Enter per-agent NVIDIA API keys:</p>", unsafe_allow_html=True)
        
        for agent_id, env_var in NVIDIA_AGENT_KEY_MAP.items():
            display_name = agent_id.replace("_", " ").title()
            key_val = st.sidebar.text_input(
                f"{display_name}",
                type="password",
                key=f"api_key_{agent_id}",
                placeholder="nvapi-...",
                help=f"Env var: {env_var}"
            )
            if key_val:
                os.environ[env_var] = key_val
        
        fallback = st.sidebar.text_input(
            "🔄 Fallback NVIDIA Key",
            type="password",
            key="nvidia_fallback",
            placeholder="nvapi-...",
            help="NVIDIA_API_KEY fallback"
        )
        if fallback:
            os.environ["NVIDIA_API_KEY"] = fallback
    
    # API Status Dashboard
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h4 style='color: #818cf8;'>📊 Agent Key Status</h4>", unsafe_allow_html=True)
    
    status = get_all_agent_status()
    all_ok = True
    for agent_id, info in status.items():
        display_name = agent_id.replace("_", " ").title()
        if info["configured"]:
            st.sidebar.markdown(
                f"<span class='api-ok'>✅ {display_name}</span> "
                f"<span class='api-key-display'>{info['key_preview']}</span>",
                unsafe_allow_html=True
            )
        else:
            st.sidebar.markdown(f"<span class='api-missing'>❌ {display_name}</span>", unsafe_allow_html=True)
            all_ok = False
    
    if all_ok:
        st.sidebar.success("All agents configured!")
    else:
        st.sidebar.warning("Some agents missing keys")
    
    st.sidebar.markdown("---")
    
    # Crew Hierarchy Info
    st.sidebar.markdown("<h4 style='color: #818cf8;'>👥 Crew Hierarchy</h4>", unsafe_allow_html=True)
    
    agents_info = [
        ("📊", "Data Scientist", "Data validation & quant signals", "layer1"),
        ("🌍", "Global News Analyst", "Macro & geopolitical analysis", "layer1"),
        ("📈", "Technical Analyst", "Chart patterns & trade levels", "layer2"),
        ("💰", "Fundamental Analyst", "Valuation & financial health", "layer2"),
        ("🧠", "Sentiment Analyst", "Market sentiment & positioning", "layer2"),
        ("🔬", "Equity Research", "Industry & competitive analysis", "layer2"),
        ("👑", "Master Analyst", "Final synthesis & decision", "layer3"),
    ]
    
    for emoji, name, desc, layer in agents_info:
        border_color = {"layer1": "#22c55e", "layer2": "#3b82f6", "layer3": "#f59e0b"}[layer]
        st.sidebar.markdown(f"""
            <div style='background: rgba(30, 41, 59, 0.6); border-left: 3px solid {border_color}; border-radius: 8px; padding: 0.7rem; margin-bottom: 0.5rem;'>
                <div style='font-weight: 600; color: #f1f5f9;'>{emoji} {name}</div>
                <div style='font-size: 0.8rem; color: #94a3b8;'>{desc}</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # About
    st.sidebar.markdown("""
        <div style='font-size: 0.8rem; color: #64748b; text-align: center;'>
            <p>Powered by <b style='color: #818cf8;'>CrewAI</b> + <b style='color: #76b900;'>NVIDIA NIM</b></p>
            <p>Not financial advice. For educational purposes only.</p>
        </div>
    """, unsafe_allow_html=True)


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    """Main application entry point"""
    
    # Render sidebar
    render_sidebar()
    
    # Header
    st.markdown("<h1 class='main-title'>📈 Master Trading Crew</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Institutional-Grade Multi-Agent Trading Intelligence System</p>", unsafe_allow_html=True)
    
    # Input section
    st.markdown("---")
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        symbol = st.text_input(
            "Enter Stock Symbol",
            placeholder="e.g., AAPL, TSLA, NVDA, MSFT",
            value="",
            key="symbol_input",
        ).upper().strip()
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_button = st.button("🚀 Launch Analysis", use_container_width=True)
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        clear_button = st.button("🔄 Clear", use_container_width=True)
    
    if clear_button:
        st.session_state.clear()
        st.rerun()
    
    # Execute analysis
    if analyze_button and symbol:
        # Validate API keys
        valid, missing = validate_all_keys()
        if not valid:
            st.error("❌ Missing API keys for some agents!")
            with st.expander("Missing Keys"):
                st.code("\\n".join(missing))
            st.info("Add keys in the sidebar or set them in `.streamlit/secrets.toml`")
            return
        
        # Validate symbol format
        if not symbol.isalpha() or len(symbol) > 5:
            st.error("⚠️ Please enter a valid stock symbol (1-5 letters, e.g., AAPL)")
            return
        
        # Run analysis
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.info("📡 Fetching market data...")
            progress_bar.progress(10)
            
            status_text.info("🧠 Deploying 7 expert agents with individual NVIDIA API keys...")
            progress_bar.progress(30)
            
            results = run_master_crew(symbol)
            
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()
            
            if "error" in results:
                st.error(f"❌ Error: {results['error']}")
                return
            
            # Render results
            render_trade_dashboard(results)
            
            # Download option
            st.markdown("---")
            report_data = {
                "ticker": symbol,
                "timestamp": datetime.now().isoformat(),
                "final_recommendation": str(results.get("crew_output", "")),
                "agent_outputs": results.get("agent_outputs", {}),
            }
            
            st.download_button(
                label="📥 Download Full Report (JSON)",
                data=json.dumps(report_data, indent=2),
                file_name=f"master_crew_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.info("Please check your API keys and try again.")
            with st.expander("🔍 Debug Details"):
                st.code(traceback.format_exc())
    
    elif analyze_button and not symbol:
        st.warning("⚠️ Please enter a stock symbol")
    
    # Default landing page
    if not symbol or not analyze_button:
        st.markdown("---")
        
        # Features showcase
        st.markdown("<h2 class='section-header'>How It Works</h2>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div class='metric-card' style='text-align: center; height: 200px;'>
                    <div style='font-size: 3rem; margin-bottom: 1rem;'>1️⃣</div>
                    <h4 style='color: #f1f5f9;'>Data Foundation</h4>
                    <p style='color: #94a3b8; font-size: 0.9rem;'>Data Scientist validates and structures all market data with institutional-grade quality checks</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class='metric-card' style='text-align: center; height: 200px;'>
                    <div style='font-size: 3rem; margin-bottom: 1rem;'>2️⃣</div>
                    <h4 style='color: #f1f5f9;'>Expert Analysis</h4>
                    <p style='color: #94a3b8; font-size: 0.9rem;'>6 specialist analysts evaluate from technical, fundamental, sentiment, macro, and research perspectives</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class='metric-card' style='text-align: center; height: 200px;'>
                    <div style='font-size: 3rem; margin-bottom: 1rem;'>3️⃣</div>
                    <h4 style='color: #f1f5f9;'>Master Decision</h4>
                    <p style='color: #94a3b8; font-size: 0.9rem;'>Chief Investment Officer synthesizes all inputs into precise entry, stop-loss, and target levels</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Sample symbols
        st.markdown("---")
        st.markdown("<h4 style='color: #94a3b8; text-align: center;'>Popular Symbols to Try</h4>", unsafe_allow_html=True)
        
        sample_cols = st.columns(8)
        sample_symbols = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "NFLX"]
        
        for col, sym in zip(sample_cols, sample_symbols):
            with col:
                if st.button(sym, key=f"sample_{sym}", use_container_width=True):
                    st.session_state.symbol_input = sym
                    st.rerun()


if __name__ == "__main__":
    main()
