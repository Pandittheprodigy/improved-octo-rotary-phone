# 📈 Master Trading Crew

**Institutional-Grade Multi-Agent Trading Intelligence System**

A sophisticated AI-powered trading analysis platform built with CrewAI and NVIDIA NIM, featuring 7 specialized analyst agents working in hierarchical collaboration to deliver precise, actionable trade recommendations.

![Master Trading Crew](https://img.shields.io/badge/Powered%20by-CrewAI-blue)
![NVIDIA NIM](https://img.shields.io/badge/AI%20Engine-NVIDIA%20NIM-green)
![Streamlit](https://img.shields.io/badge/Hosted%20on-Streamlit%20Cloud-red)

---

## 🏗️ Architecture

### 7-Agent Hierarchical System

```
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 3: DECISION                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  👑 Master Analyst (CIO)                              │  │
│  │  Final synthesis, precise entry/exit/stop levels      │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    LAYER 2: ANALYSIS                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │ 📈 Tech  │ │ 💰 Fund  │ │ 🧠 Sent  │ │ 🔬 Research  │  │
│  │ Analyst  │ │ Analyst  │ │ Analyst  │ │ Analyst      │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    LAYER 1: FOUNDATION                       │
│  ┌──────────────────────┐ ┌──────────────────────────┐     │
│  │ 📊 Data Scientist    │ │ 🌍 Global News Analyst   │     │
│  │ Data validation &    │ │ Macro, geopolitical &    │     │
│  │ quantitative signals │ │ sector analysis          │     │
│  └──────────────────────┘ └──────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Agent Roles

| Agent | Role | Expertise |
|-------|------|-----------|
| **Data Scientist** | Chief Data Scientist & Quant Engineer | Data validation, anomaly detection, statistical modeling, quantitative signal generation |
| **Global News Analyst** | Senior Global Macro & Geopolitical Analyst | Central bank policy, geopolitical risk, macro scenarios, sector impact assessment |
| **Technical Analyst** | Chief Technical Strategist | Price action, chart patterns, indicator analysis, precise level identification |
| **Fundamental Analyst** | Senior Valuation Analyst | DCF modeling, peer comparison, quality scoring (Piotroski, Altman, Beneish) |
| **Sentiment Analyst** | Market Sentiment Specialist | Options flow, retail/institutional positioning, contrarian signals, MASTER SENTIMENT SCORE |
| **Equity Research Analyst** | Senior Research Strategist | Industry analysis, competitive positioning, catalyst calendar, ESG assessment |
| **Master Analyst** | Chief Investment Officer | Synthesis of all inputs, final trade decision with entry/exit/stop levels |

---

## ⚡ Features

- **Hierarchical Multi-Agent Flow**: 3-layer architecture with data foundation → specialist analysis → master synthesis
- **Precise Trade Levels**: Exact entry zones, technical stop-losses, and 3-tier profit targets
- **Institutional-Grade Analysis**: DCF valuation, technical confluence, sentiment extremes detection
- **Risk Management**: Risk/Reward ratios, position sizing, invalidation conditions
- **Interactive Charts**: Candlestick with SMA, volume, and RSI indicators
- **Real-Time Data**: Live market data from Yahoo Finance
- **Professional Dashboard**: Dark theme with real-time metrics visualization

---

## 🚀 Deployment on Streamlit Cloud (Free Tier)

### Step 1: Get NVIDIA API Key (Free)

1. Visit [build.nvidia.com/explore/discover](https://build.nvidia.com/explore/discover)
2. Sign up for a free account
3. Generate an API key for `meta/llama-3.3-70b-instruct`
4. Copy your API key

### Step 2: Deploy to Streamlit Cloud

1. **Push code to GitHub**:
```bash
git init
git add .
git commit -m "Initial Master Trading Crew deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/master-trading-crew.git
git push -u origin main
```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Add API Key as Secret**:
   - In your Streamlit Cloud dashboard, go to your app
   - Click "⋮" → "Settings" → "Secrets"
   - Add:
```toml
NVIDIA_API_KEY = "your-nvidia-api-key-here"
```

4. **Restart the app** and it's ready!

### Alternative: Run Locally

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/master-trading-crew.git
cd master-trading-crew

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variable
export NVIDIA_API_KEY="your-nvidia-api-key-here"
# or on Windows: set NVIDIA_API_KEY=your-nvidia-api-key-here

# Launch
streamlit run app.py
```

---

## 📋 Usage

1. **Enter a stock symbol** (e.g., AAPL, TSLA, NVDA, MSFT)
2. **Click "Launch Analysis"**
3. **Wait 2-3 minutes** while the Master Crew analyzes:
   - Data validation & preparation
   - Technical, fundamental, sentiment analysis (parallel)
   - Macro context assessment
   - Final master synthesis
4. **Review the comprehensive report** with:
   - Trade direction (LONG/SHORT) with conviction score
   - Precise entry, stop-loss, and 3 profit targets
   - Risk/Reward ratios and position sizing
   - Interactive technical chart
   - Full crew analysis transcript
   - Key financial metrics

---

## 🔧 Technology Stack

| Component | Technology |
|-----------|------------|
| **Agent Framework** | CrewAI |
| **LLM Backend** | NVIDIA NIM API (Llama 3.3 70B) |
| **Data Sources** | Yahoo Finance, RSS Feeds |
| **UI Framework** | Streamlit |
| **Visualization** | Plotly |
| **Language** | Python 3.9+ |

---

## 📁 Project Structure

```
master_crew/
├── app.py                    # Streamlit main application
├── requirements.txt          # Python dependencies
├── packages.txt             # System dependencies (optional)
├── README.md                # This file
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── crew/
│   ├── __init__.py
│   ├── agents.py            # 7 agent definitions with expertise
│   ├── tasks.py             # Hierarchical task definitions
│   └── crew.py              # Crew orchestration engine
└── utils/
    ├── __init__.py
    ├── nvidia_llm.py        # NVIDIA NIM API wrapper
    └── data_fetcher.py      # Data fetching & presentation
```

---

## ⚠️ Disclaimer

**This tool is for educational and research purposes only.**

- Not financial advice
- Trading involves substantial risk of loss
- Past performance does not guarantee future results
- Always consult a qualified financial advisor before making investment decisions
- The AI-generated recommendations should be used as one of many inputs in your decision-making process

---

## 📝 License

MIT License - See LICENSE file for details.

---

## 🙏 Acknowledgments

- [CrewAI](https://github.com/joaomdmoura/crewAI) - Multi-agent framework
- [NVIDIA NIM](https://build.nvidia.com/) - Free LLM inference API
- [Streamlit](https://streamlit.io/) - UI framework
- [Yahoo Finance](https://finance.yahoo.com/) - Market data

---

**Built with precision. Trade with intelligence. 📈**