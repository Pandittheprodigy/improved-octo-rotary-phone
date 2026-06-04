"""
Master Trading Crew — Hierarchical Task Definitions
Tasks flow from Layer 1 (Data Foundation) → Layer 2 (Specialist Analysis) → Layer 3 (Master Decision)
"""
from crewai import Task
from textwrap import dedent


def create_data_validation_task(agent, ticker, raw_data):
    return Task(
        description=dedent(f"""
            ## DATA VALIDATION & QUANTITATIVE SIGNAL EXTRACTION
            
            **Asset:** {ticker}
            **Raw Data Provided:**
            {raw_data}
            
            ### Your Mission:
            1. **Validate Data Integrity**: Check for missing values, stale data,
               anomalous spikes, and suspicious volume patterns. Flag any data quality issues.
            2. **Calculate Quantitative Signals**:
               - Volatility regime (low <15%, moderate 15-30%, high >30%)
               - Trend strength (ADX-based classification)
               - Momentum score (RSI + MACD composite)
               - Volume anomaly (today vs 20D average ratio)
               - Statistical edge (mean reversion probability vs trend continuation)
            3. **Risk Metrics**:
               - Value at Risk (VaR) 95% confidence, 1-day horizon
               - Expected shortfall (CVaR)
               - Maximum adverse excursion (MAE) estimate
               - Position sizing recommendation (Kelly fraction, adjusted for volatility)
            4. **Output Format**: Present as a clean, structured report with bullet points
               and clear numerical values. No ambiguity.
            
            ### Constraints:
            - Do NOT make directional predictions — only quantify what the data shows
            - Use specific numbers, not vague descriptions
            - Flag any data that looks suspicious or unreliable
        """),
        expected_output=dedent("""
            A structured quantitative report containing:
            - Data quality assessment (PASS/CAUTION/FAIL with reasons)
            - All calculated signals with exact numerical values
            - Risk metrics with confidence intervals
            - Position sizing recommendation (conservative/moderate/aggressive)
            - Any data anomalies or warnings
        """),
        agent=agent,
    )


def create_global_macro_task(agent, ticker, raw_data):
    return Task(
        description=dedent(f"""
            ## GLOBAL MACRO & GEOPOLITICAL RISK ASSESSMENT
            
            **Asset:** {ticker}
            **Sector/Industry Context:** Extract from provided data
            **Raw Data:** {raw_data}
            
            ### Your Mission:
            1. **Federal Reserve Analysis**: Current policy stance, next meeting date,
               expected rate trajectory (dot plot interpretation), Powell's recent guidance.
            2. **Geopolitical Risk Map**: Active conflicts, trade tensions, sanctions,
               election calendars (US, EU, key emerging markets), policy uncertainty index.
            3. **Macro Data Pulse**: Latest GDP, CPI, PPI, employment, PMI trends.
               Is the economy accelerating, decelerating, or stagnating?
            4. **Cross-Asset Context**: DXY trend, 10Y Treasury yield direction,
               oil/gold/copper prices, credit spreads (HY vs IG), VIX level and term structure.
            5. **Sector-Specific Macro**: How does the current macro environment
               specifically impact this company's sector? (e.g., rising rates hurt REITs,
               help banks; strong dollar hurts exporters, helps importers)
            6. **Scenario Probabilities**:
               - Bull case: probability %, key catalysts, time horizon
               - Base case: probability %, expected path
               - Bear case: probability %, risk triggers, contagion potential
            7. **Black Swan/Grey Rhino Risks**: List 2-3 low-probability, high-impact
               events that could invalidate any trade thesis.
            
            ### Constraints:
            - Use current date context (2026)
            - Cite specific data points, not generalizations
            - Probability weights must sum to 100%
        """),
        expected_output=dedent("""
            A comprehensive macro risk report with:
            - Fed policy assessment and rate trajectory
            - Geopolitical risk score (0-100) with specific threats
            - Macro data summary table
            - Cross-asset correlation matrix
            - Sector-specific macro impact analysis
            - Bull/Base/Bear scenario probabilities (must sum to 100%)
            - Black swan/grey rhino risk list
        """),
        agent=agent,
    )


def create_technical_analysis_task(agent, ticker, raw_data):
    return Task(
        description=dedent(f"""
            ## TECHNICAL ANALYSIS — PRECISION LEVELS & PATTERN IDENTIFICATION
            
            **Asset:** {ticker}
            **Technical Data Provided:** {raw_data}
            
            ### Your Mission:
            1. **Trend Analysis**: Identify primary, intermediate, and short-term trends.
               Are they aligned (bullish/bearish/neutral)? Note any divergences.
            2. **Support & Resistance Levels**: Identify 3 key support and 3 key resistance
               levels with their significance (psychological, historical, Fibonacci, pivot).
            3. **Chart Patterns**: Scan for any recognizable patterns (H&S, triangles,
               flags, wedges, double tops/bottoms, cup & handle). Include measured move targets.
            4. **Indicator Confluence**: Evaluate all provided indicators (RSI, MACD,
               Bollinger, Stochastic, ADX). How many are bullish vs bearish vs neutral?
            5. **Volume Analysis**: Is volume confirming the price action? Identify
               accumulation vs distribution patterns.
            6. **EXACT TRADE LEVELS** (if a trade is warranted):
               - Entry Zone: Specific price range (max 1-2% width)
               - Stop Loss: At the technical invalidation level — where the thesis is definitively wrong
               - Target 1: First logical resistance/support with R:R ratio
               - Target 2: Second logical level with R:R ratio
               - Target 3: Extended target or trailing stop level
            7. **Time Frame Alignment**: Are daily, weekly, and monthly charts aligned?
               If conflicting, which time frame takes precedence?
            
            ### Constraints:
            - Stop loss MUST be at a technical level, not an arbitrary percentage
            - Entry zone must be tight and justified by technical confluence
            - If no clear pattern exists, state NO TRADE — do not force a recommendation
        """),
        expected_output=dedent("""
            A technical analysis report containing:
            - Trend assessment (primary/intermediate/short-term with alignment score)
            - Key S/R levels table with significance ratings
            - Chart patterns identified with measured move targets
            - Indicator confluence matrix (bullish/bearish/neutral count)
            - Volume analysis and confirmation status
            - Exact trade levels (if warranted) with technical justification
            - Time frame alignment assessment
            - Technical conviction score (0-100)
        """),
        agent=agent,
    )


def create_fundamental_analysis_task(agent, ticker, raw_data):
    return Task(
        description=dedent(f"""
            ## FUNDAMENTAL ANALYSIS — VALUATION & QUALITY ASSESSMENT
            
            **Asset:** {ticker}
            **Fundamental Data Provided:** {raw_data}
            
            ### Your Mission:
            1. **Valuation Framework**:
               - Relative valuation: P/E, EV/EBITDA, P/B, P/S, PEG vs sector median
               - DCF sanity check: Is the current price within a reasonable DCF range?
               - Historical valuation: Where does current valuation sit vs 5-year range?
            2. **Quality Scoring**:
               - Piotroski F-Score (0-9) with component breakdown
               - Altman Z-Score (bankruptcy risk)
               - Beneish M-Score (earnings manipulation risk)
               - Earnings quality (accruals vs operating cash flow ratio)
            3. **Financial Health**:
               - Balance sheet strength (debt/equity, current ratio, interest coverage)
               - Profitability trends (gross margin, operating margin, net margin trajectory)
               - Capital efficiency (ROE, ROIC, asset turnover)
               - Cash flow quality (FCF yield, FCF conversion rate)
            4. **Growth Assessment**:
               - Revenue growth (1Y, 3Y CAGR, 5Y CAGR)
               - Earnings growth and acceleration/deceleration
               - Guidance vs consensus vs actual beat/miss history
            5. **Intrinsic Value Estimate**: Provide a fair value range with margin of safety.
            6. **Valuation Verdict**: Undervalued (buy), Fairly Valued (hold), Overvalued (sell/short).
            
            ### Constraints:
            - Use exact ratios from data; do not estimate unless explicitly noted
            - Provide numerical scores, not vague ratings
            - If data is insufficient, state "INSUFFICIENT DATA" rather than guessing
        """),
        expected_output=dedent("""
            A fundamental analysis report containing:
            - Valuation summary table (all key ratios vs peers and history)
            - Quality scores (Piotroski, Altman, Beneish with interpretations)
            - Financial health dashboard
            - Growth trajectory analysis
            - Intrinsic value estimate with margin of safety
            - Valuation verdict with conviction level
        """),
        agent=agent,
    )


def create_sentiment_analysis_task(agent, ticker, raw_data):
    return Task(
        description=dedent(f"""
            ## SENTIMENT ANALYSIS — MASTER SENTIMENT SCORE (0-100)
            
            **Asset:** {ticker}
            **Market Data Provided:** {raw_data}
            
            ### Your Mission:
            1. **Options Market Sentiment**:
               - Put/Call ratio interpretation (skewed puts = bearish, skewed calls = bullish)
               - Implied volatility skew (fear vs greed)
               - VIX level and term structure (contango vs backwardation)
            2. **Institutional Positioning**:
               - 13F trends (accumulation vs distribution by smart money)
               - Short interest dynamics (days to cover, cost to borrow, squeeze risk)
               - Dark pool activity (block trades, institutional flow)
            3. **Retail & Social Sentiment**:
               - Social media volume and sentiment trends
               - Reddit/Twitter buzz intensity
               - Google Trends for the ticker and related terms
            4. **Analyst Consensus**:
               - Rating distribution (buy/hold/sell ratio)
               - Price target dispersion (tight = consensus, wide = uncertainty)
               - Earnings revision momentum (up vs down vs stable)
            5. **Insider Activity**:
               - Recent insider buying/selling ratio
               - Cluster buying patterns (multiple insiders buying simultaneously)
               - 10b5-1 plan vs discretionary trades
            6. **Fund Flows**:
               - Sector ETF inflows/outflows
               - Active vs passive fund rotation
               - Retail vs institutional flow divergence
            7. **MASTER SENTIMENT SCORE**: Calculate a composite 0-100 score with:
               - 0-20: Extreme Fear (potential contrarian buy)
               - 21-40: Fear (caution, watch for reversal)
               - 41-60: Neutral (no edge from sentiment)
               - 61-80: Greed (trend following, watch for exhaustion)
               - 81-100: Extreme Greed (potential contrarian sell/short)
            8. **Contrarian Signals**: Identify any extreme sentiment that suggests a reversal.
            
            ### Constraints:
            - Score must be justified with specific data points
            - If options data is unavailable, note it and weight other factors higher
            - Do not let one factor dominate — use a balanced composite
        """),
        expected_output=dedent("""
            A sentiment analysis report containing:
            - Options sentiment assessment
            - Institutional positioning summary
            - Retail/social sentiment indicators
            - Analyst consensus matrix
            - Insider activity summary
            - Fund flow analysis
            - MASTER SENTIMENT SCORE (0-100) with detailed justification
            - Contrarian signal identification (if any)
            - Sentiment trend direction (improving/deteriorating/stable)
        """),
        agent=agent,
    )


def create_equity_research_task(agent, ticker, raw_data):
    return Task(
        description=dedent(f"""
            ## EQUITY RESEARCH — INDUSTRY, CATALYST & ESG ANALYSIS
            
            **Asset:** {ticker}
            **Company & Industry Data:** {raw_data}
            
            ### Your Mission:
            1. **Industry Structure Analysis**:
               - TAM, SAM, SOM with growth rates
               - Industry lifecycle (emerging, growth, mature, decline)
               - Cyclicality assessment (defensive, cyclical, secular growth)
            2. **Competitive Landscape**:
               - Market share ranking (company vs top 5 competitors)
               - Competitive moat assessment (wide, narrow, none)
               - Pricing power and margin stability
               - Barriers to entry and threat of disruption
            3. **Business Model Quality**:
               - Revenue mix (recurring vs transactional)
               - Customer concentration risk (top 5 customers % of revenue)
               - Switching costs and retention rates
               - Network effects or platform dynamics
            4. **Catalyst Calendar (Next 90 Days)**:
               - List the 3 most important upcoming events
               - For each: date, probability of positive/negative impact, magnitude
               - Earnings date and whisper number vs consensus
            5. **Regulatory & Legal Environment**:
               - Active regulatory risks (antitrust, environmental, labor)
               - Pending litigation and potential financial impact
               - Tariff exposure and trade policy sensitivity
            6. **Management Quality**:
               - CEO tenure and track record
               - Capital allocation history (M&A success, buyback timing, dividend policy)
               - Insider ownership level and alignment with shareholders
            7. **ESG Assessment**:
               - Environmental score and carbon trajectory
               - Governance quality (board independence, audit quality, shareholder rights)
               - Social factors (labor practices, diversity, community impact)
               - ESG risk to valuation (discount/premium justified?)
            
            ### Constraints:
            - Catalyst dates must be specific (month/day if known, quarter if not)
            - Use factual data from the provided context; do not invent events
            - ESG assessment must be financially material, not ideological
        """),
        expected_output=dedent("""
            An equity research report containing:
            - Industry structure and growth profile
            - Competitive landscape map
            - Business model quality score
            - 90-day catalyst calendar with probability/impact matrix
            - Regulatory risk assessment
            - Management quality evaluation
            - ESG score with financial materiality assessment
            - Overall equity research conviction (0-100)
        """),
        agent=agent,
    )


def create_master_synthesis_task(agent, ticker, all_analyst_outputs):
    return Task(
        description=dedent(f"""
            ## MASTER SYNTHESIS — FINAL TRADE RECOMMENDATION
            
            **Asset:** {ticker}
            **Analyst Inputs:**
            {all_analyst_outputs}
            
            ### Your Mission:
            You are the Chief Investment Officer. You have received comprehensive analysis
            from 6 expert analysts. Your job is to synthesize ALL inputs into ONE final,
            actionable trade recommendation.
            
            ### Synthesis Process:
            1. **Conflict Resolution**: Identify where analysts disagree. Which input takes
               precedence? (e.g., technical bullish + fundamental bearish = which wins?)
            2. **Signal Confluence**: Count how many analysts are bullish, bearish, or neutral.
               Weight by their expertise relevance to the current market condition.
            3. **Risk-Adjusted Decision**: Even if bullish, is the risk/reward attractive?
               Even if bearish, is the downside limited?
            4. **Macro Overlay**: Does the global macro environment support or oppose this trade?
            5. **Catalyst Timing**: Are there near-term catalysts that make this trade urgent
               or suggest waiting?
            
            ### Final Output Requirements:
            
            **1. TRADE DIRECTION & CONVICTION**
            - Direction: LONG / SHORT / NO TRADE
            - Conviction Score: 0-100% (only recommend if >60% conviction)
            - If NO TRADE: Explain why and what would change your mind
            
            **2. EXACT TRADE PARAMETERS**
            - Entry Zone: Specific price range (e.g., $145.20 - $147.50)
            - Stop Loss: Exact price at technical invalidation (e.g., $142.30)
            - Target 1: Price + R:R ratio (50% of position)
            - Target 2: Price + R:R ratio (30% of position)
            - Target 3: Price or trailing stop rule (20% of position)
            - Position Sizing: Conservative/Moderate/Aggressive with % of portfolio
            - Time Horizon: Days/weeks/months with review trigger dates
            
            **3. SCENARIO ANALYSIS**
            - Bull Case: Price target, probability %, key drivers
            - Base Case: Price target, probability %, expected path
            - Bear Case: Price target, probability %, risk triggers
            
            **4. RISK MANAGEMENT**
            - Maximum Risk: $ amount or % of portfolio at risk
            - Risk/Reward Ratio: For each target
            - Correlation Risk: How this trade correlates with broader market
            - Liquidity Risk: Can the position be exited quickly if needed?
            
            **5. INVALIDATION CONDITIONS**
            - List 3 specific conditions that would cause you to exit BEFORE the stop loss
            - Include time-based invalidation (e.g., "If no move within 5 trading days, cut 50%")
            
            **6. EXECUTION NOTES**
            - Best order type (limit, market, stop-limit)
            - Optimal entry timing (pre-market, open, close, VWAP)
            - Partial entry strategy (e.g., "Enter 50% at open, 50% on pullback to $X")
            
            ### CRITICAL CONSTRAINTS:
            - If conviction is below 60%, recommend NO TRADE with explanation
            - Stop loss must be at a real technical level, not a round number or arbitrary %
            - Entry zone must be tight — no "buy anywhere" recommendations
            - Every number must be justified by analyst inputs
            - Do NOT recommend leverage or options unless explicitly justified by risk/reward
        """),
        expected_output=dedent("""
            A comprehensive final trade recommendation containing:
            - Trade direction with conviction score and justification
            - Exact entry zone, stop loss, and 3 profit targets with position allocations
            - Scenario analysis (Bull/Base/Bear) with probability weights
            - Risk management framework with max risk and R:R ratios
            - Invalidation conditions (3 specific early-exit triggers)
            - Execution strategy with order types and timing
            - If NO TRADE: Clear explanation and conditions to re-evaluate
        """),
        agent=agent,
        context=[],  # Will be populated with outputs from previous tasks
    )
