"""
Master Trading Crew - Task Definitions
Hierarchical task flow with proper dependencies
"""
from crewai import Task
from typing import Dict, Any


def create_tasks(agents: Dict[str, Any], symbol: str, data_context: Dict[str, Any]) -> list:
    """Create all tasks with hierarchical dependencies"""
    
    stock_report = data_context.get("stock_report", "")
    news_report = data_context.get("news_report", "")
    global_news_report = data_context.get("global_news_report", "")
    market_summary = data_context.get("market_summary", {})
    earnings_data = data_context.get("earnings_data", {})
    
    # ============================================================
    # PHASE 1: DATA FOUNDATION (Layer 1)
    # ============================================================
    
    data_preparation_task = Task(
        description=f"""
        ## ROLE: Chief Data Scientist - Data Foundation Phase
        
        ### OBJECTIVE
        Prepare institutional-grade data foundation for comprehensive analysis of {symbol.upper()}.
        
        ### INPUT DATA
        {stock_report}
        
        ### YOUR TASKS
        1. **Data Validation**: Verify all numerical data for consistency and reasonableness
        2. **Anomaly Detection**: Flag any suspicious values, outliers, or data gaps
        3. **Statistical Summary**: Calculate key statistics (mean, std, min, max, percentiles) for price and volume
        4. **Correlation Analysis**: Identify relationships between price, volume, and technical indicators
        5. **Data Quality Report**: Assign data quality score (0-100) with specific issues noted
        6. **Quantitative Signals**: Extract raw quantitative signals from the data:
           - Price momentum signals (short, medium, long-term)
           - Volume anomaly detection  
           - Volatility regime classification
           - Mean reversion potential
        7. **Data Package**: Structure data into clean, analysis-ready format for other analysts
        
        ### OUTPUT FORMAT
        Provide a structured data report with:
        - Data Quality Score: X/100
        - Key Statistics Table
        - Identified Anomalies (if any)
        - Quantitative Signal Summary
        - Recommended Analysis Focus Areas
        - Clean Data Summary for Analysts
        
        ### IMPORTANT
        - Be precise with all numbers
        - Flag any data concerns immediately
        - Provide statistical evidence for all claims
        - This data will drive all downstream analysis - accuracy is critical
        """,
        expected_output="A comprehensive, structured data quality report with validated statistics, anomaly flags, quantitative signals, and a clean data summary ready for analyst consumption. Include specific numbers, scores, and evidence-based observations.",
        agent=agents["data_scientist"],
    )
    
    global_macro_analysis_task = Task(
        description=f"""
        ## ROLE: Senior Global Macro & Geopolitical News Analyst
        
        ### OBJECTIVE  
        Analyze global macroeconomic conditions and their impact on {symbol.upper()}.
        
        ### INPUT DATA
        **Global News:**
        {global_news_report}
        
        **Market Summary:**
        {market_summary}
        
        **Company Info:**
        Symbol: {symbol.upper()}
        Sector: {data_context.get('raw_data', {}).get('sector', 'N/A')}
        Industry: {data_context.get('raw_data', {}).get('industry', 'N/A')}
        
        ### YOUR ANALYSIS FRAMEWORK
        1. **Central Bank Policy Assessment**:
           - Current interest rate environment and trajectory
           - Impact on sector valuation multiples
           - Liquidity conditions assessment
        
        2. **Geopolitical Risk Evaluation**:
           - Active geopolitical tensions with market relevance
           - Supply chain disruption risks
           - Sanctions and trade policy impacts
        
        3. **Macro Indicator Analysis**:
           - Inflation trajectory and expectations
           - Employment trends and consumer health
           - GDP growth outlook
        
        4. **Sector-Specific Macro Factors**:
           - Regulatory environment and pending legislation
           - Industry cyclical positioning
           - Commodity/currency exposure if relevant
        
        5. **Market Regime Classification**:
           - Risk-on vs risk-off environment
           - Volatility regime assessment
           - Liquidity conditions
        
        6. **Probability-Weighted Scenarios**:
           - Bull case: Probability %, key drivers, price impact estimate
           - Base case: Probability %, expected path
           - Bear case: Probability %, key risks, downside estimate
        
        ### OUTPUT FORMAT
        Provide structured analysis with:
        - Macro Environment Score (-10 to +10, 0 neutral)
        - Sector Tailwind/Headwind Rating (-5 to +5)
        - Key Risks with Probability Estimates
        - Scenario Analysis with Price Impact Ranges
        - Trading Implications Summary
        - Critical Events Calendar (next 30 days)
        
        ### IMPORTANT
        - Quantify everything with specific numbers
        - Assign probability percentages to scenarios
        - Focus on actionable trading implications
        - Distinguish between noise and genuine regime changes
        """,
        expected_output="A comprehensive global macro analysis with probability-weighted scenarios, quantified risk assessment, sector impact evaluation, and specific trading implications. Include the Macro Environment Score, scenario probabilities with price impact estimates, and a critical events calendar.",
        agent=agents["global_news_analyst"],
    )
    
    # ============================================================
    # PHASE 2: SPECIALIST ANALYSIS (Layer 2 - Parallel)
    # ============================================================
    
    technical_analysis_task = Task(
        description=f"""
        ## ROLE: Chief Technical Strategist & Chart Analyst
        
        ### OBJECTIVE
        Provide comprehensive technical analysis for {symbol.upper()} with precise trade levels.
        
        ### INPUT DATA
        {stock_report}
        
        **Additional Context from Data Science Team:**
        Use the validated data and quantitative signals provided in the crew context.
        
        ### YOUR ANALYSIS REQUIREMENTS
        
        1. **Trend Analysis (Primary)**:
           - Identify dominant trend using 200-day SMA, ADX, and price structure
           - Trend strength rating (1-10)
           - Trend duration and maturity assessment
        
        2. **Support & Resistance Mapping**:
           - List all key support levels with historical test count
           - List all key resistance levels with historical test count
           - Identify the most significant S/R zone
        
        3. **Momentum Analysis**:
           - RSI interpretation with divergence check
           - MACD signal and histogram analysis
           - Stochastic position and crossover status
           - Momentum score (1-10)
        
        4. **Pattern Recognition**:
           - Active chart patterns with measured moves
           - Candlestick patterns on daily/weekly timeframe
           - Pattern completion percentage
        
        5. **Volatility Assessment**:
           - Current ATR and volatility percentile
           - Bollinger Band position and squeeze status
           - Expected move calculation
        
        6. **Volume Analysis**:
           - Volume trend vs price trend divergence
           - Institutional accumulation/distribution signals
           - Volume profile key levels
        
        7. **Multi-Timeframe Confluence**:
           - Daily, weekly, and monthly alignment
           - Confluence score (1-10)
        
        8. **PRECISE TRADE LEVELS**:
           - Optimal entry zone (specific price range, max 1-2% width)
           - Invalidation stop-loss (technical level, not arbitrary %)
           - Target 1: Conservative (50% position)
           - Target 2: Moderate (30% position)
           - Target 3: Aggressive (20% trailing)
           - Risk/Reward ratio for each target
        
        9. **Technical Bias**:
           - Directional bias: STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL
           - Confidence percentage (0-100%)
           - Key levels that would change the bias
        
        ### OUTPUT FORMAT
        Structure your analysis as:
        - Executive Summary (3 bullet points)
        - Trend Assessment
        - Key Levels Table (Level | Type | Significance | Tests)
        - Momentum Scorecard
        - Pattern Status
        - Volume Analysis
        - Multi-Timeframe Summary
        - TRADE SETUP (Entry | Stop | Target 1 | Target 2 | Target 3 | R:R)
        - Technical Bias with Confidence
        - Invalidation Conditions
        
        ### CRITICAL RULES
        - Entry zones must be specific (e.g., $152.30-$154.15, NOT $150-$160)
        - Stop-loss must be at a technical invalidation level
        - Every target needs a technical basis (resistance, measured move, etc.)
        - Confidence must be justified with confluence count
        """,
        expected_output="A comprehensive technical analysis with specific trade levels including exact entry zone (1-2% width), technical stop-loss at invalidation level, three targets with R:R ratios, trend/momentum/pattern analysis, and a directional bias with confidence percentage. All numbers must be precise and justified.",
        agent=agents["technical_analyst"],
    )
    
    fundamental_analysis_task = Task(
        description=f"""
        ## ROLE: Senior Fundamental & Valuation Analyst  
        
        ### OBJECTIVE
        Conduct deep fundamental analysis and valuation for {symbol.upper()}.
        
        ### INPUT DATA
        {stock_report}
        
        **Earnings Data:**
        {earnings_data}
        
        ### YOUR ANALYSIS REQUIREMENTS
        
        1. **Business Quality Assessment**:
           - Competitive moat width (None/Narrow/Wide) and durability
           - Pricing power evidence
           - Market leadership position
           - Industry growth trajectory
        
        2. **Financial Health Analysis**:
           - Balance sheet strength rating (A-F)
           - Cash generation quality
           - Debt coverage and capacity
           - Working capital efficiency
        
        3. **Profitability Metrics**:
           - ROE, ROA, ROIC with 3-year trends
           - Margin profile (gross, operating, net)
           - Margin expansion/contraction trajectory
           - Capital intensity and efficiency
        
        4. **Growth Analysis**:
           - Revenue growth (1yr, 3yr, 5yr CAGR)
           - Earnings growth trajectory
           - Unit economics if applicable
           - TAM and market share trends
        
        5. **Valuation Analysis**:
           - Current P/E vs 5-year average P/E
           - Current P/E vs industry average
           - PEG ratio assessment
           - EV/EBITDA vs peers
           - Price-to-Book vs ROE regression
           - DCF fair value estimate (base, bull, bear)
        
        6. **Margin of Safety**:
           - Current price vs fair value estimate
           - Margin of safety percentage
           - Minimum MOS required for investment grade
        
        7. **Quality Scoring**:
           - Piotroski F-Score (0-9)
           - Altman Z-Score (bankruptcy risk)
           - Beneish M-Score (earnings manipulation risk)
        
        8. **Earnings & Revenue Estimates**:
           - Consensus EPS estimates (current quarter, FY)
           - Revenue estimates and growth implied
           - Beat/miss probability based on history
        
        9. **Fundamental Bias**:
           - Rating: STRONG BUY / BUY / HOLD / SELL / STRONG SELL
           - Fair value estimate with range
           - Upside/downside from current price
           - 12-month price target
        
        ### OUTPUT FORMAT
        - Executive Summary (3 bullet points)
        - Business Quality Scorecard
        - Financial Health Dashboard
        - Valuation Table (Metric | Current | 5Y Avg | Industry | Assessment)
        - DCF Valuation (Base | Bull | Bear with probabilities)
        - Quality Scores (Piotroski, Altman, Beneish)
        - Earnings Estimate Analysis
        - FUNDAMENTAL SETUP (Fair Value | Current Price | MOS | Upside% | Rating)
        - Key Risks to Thesis
        
        ### CRITICAL RULES
        - Fair value must have specific number with range
        - All valuation metrics must include peer comparison
        - Quality scores must use established frameworks
        - Every claim backed by specific financial data
        """,
        expected_output="A deep fundamental analysis with specific fair value estimate, margin of safety calculation, quality scores (Piotroski, Altman, Beneish), peer valuation comparison, DCF scenarios, and a definitive fundamental rating with 12-month price target. All backed by specific financial metrics.",
        agent=agents["fundamental_analyst"],
    )
    
    sentiment_analysis_task = Task(
        description=f"""
        ## ROLE: Market Sentiment & Behavioral Finance Specialist
        
        ### OBJECTIVE
        Analyze market sentiment for {symbol.upper()} across all available dimensions.
        
        ### INPUT DATA
        {stock_report}
        
        **Recent News:**
        {news_report}
        
        ### YOUR ANALYSIS REQUIREMENTS
        
        1. **Options Market Sentiment**:
           - Put/Call ratio interpretation
           - Implied volatility skew analysis
           - Unusual options activity identification
           - Max pain calculation for options expiry
        
        2. **Retail Sentiment Gauges**:
           - Social media sentiment score (parse from available data)
           - Retail positioning indicators
           - Crowd sentiment extreme detection
        
        3. **Institutional Flow Analysis**:
           - Short interest trends and days to cover
           - Short % of float assessment
           - Recent institutional buying/selling patterns
           - 13F filing trends if available
        
        4. **Analyst Sentiment**:
           - Consensus rating distribution
           - Recent upgrade/downgrade momentum
           - Estimate revision trend (rising/falling)
           - Price target vs current price gap
        
        5. **Insider Activity**:
           - Recent insider buying/selling patterns
           - Cluster buying detection
           - Insider sentiment signal
        
        6. **News Sentiment**:
           - Tone analysis of recent headlines
           - Media narrative assessment
           - FUD (fear, uncertainty, doubt) level
        
        7. **SENTIMENT SCORE CALCULATION**:
           Calculate MASTER SENTIMENT SCORE (0-100):
           - Retail sentiment (20% weight): Score ___
           - Institutional flow (25% weight): Score ___
           - Options positioning (20% weight): Score ___
           - Short interest dynamics (15% weight): Score ___
           - Analyst revisions (10% weight): Score ___
           - Insider activity (10% weight): Score ___
           - TOTAL WEIGHTED SCORE: ___
        
        8. **Contrarian Signal Assessment**:
           - Identify if sentiment is at extremes (favorable for contrarian)
           - Divergence between price and sentiment
           - Historical accuracy of current sentiment level
        
        9. **Sentiment Bias**:
           - Direction: BULLISH / NEUTRAL / BEARISH
           - Confidence: 0-100%
           - Extreme signal: FEAR / GREED / NEUTRAL
        
        ### OUTPUT FORMAT
        - Executive Summary (3 bullet points)
        - Options Sentiment Dashboard
        - Retail vs Institutional Divergence Analysis
        - Short Interest Analysis
        - Analyst Sentiment Tracker
        - MASTER SENTIMENT SCORE with breakdown
        - Contrarian Opportunity Assessment
        - Historical Context ("Similar to [date] when...")
        - SENTIMENT SETUP (Score | Signal | Bias | Confidence)
        - Key Sentiment Levels to Monitor
        
        ### CRITICAL RULES
        - MASTER SENTIMENT SCORE must be calculated with shown work
        - All sentiment dimensions must have specific scores
        - Contrarian signals must reference historical precedents
        - Confidence must reflect data quality and sample size
        """,
        expected_output="A comprehensive sentiment analysis with a calculated MASTER SENTIMENT SCORE (0-100) showing all component scores and weights, contrarian signal identification, options/retail/institutional breakdown, and a definitive sentiment bias with confidence level.",
        agent=agents["sentimental_analyst"],
    )
    
    equity_research_task = Task(
        description=f"""
        ## ROLE: Senior Equity Research & Industry Strategist
        
        ### OBJECTIVE
        Produce institutional-grade equity research on {symbol.upper()}.
        
        ### INPUT DATA
        {stock_report}
        
        **Recent News:**
        {news_report}
        
        ### YOUR ANALYSIS REQUIREMENTS
        
        1. **Industry Overview**:
           - Total Addressable Market (TAM) size and growth rate
           - Industry life cycle stage
           - Key growth drivers and headwinds
           - Regulatory environment assessment
        
        2. **Competitive Positioning**:
           - Market share and trend
           - Key competitors and relative strength
           - Competitive advantages (technology, brand, scale, network effects)
           - Threats from new entrants/substitutes
        
        3. **Management Assessment**:
           - Capital allocation track record
           - Guidance accuracy history
           - Strategic vision assessment
           - Insider ownership level
        
        4. **Thematic Analysis**:
           - Key themes driving the business (AI, cloud, EV, etc.)
           - Theme exposure quality and sustainability
           - Thematic tailwind/headwind assessment
        
        5. **Earnings Power Assessment**:
           - Normalized earnings power
           - Recurring vs transactional revenue mix
           - Customer concentration risks
           - Backlog/visibility if applicable
        
        6. **Catalyst Calendar**:
           - Upcoming earnings date and expectations
           - Product launches, FDA decisions, contract awards
           - Industry conferences and presentations
           - Potential M&A activity
        
        7. **ESG Assessment**:
           - Environmental risks and opportunities
           - Social factors and labor practices
           - Governance quality and shareholder rights
           - ESG score (1-10) with key factors
        
        8. **Institutional Recommendation**:
           - Rating: STRONG BUY / BUY / HOLD / SELL / STRONG SELL
           - 12-month price target
           - Bull case price target with assumptions
           - Bear case price target with risks
           - Expected return with probability weighting
        
        9. **Portfolio Fit Assessment**:
           - Suitable for: Growth / Value / Income / Momentum / Blend
           - Risk category: Low / Moderate / High / Speculative
           - Ideal position size recommendation
           - Correlation considerations
        
        ### OUTPUT FORMAT
        - Executive Summary (Investment thesis in 3 bullets)
        - Industry & Competitive Landscape
        - Management Scorecard
        - Thematic Drivers Analysis
        - Earnings Power & Quality Assessment
        - Catalyst Calendar (next 90 days)
        - ESG Scorecard
        - RESEARCH RECOMMENDATION (Rating | Price Target | Expected Return)
        - Bull/Bear Case Scenarios
        - Portfolio Fit Assessment
        - Key Risks (ranked by probability and impact)
        
        ### CRITICAL RULES
        - Price target must be specific with methodology
        - Catalyst dates must be precise
        - Competitive analysis must name specific competitors
        - ESG assessment must use recognized frameworks
        """,
        expected_output="Institutional-grade equity research with a definitive rating, specific 12-month price target, bull/bear scenario analysis, catalyst calendar, competitive landscape assessment, ESG scorecard, and portfolio fit recommendation.",
        agent=agents["equity_research_analyst"],
    )
    
    # ============================================================
    # PHASE 3: MASTER SYNTHESIS (Layer 3)
    # ============================================================
    
    master_synthesis_task = Task(
        description=f"""
        ## ROLE: Chief Investment Officer & Master Trade Strategist
        
        ### OBJECTIVE  
        Synthesize ALL analyst inputs into a definitive, actionable trade recommendation for {symbol.upper()}.
        
        ### CONTEXT
        You are the final decision-maker. All specialist analysts have completed their work. Your job is to:
        1. Evaluate each analyst's input critically
        2. Weight analyses based on current market regime
        3. Resolve any contradictions between analysts
        4. Produce the FINAL trade recommendation
        
        ### INPUT FROM ALL ANALYSTS (Review these outputs in context):
        
        **1. DATA SCIENTIST REPORT:**
        - Data quality validation
        - Quantitative signals and statistics
        - Anomaly flags and warnings
        
        **2. GLOBAL MACRO ANALYST REPORT:**
        - Macro Environment Score
        - Sector tailwinds/headwinds
        - Probability-weighted scenarios
        
        **3. TECHNICAL ANALYST REPORT:**
        - Trend analysis and key levels
        - Precise entry/stop/target levels
        - Technical bias and confidence
        
        **4. FUNDAMENTAL ANALYST REPORT:**
        - Fair value estimate and margin of safety
        - Quality scores (Piotroski, Altman, Beneish)
        - Fundamental rating and price target
        
        **5. SENTIMENT ANALYST REPORT:**
        - MASTER SENTIMENT SCORE (0-100)
        - Contrarian signals
        - Sentiment bias and extreme readings
        
        **6. EQUITY RESEARCH ANALYST REPORT:**
        - Institutional rating and price target
        - Catalyst calendar
        - Competitive and ESG assessment
        
        ### YOUR SYNTHESIS PROCESS
        
        1. **Cross-Validation Check**:
           - Do technical and fundamental analyses align or contradict?
           - Does sentiment confirm or warn against the directional bias?
           - Are macro conditions supportive of the trade?
           - Has the data scientist flagged any data concerns?
        
        2. **Weighted Decision Matrix**:
           - Technical Analysis: Weight __% (justify)
           - Fundamental Analysis: Weight __% (justify)
           - Sentiment Analysis: Weight __% (justify)
           - Macro Context: Weight __% (justify)
           - Equity Research: Weight __% (justify)
        
        3. **Contradiction Resolution**:
           - List any analyst disagreements
           - Explain which analysis you favor and why
           - Document your reasoning process
        
        4. **Risk Assessment**:
           - What are the top 3 risks to this trade?
           - What is the maximum expected loss?
           - What is the probability of stop-loss being hit?
        
        5. **Conviction Calculation**:
           - Start with base confidence from analyst agreement
           - Adjust for data quality
           - Adjust for macro environment clarity
           - Adjust for personal track record with similar setups
           - FINAL CONVICTION SCORE: ___%
        
        ### YOUR FINAL OUTPUT - MUST INCLUDE ALL OF THE FOLLOWING:
        
        ## === MASTER TRADE RECOMMENDATION ===
        
        **TRADE DIRECTION:** [LONG / SHORT / NEUTRAL / PASS]
        
        **CONVICTION LEVEL:** [___%]
        
        **POSITION SIZE RECOMMENDATION:** 
        - Conservative portfolio: ___% allocation
        - Moderate portfolio: ___% allocation
        - Aggressive portfolio: ___% allocation
        
        **ENTRY STRATEGY:**
        - Primary Entry: $____ (specific price or zone)
        - Alternative Entry: $____ (if primary missed)
        - Entry Rationale: [2-3 sentences explaining why this level]
        
        **STOP LOSS:**
        - Stop Level: $____ (specific technical invalidation)
        - Stop Rationale: [why this level invalidates the thesis]
        - Risk Amount: $____ (per share/unit)
        
        **PROFIT TARGETS:**
        - Target 1 (Conservative): $____ (___% gain) - Close 50% position
        - Target 2 (Moderate): $____ (___% gain) - Close 30% position  
        - Target 3 (Aggressive): $____ (___% gain) - Trail 20% with stop
        
        **RISK/REWARD RATIOS:**
        - To Target 1: 1:____
        - To Target 2: 1:____
        - To Target 3: 1:____
        
        **TRADE MANAGEMENT:**
        - Expected Hold Duration: ___ days/weeks
        - Breakeven Win Rate: ___%
        - Maximum Risk: $____ total (at full position)
        - Scale-in Strategy: [if applicable]
        
        **SCENARIO ANALYSIS:**
        - Bull Case (___% probability): Price reaches $____ by [date]
        - Base Case (___% probability): Price reaches $____ by [date]
        - Bear Case (___% probability): Price falls to $____ by [date]
        
        **INVALIDATION CONDITIONS:**
        - What market/stock event would cause immediate exit before stop?
        - What fundamental change would negate the thesis?
        - What technical breakdown signals abandon ship?
        
        **KEY LEVELS TO MONITOR:**
        - $____ (Critical support - thesis intact above)
        - $____ (Decision point - partial exit consideration)
        - $____ (Target zone - profit taking)
        
        **SYNTHESIS SUMMARY:**
        [4-5 sentences explaining your reasoning, why you agree/disagree with specific analysts, and what gives you confidence in this recommendation]
        
        **RISK ACKNOWLEDGMENT:**
        [2-3 sentences on what could go wrong and how you've accounted for it]
        
        ### CRITICAL RULES
        - You MUST provide specific prices for entry, stop, and all targets
        - Entry zone must be narrow (max 1-2% width of current price)
        - Stop must be at a technical invalidation level, not arbitrary %
        - R:R ratio must be minimum 1:2
        - Every claim must reference specific analyst input or your own reasoning
        - If conviction is below 60%, recommend PASS
        - Take full responsibility - "I recommend..." not "analysts suggest..."
        """,
        expected_output="A definitive MASTER TRADE RECOMMENDATION with specific entry price, technical stop-loss, three profit targets with R:R ratios, position sizing, scenario analysis with probabilities, conviction score, invalidation conditions, and a detailed synthesis explaining the reasoning process. Must include exact dollar amounts for all levels.",
        agent=agents["master_analyst"],
    )
    
    return [
        data_preparation_task,
        global_macro_analysis_task,
        technical_analysis_task,
        fundamental_analysis_task,
        sentiment_analysis_task,
        equity_research_task,
        master_synthesis_task,
    ]