"""
Master Trading Crew — Orchestration Engine
Manages the flow: Layer 1 (Foundation) → Layer 2 (Specialists) → Layer 3 (Master)
Uses SEQUENTIAL process with explicit context chaining for reliability.
"""
from crewai import Crew, Process
from crew.agents import (
    create_data_scientist,
    create_global_news_analyst,
    create_technical_analyst,
    create_fundamental_analyst,
    create_sentiment_analyst,
    create_equity_research_analyst,
    create_master_analyst,
)
from crew.tasks import (
    create_data_validation_task,
    create_global_macro_task,
    create_technical_analysis_task,
    create_fundamental_analysis_task,
    create_sentiment_analysis_task,
    create_equity_research_task,
    create_master_synthesis_task,
)
from utils.data_fetcher import StockDataFetcher


class MasterTradingCrew:
    """
    Institutional-grade multi-agent trading intelligence system.
    7 agents, 7 tasks, 3-layer sequential execution with context chaining.
    """
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.data_fetcher = StockDataFetcher()
        self.raw_data = None
        self.results = {}
        
        # Initialize all agents with their own API keys
        self.agents = {
            "data_scientist": create_data_scientist(),
            "global_news_analyst": create_global_news_analyst(),
            "technical_analyst": create_technical_analyst(),
            "fundamental_analyst": create_fundamental_analyst(),
            "sentiment_analyst": create_sentiment_analyst(),
            "equity_research_analyst": create_equity_research_analyst(),
            "master_analyst": create_master_analyst(),
        }
    
    def fetch_data(self) -> dict:
        """Fetch and format all market data."""
        data = self.data_fetcher.get_stock_data(self.ticker)
        if "error" in data:
            return data
        
        formatted = self.data_fetcher.format_for_agents(data)
        news = self.data_fetcher.get_news_summary(self.ticker)
        
        self.raw_data = {
            "market_data": formatted,
            "news": news,
            "full_data": data,
        }
        return self.raw_data
    
    def build_crew(self) -> Crew:
        """Build the crew with sequential process and explicit context chaining."""
        raw_text = self.raw_data["market_data"] if self.raw_data else ""
        
        # Layer 1: Data Foundation
        data_task = create_data_validation_task(
            self.agents["data_scientist"], self.ticker, raw_text
        )
        macro_task = create_global_macro_task(
            self.agents["global_news_analyst"], self.ticker, raw_text
        )
        
        # Layer 2: Specialist Analysis
        tech_task = create_technical_analysis_task(
            self.agents["technical_analyst"], self.ticker, raw_text
        )
        fund_task = create_fundamental_analysis_task(
            self.agents["fundamental_analyst"], self.ticker, raw_text
        )
        sent_task = create_sentiment_analysis_task(
            self.agents["sentiment_analyst"], self.ticker, raw_text
        )
        research_task = create_equity_research_task(
            self.agents["equity_research_analyst"], self.ticker, raw_text
        )
        
        # Layer 3: Master Synthesis
        master_task = create_master_synthesis_task(
            self.agents["master_analyst"], self.ticker, "[All analyst outputs will be injected here]"
        )
        
        # Context chaining: Layer 2 depends on Layer 1
        tech_task.context = [data_task, macro_task]
        fund_task.context = [data_task, macro_task]
        sent_task.context = [data_task, macro_task]
        research_task.context = [data_task, macro_task]
        
        # Layer 3 depends on all Layer 2
        master_task.context = [tech_task, fund_task, sent_task, research_task]
        
        # SEQUENTIAL process ensures reliable context passing
        crew = Crew(
            agents=list(self.agents.values()),
            tasks=[data_task, macro_task, tech_task, fund_task, sent_task, research_task, master_task],
            process=Process.sequential,
            verbose=True,
            memory=True,
            max_rpm=10,
        )
        
        return crew
    
    def run(self) -> dict:
        """Execute the full crew workflow."""
        data_result = self.fetch_data()
        if "error" in data_result:
            return {"error": data_result["error"]}
        
        crew = self.build_crew()
        result = crew.kickoff()
        
        self.results = {
            "ticker": self.ticker,
            "raw_data": self.raw_data,
            "crew_output": result,
            "agent_outputs": {},
        }
        
        # Extract individual task outputs
        for task in crew.tasks:
            if hasattr(task, 'output') and task.output:
                self.results["agent_outputs"][task.agent.role] = str(task.output)
        
        return self.results


def run_master_crew(ticker: str) -> dict:
    """Convenience function to run the full crew analysis."""
    crew = MasterTradingCrew(ticker)
    return crew.run()
