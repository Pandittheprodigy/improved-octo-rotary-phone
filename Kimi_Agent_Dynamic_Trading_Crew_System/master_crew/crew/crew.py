"""
Master Trading Crew - Crew Orchestration
Hierarchical multi-agent system for institutional-grade trade recommendations
"""
from crewai import Crew, Process
from crew.agents import create_agents
from crew.tasks import create_tasks
from utils.data_fetcher import fetch_all_data
import streamlit as st


class MasterTradingCrew:
    """
    Orchestrates the complete Master Trading Crew workflow.
    Manages agent creation, task sequencing, and result compilation.
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.agents = None
        self.tasks = None
        self.crew = None
        self.data_context = None
        self.results = {}
    
    def prepare_data(self):
        """Phase 0: Fetch and prepare all data"""
        with st.spinner("📊 Fetching market data and news..."):
            self.data_context = fetch_all_data(self.symbol)
        
        if "error" in self.data_context.get("raw_data", {}):
            return False, self.data_context["raw_data"]["error"]
        
        return True, "Data prepared successfully"
    
    def setup_crew(self):
        """Setup the complete crew with agents and tasks"""
        # Create all agents
        self.agents = create_agents()
        
        # Create tasks with data context
        self.tasks = create_tasks(self.agents, self.symbol, self.data_context)
        
        # Create the crew with sequential process (hierarchical flow)
        self.crew = Crew(
            agents=list(self.agents.values()),
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=True,
            max_rpm=30,  # Rate limiting for API
            cache=True,
        )
    
    def run(self):
        """Execute the complete crew workflow"""
        # Step 1: Prepare data
        success, message = self.prepare_data()
        if not success:
            return {"error": message}
        
        # Step 2: Setup crew
        self.setup_crew()
        
        # Step 3: Execute
        try:
            with st.spinner("🧠 Master Crew analyzing... This may take 2-3 minutes..."):
                result = self.crew.kickoff()
                
            # Compile comprehensive results
            self.results = {
                "symbol": self.symbol,
                "company_name": self.data_context.get("raw_data", {}).get("company_name", self.symbol),
                "current_price": self.data_context.get("raw_data", {}).get("current_price", 0),
                "crew_output": result,
                "raw_data": self.data_context.get("raw_data", {}),
                "market_summary": self.data_context.get("market_summary", {}),
                "earnings_data": self.data_context.get("earnings_data", {}),
                "technical_indicators": self.data_context.get("raw_data", {}).get("technical_indicators", {}),
                "news_items": self.data_context.get("news_items", []),
            }
            
            return self.results
            
        except Exception as e:
            return {"error": f"Crew execution error: {str(e)}"}


def run_master_crew(symbol: str):
    """
    Convenience function to run the Master Trading Crew.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
    
    Returns:
        Dictionary with complete analysis and trade recommendation
    """
    crew = MasterTradingCrew(symbol)
    return crew.run()