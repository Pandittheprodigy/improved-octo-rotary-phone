"""
LLM Provider Factory — NVIDIA NIM Only (Per-Agent API Keys)
Each agent uses its own dedicated NVIDIA API key for isolation and rate limit management.

FREE TIER: 6,000 requests/month at build.nvidia.com
"""
import os
from typing import Optional

# ============================================================
# AGENT-TO-API-KEY MAPPING — NVIDIA NIM
# ============================================================
NVIDIA_AGENT_KEY_MAP = {
    "data_scientist": "NVIDIA_API_KEY_DATA_SCIENTIST",
    "global_news_analyst": "NVIDIA_API_KEY_GLOBAL_NEWS",
    "technical_analyst": "NVIDIA_API_KEY_TECHNICAL",
    "fundamental_analyst": "NVIDIA_API_KEY_FUNDAMENTAL",
    "sentiment_analyst": "NVIDIA_API_KEY_SENTIMENT",
    "equity_research_analyst": "NVIDIA_API_KEY_EQUITY_RESEARCH",
    "master_analyst": "NVIDIA_API_KEY_MASTER",
}

DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"
FALLBACK_MODEL = "meta/llama-3.1-70b-instruct"


def get_agent_llm(agent_name: str, temperature: float = 0.2, model: Optional[str] = None):
    """
    Create a ChatNVIDIA LLM instance for a specific agent using its dedicated API key.
    
    Args:
        agent_name: One of the keys in NVIDIA_AGENT_KEY_MAP
        temperature: Creativity vs determinism (0.0-1.0)
        model: Override default model
    
    Returns:
        ChatNVIDIA: Configured LLM instance
    
    Raises:
        ValueError: If agent_name is unknown or API key is missing
    """
    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
    except ImportError:
        raise ImportError("Install: pip install langchain-nvidia-ai-endpoints")
    
    if agent_name not in NVIDIA_AGENT_KEY_MAP:
        raise ValueError(
            f"Unknown agent: {agent_name}. Valid: {list(NVIDIA_AGENT_KEY_MAP.keys())}"
        )
    
    env_var = NVIDIA_AGENT_KEY_MAP[agent_name]
    api_key = os.getenv(env_var)
    
    if not api_key:
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError(
                f"API key missing for {agent_name}.\\n"
                f"Set {env_var} in .streamlit/secrets.toml or environment."
            )
    
    model_id = model or DEFAULT_MODEL
    
    return ChatNVIDIA(
        model=model_id,
        api_key=api_key,
        temperature=temperature,
        max_tokens=4096,
        top_p=0.9,
    )


def get_all_agent_status() -> dict:
    """Check which agents have their API keys configured."""
    status = {}
    for agent, env_var in NVIDIA_AGENT_KEY_MAP.items():
        key = os.getenv(env_var) or os.getenv("NVIDIA_API_KEY")
        status[agent] = {
            "configured": bool(key),
            "env_var": env_var,
            "key_preview": f"{key[:12]}..." if key else None,
        }
    return status


def validate_all_keys() -> tuple[bool, list[str]]:
    """Validate all agent API keys are present. Returns (all_ok, missing_agents)."""
    missing = []
    fallback = os.getenv("NVIDIA_API_KEY")
    
    for agent, env_var in NVIDIA_AGENT_KEY_MAP.items():
        if not os.getenv(env_var) and not fallback:
            missing.append(f"{agent} ({env_var})")
    
    return len(missing) == 0, missing
