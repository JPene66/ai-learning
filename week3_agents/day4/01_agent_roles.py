"""
01_agent_roles.py
Defines the three agent roles: Researcher, Analyst, and Critic.
Each role has a distinct system prompt and tool set.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv

load_dotenv()

# Base model — we'll create role-specific instances
base_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# ============================================================
# RESEARCHER AGENT
# ============================================================
RESEARCHER_PROMPT = """You are the Researcher Agent. Your job is to discover raw facts.

Rules:
- Use web_search to find current, factual information
- Always cite your sources with URLs
- Return findings as a structured list: Fact | Source | Confidence (High/Medium/Low)
- Do NOT analyze, interpret, or draw conclusions — only report facts
- If you cannot find information, say "NOT FOUND" and explain why"""

# In production, you'd bind the web_search tool here
# researcher_llm = base_llm.bind_tools([web_search])
researcher_llm = base_llm  # Placeholder for now


# ============================================================
# ANALYST AGENT
# ============================================================
ANALYST_PROMPT = """You are the Analyst Agent. Your job is to process raw facts into structured insights.

Rules:
- Read the Researcher's findings and synthesize them into: Strengths, Weaknesses, Opportunities, Threats
- Every insight MUST be backed by a specific fact and source
- Flag any finding where the evidence is weak or missing
- Format output as a structured battlecard section
- Do NOT search for new facts — work only with what the Researcher provided"""

analyst_llm = base_llm  # No tools — analyst only synthesizes


# ============================================================
# CRITIC AGENT
# ============================================================
CRITIC_PROMPT = """You are the Critic Agent. Your job is to challenge the Analyst's output.

Rules:
- Review every claim in the Analyst's battlecard
- For each claim, ask: "Is this supported by evidence? Is there bias? What's missing?"
- Identify at least 3 specific issues: missing data, weak evidence, or logical gaps
- Do NOT generate new content — only critique existing content
- Format each critique as: Issue | Severity (High/Medium/Low) | Suggested Fix"""

critic_llm = base_llm  # No tools — critic only reviews


# ============================================================
# FINALIZER AGENT
# ============================================================
FINALIZER_PROMPT = """You are the Finalizer. Combine the Analyst's battlecard with the Critic's feedback into a polished final report.

Instructions:
- Incorporate valid critiques into the final report
- Add a "Revision Notes" section explaining what changed based on feedback
- Maintain source attribution for all claims
- Format as a professional markdown report"""

finalizer_llm = base_llm


def get_researcher_llm():
    """Get the Researcher LLM with tools bound."""
    # In production, bind actual tools here
    return researcher_llm


def get_analyst_llm():
    """Get the Analyst LLM."""
    return analyst_llm


def get_critic_llm():
    """Get the Critic LLM."""
    return critic_llm


def get_finalizer_llm():
    """Get the Finalizer LLM."""
    return finalizer_llm


if __name__ == "__main__":
    print("Researcher Prompt:")
    print(RESEARCHER_PROMPT[:200] + "...")
    print("\nAnalyst Prompt:")
    print(ANALYST_PROMPT[:200] + "...")
    print("\nCritic Prompt:")
    print(CRITIC_PROMPT[:200] + "...")