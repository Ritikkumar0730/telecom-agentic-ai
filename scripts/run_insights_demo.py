"""
Demo: LLM-generated business insights briefing.

Run with:
    python scripts/run_insights_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import get_llm
from data.seed import get_connection
from agents.insights_agent import InsightsAgent


def main():
    llm = get_llm()
    conn = get_connection()
    agent = InsightsAgent(llm, conn)
    print(agent.daily_briefing())


if __name__ == "__main__":
    main()
