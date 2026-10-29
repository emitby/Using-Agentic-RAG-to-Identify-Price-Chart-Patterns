"""Agentic RAG chart-pattern prototype.

Pulls stock prices, detects chart-pattern candidates with deterministic code, and uses
three Claude agents (Researcher, Chart Analyst, Verifier) grounded in a corpus built from
StockCharts ChartSchool, Investopedia and ThePatternSite to decide which patterns to draw,
annotate them with cited descriptions, and give a bullish/bearish verdict.
"""
