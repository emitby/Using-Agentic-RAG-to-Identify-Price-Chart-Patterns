"""Token and cost tracking for every Claude API call, attributed by agent and by tool.

Two append-only logs are written under logs/:

* token_usage.jsonl: one line per API call with exact token counts from the API's
  `usage` field (input, output, cache write, cache read), the dollar cost, the agent,
  the turn number, the stop reason, and which tools the response asked for.
* tool_calls.jsonl: one line per tool execution with the arguments, result size,
  estimated result tokens, duration, and whether it errored.

Attribution "by tool" works in two parts, because a tool costs money in two places:
1. **Calling it.** The API call whose response requested tools is charged to those tools,
   split evenly if several were requested at once. Its output tokens are the model
   deciding to call the tool and writing the arguments. A response that requests no tool
   is charged to "final_answer".
2. **Reading its result.** A tool's result text is sent back as input on the next call.
   Its size is estimated at 4 characters per token (the API reports input only as a
   total) and priced at the input rate. This is labeled as an estimate everywhere.

A per-run budget stops the run before the next API call once spend reaches the cap.
"""

from datetime import datetime, timezone

from rag_charts import config
from rag_charts.guardrails import GuardrailError, append_jsonl

CHARS_PER_TOKEN = 4  # rough English average; used only for tool-result estimates


class BudgetExceeded(GuardrailError):
    """Raised when a run's tracked spend reaches its budget."""


def price_for(model: str) -> dict:
    """Return the per-million-token prices for a model, falling back to the main model.

    Args:
        model: Model id reported by the API (a refusal fallback may differ from the one
            requested).

    Returns:
        Dict with input, output, cache_write, cache_read prices in USD per 1M tokens.
    """
    return config.PRICING.get(model, config.PRICING[config.MODEL])


def call_cost(model: str, usage) -> tuple[dict, float]:
    """Compute token counts and dollar cost for one API response.

    Args:
        model: Model id from the response.
        usage: The response's `usage` object (fields may be None).

    Returns:
        (token counts dict, cost in USD).
    """
    tokens = {
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "cache_write_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
    }
    rates = price_for(model)
    cost = (tokens["input_tokens"] * rates["input"]
            + tokens["output_tokens"] * rates["output"]
            + tokens["cache_write_tokens"] * rates["cache_write"]
            + tokens["cache_read_tokens"] * rates["cache_read"]) / 1_000_000
    return tokens, cost


class CostTracker:
    """Accumulates and logs the cost of one pipeline or eval run.

    Attributes:
        run_id: Identifier written on every log line.
        budget_usd: Spend cap for the run.
        total_usd: Spend so far.
        calls: In-memory copy of every API-call record.
        tools: In-memory copy of every tool-call record.
        log: When False, nothing is written to disk (used by offline tests).
    """

    def __init__(self, run_id: str, budget_usd: float, log: bool = True):
        self.run_id = run_id
        self.budget_usd = budget_usd
        self.total_usd = 0.0
        self.calls: list[dict] = []
        self.tools: list[dict] = []
        self.log = log

    def check_budget(self) -> None:
        """Raise `BudgetExceeded` if the run has already spent its budget.

        Called before every API request, so the overshoot is at most one call.
        """
        if self.total_usd >= self.budget_usd:
            raise BudgetExceeded(f"budget: run {self.run_id} spent ${self.total_usd:.2f} "
                                 f"of its ${self.budget_usd:.2f} cap")

    def record_call(self, agent: str, turn: int, response, requested_tools: list[str]) -> dict:
        """Log one API response's usage and cost.

        Args:
            agent: Agent name (researcher, analyst, verifier, ...).
            turn: Turn number within the agent's loop (1-based).
            response: The API response object.
            requested_tools: Names of tools the response asked to call.

        Returns:
            The record written.

        Side effects:
            Updates totals; appends to logs/token_usage.jsonl when logging is on.
        """
        model = getattr(response, "model", config.MODEL) or config.MODEL
        tokens, cost = call_cost(model, response.usage)
        self.total_usd += cost
        record = {
            "time": datetime.now(timezone.utc).isoformat(), "run_id": self.run_id,
            "agent": agent, "turn": turn, "model": model,
            "stop_reason": getattr(response, "stop_reason", None),
            **tokens, "cost_usd": round(cost, 6),
            "requested_tools": requested_tools,
            "attributed_to": requested_tools or ["final_answer"],
        }
        self.calls.append(record)
        if self.log:
            append_jsonl(config.USAGE_LOG_FILE, record)
        return record

    def record_tool(self, agent: str, tool: str, args: dict, result: str, error: bool,
                    seconds: float) -> dict:
        """Log one tool execution and the estimated cost of feeding its result back.

        Args:
            agent: Agent that called the tool.
            tool: Tool name.
            args: Tool arguments.
            result: Text returned to the model.
            error: Whether the tool raised.
            seconds: Wall-clock duration.

        Returns:
            The record written.

        Side effects:
            Appends to logs/tool_calls.jsonl when logging is on.
        """
        est_tokens = len(result) // CHARS_PER_TOKEN
        record = {
            "time": datetime.now(timezone.utc).isoformat(), "run_id": self.run_id,
            "agent": agent, "tool": tool, "args": args, "error": error,
            "result_chars": len(result), "est_result_tokens": est_tokens,
            "est_result_input_cost_usd": round(
                est_tokens * price_for(config.MODEL)["input"] / 1_000_000, 6),
            "seconds": round(seconds, 3),
        }
        self.tools.append(record)
        if self.log:
            append_jsonl(config.TOOL_LOG_FILE, record)
        return record

    def summary(self) -> dict:
        """Totals for this run by agent, by tool, and overall.

        Returns:
            Dict with total cost and tokens, per-agent costs, per-tool call-attributed costs
            and estimated result-ingestion costs, and API/tool call counts.
        """
        by_agent: dict[str, float] = {}
        by_tool: dict[str, dict] = {}
        totals = {"input_tokens": 0, "output_tokens": 0, "cache_write_tokens": 0,
                  "cache_read_tokens": 0}
        for call in self.calls:
            by_agent[call["agent"]] = by_agent.get(call["agent"], 0) + call["cost_usd"]
            for key in totals:
                totals[key] += call[key]
            share = call["cost_usd"] / len(call["attributed_to"])
            for name in call["attributed_to"]:
                slot = by_tool.setdefault(name, {"calls": 0, "call_cost_usd": 0.0,
                                                 "est_result_cost_usd": 0.0})
                slot["call_cost_usd"] += share
        for tool in self.tools:
            slot = by_tool.setdefault(tool["tool"], {"calls": 0, "call_cost_usd": 0.0,
                                                     "est_result_cost_usd": 0.0})
            slot["calls"] += 1
            slot["est_result_cost_usd"] += tool["est_result_input_cost_usd"]
        return {
            "run_id": self.run_id, "total_cost_usd": round(self.total_usd, 4),
            "budget_usd": self.budget_usd, "api_calls": len(self.calls),
            "tool_calls": len(self.tools), **totals,
            "by_agent_usd": {k: round(v, 4) for k, v in by_agent.items()},
            "by_tool": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                            for kk, vv in v.items()} for k, v in by_tool.items()},
        }
