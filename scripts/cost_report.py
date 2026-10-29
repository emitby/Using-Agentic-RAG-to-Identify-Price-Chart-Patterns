"""Summarize every logged Claude API call by run, agent, tool and model.

Reads the append-only logs written by costs.CostTracker:
    logs/token_usage.jsonl   one line per API call (exact tokens and cost)
    logs/tool_calls.jsonl    one line per tool execution (result size, estimate)

Writes:
    outputs/cost_by_run.csv, cost_by_agent.csv, cost_by_tool.csv, cost_by_model.csv
    outputs/cost_summary.json
    outputs/cost_breakdown.png   (cost by agent and by tool, all runs)

Tool attribution follows costs.py: an API call is charged to the tools it requested
(split evenly), or to "final_answer" if it requested none, so per-tool call costs sum
exactly to the real total. The separate "est. result cost" column estimates how much of
later calls' input was spent re-reading each tool's output. That money is already inside
the call costs, so the two columns must never be added together.

Usage:
    ./run.sh scripts/cost_report.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from rag_charts import config  # noqa: E402
from rag_charts.guardrails import ensure_dir, read_jsonl, safe_path, write_json  # noqa: E402

BAR_COLOR = "#2f5d9e"   # one hue: each panel is a single series, so no legend is needed
INK, MUTED = "#222222", "#666666"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both logs as DataFrames (empty frames if nothing has been logged yet)."""
    calls = pd.DataFrame(read_jsonl(config.USAGE_LOG_FILE))
    tools = pd.DataFrame(read_jsonl(config.TOOL_LOG_FILE))
    return calls, tools


def by_tool(calls: pd.DataFrame, tools: pd.DataFrame) -> pd.DataFrame:
    """Per-tool call-attributed cost plus estimated result-ingestion cost."""
    rows = []
    for _, call in calls.iterrows():
        targets = call["attributed_to"]
        for name in targets:
            rows.append({"tool": name, "call_cost_usd": call["cost_usd"] / len(targets)})
    attributed = pd.DataFrame(rows).groupby("tool")["call_cost_usd"].sum()
    if tools.empty:
        return attributed.to_frame().assign(calls=0, est_result_cost_usd=0.0)
    usage = tools.groupby("tool").agg(calls=("tool", "size"),
                                      est_result_tokens=("est_result_tokens", "sum"),
                                      est_result_cost_usd=("est_result_input_cost_usd", "sum"))
    table = usage.join(attributed, how="outer").fillna(0)
    return table.sort_values("call_cost_usd", ascending=False)


def run_kind(run_id: str) -> str:
    """Label a run as an eval run or a pipeline run from its id."""
    return "eval" if str(run_id).startswith("eval") else "pipeline"


def plot(agent_cost: pd.Series, tool_cost: pd.Series, out_path) -> str:
    """Two horizontal bar panels: total cost by agent, and by tool (all runs).

    Horizontal bars because the category names are long; values are labeled at the bar
    ends in ink color, and the grid is kept faint so the bars carry the reading.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, series, title in ((axes[0], agent_cost, "Cost by agent"),
                              (axes[1], tool_cost, "Cost by tool (attributed API calls)")):
        series = series.sort_values()
        ax.barh(series.index, series.values, color=BAR_COLOR, height=0.55)
        for y, value in enumerate(series.values):
            ax.text(value, y, f"  ${value:.2f}", va="center", fontsize=9, color=INK)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("USD, all logged runs", color=MUTED, fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(axis="x", alpha=0.2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlim(0, series.max() * 1.25)
    fig.tight_layout()
    target = safe_path(out_path)
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return str(target)


def main() -> None:
    """Build every table, the chart, and the JSON summary; print the headline numbers."""
    calls, tools = load()
    if calls.empty:
        print("No API calls logged yet.")
        return
    ensure_dir(config.OUTPUT_DIR)
    token_cols = ["input_tokens", "output_tokens", "cache_write_tokens", "cache_read_tokens"]
    calls["kind"] = calls["run_id"].map(run_kind)

    runs = calls.groupby(["run_id", "kind"]).agg(
        api_calls=("cost_usd", "size"), cost_usd=("cost_usd", "sum"),
        **{c: (c, "sum") for c in token_cols}).reset_index()
    agents = calls.groupby("agent").agg(api_calls=("cost_usd", "size"),
                                        cost_usd=("cost_usd", "sum"),
                                        **{c: (c, "sum") for c in token_cols})
    models = calls.groupby("model").agg(api_calls=("cost_usd", "size"),
                                        cost_usd=("cost_usd", "sum"))
    tools_table = by_tool(calls, tools)

    runs.to_csv(safe_path(config.OUTPUT_DIR / "cost_by_run.csv"), index=False)
    agents.to_csv(safe_path(config.OUTPUT_DIR / "cost_by_agent.csv"))
    tools_table.to_csv(safe_path(config.OUTPUT_DIR / "cost_by_tool.csv"))
    models.to_csv(safe_path(config.OUTPUT_DIR / "cost_by_model.csv"))
    chart = plot(agents["cost_usd"], tools_table["call_cost_usd"],
                 config.OUTPUT_DIR / "cost_breakdown.png")

    pipeline_runs = runs[runs["kind"] == "pipeline"]
    summary = {
        "total_cost_usd": round(float(calls["cost_usd"].sum()), 4),
        "api_calls": int(len(calls)), "tool_calls": int(len(tools)),
        "runs": int(len(runs)),
        "avg_pipeline_run_usd": round(float(pipeline_runs["cost_usd"].mean()), 4) if len(pipeline_runs) else None,
        "cache_read_share_of_input": round(float(
            calls["cache_read_tokens"].sum()
            / max(calls[["input_tokens", "cache_write_tokens", "cache_read_tokens"]].sum().sum(), 1)), 3),
        "by_agent_usd": agents["cost_usd"].round(4).to_dict(),
        "by_tool_usd": tools_table["call_cost_usd"].round(4).to_dict(),
        "by_tool_est_result_usd": tools_table["est_result_cost_usd"].round(4).to_dict(),
        "by_model_usd": models["cost_usd"].round(4).to_dict(),
        "chart": chart,
    }
    write_json(config.OUTPUT_DIR / "cost_summary.json", summary)
    print(runs.to_string(index=False))
    print(tools_table.round(4).to_string())
    print({k: v for k, v in summary.items() if not k.startswith("by_")})


if __name__ == "__main__":
    main()
