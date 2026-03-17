"""
Kalshi AI Trading Bot — MCP Server
====================================
Exposes the debate pipeline and Kalshi market data as MCP tools so that an
orchestrating agent (OpenClaw, Claude Desktop, etc.) can call them on demand
and use the results as context for follow-up conversation.

Tools
-----
list_markets(series, min_volume, limit)
    Discover live Kalshi markets.  Cheap — pure REST, no LLM spend.

get_market_ladder(event_ticker)
    Return the full probability ladder for a multi-strike event (e.g. all
    CPI thresholds for a given month).  Lets the agent read the implied
    distribution before committing to a single-strike analysis.

analyze_market(ticker, paper_trade)
    Run the full 7-agent debate on a specific market: KnowledgeResearcher
    (RAG + worldview + web), ForecasterAgent, NewsAnalystAgent,
    BullResearcher, BearResearcher, RiskManagerAgent, TraderAgent.
    Returns the structured result including per-agent probabilities, EV,
    position sizing, and the full trader reasoning.  Expensive (~3-5 min,
    ~$0.05-0.10 in LLM costs) — call deliberately.

Usage
-----
Stdio (Claude Desktop / OpenClaw):
    python -m src.mcp_server

TCP (remote agents / testing):
    python -m src.mcp_server --transport streamable-http --port 8765
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_request

# Ensure project root is on the path when run as __main__
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"))

import structlog
import logging
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))

from src.clients.kalshi_client import KalshiClient
from src.clients.model_router import ModelRouter
from src.agents.debate import DebateRunner
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("mcp_server")

# ---------------------------------------------------------------------------
# Role → model assignments (mirrors the test runs)
# ---------------------------------------------------------------------------
ROLE_MODEL_MAP: dict[str, str] = {
    "forecaster":           "anthropic/claude-sonnet-4-5",
    "news_analyst":         "anthropic/claude-sonnet-4-5",
    "knowledge_researcher": "mistralai/mistral-nemo",
    "bull_researcher":      "openai/o3",
    "bear_researcher":      "google/gemini-3-flash-preview",
    "risk_manager":         "deepseek/deepseek-v3.2",
    "trader":               "anthropic/claude-sonnet-4-5",
}

# ---------------------------------------------------------------------------
# Shared clients (initialised once at server startup via lifespan)
# ---------------------------------------------------------------------------
_kalshi: KalshiClient | None = None
_router: ModelRouter | None = None


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Initialise and cleanly shut down shared clients."""
    global _kalshi, _router
    logger.info("MCP server starting — initialising clients")
    _kalshi = KalshiClient(read_only=True)   # read_only=True → skips key loading, GET endpoints only
    _router = ModelRouter()
    logger.info("MCP server ready")
    try:
        yield
    finally:
        logger.info("MCP server shutting down")
        if _kalshi:
            await _kalshi.close()
        if _router:
            await _router.close()


# ---------------------------------------------------------------------------
# MCP server instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    name="kalshi-trading-analyst",
    instructions=(
        "You have access to a Kalshi prediction-market analysis system backed "
        "by a 7-agent AI debate pipeline (Forecaster, News Analyst, Knowledge "
        "Researcher with RAG + worldview, Bull Researcher, Bear Researcher, "
        "Risk Manager, Trader).  Use list_markets or get_market_ladder to "
        "discover markets, then call analyze_market on the ticker you want to "
        "examine in depth.  analyze_market is expensive — call it deliberately."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helper: build per-role completion callables from the shared router
# ---------------------------------------------------------------------------
def _make_completions(market_id: str) -> dict[str, Any]:
    """Return a dict of role -> async get_completion callable."""
    assert _router is not None, "Router not initialised"

    def _make_fn(model: str):
        async def _fn(prompt: str) -> str:
            result = await _router.get_completion(
                prompt=prompt,
                model=model,
                strategy="mcp_debate",
                query_type="agent_analysis",
                market_id=market_id,
            )
            return result or ""
        return _fn

    return {role: _make_fn(model) for role, model in ROLE_MODEL_MAP.items()}


# ---------------------------------------------------------------------------
# Tool 1 — list_markets
# ---------------------------------------------------------------------------
@mcp.tool
async def list_markets(
    series: str = "",
    min_volume: int = 1000,
    limit: int = 25,
) -> dict:
    """
    Discover live, priced Kalshi prediction markets.

    Args:
        series: Optional Kalshi series ticker prefix to filter by
                (e.g. "KXCPI", "KXFED", "KXGDP").  Leave empty to
                browse all non-sports active markets.
        min_volume: Only return markets with at least this many contracts
                    traded.  Default 1 000.
        limit: Maximum number of markets to return.  Default 25, max 100.

    Returns:
        {
          "markets": [
            {
              "ticker": str,
              "title": str,
              "yes_mid": float,   # midpoint of bid/ask in dollars (0-1)
              "yes_ask": float,
              "yes_bid": float,
              "volume": int,
              "open_interest": int,
              "close_date": str,  # YYYY-MM-DD
              "rules": str,
            }, ...
          ],
          "count": int,
          "note": str,
        }
    """
    assert _kalshi is not None, "Kalshi client not initialised"
    limit = min(int(limit), 100)

    params: dict[str, Any] = {"limit": min(limit * 4, 200)}  # over-fetch then filter
    if series:
        params["series_ticker"] = series

    resp = await _kalshi.get_markets(**params)
    raw = resp.get("markets", [])

    results = []
    for m in raw:
        ya = float(m.get("yes_ask_dollars") or 0)
        yb = float(m.get("yes_bid_dollars") or 0)
        vol = float(m.get("volume_fp") or 0)
        status = m.get("status", "")

        # Filter: active, priced on both sides, meets volume floor
        if status != "active" or ya <= 0.01 or ya >= 0.99 or vol < min_volume:
            continue

        results.append({
            "ticker":        m.get("ticker", ""),
            "title":         m.get("title", ""),
            "yes_mid":       round((ya + yb) / 2, 3),
            "yes_ask":       round(ya, 3),
            "yes_bid":       round(yb, 3),
            "volume":        int(vol),
            "open_interest": int(float(m.get("open_interest_fp") or 0)),
            "close_date":    (m.get("close_time") or "")[:10],
            "rules":         (m.get("rules_primary") or "")[:200],
        })

        if len(results) >= limit:
            break

    return {
        "markets": results,
        "count":   len(results),
        "note": (
            "Prices in dollars (0-1 scale). yes_mid is the bid/ask midpoint. "
            "Use get_market_ladder for a full strike ladder, or analyze_market "
            "to run the full debate on a specific ticker."
        ),
    }


# ---------------------------------------------------------------------------
# Tool 2 — get_market_ladder
# ---------------------------------------------------------------------------
@mcp.tool
async def get_market_ladder(event_ticker: str) -> dict:
    """
    Return the full probability ladder for a multi-strike Kalshi event.

    Fetches every market in the event and sorts them by strike, giving you
    the market's implied cumulative probability distribution (e.g. all CPI
    thresholds for March 2026, or all GDP thresholds for Q1 2026).

    Args:
        event_ticker: The Kalshi event ticker (e.g. "KXCPI-26MAR",
                      "KXGDP-26APR30", "KXFED-26JUN").

    Returns:
        {
          "event_ticker": str,
          "strikes": [
            {
              "ticker": str,
              "threshold": str,   # human-readable strike label
              "yes_mid": float,
              "yes_ask": float,
              "yes_bid": float,
              "volume": int,
              "close_date": str,
              "rules": str,
            }, ...
          ],
          "count": int,
          "interpretation": str,
        }
    """
    assert _kalshi is not None, "Kalshi client not initialised"

    resp = await _kalshi.get_markets(limit=50, event_ticker=event_ticker)
    raw = resp.get("markets", [])

    strikes = []
    for m in raw:
        ya = float(m.get("yes_ask_dollars") or 0)
        yb = float(m.get("yes_bid_dollars") or 0)
        vol = float(m.get("volume_fp") or 0)
        strikes.append({
            "ticker":     m.get("ticker", ""),
            "threshold":  m.get("subtitle") or m.get("yes_sub_title") or m.get("ticker", ""),
            "yes_mid":    round((ya + yb) / 2, 3),
            "yes_ask":    round(ya, 3),
            "yes_bid":    round(yb, 3),
            "volume":     int(vol),
            "close_date": (m.get("close_time") or "")[:10],
            "rules":      (m.get("rules_primary") or "")[:200],
        })

    # Sort by yes_mid descending (highest-probability / lowest strike first)
    strikes.sort(key=lambda x: x["yes_mid"], reverse=True)

    return {
        "event_ticker":   event_ticker,
        "strikes":        strikes,
        "count":          len(strikes),
        "interpretation": (
            "Each row is a cumulative threshold: 'YES mid = 0.70' means the "
            "market implies a 70% chance the outcome exceeds that threshold. "
            "The gap between adjacent strikes approximates the probability mass "
            "the market assigns to the outcome falling in that range. "
            "Use analyze_market on a specific ticker for an AI probability estimate."
        ),
    }


# ---------------------------------------------------------------------------
# Tool 3 — analyze_market
# ---------------------------------------------------------------------------
@mcp.tool
async def analyze_market(
    ticker: str,
    paper_trade: bool = False,
) -> dict:
    """
    Run the full 7-agent AI debate on a specific Kalshi market.

    Agents (run in structured sequence):
      1. ForecasterAgent    — base-rate + current conditions probability
      2. NewsAnalystAgent   — sentiment and news impact direction
      3. KnowledgeResearcher — RAG retrieval from document library,
                               DuckDuckGo web search, and worldview injection
      4. BullResearcher     — strongest case for YES
      5. BearResearcher     — strongest case for NO
      6. RiskManagerAgent   — EV calculation and position sizing
      7. TraderAgent        — final synthesis through the worldview lens

    This call is EXPENSIVE in time (~3-5 minutes) and LLM cost (~$0.05-0.15).
    Call it deliberately on markets you have genuine interest in.

    Args:
        ticker:      Kalshi market ticker (e.g. "KXCPI-26MAR-T0.7").
                     Use list_markets or get_market_ladder to find tickers.
        paper_trade: If True, log the result as a paper trade in the local
                     database.  Default False (analysis only).

    Returns:
        {
          "market": { ticker, title, yes_price, no_price, volume, days_to_expiry },
          "decision": {
            "action":           "BUY" | "SELL" | "SKIP",
            "side":             "YES" | "NO",
            "limit_price_cents": int,
            "confidence":       float,
            "position_size_pct": float,
            "ev_per_contract":  float,   # expected value in dollars per $1 contract
          },
          "agents": {
            "forecaster":    { probability, confidence, base_rate, reasoning },
            "news_analyst":  { sentiment, relevance, impact_direction, reasoning },
            "knowledge_researcher": { worldview_applies, key_frameworks, reasoning },
            "bull_researcher": { probability, probability_floor, confidence, key_arguments, reasoning },
            "bear_researcher": { probability, probability_ceiling, confidence, key_arguments, reasoning },
            "risk_manager":  { ev_estimate, risk_score, should_trade, recommended_size_pct, reasoning },
            "trader":        { action, side, limit_price, confidence, reasoning },
          },
          "trader_reasoning": str,   # Full trader reasoning (pre-transcript)
          "elapsed_seconds":  float,
          "error":            str | None,
        }
    """
    assert _kalshi is not None, "Kalshi client not initialised"
    assert _router is not None,  "Model router not initialised"

    # ── 1. Fetch live market data from Kalshi ────────────────────────────
    try:
        resp = await _kalshi.get_market(ticker)
        raw = resp.get("market", {})
    except Exception as exc:
        return {"error": f"Failed to fetch market {ticker!r}: {exc}"}

    if not raw:
        return {"error": f"Market {ticker!r} not found on Kalshi."}

    ya  = float(raw.get("yes_ask_dollars") or 0)
    yb  = float(raw.get("yes_bid_dollars") or 0)
    na  = float(raw.get("no_ask_dollars")  or 0)
    nb  = float(raw.get("no_bid_dollars")  or 0)
    vol = float(raw.get("volume_fp")       or 0)
    oi  = float(raw.get("open_interest_fp") or 0)

    close_time = raw.get("close_time") or raw.get("expiration_time") or ""
    days_to_expiry: float = 30.0
    if close_time:
        from datetime import datetime, timezone
        try:
            expiry_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            now_dt    = datetime.now(timezone.utc)
            days_to_expiry = max(0.0, (expiry_dt - now_dt).total_seconds() / 86400)
        except ValueError:
            pass

    market_data = {
        "ticker":         raw.get("ticker", ticker),
        "title":          raw.get("title", ""),
        "rules":          (raw.get("rules_primary") or "") + " " + (raw.get("rules_secondary") or ""),
        "yes_price":      round((ya + yb) / 2, 3),
        "no_price":       round((na + nb) / 2, 3),
        "volume":         int(vol),
        "open_interest":  int(oi),
        "category":       "economics",   # default; Kalshi v2 doesn't return category
        "days_to_expiry": round(days_to_expiry, 1),
        "expiration_ts":  0,
    }

    # ── 2. Run the debate ────────────────────────────────────────────────
    completions = _make_completions(ticker)
    runner = DebateRunner()

    try:
        result = await runner.run_debate(
            market_data=market_data,
            get_completions=completions,
            context={"portfolio": {"cash": 1000.0}},
        )
    except Exception as exc:
        return {
            "market": market_data,
            "error":  f"Debate pipeline failed: {exc}",
        }

    # ── 3. Compute EV for the recommended trade ──────────────────────────
    action = result.get("action", "SKIP")
    side   = result.get("side", "YES")
    limit_cents = result.get("limit_price", 50)

    ev_per_contract: float = 0.0
    step_results = result.get("step_results", {})

    probs = []
    for role in ("forecaster", "bull_researcher", "bear_researcher"):
        sr = step_results.get(role) or {}
        p  = sr.get("probability")
        c  = sr.get("confidence")
        if p is not None and c is not None:
            try:
                probs.append((float(p), float(c)))
            except (TypeError, ValueError):
                pass

    if probs:
        tw = sum(w for _, w in probs)
        cw_yes = sum(p * w for p, w in probs) / tw if tw > 0 else 0.5
        cw_no  = 1.0 - cw_yes
        entry  = limit_cents / 100.0

        if action != "SKIP":
            if action == "BUY" and side == "YES":
                ev_per_contract = cw_yes * (1.0 - entry) - cw_no * entry
            elif action == "BUY" and side == "NO":
                ev_per_contract = cw_no * (1.0 - entry) - cw_yes * entry
            elif action == "SELL" and side == "YES":
                ev_per_contract = cw_no * entry - cw_yes * (1.0 - entry)
            else:  # SELL NO
                ev_per_contract = cw_yes * entry - cw_no * (1.0 - entry)

    # ── 4. Shape the response ─────────────────────────────────────────────
    def _agent_summary(role: str, fields: list[str]) -> dict:
        sr = step_results.get(role) or {}
        if "error" in sr:
            return {"error": sr["error"]}
        out = {}
        for f in fields:
            v = sr.get(f)
            if v is not None:
                out[f] = v
        # Truncate reasoning so it's readable but not overwhelming
        rsn = sr.get("reasoning", "")
        out["reasoning"] = rsn[:800] if rsn else ""
        return out

    agents_summary = {
        "forecaster": _agent_summary("forecaster", [
            "probability", "confidence", "base_rate", "side"
        ]),
        "news_analyst": _agent_summary("news_analyst", [
            "sentiment", "relevance", "impact_direction", "key_factors"
        ]),
        "knowledge_researcher": _agent_summary("knowledge_researcher", [
            "worldview_applies", "key_frameworks", "knowledge_citations"
        ]),
        "bull_researcher": _agent_summary("bull_researcher", [
            "probability", "probability_floor", "confidence", "key_arguments", "catalysts"
        ]),
        "bear_researcher": _agent_summary("bear_researcher", [
            "probability", "probability_ceiling", "confidence", "key_arguments", "risk_factors"
        ]),
        "risk_manager": _agent_summary("risk_manager", [
            "ev_estimate", "risk_score", "should_trade",
            "recommended_size_pct", "max_loss_pct", "edge_durability_hours"
        ]),
        "trader": _agent_summary("trader", [
            "action", "side", "limit_price", "confidence", "position_size_pct"
        ]),
    }

    # Full trader reasoning, trimmed before the raw transcript block
    full_reasoning = result.get("reasoning", "")
    cut = full_reasoning.find("--- DEBATE TRANSCRIPT ---")
    trader_reasoning = full_reasoning[:cut].strip() if cut > 0 else full_reasoning[:3000]

    return {
        "market": {
            "ticker":         market_data["ticker"],
            "title":          market_data["title"],
            "yes_price":      market_data["yes_price"],
            "no_price":       market_data["no_price"],
            "volume":         market_data["volume"],
            "open_interest":  market_data["open_interest"],
            "days_to_expiry": market_data["days_to_expiry"],
        },
        "decision": {
            "action":            action,
            "side":              side,
            "limit_price_cents": limit_cents,
            "confidence":        result.get("confidence", 0.0),
            "position_size_pct": result.get("position_size_pct", 0.0),
            "ev_per_contract":   round(ev_per_contract, 4),
        },
        "agents":          agents_summary,
        "trader_reasoning": trader_reasoning,
        "elapsed_seconds":  result.get("elapsed_seconds", 0.0),
        "error":            result.get("error"),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kalshi AI Trading Bot — MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="Transport to use (default: stdio for Claude Desktop / OpenClaw)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Host for streamable-http transport (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="Port for streamable-http transport (default: 8765)",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(
            transport="streamable-http",
            host=args.host,
            port=args.port,
        )
