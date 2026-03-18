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
import time
import uuid
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

# ---------------------------------------------------------------------------
# In-memory job store for async debate tracking
# ---------------------------------------------------------------------------
# job_id -> {
#   "status":   "queued" | "running" | "complete" | "error"
#   "ticker":   str
#   "started":  float (epoch)
#   "elapsed":  float | None
#   "step":     str   (current debate step)
#   "steps_done": list[str]
#   "result":   dict | None
#   "error":    str | None
# }
_jobs: dict[str, dict[str, Any]] = {}
_JOB_TTL_SECONDS = 3600  # prune completed jobs after 1 hour


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

# Update server instructions to reflect async workflow
mcp.instructions = (
    "You have access to a Kalshi prediction-market analysis system backed "
    "by a 7-agent AI debate pipeline.\n\n"
    "WORKFLOW:\n"
    "1. Use list_markets(category=...) to discover live markets.\n"
    "2. Use get_market_ladder(event_ticker) to see the full probability ladder for an event.\n"
    "3. Call analyze_market(ticker) to START the debate — it returns immediately with a job_id.\n"
    "4. Call get_analysis_status(job_id) every ~60 seconds to check progress.\n"
    "   When status=='complete' the full result is in the response.\n\n"
    "DO NOT wait silently after analyze_market — poll with get_analysis_status "
    "and report progress to the user every minute."
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
# Sports/entertainment series prefixes to always exclude.
# Kalshi's unfiltered market list leads with these; we never want them.
# ---------------------------------------------------------------------------
_SPORTS_PREFIXES = (
    "KXMVE",   # NBA player props (MVE = multi-variate events)
    "KXNBA",   # NBA game markets
    "KXNFL",   # NFL
    "KXNHL",   # NHL
    "KXMLB",   # MLB
    "KXNCAA",  # College sports
    "KXSOCCER","KXMLS",
    "KXTENNIS","KXWTA","KXATP",
    "KXGOLF",  "KXPGA",
    "KXNASCAR","KXF1",
    "KXBOXING","KXMMA","KXUFC",
    "KXOLYMPIC",
    "KXENTERTAIN",
    "KXAWARDS","KXOSCARS","KXEMMYS","KXGRAMMYS",
    "KXREALITY",
)

# ---------------------------------------------------------------------------
# All known non-sports series on Kalshi, grouped by category.
# These are fetched by series_ticker which is server-side and fast.
# Political/geo markets are sparse and scattered; we include every known
# series and refresh this list as new ones appear.
# ---------------------------------------------------------------------------
_SERIES_BY_CATEGORY: dict[str, list[str]] = {
    "macro": [
        "KXFED", "KXCPI", "KXGDP", "KXUNRATE", "KXPCE",
        "KXPPI", "KXNFP", "KXSPX", "KXNASDAQ", "KXGOLD",
        "KXOIL", "KXUSD", "KXRECESSION", "KXDEFICIT",
    ],
    "crypto": [
        "KXBTC", "KXETH", "KXCRYPTO",
    ],
    "policy": [
        # US executive / regulatory — confirmed to exist on Kalshi
        "KXFDA", "KXSEC", "KXDEFICIT",
        # Try these; they return empty if no markets exist yet
        "KXTRUMP", "KXTARIFF", "KXDEBT", "KXDOGE",
        "KXELON", "KXCONGRESS", "KXSENATE", "KXHOUSE",
        "KXIMMIGRATION", "KXBORDER", "KXBUDGET",
        "KXDOJ", "KXCFTC", "KXFBI", "KXCIA",
    ],
    "geo": [
        # Confirmed to exist
        "KXUKRAINE",
        # Try these
        "KXIRAN", "KXCHINA", "KXTAIWAN", "KXNATO",
        "KXISRAEL", "KXGAZA", "KXRUSSIA",
        "KXNORTHKOREA", "KXMIDEAST",
    ],
    "ai": [
        "KXAI", "KXOPENAI", "KXANTHROPICS",
        "KXTECH", "KXANTITRUST",
    ],
}
# "all" = every series across every category
_ALL_SERIES: list[str] = list({
    s for series_list in _SERIES_BY_CATEGORY.values() for s in series_list
})


def _is_sports(ticker: str) -> bool:
    """Return True if the ticker belongs to a sports/entertainment series."""
    t = ticker.upper()
    return any(t.startswith(p) for p in _SPORTS_PREFIXES)


# ---------------------------------------------------------------------------
# Tool 1 — list_markets
# ---------------------------------------------------------------------------
@mcp.tool
async def list_markets(
    series: str = "",
    category: str = "macro",
    min_volume: int = 1000,
    limit: int = 25,
) -> dict:
    """
    Discover live, priced Kalshi prediction markets.

    Sports and entertainment markets are always excluded.  By default only
    macro / policy markets are returned (economics, Fed, crypto, geopolitics).

    Args:
        series: Optional specific Kalshi series ticker to filter by
                (e.g. "KXCPI", "KXFED", "KXGDP").  When provided, only
                markets from that series are returned and the category
                filter is ignored.
        category: Broad category filter when no series is specified.
                  "macro"  — economics, Fed, inflation, GDP, jobs (default)
                  "crypto" — Bitcoin, Ethereum, crypto prices
                  "geo"    — geopolitics, foreign policy, war/conflict
                  "policy" — US politics, tariffs, regulation, executive actions,
                             cabinet confirmations, DOGE, executive orders
                  "ai"     — AI policy, regulation, model releases
                  "all"    — everything non-sports
        min_volume: Minimum number of contracts traded.  Default 1 000.
        limit: Maximum markets to return.  Default 25, max 100.

    Returns:
        {
          "markets": [
            {
              "ticker": str,
              "title": str,
              "yes_mid": float,
              "yes_ask": float,
              "yes_bid": float,
              "volume": int,
              "open_interest": int,
              "close_date": str,
              "rules": str,
            }, ...
          ],
          "count": int,
          "note": str,
        }
    """
    assert _kalshi is not None, "Kalshi client not initialised"
    limit = min(int(limit), 100)
    cat = category.lower()

    raw_markets: list[dict] = []
    seen: set[str] = set()

    if series:
        # Explicit series — single targeted fetch via markets endpoint
        resp = await _kalshi.get_markets(
            limit=min(limit * 4, 200),
            series_ticker=series.upper(),
        )
        raw_markets = resp.get("markets", [])

    elif cat == "macro":
        # Macro has clean, confirmed series prefixes — use markets endpoint
        for s in _SERIES_BY_CATEGORY["macro"]:
            try:
                resp = await _kalshi.get_markets(limit=50, series_ticker=s)
                for m in resp.get("markets", []):
                    t = m.get("ticker","")
                    if t not in seen:
                        seen.add(t)
                        raw_markets.append(m)
            except Exception:
                continue

    else:
        # For political/geo/crypto/ai/all: use the events endpoint which
        # surfaces markets that the /markets endpoint misclassifies as
        # finalized or doesn't surface at all.
        # Fetch events with nested markets, then filter by title keywords.
        kw_map: dict[str, tuple[str,...]] = {
            "policy":  ("tariff","executive","congress","senate","trump","cabinet",
                        "confirmed","legislation","immigration","border","budget",
                        "deficit","department","doge","sanction","trade deal",
                        "executive order","elon","bill passed","signed"),
            "geo":     ("ukraine","russia","iran","china overtake","taiwan","nato",
                        "israel","gaza","ceasefire","nuclear deal","troops",
                        "invasion","north korea","south korea","middle east",
                        "military strike","foreign policy","sanctions on",
                        "level 4","state department","free trade agreement"),
            "crypto":  ("bitcoin","ethereum","btc","eth","crypto","stablecoin",
                        "blockchain","coinbase","binance"),
            "ai":      ("artificial intelligence"," ai ","openai","anthropic",
                        "large language model","chatgpt","ai regulation","ai act",
                        "deepmind","llm"),
            "all":     (),  # no keyword filter — take everything non-sports
        }
        keywords = kw_map.get(cat, ())

        cursor = None
        pages = 0
        while pages < 30:  # up to 30 × 200 = 6 000 events
            try:
                resp = await _kalshi.get_events(
                    limit=200,
                    cursor=cursor,
                    with_nested_markets=True,
                )
            except Exception:
                break

            events = resp.get("events", [])
            for event in events:
                event_title = (event.get("title") or "").lower()
                # Skip if keyword filter applies and title doesn't match
                if keywords and not any(kw in event_title for kw in keywords):
                    continue
                # Expand nested markets
                for m in event.get("markets", []):
                    t = m.get("ticker","")
                    if t not in seen:
                        seen.add(t)
                        raw_markets.append(m)

            cursor = resp.get("cursor")
            pages += 1
            if not cursor or len(events) == 0:
                break
            # Stop early once we have plenty of candidates
            if len(raw_markets) >= limit * 8:
                break

    # ── Filter and shape ──────────────────────────────────────────────────────
    results = []
    for m in raw_markets:
        ticker = m.get("ticker", "")

        if _is_sports(ticker):
            continue

        ya  = float(m.get("yes_ask_dollars") or 0)
        yb  = float(m.get("yes_bid_dollars") or 0)
        vol = float(m.get("volume_fp") or 0)

        if m.get("status") != "active" or ya <= 0.01 or ya >= 0.99 or vol < min_volume:
            continue

        results.append({
            "ticker":        ticker,
            "title":         m.get("title", ""),
            "yes_mid":       round((ya + yb) / 2, 3),
            "yes_ask":       round(ya, 3),
            "yes_bid":       round(yb, 3),
            "volume":        int(vol),
            "open_interest": int(float(m.get("open_interest_fp") or 0)),
            "close_date":    (m.get("close_time") or "")[:10],
            "rules":         (m.get("rules_primary") or "")[:200],
        })

    results.sort(key=lambda x: x["volume"], reverse=True)
    results = results[:limit]

    note = (
        "Sports and entertainment markets are always excluded. "
        "Prices in dollars (0-1 scale). yes_mid is the bid/ask midpoint. "
        "category options: macro (default), crypto, geo, policy, ai, all. "
        "Use get_market_ladder for a full strike ladder, or analyze_market "
        "to run the full debate on a specific ticker."
    )
    if results == [] and not series:
        note = (
            f"No liquid markets found for category='{cat}' at min_volume={min_volume}. "
            "Kalshi's political and geopolitical markets are currently sparse. "
            "Try min_volume=0 to see all available markets in this category, "
            "or use category='macro' for the most liquid markets."
        )

    return {
        "markets": results,
        "count":   len(results),
        "note":    note,
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
def _build_final_response(market_data: dict, result: dict) -> dict:
    """Shape the debate result dict into the standard tool response format."""
    action      = result.get("action", "SKIP")
    side        = result.get("side", "YES")
    limit_cents = result.get("limit_price", 50)
    step_results = result.get("step_results", {})

    # Confidence-weighted EV
    ev_per_contract: float = 0.0
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
        tw     = sum(w for _, w in probs)
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
            else:
                ev_per_contract = cw_yes * entry - cw_no * (1.0 - entry)

    def _agent_summary(role: str, fields: list[str]) -> dict:
        sr = step_results.get(role) or {}
        if "error" in sr:
            return {"error": sr["error"]}
        out = {f: sr[f] for f in fields if sr.get(f) is not None}
        rsn = sr.get("reasoning", "")
        out["reasoning"] = rsn[:800] if rsn else ""
        return out

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
        "agents": {
            "forecaster":           _agent_summary("forecaster",        ["probability","confidence","base_rate","side"]),
            "news_analyst":         _agent_summary("news_analyst",       ["sentiment","relevance","impact_direction","key_factors"]),
            "knowledge_researcher": _agent_summary("knowledge_researcher",["worldview_applies","key_frameworks","knowledge_citations"]),
            "bull_researcher":      _agent_summary("bull_researcher",    ["probability","probability_floor","confidence","key_arguments","catalysts"]),
            "bear_researcher":      _agent_summary("bear_researcher",    ["probability","probability_ceiling","confidence","key_arguments","risk_factors"]),
            "risk_manager":         _agent_summary("risk_manager",       ["ev_estimate","risk_score","should_trade","recommended_size_pct","max_loss_pct","edge_durability_hours"]),
            "trader":               _agent_summary("trader",             ["action","side","limit_price","confidence","position_size_pct"]),
        },
        "trader_reasoning": trader_reasoning,
        "elapsed_seconds":  result.get("elapsed_seconds", 0.0),
        "error":            result.get("error"),
    }


async def _fetch_market_data(ticker: str) -> tuple[dict | None, str | None]:
    """Fetch and normalise Kalshi market data. Returns (market_data, error_str)."""
    assert _kalshi is not None
    try:
        resp = await _kalshi.get_market(ticker)
        raw  = resp.get("market", {})
    except Exception as exc:
        return None, f"Failed to fetch market {ticker!r}: {exc}"
    if not raw:
        return None, f"Market {ticker!r} not found on Kalshi."

    ya  = float(raw.get("yes_ask_dollars") or 0)
    yb  = float(raw.get("yes_bid_dollars") or 0)
    na  = float(raw.get("no_ask_dollars")  or 0)
    nb  = float(raw.get("no_bid_dollars")  or 0)
    vol = float(raw.get("volume_fp")       or 0)
    oi  = float(raw.get("open_interest_fp") or 0)

    close_time     = raw.get("close_time") or raw.get("expiration_time") or ""
    days_to_expiry = 30.0
    if close_time:
        from datetime import datetime, timezone
        try:
            expiry_dt      = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            days_to_expiry = max(0.0, (expiry_dt - datetime.now(timezone.utc)).total_seconds() / 86400)
        except ValueError:
            pass

    return {
        "ticker":         raw.get("ticker", ticker),
        "title":          raw.get("title", ""),
        "rules":          (raw.get("rules_primary") or "") + " " + (raw.get("rules_secondary") or ""),
        "yes_price":      round((ya + yb) / 2, 3),
        "no_price":       round((na + nb) / 2, 3),
        "volume":         int(vol),
        "open_interest":  int(oi),
        "category":       "economics",
        "days_to_expiry": round(days_to_expiry, 1),
        "expiration_ts":  0,
    }, None


async def _run_debate_job(job_id: str, market_data: dict) -> None:
    """Background coroutine: run the full debate and write results to _jobs."""
    job = _jobs[job_id]
    ticker = market_data["ticker"]

    # Ordered debate steps — mirrors DebateRunner's internal sequence
    STEPS = [
        "knowledge_researcher",
        "forecaster",
        "news_analyst",
        "bull_researcher",
        "bear_researcher",
        "risk_manager",
        "trader",
    ]

    # Patch DebateRunner to report progress into the job store.
    # We wrap each completion callable so we can detect which step just started.
    completions = _make_completions(ticker)
    wrapped: dict[str, Any] = {}
    for role, fn in completions.items():
        async def _w(prompt: str, _role: str = role, _fn: Any = fn) -> str:
            _jobs[job_id]["step"] = _role
            if _role not in _jobs[job_id]["steps_done"]:
                pass  # will be added after completion
            result_text = await _fn(prompt)
            if _role not in _jobs[job_id]["steps_done"]:
                _jobs[job_id]["steps_done"].append(_role)
            return result_text
        wrapped[role] = _w

    runner = DebateRunner()
    try:
        job["status"] = "running"
        result = await runner.run_debate(
            market_data=market_data,
            get_completions=wrapped,
            context={"portfolio": {"cash": 1000.0}},
        )
        job["result"]  = _build_final_response(market_data, result)
        job["status"]  = "complete"
        job["elapsed"] = time.time() - job["started"]
        job["step"]    = "done"
    except Exception as exc:
        job["status"] = "error"
        job["error"]  = str(exc)
        job["elapsed"] = time.time() - job["started"]

    # Prune old completed jobs
    cutoff = time.time() - _JOB_TTL_SECONDS
    stale  = [jid for jid, j in _jobs.items() if j.get("started", 0) < cutoff]
    for jid in stale:
        _jobs.pop(jid, None)


@mcp.tool
async def analyze_market(
    ticker: str,
    paper_trade: bool = False,
) -> dict:
    """
    Start the full 7-agent AI debate on a specific Kalshi market.

    This call returns IMMEDIATELY with a job_id.  The debate runs in the
    background (~5-15 minutes).  Use get_analysis_status(job_id) to poll
    for progress and retrieve the result when it's ready.

    Agents (run in sequence):
      1. KnowledgeResearcher — RAG + worldview + DuckDuckGo web search
      2. ForecasterAgent     — base-rate + current conditions probability
      3. NewsAnalystAgent    — sentiment and news impact direction
      4. BullResearcher      — strongest case for YES
      5. BearResearcher      — strongest case for NO
      6. RiskManagerAgent    — EV calculation and position sizing
      7. TraderAgent         — final synthesis through the worldview lens

    Args:
        ticker:      Kalshi market ticker (e.g. "KXCPI-26MAR-T0.7").
                     Use list_markets or get_market_ladder to find tickers.
        paper_trade: Reserved for future use.  Default False.

    Returns:
        {
          "job_id":   str,   # pass to get_analysis_status to poll for results
          "ticker":   str,
          "title":    str,
          "status":   "queued",
          "message":  str,
        }
    """
    assert _kalshi is not None, "Kalshi client not initialised"
    assert _router is not None, "Model router not initialised"

    market_data, err = await _fetch_market_data(ticker)
    if err:
        return {"error": err}

    job_id = str(uuid.uuid4())[:8]   # short, human-readable
    _jobs[job_id] = {
        "status":     "queued",
        "ticker":     ticker,
        "title":      market_data["title"],
        "started":    time.time(),
        "elapsed":    None,
        "step":       "queued",
        "steps_done": [],
        "result":     None,
        "error":      None,
    }

    # Fire and forget — runs concurrently while the agent does other things
    asyncio.create_task(_run_debate_job(job_id, market_data))

    return {
        "job_id":  job_id,
        "ticker":  ticker,
        "title":   market_data["title"],
        "status":  "queued",
        "message": (
            f"Debate started (job_id={job_id!r}).  "
            "Call get_analysis_status(job_id) to check progress.  "
            "Expected completion: 5-15 minutes.  "
            "Suggested polling interval: every 60 seconds."
        ),
    }


# ---------------------------------------------------------------------------
# Tool 4 — get_analysis_status
# ---------------------------------------------------------------------------
@mcp.tool
async def get_analysis_status(job_id: str) -> dict:
    """
    Check the status of a debate analysis started by analyze_market.

    Poll this every 60 seconds after calling analyze_market.  When
    status == "complete" the full result is included in the response.

    Args:
        job_id: The job ID returned by analyze_market.

    Returns:
        While running:
        {
          "job_id":      str,
          "ticker":      str,
          "status":      "queued" | "running",
          "current_step": str,    # which agent is currently running
          "steps_done":  list[str],
          "elapsed_seconds": float,
          "message":     str,
        }

        When complete:
        {
          "job_id":      str,
          "ticker":      str,
          "status":      "complete",
          "elapsed_seconds": float,
          "result":      { ... full analyze_market result ... }
        }

        On error:
        {
          "job_id":  str,
          "status":  "error",
          "error":   str,
        }
    """
    job = _jobs.get(job_id)
    if job is None:
        return {
            "job_id": job_id,
            "status": "not_found",
            "error":  (
                f"No job found with id={job_id!r}.  "
                "Jobs expire after 1 hour.  "
                "Use analyze_market to start a new one."
            ),
        }

    elapsed = time.time() - job["started"]
    status  = job["status"]

    if status == "complete":
        return {
            "job_id":          job_id,
            "ticker":          job["ticker"],
            "status":          "complete",
            "elapsed_seconds": round(job["elapsed"] or elapsed, 1),
            "result":          job["result"],
        }

    if status == "error":
        return {
            "job_id":  job_id,
            "ticker":  job["ticker"],
            "status":  "error",
            "elapsed_seconds": round(elapsed, 1),
            "error":   job.get("error", "Unknown error"),
        }

    # Still running or queued
    steps_done = job["steps_done"]
    current    = job["step"]
    all_steps  = [
        "knowledge_researcher", "forecaster", "news_analyst",
        "bull_researcher", "bear_researcher", "risk_manager", "trader",
    ]
    remaining = [s for s in all_steps if s not in steps_done]

    return {
        "job_id":          job_id,
        "ticker":          job["ticker"],
        "title":           job.get("title", ""),
        "status":          status,
        "current_step":    current,
        "steps_done":      steps_done,
        "steps_remaining": remaining,
        "elapsed_seconds": round(elapsed, 1),
        "message": (
            f"Debate in progress.  "
            f"Currently running: {current}.  "
            f"Completed: {len(steps_done)}/7 agents.  "
            f"Elapsed: {elapsed:.0f}s.  "
            "Call get_analysis_status again in ~60 seconds."
        ),
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
