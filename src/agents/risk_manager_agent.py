"""
Risk Manager Agent -- evaluates risk/reward and recommends position sizing.

Uses DeepSeek (via OpenRouter) by default.  Focuses on:
- Expected value calculation for the SPECIFIC proposed trade direction
- Position sizing recommendation
- Risk assessment (1-10 scale)
"""

from src.agents.base_agent import BaseAgent


class RiskManagerAgent(BaseAgent):
    """Evaluates risk/reward profile and recommends position sizing."""

    AGENT_NAME = "risk_manager"
    AGENT_ROLE = "risk_manager"
    DEFAULT_MODEL = "deepseek/deepseek-v3.2"

    SYSTEM_PROMPT = (
        "You are a quantitative risk manager for a prediction-market trading "
        "desk. Your job is to evaluate whether a SPECIFIC proposed trade has "
        "acceptable risk/reward and to recommend position sizing.\n\n"
        "IMPORTANT: You will be given an explicit PROPOSED TRADE (side and "
        "entry price). Evaluate EV for THAT trade, not for the opposite side.\n\n"
        "You must consider:\n"
        "1. EXPECTED VALUE (EV) -- For the proposed trade:\n"
        "   - If BUYING YES at price P: EV = (consensus_prob_yes * (1-P)) - ((1-consensus_prob_yes) * P)\n"
        "   - If SELLING YES at price P: EV = ((1-consensus_prob_yes) * P) - (consensus_prob_yes * (1-P))\n"
        "   - If BUYING NO at price P: EV = (consensus_prob_no * (1-P)) - ((1-consensus_prob_no) * P)\n"
        "   - If SELLING NO at price P: EV = ((1-consensus_prob_no) * P) - (consensus_prob_no * (1-P))\n"
        "   Only trades with positive EV should be taken.\n"
        "2. RISK SCORE -- Rate the overall risk from 1 (very safe) to 10 (very "
        "   risky). Consider: liquidity, time to expiry, volatility, information "
        "   quality, and model disagreement.\n"
        "3. POSITION SIZE -- Recommend what percentage of available capital to "
        "   allocate (0-100%). Use fractional Kelly criterion logic: higher EV "
        "   and lower risk = larger position.\n"
        "4. WORST CASE -- What is the maximum loss, and is it acceptable?\n"
        "5. EDGE DURABILITY -- How long will the informational edge last?\n\n"
        "Return your analysis as a JSON object (inside a ```json``` code block) "
        "with the following keys:\n"
        '  "risk_score": float (1.0-10.0),\n'
        '  "recommended_size_pct": float (0.0-100.0, percent of capital),\n'
        '  "ev_estimate": float (expected value as a decimal, e.g. 0.15 = 15%),\n'
        '  "max_loss_pct": float (worst case loss as percent of position),\n'
        '  "edge_durability_hours": float (estimated hours the edge lasts),\n'
        '  "should_trade": boolean (true if trade meets risk criteria),\n'
        '  "reasoning": string (detailed risk analysis)'
    )

    @staticmethod
    def _derive_proposed_trade(
        market_data: dict, context: dict
    ) -> tuple[str, str, float, float]:
        """
        Derive the proposed trade direction from agent consensus BEFORE asking
        the LLM to evaluate EV, so the LLM is never left to guess the direction.

        Returns:
            (action, side, entry_price_cents, consensus_prob_yes)
        """
        prob_weight_pairs = []

        fc = context.get("forecaster_result") or {}
        if fc.get("probability") is not None:
            try:
                prob_weight_pairs.append(
                    (float(fc["probability"]), float(fc.get("confidence", 0.5)))
                )
            except (TypeError, ValueError):
                pass

        bull = context.get("bull_result") or {}
        if bull.get("probability") is not None:
            try:
                prob_weight_pairs.append(
                    (float(bull["probability"]), float(bull.get("confidence", 0.5)))
                )
            except (TypeError, ValueError):
                pass

        bear = context.get("bear_result") or {}
        if bear.get("probability") is not None:
            try:
                prob_weight_pairs.append(
                    (float(bear["probability"]), float(bear.get("confidence", 0.5)))
                )
            except (TypeError, ValueError):
                pass

        if prob_weight_pairs:
            total_weight = sum(w for _, w in prob_weight_pairs)
            if total_weight > 0:
                consensus_yes = sum(p * w for p, w in prob_weight_pairs) / total_weight
            else:
                consensus_yes = sum(p for p, _ in prob_weight_pairs) / len(prob_weight_pairs)
        else:
            consensus_yes = float(market_data.get("yes_price", 0.5))

        consensus_yes = max(0.01, min(0.99, consensus_yes))

        market_yes = float(market_data.get("yes_price", 0.5))
        market_no  = float(market_data.get("no_price",  0.5))

        yes_edge = consensus_yes - market_yes
        no_edge  = (1.0 - consensus_yes) - market_no

        if yes_edge >= no_edge and yes_edge > 0:
            return "BUY", "YES", market_yes * 100, consensus_yes
        elif no_edge > yes_edge and no_edge > 0:
            return "BUY", "NO", market_no * 100, consensus_yes
        else:
            if abs(yes_edge) >= abs(no_edge):
                return "SELL", "YES", market_yes * 100, consensus_yes
            else:
                return "SELL", "NO", market_no * 100, consensus_yes

    def _build_prompt(self, market_data: dict, context: dict) -> str:
        summary = self.format_market_summary(market_data)

        # Derive the proposed trade so EV can be computed unambiguously
        action, side, entry_price_cents, consensus_yes = self._derive_proposed_trade(
            market_data, context
        )
        consensus_no = 1.0 - consensus_yes
        entry_price_frac = entry_price_cents / 100.0

        if action == "BUY" and side == "YES":
            ev_hint = consensus_yes * (1.0 - entry_price_frac) - consensus_no * entry_price_frac
            ev_formula = (
                f"EV = P(YES)*profit_if_yes - P(NO)*cost\n"
                f"   = {consensus_yes:.3f} * {(1-entry_price_frac):.3f} - {consensus_no:.3f} * {entry_price_frac:.3f}\n"
                f"   = {ev_hint:.4f}"
            )
        elif action == "BUY" and side == "NO":
            ev_hint = consensus_no * (1.0 - entry_price_frac) - consensus_yes * entry_price_frac
            ev_formula = (
                f"EV = P(NO)*profit_if_no - P(YES)*cost\n"
                f"   = {consensus_no:.3f} * {(1-entry_price_frac):.3f} - {consensus_yes:.3f} * {entry_price_frac:.3f}\n"
                f"   = {ev_hint:.4f}"
            )
        elif action == "SELL" and side == "YES":
            ev_hint = consensus_no * entry_price_frac - consensus_yes * (1.0 - entry_price_frac)
            ev_formula = (
                f"EV = P(NO)*premium_received - P(YES)*payout_owed\n"
                f"   = {consensus_no:.3f} * {entry_price_frac:.3f} - {consensus_yes:.3f} * {(1-entry_price_frac):.3f}\n"
                f"   = {ev_hint:.4f}"
            )
        else:  # SELL NO
            ev_hint = consensus_yes * entry_price_frac - consensus_no * (1.0 - entry_price_frac)
            ev_formula = (
                f"EV = P(YES)*premium_received - P(NO)*payout_owed\n"
                f"   = {consensus_yes:.3f} * {entry_price_frac:.3f} - {consensus_no:.3f} * {(1-entry_price_frac):.3f}\n"
                f"   = {ev_hint:.4f}"
            )

        proposed_trade_section = (
            f"\n\n--- PROPOSED TRADE ---\n"
            f"Action: {action} {side}\n"
            f"Entry price: {entry_price_cents:.1f}¢\n"
            f"Consensus P(YES): {consensus_yes:.3f} | Consensus P(NO): {consensus_no:.3f}\n"
            f"Pre-computed EV for this trade:\n  {ev_formula}\n"
            f"Evaluate THIS trade. Do not flip to the opposite side.\n"
            f"--- END PROPOSED TRADE ---"
        )

        pieces = []

        fc = context.get("forecaster_result") or {}
        if fc.get("probability") is not None:
            pieces.append(
                f"Forecaster: YES prob={fc.get('probability', '?')}, "
                f"confidence={fc.get('confidence', '?')}"
            )

        if context.get("bull_result"):
            bull = context["bull_result"]
            pieces.append(
                f"Bull Researcher: YES prob={bull.get('probability', '?')}, "
                f"floor={bull.get('probability_floor', '?')}"
            )

        if context.get("bear_result"):
            bear = context["bear_result"]
            pieces.append(
                f"Bear Researcher: YES prob={bear.get('probability', '?')}, "
                f"ceiling={bear.get('probability_ceiling', '?')}"
            )

        if context.get("news_result"):
            news = context["news_result"]
            pieces.append(
                f"News Analyst: sentiment={news.get('sentiment', '?')}, "
                f"relevance={news.get('relevance', '?')}, "
                f"direction={news.get('impact_direction', '?')}"
            )

        agents_section = ""
        if pieces:
            agents_section = (
                "\n\n--- OTHER AGENTS' ASSESSMENTS ---\n"
                + "\n".join(f"- {p}" for p in pieces)
                + "\n--- END ASSESSMENTS ---"
            )

        portfolio_section = ""
        if context.get("portfolio"):
            pf = context["portfolio"]
            portfolio_section = (
                f"\n\nPortfolio: cash=${pf.get('cash', 0):,.2f}, "
                f"max_position_pct={pf.get('max_position_pct', 5)}%, "
                f"existing_positions={pf.get('existing_positions', 0)}"
            )

        return (
            f"Evaluate the risk/reward for the following prediction market "
            f"trade.\n\n"
            f"{summary}{proposed_trade_section}{agents_section}{portfolio_section}\n\n"
            f"The proposed trade and its pre-computed EV are shown above. "
            f"Verify the EV calculation, assess overall risk, and recommend "
            f"position sizing for THAT specific trade.\n"
            f"Return ONLY a JSON object inside a ```json``` code block."
        )

    def _parse_result(self, raw_json: dict) -> dict:
        risk_score = self.clamp(raw_json.get("risk_score", 5.0), lo=1.0, hi=10.0)
        recommended_size_pct = self.clamp(
            raw_json.get("recommended_size_pct", 1.0), lo=0.0, hi=100.0
        )
        ev_estimate = float(raw_json.get("ev_estimate", 0.0))
        max_loss_pct = self.clamp(
            raw_json.get("max_loss_pct", 100.0), lo=0.0, hi=100.0
        )
        edge_durability = max(0.0, float(raw_json.get("edge_durability_hours", 24.0)))
        should_trade = bool(raw_json.get("should_trade", False))
        reasoning = str(raw_json.get("reasoning", "No reasoning provided."))

        return {
            "risk_score": risk_score,
            "recommended_size_pct": recommended_size_pct,
            "ev_estimate": ev_estimate,
            "max_loss_pct": max_loss_pct,
            "edge_durability_hours": edge_durability,
            "should_trade": should_trade,
            "reasoning": reasoning,
        }
