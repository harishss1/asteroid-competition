"""
Asteroid Auction Challenge — Competition Strategy FINAL
Team: Harish Corp

Architecture: XGBoost for all 5 models, 7 interaction features.
This is the v2 configuration confirmed best by verify_model.py:
  - Value MAE=9.62, corr=0.9992
  - thresh=0.35: precision=0.989, recall=1.000 (catastrophes reliably blocked)

ML Interface:
  load_model()      — called ONCE at tournament start, 30s timeout
  price_asteroids() — called each round, 2s timeout
"""

import os
import numpy as np

STRATEGY_NAME = "Harish Corp"

# ── Feature drop list (must match train_model.py exactly) ────────────────────
_DROP = {
    "asteroid_id", "time_period", "lucky_number", "social_sentiment_score",
    "communication_delay", "orbital_period", "aphelion_distance", "perihelion_distance",
    "estimated_volume", "surface_gravity",
    "mineral_value", "extraction_yield", "extraction_delay",
    "catastrophe_type", "toxic_outgassing_impact",
}

# Valid categorical values — unknown values get a safe default
_VALID_SPECTRAL   = {"C-type", "M-type", "S-type", "X-type"}
_VALID_REGION     = {"inner", "main", "outer"}
_VALID_PROBE      = {"active_flyby", "drill_core", "landing", "passive"}
_DEFAULT_SPECTRAL = "S-type"
_DEFAULT_REGION   = "main"
_DEFAULT_PROBE    = "active_flyby"

# Catastrophe penalty constants (from competition rules)
_PENALTY_VOID     = 100.0
_PENALTY_COLLAPSE = 200.0
_PENALTY_OUTGAS   = 300.0
_OUTGAS_NEIGHBOUR = 10.0

# type_model class indices (0-indexed, alphabetical):
# 0=structural_collapse, 1=toxic_outgassing, 2=void_rock
_TYPE_COLLAPSE = 0
_TYPE_OUTGAS   = 1
_TYPE_VOID     = 2


# ── load_model ────────────────────────────────────────────────────────────────
def load_model():
    """Called ONCE at tournament start. 30-second timeout."""
    import joblib
    model_dir  = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_dir, "model.joblib")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


# ── Feature preparation ───────────────────────────────────────────────────────
def _add_interaction_features(row):
    """
    Compute the 7 interaction features engineered during training.
    Must match train_model.py exactly — same formula, same column names.
    Validated: composite_risk corr=+0.212 vs best raw feature -0.174.
    """
    si  = float(row.get("structural_integrity", 0.7))
    vc  = float(row.get("volatile_content", 0.2))
    po  = float(row.get("porosity", 0.3))
    sc  = float(row.get("survey_confidence", 0.6))
    den = float(row.get("density", 4.0))
    eh  = float(row.get("environmental_hazard_rating", 0.3))

    row["interact_composite_risk"]        = (1.0 - si) * 0.4 + vc * 0.3 + po * 0.3
    row["interact_volatile_x_integrity"]  = vc * (1.0 - si)
    row["interact_low_integrity_x_poros"] = (1.0 - si) * po
    row["interact_survey_x_integrity"]    = sc * si
    row["interact_volatile_x_porosity"]   = vc * po
    row["interact_env_hazard_x_volatile"] = eh * vc
    row["interact_density_porosity_risk"] = po / (den + 1e-9)
    return row


def _build_df(asteroids, feature_cols):
    """
    Convert list of feature dicts to DataFrame aligned to training columns.
    Categorical columns kept as strings (pipeline OrdinalEncoder expects strings).
    Interaction features computed inline.
    """
    import pandas as pd

    rows = []
    for feat in asteroids:
        row = {}
        for k, v in feat.items():
            if k in _DROP:
                continue
            if k == "spectral_class":
                s = str(v)
                row[k] = s if s in _VALID_SPECTRAL else _DEFAULT_SPECTRAL
            elif k == "belt_region":
                s = str(v)
                row[k] = s if s in _VALID_REGION else _DEFAULT_REGION
            elif k == "probe_type":
                s = str(v)
                row[k] = s if s in _VALID_PROBE else _DEFAULT_PROBE
            else:
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    row[k] = 0.0

        row = _add_interaction_features(row)
        rows.append(row)

    df = pd.DataFrame(rows)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    return df[feature_cols]


# ── Heuristic fallback (if model not loaded) ──────────────────────────────────
def _heuristic_value(features):
    """
    Fallback using third-party estimates directly.
    ai_valuation_estimate: corr=0.971 with mineral_value, overestimates by ~$83.
    analyst_consensus_estimate: corr=0.874.
    """
    ai_est      = float(features.get("ai_valuation_estimate", 0) or 0)
    analyst_est = float(features.get("analyst_consensus_estimate", 0) or 0)

    if ai_est > 0 and analyst_est > 0:
        blended = 0.65 * ai_est + 0.35 * analyst_est
    elif ai_est > 0:
        blended = ai_est
    elif analyst_est > 0:
        blended = analyst_est
    else:
        blended = 0.0

    return max(0.0, blended * 0.75 - 20.0)


# ── Core bidding logic ────────────────────────────────────────────────────────
def price_asteroids(asteroids, capital, round_info, model=None):
    """
    Called each round with 2-second timeout.
    Returns list of bid amounts (same length as asteroids), 0 to pass.
    """
    if not asteroids:
        return []

    n             = len(asteroids)
    round_num     = round_info.get("round_number", 1)
    total_rounds  = round_info.get("total_rounds", 50)
    rounds_left   = total_rounds - round_num
    risk_free     = round_info.get("risk_free_rate", 0.002)
    num_pending   = round_info.get("num_pending_extractions", 0)
    pending_rev   = round_info.get("pending_revenue", 0.0)
    n_competitors = round_info.get("num_active_competitors", 5)

    # ── Step 1: Model predictions ─────────────────────────────────────────────
    if model is not None:
        try:
            feature_cols = model["all_feature_cols"]
            df_batch     = _build_df(asteroids, feature_cols)

            pred_value = model["value_model"].predict(df_batch)
            pred_yield = model["yield_model"].predict(df_batch)
            pred_delay = model["delay_model"].predict(df_batch)

            # Two-stage catastrophe probabilities
            # Stage 1: P(any catastrophe)
            p_any_cat = model["binary_cat_model"].predict_proba(df_batch)[:, 1]

            # Stage 2: P(type | catastrophe) — 3-class 0-indexed
            # 0=structural_collapse, 1=toxic_outgassing, 2=void_rock
            type_probs       = model["type_model"].predict_proba(df_batch)
            p_collapse_given = type_probs[:, _TYPE_COLLAPSE]
            p_outgas_given   = type_probs[:, _TYPE_OUTGAS]
            p_void_given     = type_probs[:, _TYPE_VOID]

            # Normalize (safety)
            type_sum         = p_collapse_given + p_outgas_given + p_void_given + 1e-9
            p_collapse_given = p_collapse_given / type_sum
            p_outgas_given   = p_outgas_given   / type_sum
            p_void_given     = p_void_given     / type_sum

            # Joint probabilities
            p_none     = 1.0 - p_any_cat
            p_collapse = p_any_cat * p_collapse_given
            p_outgas   = p_any_cat * p_outgas_given
            p_void     = p_any_cat * p_void_given

        except Exception:
            model = None

    if model is None:
        pred_value = np.array([_heuristic_value(f) for f in asteroids])
        pred_yield = np.array([
            max(0.5, min(1.15, float(f.get("equipment_compatibility", 0.7)) * 1.2))
            for f in asteroids
        ])
        pred_delay = np.full(n, 8.0)
        p_none     = np.full(n, 0.83)
        p_collapse = np.full(n, 0.07)
        p_outgas   = np.full(n, 0.05)
        p_void     = np.full(n, 0.05)

    # ── Step 2: Cluster-aware toxic outgassing penalty ────────────────────────
    cluster_ids    = [f.get("cluster_id", -1) for f in asteroids]
    cluster_counts = {}
    for cid in cluster_ids:
        cluster_counts[cid] = cluster_counts.get(cid, 0) + 1

    # ── Step 3: Risk-adjusted expected value ──────────────────────────────────
    expected_recovered = np.zeros(n)
    for i in range(n):
        ev_clean     = max(0.0, float(pred_value[i])) * max(0.477, min(1.15, float(pred_yield[i])))
        cid          = cluster_ids[i]
        cluster_size = cluster_counts.get(cid, 1)
        outgas_pen   = _PENALTY_OUTGAS + _OUTGAS_NEIGHBOUR * (cluster_size - 1)

        expected_recovered[i] = (
            p_none[i]     * ev_clean
            - p_void[i]     * _PENALTY_VOID
            - p_collapse[i] * _PENALTY_COLLAPSE
            - p_outgas[i]   * outgas_pen
        )

    # ── Step 4: Time-value discount ───────────────────────────────────────────
    delays           = np.clip(pred_delay, 4, 11)
    discount_factors = 1.0 / (1.0 + risk_free) ** delays
    discounted_ev    = expected_recovered * discount_factors

    # ── Step 5: Competitive bid shading ──────────────────────────────────────
    # Optimized by simulator: base shade 0.54-0.62 range (10% below original)
    market_history  = round_info.get("market_history")
    competitive_adj = 1.0
    if market_history:
        your_wins      = market_history.get("your_total_wins", 0)
        rounds_done    = market_history.get("rounds_completed", 1)
        offered_so_far = market_history.get("cumulative_asteroids_offered", rounds_done * n)
        win_rate       = your_wins / max(1, offered_so_far)
        if win_rate < 0.08:
            competitive_adj = 1.15
        elif win_rate < 0.15:
            competitive_adj = 1.07
        elif win_rate > 0.40:
            competitive_adj = 0.88
        elif win_rate > 0.25:
            competitive_adj = 0.94

    # Simulator found 0.90x of original shade was optimal
    base_shade   = 0.54 + 0.009 * min(n_competitors, 8)
    shade_factor = base_shade * competitive_adj

    # ── Step 6: Late-game adjustment ─────────────────────────────────────────
    late_game_mult = np.ones(n)
    if rounds_left <= 5:
        late_game_mult[:] = 1.15
    elif rounds_left <= 10:
        late_game_mult[:] = 1.07
    elif rounds_left <= 15:
        late_game_mult[:] = 1.03

    # ── Step 7: Liquidity management ─────────────────────────────────────────
    liq_mult = 1.0
    if num_pending >= 6:
        liq_mult = 0.70
    elif num_pending >= 4:
        liq_mult = 0.85
    elif num_pending >= 2:
        liq_mult = 0.95

    # ── Step 8: Raw bids ──────────────────────────────────────────────────────
    raw_bids = discounted_ev * shade_factor * late_game_mult * liq_mult

    # Floor: skip bids below minimum profit threshold
    min_profit_threshold = max(15.0, capital * 0.005)
    raw_bids = np.where(raw_bids < min_profit_threshold, 0.0, raw_bids)

    # Hard filter: skip asteroids with >35% total catastrophe probability
    # At this threshold v2 XGBoost showed precision=0.989, recall=1.000
    p_catastrophe = 1.0 - p_none
    raw_bids = np.where(p_catastrophe > 0.35, 0.0, raw_bids)

    # Per-bid cap: no single asteroid absorbs more than 18% of capital
    per_bid_cap = capital * 0.18
    raw_bids    = np.minimum(raw_bids, per_bid_cap)

    # ── Step 9: Portfolio budget allocation ──────────────────────────────────
    # Spend at most 65% of capital per round (simulator: 60-70% flat, midpoint)
    # Greedy allocation rather than uniform scaling
    budget_cap = capital * 0.65
    total_raw  = raw_bids.sum()

    if total_raw > budget_cap:
        order       = np.argsort(-raw_bids)
        final_bids  = np.zeros(n)
        budget_used = 0.0
        for idx in order:
            if raw_bids[idx] <= 0:
                break
            if budget_used + raw_bids[idx] <= budget_cap:
                final_bids[idx]  = raw_bids[idx]
                budget_used     += raw_bids[idx]
            else:
                remaining = budget_cap - budget_used
                if remaining > min_profit_threshold:
                    final_bids[idx] = remaining
                    budget_used    += remaining
                break
    else:
        final_bids = raw_bids.copy()

    return final_bids.tolist()