"""
Asteroid Auction Challenge — Competition Strategy (No Trap Features)
Team: Harish Corp

Removed trap features: ai_valuation_estimate, analyst_consensus_estimate,
                       media_hype_score, lucky_number, social_sentiment_score

New value proxy features replace lost signal:
  eng_mass_x_mineral_score  (corr=0.676) — best single predictor
  eng_weighted_mineral_score (corr=0.528)
  eng_cycle_adj_score        (corr=0.525)
  eng_mineral_concentration  (corr=0.461)
  eng_value_density          (corr=0.448)

ML Interface:
  load_model()      — called ONCE at tournament start, 30s timeout
  price_asteroids() — called each round, 2s timeout
"""

import os
import numpy as np

STRATEGY_NAME = "Harish Corp"

# ── Feature drop list (must match train_model.py exactly) ────────────────────
_DROP = {
    "asteroid_id", "time_period",
    # trap features — explicitly excluded per competition feedback
    "ai_valuation_estimate", "analyst_consensus_estimate",
    "media_hype_score", "lucky_number", "social_sentiment_score",
    # redundant orbital
    "communication_delay", "orbital_period", "aphelion_distance", "perihelion_distance",
    # redundant physical
    "estimated_volume", "surface_gravity",
    # target columns
    "mineral_value", "extraction_yield", "extraction_delay",
    "catastrophe_type", "toxic_outgassing_impact",
}

# Valid categorical values
_VALID_SPECTRAL   = {"C-type", "M-type", "S-type", "X-type"}
_VALID_REGION     = {"inner", "main", "outer"}
_VALID_PROBE      = {"active_flyby", "drill_core", "landing", "passive"}
_DEFAULT_SPECTRAL = "S-type"
_DEFAULT_REGION   = "main"
_DEFAULT_PROBE    = "active_flyby"

# Catastrophe penalty constants
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


# ── Feature engineering ───────────────────────────────────────────────────────
def _engineer_features(row):
    """
    Compute all engineered features. Must match train_model.py exactly.

    Value proxy features (replace lost ai_valuation_estimate signal):
      eng_mass_x_mineral_score: best predictor, corr=0.676 with mineral_value
      eng_weighted_mineral_score: sum(sig * price), corr=0.528

    Catastrophe interaction features (unchanged from previous version).
    """
    # Raw values needed for engineering
    iron   = float(row.get("mineral_signature_iron", 0) or 0)
    nickel = float(row.get("mineral_signature_nickel", 0) or 0)
    cobalt = float(row.get("mineral_signature_cobalt", 0) or 0)
    plat   = float(row.get("mineral_signature_platinum", 0) or 0)
    rare   = float(row.get("mineral_signature_rare_earth", 0) or 0)
    water  = float(row.get("water_ice_fraction", 0) or 0)

    p_iron  = float(row.get("mineral_price_iron", 50) or 50)
    p_nick  = float(row.get("mineral_price_nickel", 200) or 200)
    p_cob   = float(row.get("mineral_price_cobalt", 500) or 500)
    p_plat  = float(row.get("mineral_price_platinum", 2000) or 2000)
    p_rare  = float(row.get("mineral_price_rare_earth", 800) or 800)
    p_water = float(row.get("mineral_price_water", 100) or 100)

    mass  = float(row.get("mass", 100) or 100)
    cycle = float(row.get("economic_cycle_indicator", 1.0) or 1.0)
    cryst = float(row.get("crystalline_fraction", 0.4) or 0.4)

    si  = float(row.get("structural_integrity", 0.7) or 0.7)
    vc  = float(row.get("volatile_content", 0.2) or 0.2)
    po  = float(row.get("porosity", 0.3) or 0.3)
    sc  = float(row.get("survey_confidence", 0.6) or 0.6)
    den = float(row.get("density", 4.0) or 4.0)
    eh  = float(row.get("environmental_hazard_rating", 0.3) or 0.3)

    # ── Value proxy features ──────────────────────────────────────────────────
    weighted_mineral = (
        iron * p_iron + nickel * p_nick + cobalt * p_cob
        + plat * p_plat + rare * p_rare + water * p_water
    )
    log_mass = float(np.log1p(mass))

    row["eng_weighted_mineral_score"] = weighted_mineral
    row["eng_mass_x_mineral_score"]   = log_mass * weighted_mineral
    row["eng_cycle_adj_score"]        = weighted_mineral * cycle
    row["eng_mineral_concentration"]  = iron + nickel + cobalt + plat + rare
    row["eng_value_density"]          = weighted_mineral * cryst
    row["eng_mass_cycle_mineral"]     = log_mass * cycle * weighted_mineral
    row["eng_platinum_mass"]          = plat * p_plat * log_mass

    # ── Catastrophe interaction features ─────────────────────────────────────
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
    All engineered features computed inline.
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

        row = _engineer_features(row)
        rows.append(row)

    df = pd.DataFrame(rows)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    return df[feature_cols]


# ── Heuristic fallback (if model not loaded) ──────────────────────────────────
def _heuristic_value(features):
    """
    Fallback without any third-party estimates.
    Uses weighted mineral score × mass proxy directly.
    """
    iron   = float(features.get("mineral_signature_iron", 0) or 0)
    nickel = float(features.get("mineral_signature_nickel", 0) or 0)
    cobalt = float(features.get("mineral_signature_cobalt", 0) or 0)
    plat   = float(features.get("mineral_signature_platinum", 0) or 0)
    rare   = float(features.get("mineral_signature_rare_earth", 0) or 0)
    water  = float(features.get("water_ice_fraction", 0) or 0)
    p_iron  = float(features.get("mineral_price_iron", 50) or 50)
    p_nick  = float(features.get("mineral_price_nickel", 200) or 200)
    p_cob   = float(features.get("mineral_price_cobalt", 500) or 500)
    p_plat  = float(features.get("mineral_price_platinum", 2000) or 2000)
    p_rare  = float(features.get("mineral_price_rare_earth", 800) or 800)
    p_water = float(features.get("mineral_price_water", 100) or 100)
    mass    = float(features.get("mass", 100) or 100)
    cycle   = float(features.get("economic_cycle_indicator", 1.0) or 1.0)

    weighted = (iron*p_iron + nickel*p_nick + cobalt*p_cob
                + plat*p_plat + rare*p_rare + water*p_water)
    estimated = np.log1p(mass) * weighted * cycle * 0.5
    return max(0.0, estimated)


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
            p_any_cat = model["binary_cat_model"].predict_proba(df_batch)[:, 1]

            type_probs       = model["type_model"].predict_proba(df_batch)
            p_collapse_given = type_probs[:, _TYPE_COLLAPSE]
            p_outgas_given   = type_probs[:, _TYPE_OUTGAS]
            p_void_given     = type_probs[:, _TYPE_VOID]

            type_sum         = p_collapse_given + p_outgas_given + p_void_given + 1e-9
            p_collapse_given = p_collapse_given / type_sum
            p_outgas_given   = p_outgas_given   / type_sum
            p_void_given     = p_void_given     / type_sum

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
    # With lower prediction accuracy (R²=0.785 vs 0.943), more uncertainty
    # means we should shade bids slightly more conservatively
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

    # Slightly more conservative shade given higher prediction uncertainty
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

    min_profit_threshold = max(15.0, capital * 0.005)
    raw_bids = np.where(raw_bids < min_profit_threshold, 0.0, raw_bids)

    # Hard filter: catastrophe probability > 35%
    p_catastrophe = 1.0 - p_none
    raw_bids = np.where(p_catastrophe > 0.35, 0.0, raw_bids)

    # Per-bid cap: 18% of capital max
    per_bid_cap = capital * 0.18
    raw_bids    = np.minimum(raw_bids, per_bid_cap)

    # ── Step 9: Portfolio budget allocation ──────────────────────────────────
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