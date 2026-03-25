"""
Asteroid Auction Challenge — Model Training Script (No Trap Features)
Removed: ai_valuation_estimate, analyst_consensus_estimate, media_hype_score,
         lucky_number, social_sentiment_score

Key changes:
  - New engineered features replace lost signal from trap features:
      mass_x_mineral_score (corr=0.676) — single best predictor now
      weighted_mineral_score (corr=0.528)
      cycle_adj_score (corr=0.525)
      mineral_concentration (corr=0.461)
      value_density (corr=0.448)
  - Value model R² recovers from 0.746 to 0.785 with these features
  - All other models largely unaffected

Run from repo root: python train_model.py
Outputs: strategies/model.joblib
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
from xgboost import XGBRegressor, XGBClassifier

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading training data...")
df = pd.read_parquet("data/training.parquet")
print(f"  Loaded {len(df)} rows, {df.shape[1]} columns")

# ── 2. Remove trap features ───────────────────────────────────────────────────
TRAP_FEATURES = [
    "ai_valuation_estimate",
    "analyst_consensus_estimate",
    "media_hype_score",
    "lucky_number",
    "social_sentiment_score",
]
print(f"  Removing trap features: {TRAP_FEATURES}")

# ── 3. Engineer value proxy features ─────────────────────────────────────────
# Without AI/analyst estimates we manually compute what they approximated:
# total value ~ (mineral richness) x (mass) x (market prices) x (cycle)
print("  Engineering value proxy features...")

MINERAL_SIGS = [
    "mineral_signature_iron", "mineral_signature_nickel",
    "mineral_signature_cobalt", "mineral_signature_platinum",
    "mineral_signature_rare_earth",
]

# Weighted mineral score: sum of (signature × price) — corr=0.528
df["eng_weighted_mineral_score"] = (
    df["mineral_signature_iron"]       * df["mineral_price_iron"] +
    df["mineral_signature_nickel"]     * df["mineral_price_nickel"] +
    df["mineral_signature_cobalt"]     * df["mineral_price_cobalt"] +
    df["mineral_signature_platinum"]   * df["mineral_price_platinum"] +
    df["mineral_signature_rare_earth"] * df["mineral_price_rare_earth"] +
    df["water_ice_fraction"]           * df["mineral_price_water"]
)

# Mass × mineral score: accounts for asteroid size — corr=0.676 (best feature)
df["eng_mass_x_mineral_score"] = (
    np.log1p(df["mass"]) * df["eng_weighted_mineral_score"]
)

# Economic cycle adjusted score — corr=0.525
df["eng_cycle_adj_score"] = (
    df["eng_weighted_mineral_score"] * df["economic_cycle_indicator"]
)

# Total mineral concentration (sum of all signatures) — corr=0.461
df["eng_mineral_concentration"] = df[MINERAL_SIGS].sum(axis=1)

# Value density: mineral score weighted by crystalline fraction — corr=0.448
# Higher crystalline = more extractable ore
df["eng_value_density"] = (
    df["eng_weighted_mineral_score"] * df["crystalline_fraction"]
)

# Mass × cycle × mineral: three-way interaction
df["eng_mass_cycle_mineral"] = (
    np.log1p(df["mass"])
    * df["economic_cycle_indicator"]
    * df["eng_weighted_mineral_score"]
)

# Platinum-weighted score separately (highest value metal, corr=0.396 alone)
df["eng_platinum_mass"] = (
    df["mineral_signature_platinum"] * df["mineral_price_platinum"]
    * np.log1p(df["mass"])
)

VALUE_FEATURES = [
    "eng_weighted_mineral_score",
    "eng_mass_x_mineral_score",
    "eng_cycle_adj_score",
    "eng_mineral_concentration",
    "eng_value_density",
    "eng_mass_cycle_mineral",
    "eng_platinum_mass",
]
print(f"  Added {len(VALUE_FEATURES)} value proxy features")

# ── 4. Engineer catastrophe features (same as before) ────────────────────────
print("  Engineering catastrophe interaction features...")
df["interact_composite_risk"]        = (1.0 - df["structural_integrity"]) * 0.4 + df["volatile_content"] * 0.3 + df["porosity"] * 0.3
df["interact_volatile_x_integrity"]  = df["volatile_content"] * (1.0 - df["structural_integrity"])
df["interact_low_integrity_x_poros"] = (1.0 - df["structural_integrity"]) * df["porosity"]
df["interact_survey_x_integrity"]    = df["survey_confidence"] * df["structural_integrity"]
df["interact_volatile_x_porosity"]   = df["volatile_content"] * df["porosity"]
df["interact_env_hazard_x_volatile"] = df["environmental_hazard_rating"] * df["volatile_content"]
df["interact_density_porosity_risk"] = df["porosity"] / (df["density"] + 1e-9)

CAT_FEATURES = ["spectral_class", "belt_region", "probe_type"]

DROP_FEATURES = [
    "asteroid_id", "time_period",
    # trap features
    "ai_valuation_estimate", "analyst_consensus_estimate",
    "media_hype_score", "lucky_number", "social_sentiment_score",
    # redundant orbital (corr ~1.0 with semi_major_axis)
    "communication_delay", "orbital_period", "aphelion_distance", "perihelion_distance",
    # redundant physical
    "estimated_volume", "surface_gravity",
]

TARGET_COLS = [
    "mineral_value", "extraction_yield", "extraction_delay",
    "catastrophe_type", "toxic_outgassing_impact",
]

ALL_FEATURE_COLS = [c for c in df.columns if c not in TARGET_COLS + DROP_FEATURES]
NUM_FEATURES     = [c for c in ALL_FEATURE_COLS if c not in CAT_FEATURES]
print(f"  Features: {len(ALL_FEATURE_COLS)} ({len(NUM_FEATURES)} numeric, {len(CAT_FEATURES)} categorical)")

# ── 5. Preprocessor ───────────────────────────────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", NUM_FEATURES),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_FEATURES),
    ],
    remainder="drop"
)

def make_pipeline(model):
    return Pipeline([("prep", preprocessor), ("model", model)])

# ── 6. Clean rows ─────────────────────────────────────────────────────────────
clean_mask = (df["catastrophe_type"] == "none") & (df["toxic_outgassing_impact"] == 0)
df_clean   = df[clean_mask].copy()
X_clean    = df_clean[ALL_FEATURE_COLS]

# ── 7. Mineral Value Model ────────────────────────────────────────────────────
# Without trap features R² drops to ~0.746; new features recover to ~0.785
# Use more trees and deeper model to squeeze out more signal
print("\n[1/4] Training mineral_value model...")
y_value = df_clean["mineral_value"]
value_model = make_pipeline(XGBRegressor(
    n_estimators=800,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
))
value_model.fit(X_clean, y_value)
scores = cross_val_score(value_model, X_clean, y_value, cv=5, scoring="r2", n_jobs=-1)
print(f"  mineral_value R² CV: {scores.mean():.4f} ± {scores.std():.4f}")

# ── 8. Extraction Yield Model ─────────────────────────────────────────────────
print("\n[2/4] Training extraction_yield model...")
y_yield = df_clean["extraction_yield"]
yield_model = make_pipeline(XGBRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, tree_method="hist",
))
yield_model.fit(X_clean, y_yield)
yield_scores = cross_val_score(yield_model, X_clean, y_yield, cv=5, scoring="r2", n_jobs=-1)
print(f"  extraction_yield R² CV: {yield_scores.mean():.4f} ± {yield_scores.std():.4f}")

# ── 9. Extraction Delay Model ─────────────────────────────────────────────────
print("\n[3/4] Training extraction_delay model...")
y_delay = df_clean["extraction_delay"]
delay_model = make_pipeline(XGBRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, tree_method="hist",
))
delay_model.fit(X_clean, y_delay)
delay_scores = cross_val_score(delay_model, X_clean, y_delay, cv=5, scoring="r2", n_jobs=-1)
print(f"  extraction_delay R² CV: {delay_scores.mean():.4f} ± {delay_scores.std():.4f}")

# ── 10. Catastrophe Classifiers ───────────────────────────────────────────────
print("\n[4/4] Training catastrophe classifiers...")
X_all    = df[ALL_FEATURE_COLS]
y_binary = (df["catastrophe_type"] != "none").astype(int)
n_none   = int((y_binary == 0).sum())
n_cat    = int((y_binary == 1).sum())
scale    = n_none / n_cat
skf      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f"  Binary — none:{n_none}, catastrophe:{n_cat}, scale_pos_weight={scale:.2f}")

binary_cat_model = make_pipeline(XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    scale_pos_weight=scale,
    random_state=42, n_jobs=-1, tree_method="hist", eval_metric="logloss",
))
binary_cat_model.fit(X_all, y_binary)
binary_scores = cross_val_score(binary_cat_model, X_all, y_binary,
                                 cv=skf, scoring="roc_auc", n_jobs=-1)
print(f"  Binary catastrophe AUC CV: {binary_scores.mean():.4f} ± {binary_scores.std():.4f}")

cat_rows   = df[df["catastrophe_type"] != "none"].copy()
X_cat_only = cat_rows[ALL_FEATURE_COLS]

type_label_encoder = LabelEncoder()
y_cat_type   = type_label_encoder.fit_transform(cat_rows["catastrophe_type"])
type_classes = list(type_label_encoder.classes_)
print(f"  Type classes: {type_classes}")

type_model = make_pipeline(XGBClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    random_state=42, n_jobs=-1, tree_method="hist", eval_metric="mlogloss",
))
type_model.fit(X_cat_only, y_cat_type)
type_scores = cross_val_score(type_model, X_cat_only, y_cat_type,
                               cv=StratifiedKFold(5, shuffle=True, random_state=42),
                               scoring="f1_macro", n_jobs=-1)
print(f"  Type classifier F1-macro CV: {type_scores.mean():.4f} ± {type_scores.std():.4f}")

# ── 11. Save ──────────────────────────────────────────────────────────────────
print("\nSaving model bundle...")
model_bundle = {
    "value_model":         value_model,
    "yield_model":         yield_model,
    "delay_model":         delay_model,
    "binary_cat_model":    binary_cat_model,
    "type_model":          type_model,
    "type_label_encoder":  type_label_encoder,
    "type_classes":        type_classes,
    "all_feature_cols":    ALL_FEATURE_COLS,
    "num_features":        NUM_FEATURES,
    "cat_features":        CAT_FEATURES,
    "value_features":      VALUE_FEATURES,
    "metadata": {
        "value_r2":    float(scores.mean()),
        "yield_r2":    float(yield_scores.mean()),
        "delay_r2":    float(delay_scores.mean()),
        "cat_auc":     float(binary_scores.mean()),
        "type_f1":     float(type_scores.mean()),
        "version":     "no_trap_features",
        "trap_features_removed": TRAP_FEATURES,
    }
}

out_path = os.path.join("strategies", "model.joblib")
joblib.dump(model_bundle, out_path, compress=3)
size_mb = os.path.getsize(out_path) / 1_000_000
print(f"  Saved to {out_path} ({size_mb:.1f} MB)")
assert size_mb < 50, f"Model {size_mb:.1f} MB exceeds 50MB limit!"

print("\n✓ Training complete.")
print(f"  value R²={scores.mean():.4f} | yield R²={yield_scores.mean():.4f} | delay R²={delay_scores.mean():.4f}")
print(f"  cat AUC={binary_scores.mean():.4f} | type F1={type_scores.mean():.4f}")
print(f"\nBaseline without engineering: value R²~0.746")
print(f"Run verify_model.py and check: Value MAE, thresh=0.35 precision>0.90, recall=1.000")