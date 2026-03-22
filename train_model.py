"""
Asteroid Auction Challenge — Model Training Script FINAL
This is v2 — the best performing version confirmed by verify_model.py:
  - value MAE=9.62, corr=0.9992
  - thresh=0.35: precision=0.989, recall=1.000
  - Total bid behavior: correct (catastrophes skipped, negatives skipped)

Architecture:
  - value/yield/delay: XGBRegressor (R² 0.943/0.971/0.938)
  - binary catastrophe: XGBClassifier with scale_pos_weight (AUC 0.6475)
  - type classifier: XGBClassifier on catastrophe rows only (F1 0.4335)
  - 7 interaction features (composite_risk, etc.)

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

# ── 2. Engineer 7 interaction features ───────────────────────────────────────
# These were validated in deep_diagnostic.py:
#   composite_risk corr=+0.212 vs best raw feature structural_integrity: -0.174
# The 5 additional v3 features were dropped — they hurt value model MAE (9.62->16.00)
print("  Engineering interaction features...")
df["interact_composite_risk"]         = (1.0 - df["structural_integrity"]) * 0.4 + df["volatile_content"] * 0.3 + df["porosity"] * 0.3
df["interact_volatile_x_integrity"]   = df["volatile_content"] * (1.0 - df["structural_integrity"])
df["interact_low_integrity_x_poros"]  = (1.0 - df["structural_integrity"]) * df["porosity"]
df["interact_survey_x_integrity"]     = df["survey_confidence"] * df["structural_integrity"]
df["interact_volatile_x_porosity"]    = df["volatile_content"] * df["porosity"]
df["interact_env_hazard_x_volatile"]  = df["environmental_hazard_rating"] * df["volatile_content"]
df["interact_density_porosity_risk"]  = df["porosity"] / (df["density"] + 1e-9)

INTERACTION_FEATURES = [
    "interact_composite_risk",
    "interact_volatile_x_integrity",
    "interact_low_integrity_x_poros",
    "interact_survey_x_integrity",
    "interact_volatile_x_porosity",
    "interact_env_hazard_x_volatile",
    "interact_density_porosity_risk",
]
print(f"  Added {len(INTERACTION_FEATURES)} interaction features")

# ── 3. Feature sets ───────────────────────────────────────────────────────────
CAT_FEATURES = ["spectral_class", "belt_region", "probe_type"]
DROP_FEATURES = [
    "asteroid_id", "time_period", "lucky_number", "social_sentiment_score",
    "communication_delay", "orbital_period", "aphelion_distance", "perihelion_distance",
    "estimated_volume", "surface_gravity",
]
TARGET_COLS = [
    "mineral_value", "extraction_yield", "extraction_delay",
    "catastrophe_type", "toxic_outgassing_impact",
]

ALL_FEATURE_COLS = [c for c in df.columns if c not in TARGET_COLS + DROP_FEATURES]
NUM_FEATURES     = [c for c in ALL_FEATURE_COLS if c not in CAT_FEATURES]
print(f"  Features: {len(ALL_FEATURE_COLS)} ({len(NUM_FEATURES)} numeric, {len(CAT_FEATURES)} categorical)")

# ── 4. Preprocessor ───────────────────────────────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", NUM_FEATURES),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_FEATURES),
    ],
    remainder="drop"
)

def make_pipeline(model):
    return Pipeline([("prep", preprocessor), ("model", model)])

# ── 5. Clean rows ─────────────────────────────────────────────────────────────
clean_mask = (df["catastrophe_type"] == "none") & (df["toxic_outgassing_impact"] == 0)
df_clean   = df[clean_mask].copy()
X_clean    = df_clean[ALL_FEATURE_COLS]

# ── 6. Mineral Value Model ────────────────────────────────────────────────────
print("\n[1/4] Training mineral_value model...")
y_value = df_clean["mineral_value"]
value_model = make_pipeline(XGBRegressor(
    n_estimators=600, max_depth=7, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, tree_method="hist",
))
value_model.fit(X_clean, y_value)
scores = cross_val_score(value_model, X_clean, y_value, cv=5, scoring="r2", n_jobs=-1)
print(f"  mineral_value R² CV: {scores.mean():.4f} ± {scores.std():.4f}")

# ── 7. Extraction Yield Model ─────────────────────────────────────────────────
print("\n[2/4] Training extraction_yield model...")
y_yield = df_clean["extraction_yield"]
yield_model = make_pipeline(XGBRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, tree_method="hist",
))
yield_model.fit(X_clean, y_yield)
yield_scores = cross_val_score(yield_model, X_clean, y_yield, cv=5, scoring="r2", n_jobs=-1)
print(f"  extraction_yield R² CV: {yield_scores.mean():.4f} ± {yield_scores.std():.4f}")

# ── 8. Extraction Delay Model ─────────────────────────────────────────────────
print("\n[3/4] Training extraction_delay model...")
y_delay = df_clean["extraction_delay"]
delay_model = make_pipeline(XGBRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, tree_method="hist",
))
delay_model.fit(X_clean, y_delay)
delay_scores = cross_val_score(delay_model, X_clean, y_delay, cv=5, scoring="r2", n_jobs=-1)
print(f"  extraction_delay R² CV: {delay_scores.mean():.4f} ± {delay_scores.std():.4f}")

# ── 9. Catastrophe Classifiers (XGBoost — best calibration) ──────────────────
print("\n[4/4] Training catastrophe classifiers (XGBoost)...")
X_all = df[ALL_FEATURE_COLS]

# Stage 1: Binary — any catastrophe?
y_binary = (df["catastrophe_type"] != "none").astype(int)
n_none   = int((y_binary == 0).sum())
n_cat    = int((y_binary == 1).sum())
scale    = n_none / n_cat
print(f"  Binary — none:{n_none}, catastrophe:{n_cat}, scale_pos_weight={scale:.2f}")

binary_cat_model = make_pipeline(XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    scale_pos_weight=scale,
    random_state=42, n_jobs=-1, tree_method="hist", eval_metric="logloss",
))
binary_cat_model.fit(X_all, y_binary)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
binary_scores = cross_val_score(binary_cat_model, X_all, y_binary,
                                 cv=skf, scoring="roc_auc", n_jobs=-1)
print(f"  Binary catastrophe AUC CV: {binary_scores.mean():.4f} ± {binary_scores.std():.4f}")

# Stage 2: Type — which catastrophe type? (catastrophe rows only)
# 0=structural_collapse, 1=toxic_outgassing, 2=void_rock
cat_rows   = df[df["catastrophe_type"] != "none"].copy()
X_cat_only = cat_rows[ALL_FEATURE_COLS]

type_label_encoder = LabelEncoder()
y_cat_type  = type_label_encoder.fit_transform(cat_rows["catastrophe_type"])
type_classes = list(type_label_encoder.classes_)
print(f"  Type classes (0-indexed): {type_classes}")

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

# ── 10. Calibration check ─────────────────────────────────────────────────────
print("\nCalibration stats...")
residuals   = y_value.values - value_model.predict(X_clean)
print(f"  Value residual mean={residuals.mean():.2f}, std={residuals.std():.2f}")
delay_preds = delay_model.predict(X_clean)
print(f"  Delay range: {delay_preds.min():.1f} – {delay_preds.max():.1f}")

# ── 11. Bundle and save ───────────────────────────────────────────────────────
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
    "interaction_features": INTERACTION_FEATURES,
    "metadata": {
        "value_r2":    float(scores.mean()),
        "yield_r2":    float(yield_scores.mean()),
        "delay_r2":    float(delay_scores.mean()),
        "cat_auc":     float(binary_scores.mean()),
        "type_f1":     float(type_scores.mean()),
        "version":     "final_v2_xgb",
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
print("\nExpected from verify_model.py:")
print("  Value MAE ~9.6, thresh=0.35 precision>0.98, recall=1.000")