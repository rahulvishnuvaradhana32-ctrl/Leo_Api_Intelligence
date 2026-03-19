# Predictive Reliability Modeling for Banking API Systems

This project implements **predictive reliability modeling for real-time financial data APIs** used by private banking firms. It demonstrates proactive failure prediction with uncertainty quantification for high-frequency trading and price update systems.

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Data loading & exploratory analysis | Complete |
| Phase 2 | Feature engineering & baseline models (LogReg, RF, XGBoost) | Complete |
| Phase 3 | Dataset expansion, cleaning & cross-API feature engineering | Complete |
| Phase 4 | MultiHorizonLSTM v5 — bidirectional, attention, focal loss | Complete |
| Phase 5 | Evaluation, conformal prediction, agent simulation, ablation | Complete |
| Phase 6 | v6 dataset (cross-API features) + status_code removal | Complete |
| Phase 7 | Retrain on v6 dataset (43 features) | Pending |

---

## Architecture — MultiHorizonLSTM v5

```
Input (B, 30, 43)
    ↓
2-Layer Bidirectional LSTM  [hidden=128 per direction → 256 out]
    ↓
LayerNorm  [applied on full sequence: B, T, 256]
    ↓
AttentionPooling  [learned scalar scores per timestep → weighted sum → B, 256]
    ↓
Dropout (p=0.3)
    ↓
3 × Linear heads  → sigmoid → failure prob at horizon 1, 5, 15
```

**Training details:**
- Loss: FocalLoss (gamma=2.0) with per-horizon pos_weight (cap=10 for h=1,5 | cap=20 for h=15)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=2, min_lr=1e-5)
- Stratified 80/10/10 split; scaler fitted on first 80% only (no leakage)
- Dataset: `data/banking_api_features_v6.csv` — 1,220,008 rows, 43 features
- Epochs: 30 | Best val loss: 0.2325 | Training time: ~9 hours (CPU)

---

## Results Summary

### Model Comparison (test set)

| Model | AUC-ROC | Notes |
|-------|---------|-------|
| Logistic Regression | 0.7001 | Linear baseline |
| Random Forest | 0.7072 | 100 trees |
| XGBoost | 0.7245 | Best non-sequential baseline |
| **LSTM v5 (avg)** | **0.7756** | **+7.0% over XGBoost** |

### Per-Horizon LSTM Performance

| Horizon | AUC-ROC | PR-AUC | Precision@100 |
|---------|---------|--------|--------------|
| 1-step  | 0.7778  | 0.4156 | 1.00 |
| 5-step  | 0.7760  | 0.4105 | 1.00 |
| 15-step | 0.7728  | 0.4098 | 1.00 |

### Conformal Prediction (ICP, alpha=0.10)

| Horizon | Coverage | Interval Width |
|---------|----------|---------------|
| 1-step  | 90.3%    | 0.9908 |
| 5-step  | 90.3%    | 0.9908 |
| 15-step | 90.3%    | 0.9908 |

Target coverage: 90.0% — all horizons meet the guarantee.

### Agent Simulation (1,000 transactions)

| Metric | Proactive (LSTM) | Reactive (Baseline) |
|--------|-----------------|---------------------|
| Failure rate | **7.80%** | 13.70% |
| Failure reduction | **-43.1%** | — |
| Cost / 1,000 tx | $3,900 | $6,850 |
| Annual cost saving | **$767,000** | — |
| Latency delta | +0.116s | — |

---

## Dataset

### banking_api_features_v6.csv (current training dataset)

| Property | Value |
|----------|-------|
| Total rows | 1,220,008 |
| Columns | 50 (43 used as features) |
| Failure rate | 13.88% |
| Date range | 2023-01-01 to 2024-12-31 |
| File size | ~430 MB |

**API distribution (balanced):**

| API | Share |
|-----|-------|
| transaction_api | 25.0% |
| market_data_api | 25.0% |
| stock_price_api | ~10.3% |
| crypto_api | ~10.4% |
| forex_api | ~10.4% |

**Data pipeline:**
1. Original synthetic CSV (802,874 rows)
2. DB rows merged + deduplicated (+417,134 rows after dedup)
3. Null `error_type` fixed via backfill / api_name mapping
4. transaction_api + market_data_api jointly downsampled to ≤25% each
5. 5 cross-API correlation features added (1-minute bucket pivot with ffill)
6. `status_code` removed from feature set (r=0.98 with label — data leakage)

---

## Feature Engineering (43 features)

```
Core telemetry         : response_time, request_count
Calendar               : hour, day_of_week, is_market_hours, is_financial_peak,
                         is_weekend, is_holiday
Rolling statistics     : response_time_rolling_mean, response_time_rolling_std,
                         error_rate_rolling, response_time_variance, error_volatility
Lag features           : response_time_lag_1, response_time_lag_5, error_rate_lag_1
EMA features           : response_time_ema_10, response_time_ema_30, error_rate_ema_10
Cyclical encoding      : hour_sin, hour_cos, dow_sin, dow_cos
API flags              : high_frequency_api, api_complexity
Synthetic event signals: error_rate_boost, rt_multiplier
Differential features  : latency_diff_1, latency_diff_5,
                         error_rate_diff_1, error_rate_diff_5
Spike/burst indicators : latency_spike, error_burst, instability_index
Trend slopes           : latency_slope, error_slope
Traffic patterns       : traffic_change, burst_ratio
Cross-API correlation  : avg_error_rate_others, max_error_rate_others,
                         n_apis_elevated, corr_with_similar_api,
                         systemic_stress_index
```

---

## Project Structure

```
FCE_project/
├── data/
│   ├── banking_api_features.csv           # Original synthetic data
│   ├── banking_api_features_clean.csv     # Cleaned & balanced (1.22M rows, 45 cols)
│   └── banking_api_features_v6.csv        # + 5 cross-API features (50 cols) ← training
├── models/
│   ├── stress_test_best_model.pth         # LSTM v5 weights
│   ├── lstm_results.json                  # Training metrics (AUC, PR-AUC, per-horizon)
│   ├── evaluation_results.json            # Baseline comparison
│   ├── conformal_results.json             # ICP coverage + interval widths
│   ├── agent_simulation_results.json      # Proactive vs reactive metrics
│   ├── ablation_results.json              # Feature ablation AUC table
│   └── scaler.pkl                         # Fitted StandardScaler (joblib format)
├── scripts/
│   ├── run_lstm_training.py               # LSTM v5 training pipeline (43 features)
│   ├── evaluate_lstm.py                   # Baseline comparison (LR / RF / XGBoost / LSTM)
│   ├── conformal_prediction.py            # Inductive conformal predictor
│   ├── agent_simulation.py                # Proactive vs reactive simulation
│   ├── ablation_study.py                  # Feature group ablation
│   ├── self_improving_pipeline.py         # Drift detection + auto-retrain loop
│   ├── data_audit.py                      # Dataset audit (distributions, coverage)
│   ├── fix_dataset.py                     # Dataset cleaning & balancing
│   └── add_cross_api_features.py          # 5 cross-API correlation features
├── notebooks/
│   └── api_reliability_modeling.ipynb     # Phase 1-2 analysis notebook
├── requirements.txt
└── EVALUATION.md                          # Detailed evaluation report
```

---

## Quick Start

### Full pipeline (v6 dataset, 43 features)

```powershell
# Activate venv
.venv\Scripts\Activate

# Step 1 — Train LSTM v5 on v6 dataset
python scripts/run_lstm_training.py --data data/banking_api_features_v6.csv

# Step 2 — Evaluate vs baselines
python scripts/evaluate_lstm.py

# Step 3 — Ablation study
python scripts/ablation_study.py

# Step 4 — Conformal prediction (uncertainty quantification)
python scripts/conformal_prediction.py

# Step 5 — Agent simulation (proactive vs reactive)
python scripts/agent_simulation.py --n_transactions 1000 --seed 42

# Step 6 — Self-improving pipeline (dry run)
python scripts/self_improving_pipeline.py --dry_run
```

### Data pipeline (if regenerating from scratch)

```powershell
# Audit original dataset
python scripts/data_audit.py

# Clean + balance dataset (produces banking_api_features_clean.csv)
python scripts/fix_dataset.py

# Add cross-API features (produces banking_api_features_v6.csv)
python scripts/add_cross_api_features.py
```

---

## Dependencies

Core dependencies (`requirements.txt`):
- **torch**: Deep learning framework (LSTM, FocalLoss, attention)
- **pandas, numpy**: Data manipulation
- **scikit-learn**: Baseline models (LogReg, RF), scalers, metrics
- **xgboost**: Gradient boosting baseline
- **joblib**: Model/scaler serialisation
- **matplotlib, seaborn**: Visualisation

---

## Novel Contributions

1. **Multi-Horizon Probabilistic Forecasting**: Simultaneous 1/5/15-step ahead failure prediction from a single model
2. **AttentionPooling over LSTM sequence**: Learns which timesteps are most predictive rather than using only the final hidden state
3. **Per-horizon Focal Loss**: Class-imbalance correction tuned independently per prediction horizon (pos_weight cap=20 for 15-step)
4. **Conformal Prediction wrapper**: Guaranteed 90% coverage on uncertainty intervals without distribution assumptions
5. **Cross-API systemic stress features**: Detect simultaneous degradation across APIs (DDoS, vendor outage) invisible to single-API models
6. **Proactive agent simulation**: Demonstrates 43.1% failure reduction and $767k/year cost saving over reactive baseline

---

## References

- Hochreiter & Schmidhuber (1997) — Long Short-Term Memory
- Bahdanau et al. (2015) — Attention mechanism
- Lin et al. (2017) — Focal Loss for dense object detection
- Shafer & Vovk (2008) — Conformal Prediction
- de Prado (2018) — Advances in Financial Machine Learning

---

## License

MIT License
