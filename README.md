# CreditLens — Corporate Loan Portfolio Risk Analytics

A comprehensive end-to-end credit risk analysis of a synthetic corporate loan portfolio, covering exploratory data analysis, feature engineering, predictive modelling, portfolio-level risk quantification, and regulatory frameworks.

---

## Project Overview

A commercial bank holds a portfolio of corporate loans. This project answers a structured credit risk question bank across six analytical parts:

| Part | Focus |
|------|-------|
| A | Problem Framing & Business Context |
| B | Exploratory Data Analysis |
| C | Feature Engineering & Selection |
| D | Predictive Modelling |
| E | Portfolio-Level Risk Analysis |
| F | Investment & Business Decision Framework |
| Bonus | CECL/IFRS 9, Merton KMV, Credit Migration |

---

## Dataset

**File:** `synthetic_portfolio_risk_data.csv`

~5,000 synthetic corporate loan records with features across six categories:

- **Income Statement:** revenue, COGS, gross profit, EBIT, net income, interest expense
- **Balance Sheet:** total assets, total debt, equity, working capital, retained earnings
- **Liquidity Ratios:** cash ratio, quick ratio, current ratio
- **Leverage & Coverage:** debt-to-assets, debt-to-equity, DSCR, interest coverage
- **Market Indicators:** beta, annualised volatility, max drawdown, market cap
- **Portfolio Metrics:** Sharpe ratio, Sortino ratio, alpha, R-squared, portfolio weight

**Target variable:** `is_default` (binary) — 17.16% overall default rate

---

## Part A — Problem Framing

- Defined the dual objective: predict default at the **loan level** (PD model) and manage risk at the **portfolio level** (concentration, EL, VaR)
- Categorised all 40+ attributes by data type and availability at origination
- Hypothesised the top 5 predictors before any modelling: DSCR, interest coverage, debt-to-assets, cash ratio, Altman Z-Score

---

## Part B — Exploratory Data Analysis

### Default Rate & Sector Patterns
- Overall default rate: **17.16%** (is_default), NPL rate: **3.62%** (npl_status differ because NPL is a broader watchlist bucket)
- Real Estate (28.2%) and Energy (27.3%) have the highest sector default rates
- Utilities (17.5%) and Healthcare (19.2%) are the safest sectors
- Sector differences persist even after controlling for leverage quartiles — genuine sector risk, not just a leverage proxy

### Loan Exposure
- Portfolio total exposure: **$4.69 trillion** (synthetic scale)
- HHI concentration index computed by sector — flags whether any single sector dominates
- Right-skewed distribution: a small number of very large loans carry disproportionate risk

### Debt Servicing (DSCR & Interest Coverage)
- Loans with DSCR < 1.0 (cannot service debt from operations) default at 3–4× the rate of DSCR > 2.0 loans
- Significant proportion of portfolio has interest coverage < 1.5×

### Leverage & Altman Z-Score
- Derived debt/equity from debt/assets: `D/E = D/A ÷ (1 − D/A)`
- **Altman Z-Score** computed as a distress indicator:

  ```
  Z = 1.2×(WC/TA) + 1.4×(RE/TA) + 3.3×(EBIT/TA) + 0.6×(MktCap/Debt) + 1.0×(Revenue/TA)
  ```

  - Z > 2.99 → Safe zone | Z 1.81–2.99 → Grey zone | Z < 1.81 → Distress zone

### Liquidity Trap
- Identified firms with high current ratio but low cash ratio — assets locked in illiquid inventory, not cash
- These firms appear solvent on traditional metrics but are vulnerable in stressed markets

### Market Indicators & Distance-to-Default
- Defaulters show higher beta, higher annualised volatility, and worse max drawdown
- **Merton Distance-to-Default** approximated: models equity as a call option on firm assets; DD measures standard deviations from insolvency

### Data Quality
- Handled infinite values from ratio calculations (division by zero in interest coverage, D/E)
- Identified and documented **data leakage variables**: `days_past_due`, `npl_status`, `credit_rating` — these are not available at origination and cannot be used as model features

---

## Part C — Feature Engineering

### Information Value (Weight of Evidence)
- Computed IV for all features to rank predictive power
- IV > 0.5 flagged as suspiciously strong (possible leakage); IV < 0.02 dropped as useless

### Engineered Features
Three composite features added:
1. **`altman_z`** — Altman Z-Score (described above)
2. **`debt_service_ratio`** — normalised DSCR variant
3. **`stress_composite`** — combines leverage, liquidity, and coverage signals

---

## Part D — Predictive Modelling

### Train / Test Split
- 80/20 stratified split preserving the 17% default rate in both sets
- All leakage columns excluded from the feature matrix

### Models

| Model | AUC-ROC | AUC-PR | F1 |
|-------|---------|--------|----|
| Logistic Regression | 0.780 | 0.515 | 0.505 |
| Decision Tree (depth=5) | 0.794 | 0.535 | 0.607 |
| Random Forest | **0.816** | **0.600** | **0.610** |
| XGBoost | 0.816 | 0.584 | 0.587 |

### Model Selection Rationale
- **XGBoost** used for portfolio PD estimates — highest AUC-ROC, handles class imbalance via `scale_pos_weight`
- **Logistic Regression** retained as regulatory baseline — SR 11-7 requires interpretable models for CECL/IFRS 9
- **SHAP values** computed for XGBoost to explain individual predictions and satisfy model governance requirements
- **VIF** checked on logistic regression features to detect multicollinearity

---

## Part E — Portfolio-Level Risk

### Expected Loss (EL = PD × LGD × EAD)

| Metric | Value |
|--------|-------|
| Total Portfolio Exposure | $4,687,270M |
| Total Expected Loss | $1,063,983M |
| EL as % of Exposure | **22.70%** |

Top EL sectors: Real Estate (28.2%), Energy (27.3%), Technology (25.7%)

### Portfolio Risk Metrics
- Weighted average volatility and portfolio VaR computed
- Loans contributing disproportionately to VaR identified as hedging/sale candidates

### Stress Testing
Three macro scenarios applied (revenue shock, interest rate shock, combined):
- Stress PD recalculated under each scenario
- Incremental EL quantified — shows buffer required above baseline provisions

---

## Part F — Business Decisions

### Loan Pricing (Risk-Adjusted Return)
```
Risk-Adjusted Return = (1 − PD) × Interest Income − PD × LGD × EAD
```
Loans where the spread does not compensate for PD are flagged for repricing or rejection.

### Basel III RWA
- Each loan's PD mapped to an implied credit rating
- Regulatory risk weights applied to compute **Risk-Weighted Assets**
- Capital Adequacy Ratio impact assessed

### Combined Classification + Regression Strategy
- Strategy 1: Lend only to predicted non-defaulters (PD < 0.5)
- Strategy 2: Use PD-tiered pricing
- Return comparison shows risk-adjusted superiority of model-guided allocation

### Regulatory & Ethical Considerations
SR 11-7 model validation framework requirements documented:
- Conceptual soundness
- Ongoing monitoring and back-testing
- Bias and fairness testing
- Independent validation

---

## Bonus: Advanced Topics

### CECL / IFRS 9 Staging
Loans staged according to IFRS 9 framework:
- **Stage 1** (PD ≤ 5%, DPD = 0): 12-month ECL provisioning
- **Stage 2** (PD 5–20% or DPD 1–29): Lifetime ECL, performing
- **Stage 3** (PD > 20% or DPD ≥ 30): Lifetime ECL, impaired

### Full Merton KMV Distance-to-Default
Iterative solver estimates true asset value and asset volatility from observable equity market data using the Black-Scholes-Merton framework.

### Credit Migration Matrix
Simulated transition matrix showing probability of loans migrating between rating grades — used for multi-year loss projections and stress scenario calibration.

---

## Repository Structure

```
├── credit_risk_analysis.ipynb      # Main analysis notebook (81 cells)
├── credit_risk_question_bank.md    # Question bank the notebook answers
├── synthetic_portfolio_risk_data.csv  # Synthetic loan portfolio dataset
└── README.md                       # This file
```

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
scipy
statsmodels
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap scipy statsmodels
```

---

## Key Takeaways

1. **DSCR and Altman Z-Score are the strongest individual predictors** of corporate default in this dataset
2. **Real Estate and Energy** sectors carry the highest credit risk; **Utilities** the lowest
3. **Random Forest and XGBoost** both achieve AUC-ROC ~0.82, significantly outperforming logistic regression
4. **22.7% of portfolio exposure** is at expected loss — requiring substantial regulatory provisioning
5. **Stress testing reveals non-linear risk amplification** — combined shocks produce more than additive EL increases
6. **Model governance matters**: leakage removal, SHAP explainability, and VIF checks are not optional in regulated environments
