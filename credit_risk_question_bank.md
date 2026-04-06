# Credit Risk Analysis — Comprehensive Question Bank
## Corporate Loan Portfolio Edition

*Designed for the `synthetic_portfolio_risk_data.csv` dataset (5,000 publicly traded companies)*

---

## Part A: Problem Framing & Business Context

### Question 1: Defining the Objective

**(a)** You are advising a commercial bank's credit risk department. The bank holds a portfolio of corporate loans to publicly traded companies. Define the analytical objectives:

- What is the business problem the bank faces? Frame it in terms of both individual loan decisions and aggregate portfolio management.
- What does "better" vs. "worse" mean in this context? Define at least two evaluation perspectives: one from a risk-minimization lens and one from a return-maximization lens.
- What are the target variables for predictive modeling? Why might you need more than one target?
- How does this corporate lending problem differ fundamentally from consumer/P2P lending in terms of data availability, loss severity, and regulatory requirements (Basel III/IV, CECL, IFRS 9)?

**(b)** Examine the data attributes. Categorize them into meaningful groups:

- Which columns represent **borrower fundamentals** (income statement, balance sheet)?
- Which represent **market-based signals**?
- Which represent **bank-specific loan characteristics**?
- Which are **derived risk metrics** vs. **raw inputs**?
- Before any analysis, hypothesize: which 5 attributes do you expect to be most predictive of default, and why?

---

## Part B: Data Exploration

### Question 2: Exploratory Data Analysis

**(i) Default Rates and Industry Patterns**

- What is the overall default rate (`is_default`) in the portfolio? What is the NPL rate (`npl_status`)? Why might these differ?
- How does the default rate vary across `industry_sector`? Which sectors show the highest and lowest default rates?
- Within the highest-default sector, what is the distribution of key financial ratios (DSCR, interest coverage, debt-to-equity) compared to the lowest-default sector?
- Does the relationship between sector and default hold after controlling for leverage? (Hint: segment by `debt_to_assets` quartiles within each sector.)

**(ii) Loan Exposure Analysis**

- What is the distribution of `total_gross_loan_amount` across the portfolio? Are there concentration risks (a few very large exposures)?
- How does loan size relate to default status? Are larger loans more or less likely to default?
- Calculate the **Herfindahl-Hirschman Index (HHI)** of loan concentration by sector: HHI = Σ(sector_share²). Is the portfolio well-diversified?
- Plot the cumulative exposure curve: what percentage of total loan exposure comes from the top 10% of borrowers?

**(iii) Interest Expense and Debt Servicing**

- Examine the distribution of `interest_coverage` ratio. What proportion of borrowers have interest coverage below 1.5x (a common distress threshold)?
- How does `dscr` (debt service coverage ratio) vary by industry? Produce summary statistics (mean, median, std, min, max) by sector.
- Is there a clear threshold in DSCR below which default probability spikes? Plot default rate by DSCR decile.
- Compare `interest_coverage` vs. `dscr` as distress indicators: which better separates defaulters from non-defaulters? Use ROC-AUC for a single-variable classifier.

**(iv) Leverage and Capital Structure**

- Examine the joint distribution of `debt_to_assets` and `debt_to_equity`. How are they related mathematically? Can one be derived from the other?
- Segment borrowers into leverage buckets (Low: D/A < 0.4, Medium: 0.4–0.65, High: > 0.65). How does default rate change across buckets?
- Within highly leveraged borrowers (D/A > 0.65), which additional ratio best discriminates defaulters from survivors? Test `cash_ratio`, `current_ratio`, and `interest_coverage`.
- The Altman Z-Score combines multiple ratios into a single bankruptcy predictor. Construct a simplified Z-Score proxy using available columns and test its predictive power.

**(v) Liquidity Analysis**

- Compare `cash_ratio`, `quick_ratio`, and `current_ratio` distributions. How do they rank in terms of conservatism?
- For defaulted companies, what was the median `cash_ratio` and `current_ratio` at the time of observation? How does this compare to non-defaulted companies?
- Is there a "liquidity trap" — companies with high current ratios but low cash ratios (i.e., liquidity tied up in inventory)? How does default rate compare for these companies?
- Calculate a derived "liquidity gap" = `current_ratio` − `cash_ratio`. Does this gap predict default?

**(vi) Profitability and Efficiency**

- How do `net_profit_margin`, `roa`, and `roe` differ between defaulted and non-defaulted firms? Show distributions.
- Is `roe` misleading for highly leveraged firms? (A firm with thin equity can show high ROE.) Examine ROE by leverage bucket.
- How do efficiency ratios (`asset_turnover`, `inventory_turnover`, `payable_turnover`) relate to default? Do operationally efficient firms default less?
- Construct a "profitability score" combining `net_profit_margin`, `roa`, and `asset_turnover`. Does it outperform any single ratio?

**(vii) Market-Based Risk Indicators**

- How does `beta` relate to default? Are high-beta firms riskier from a credit perspective, or is systematic risk a different dimension from credit risk?
- Examine `volatility_annualized`: is higher equity volatility associated with higher default rates?
- Is `max_drawdown` predictive of default? What is the default rate for companies with drawdowns > 50%?
- The Merton model suggests that equity is a call option on firm assets. Can you combine `share_price`, `shares_outstanding`, `total_liabilities`, and `volatility_annualized` to estimate a Distance-to-Default? How does this compare to simpler ratio-based predictors?

**(viii) Portfolio Risk Metrics**

- What is the distribution of `sharpe_ratio` and `sortino_ratio` in the portfolio? What proportion of companies have negative Sharpe ratios?
- How does `alpha` relate to default? Do companies that underperform their benchmark default more?
- Examine `r_squared`: for firms with low R² (idiosyncratic risk dominance), is default more or less common?
- Calculate the portfolio-level weighted average beta using `portfolio_weight`: what is the overall portfolio's systematic risk exposure?

### Question 2(b): Missing Values & Data Quality

- Identify which columns have missing, infinite, or undefined values (e.g., PE ratio when EPS ≤ 0).
- For each such column, explain *why* the value is missing and propose a handling strategy:
  - Should you impute, flag, or exclude?
  - Is missingness itself informative? (e.g., negative EPS → no valid PE ratio signals distress.)
- Create binary indicator variables for key "missingness signals" and test if they predict default.
- Which variables would you exclude entirely from modeling? Justify with data quality metrics.

### Question 2(c): Data Leakage

- Identify which variables in the dataset would **not be available at the time of loan origination**. These include:
  - `days_past_due`, `provision_for_credit_losses`, `npl_status` — these are outcomes, not predictors.
  - `is_default`, `portfolio_weight` — target variables.
- Are any of the calculated ratios contaminated by leakage? For example, does `cost_of_risk` use post-origination data?
- In a real deployment, which market data variables might change between loan origination and the model's training window? How would you handle this temporal mismatch?
- Design a proper temporal train/test split: if you were training on 2022 data to predict 2023 defaults, which variables would need to be lagged?

---

## Part C: Feature Engineering & Selection

### Question 3: Univariate Analysis

- For each candidate predictor, compute a measure of association with `is_default`:
  - For continuous predictors: use the AUC of a single-variable logistic regression, or the Kolmogorov-Smirnov (KS) statistic.
  - For categorical predictors (`industry_sector`): use chi-squared test or Weight of Evidence (WoE).
- Rank all variables by their individual predictive power. Do the top 5 match your hypothesis from Question 1(b)?
- If any variable is *too* predictive (AUC > 0.90 alone), investigate whether it has a leakage issue.
- Calculate the Weight of Evidence (WoE) and Information Value (IV) for each variable. Which variables have IV > 0.3 (strong predictors)?

### Question 3(b): Derived Features

Generate at least 5 new features and justify each:

1. **Altman Z-Score proxy:** Combine working capital/total assets, retained earnings proxy, EBIT/total assets, market cap/total liabilities, revenue/total assets. Map to distress zones.
2. **Debt service buffer:** `dscr` − 1.0 (how much cushion above breakeven).
3. **Cash burn rate indicator:** `cash` / `operating_profit` (months of cash runway if profits stop).
4. **Market-implied leverage:** `total_liabilities` / (`share_price` × `shares_outstanding` + `total_liabilities`).
5. **Return vs. risk efficiency:** `roa` / `volatility_annualized`.

For each, analyze its distribution, relationship with default, and incremental predictive value over existing features.

---

## Part D: Predictive Modeling

### Question 4: Train/Validation Split

**(a)** Split the data into training and validation sets:

- What ratio do you use (70/30? 80/20?)? Justify.
- Given the default rate (~17%), do you need stratified sampling? Why?
- Would a temporal split be more appropriate than random? Discuss the tradeoffs.

**(b)** Define your evaluation framework:

- Why is accuracy a poor metric for this problem? Demonstrate with the "predict all non-default" baseline.
- Which metrics do you prioritize: AUC-ROC, AUC-PR, F1, recall, precision? Justify based on the business cost asymmetry (cost of missing a default vs. cost of rejecting a good loan).
- Define a custom cost function: if a default costs the bank $X in loss and a rejected good loan costs $Y in lost interest income, what is the optimal classification threshold?

### Question 5: Logistic Regression Baseline

- Fit a logistic regression model. Which variables are statistically significant?
- Interpret the coefficients: for a 1-unit increase in `debt_to_equity`, how does the log-odds of default change?
- Check for multicollinearity using VIF. Which variables are collinear, and how do you handle this?
- Compare in-sample vs. out-of-sample performance. Is there evidence of overfitting?

### Question 6: Tree-Based Models

**(a) Decision Tree:**

- Train a classification tree (e.g., `sklearn.tree.DecisionTreeClassifier`). Experiment with `max_depth`, `min_samples_leaf`, and `criterion`.
- Visualize the tree. What is the first split variable? Does this make business sense?
- Compare pruned vs. unpruned trees. Which generalizes better?

**(b) Random Forest:**

- Train a random forest. Experiment with `n_estimators`, `max_features`, `max_depth`.
- Extract variable importance (Gini importance and permutation importance). How do these differ?
- Does the random forest outperform the single tree? By how much (in AUC)?

**(c) Gradient Boosted Trees (XGBoost/LightGBM):**

- Train a GBM. Experiment with `learning_rate`, `n_estimators`, `max_depth`, `subsample`, `colsample_bytree`.
- Use early stopping on validation loss. How many rounds are optimal?
- Compare SHAP values for the top 10 features. Are there non-linear relationships or interactions the GBM captures that logistic regression misses?
- Does GBM outperform Random Forest? Is the improvement worth the added complexity?

### Question 7: Model Comparison & Selection

- Tabulate all models with: AUC-ROC, AUC-PR, F1 (at optimal threshold), Precision@90% Recall, Recall@90% Precision.
- Plot ROC curves and Precision-Recall curves on the same axes.
- Which model would you recommend for production deployment, and why? Consider not just performance but also interpretability, regulatory requirements (SR 11-7), and computational cost.

---

## Part E: Portfolio-Level Risk Analysis

### Question 8: Loss Given Default (LGD) & Expected Loss

- For defaulted loans, calculate the recovery rate: what fraction of `total_gross_loan_amount` might the bank recover? (Assume recovery = 40% for senior unsecured, varying by industry.)
- Calculate **Expected Loss (EL)** for each loan: EL = PD × LGD × EAD, where PD comes from your best model, LGD is assumed, and EAD = `total_gross_loan_amount`.
- What is the total Expected Loss for the portfolio? Express as a percentage of total exposure.
- How does EL distribute across industries? Which sectors contribute disproportionately to portfolio risk?

### Question 9: Portfolio Risk Metrics

- Using `portfolio_weight` and `volatility_annualized`, calculate the portfolio's weighted average volatility. Why is this an overestimate of true portfolio volatility (hint: diversification)?
- Calculate the portfolio's weighted average beta. Is the loan book tilted toward cyclical or defensive industries?
- Using `var_95_amount` and portfolio weights, estimate the portfolio's aggregate Value-at-Risk. What are the limitations of summing individual VaRs?
- If the bank wanted to reduce portfolio VaR by 10%, which loans would you recommend selling or hedging? Consider both concentration risk and individual risk contribution.

### Question 10: Stress Testing

- Define 3 macroeconomic stress scenarios:
  - **Mild recession:** Revenue drops 10%, interest rates rise 200bps.
  - **Severe recession:** Revenue drops 25%, interest rates rise 400bps, equity prices fall 40%.
  - **Sector shock:** Energy/Materials revenue drops 50%, other sectors drop 10%.
- For each scenario, recalculate `interest_coverage`, `dscr`, and `debt_to_equity`. How many additional defaults would your model predict under each scenario?
- What is the portfolio's stressed Expected Loss under each scenario?
- Which industries are most vulnerable in each scenario? Recommend portfolio rebalancing actions.

---

## Part F: Investment & Business Decision Framework

### Question 11: Loan Pricing and Risk-Adjusted Returns

- Using your default probability estimates, calculate the **risk-adjusted return** for each loan: Risk-Adjusted Return = (1 − PD) × Interest Income − PD × LGD × EAD.
- Which loans have the highest risk-adjusted returns? Are they from the highest or lowest risk segments?
- Is there a "sweet spot" — moderate risk loans (e.g., PD between 5–15%) that offer the best risk-return tradeoff?
- How does your model-based loan selection compare to the existing grade system implied by `debt_to_assets` buckets?

### Question 12: Capital Adequacy

- Under Basel III, banks must hold capital against credit risk. Using the Standardized Approach, calculate the Risk-Weighted Assets (RWA) for this portfolio. (Assume corporate risk weights by credit quality: AAA-AA = 20%, A = 50%, BBB = 100%, Below BB = 150%.)
- Map your PD estimates to implied credit ratings. What is the total RWA?
- If the bank's minimum capital ratio is 10.5% (including buffers), how much capital must be held against this loan book?
- If the bank wanted to improve its capital ratio without reducing lending, which loans should it securitize or sell?

### Question 13: Combining Classification and Regression Models

- You have a classification model (predicting `is_default`) and potentially a regression model (predicting `portfolio_weight` or returns). How would you combine these for investment decisions?
- Approach 1: Invest in all loans predicted "non-default" → calculate expected portfolio return.
- Approach 2: Rank loans by predicted risk-adjusted return, invest in top N → calculate expected portfolio return.
- Approach 3: Optimize for a target Sharpe ratio using predicted PDs and returns → compare portfolio performance.
- Which approach maximizes risk-adjusted portfolio returns? Which is most robust to model errors?

### Question 14: Regulatory and Ethical Considerations

- How should model validation be structured under SR 11-7 / SS1/23 guidance? What documentation is required?
- What are the risks of using market data (stock prices, beta) in credit models? How might market-based signals fail in a crisis?
- Discuss model risk: if your GBM has 5% higher AUC but is less interpretable than logistic regression, which should the bank use for regulatory capital calculations? Why might the answer differ for internal risk management?
- How would you monitor model performance in production? Define specific trigger thresholds for model retraining (e.g., PSI > 0.2 on key features, AUC degradation > 5%).

---

## Bonus: Advanced Topics

### B1: PD × EAD × LGD Framework (CECL / IFRS 9)

- Under CECL (ASC 326), banks must estimate lifetime expected credit losses at origination. Using your PD model, estimate 1-year and lifetime PDs for each loan.
- How would you build a Probability of Default term structure (PD curves over time)?
- Under IFRS 9, loans are staged: Stage 1 (performing), Stage 2 (significant increase in credit risk), Stage 3 (credit-impaired). Using your data, define staging criteria and calculate provisions for each stage.

### B2: Merton/KMV Distance-to-Default

- Using `share_price`, `shares_outstanding`, `total_liabilities`, and `volatility_annualized`, estimate the Distance-to-Default for each company.
- How does this market-implied default measure compare to your accounting-ratio-based model?
- Can you combine both approaches into a hybrid model? Does it outperform either alone?

### B3: Credit Migration Analysis

- If you had multi-period data, how would you build a credit migration (transition) matrix?
- What would the matrix tell you about portfolio dynamics (upgrades vs. downgrades)?
- How would you use the migration matrix for multi-year loss forecasting?

---

*This question bank covers: EDA, feature engineering, supervised classification, regression, portfolio analytics, stress testing, regulatory capital, and model governance — the full stack of credit risk analysis.*
