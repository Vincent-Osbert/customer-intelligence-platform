# Model Card: Customer Intelligence Platform

## 1. Architecture Overview

The **Customer Intelligence Platform** employs a two-tier machine learning architecture:
1. **Tier 1 (Supervised Classification):** Three independent binary ensemble classifiers predict core customer behavioral dimensions:
   * $\mathcal{M}_1$: `IsHighValue` (high lifetime monetary contribution)
   * $\mathcal{M}_2$: `IsFrequent` (consistently high purchase cadence)
   * $\mathcal{M}_3$: `IsChurning` (imminent risk of disengagement / defection)
2. **Tier 2 (Unsupervised Clustering):** K-Means clusters customers using their calibrated probability vector $[\hat{p}_{\text{high}}, \hat{p}_{\text{freq}}, \hat{p}_{\text{churn}}]$, mapping them directly into actionable commercial segments.

---

## 2. Supervised Classification Models

### 2.1 Candidate Evaluation & Model Selection

All candidate pipelines were evaluated using 5-fold stratified cross-validation on an 80% training split, followed by held-out evaluation on a 20% test split. Standard scaling was applied to all inputs within each pipeline fold to prevent data leakage.

| Target Label | Class Distribution | Candidate Model | 5-Fold CV PR AUC (Mean ± Std) | 5-Fold CV ROC AUC (Mean) | Selected Model | Test Set PR AUC | Test Set ROC AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **IsHighValue** | 25.0% Positive | **XGBoost Classifier**<br>Random Forest<br>Logistic Regression | **0.9996 ± 0.0004**<br>0.9968 ± 0.0009<br>0.9706 ± 0.0055 | **0.9999**<br>0.9990<br>0.9912 | **XGBoost** | **0.9992** | **0.9997** |
| **IsFrequent** | 65.6% Positive | **XGBoost Classifier**<br>Random Forest<br>Logistic Regression | **1.0000 ± 0.0000**<br>0.9999 ± 0.0000<br>0.9974 ± 0.0007 | **1.0000**<br>0.9998<br>0.9938 | **XGBoost** | **1.0000** | **0.9999** |
| **IsChurning** | 13.9% Positive | **Random Forest**<br>XGBoost Classifier<br>Logistic Regression | **0.4517 ± 0.0483**<br>0.4386 ± 0.0515<br>0.2683 ± 0.0256 | **0.8725**<br>0.8575<br>0.7503 | **Random Forest** | **0.4233** | **0.8835** |

### 2.2 Why PR AUC Over ROC AUC?

In customer analytics, target classes are almost always imbalanced:
* `IsChurning` accounts for only **13.9%** of the population.
* `IsHighValue` accounts for **25.0%** of the population.

**The Problem with ROC AUC:** ROC AUC measures the False Positive Rate ($\frac{FP}{FP + TN}$) on the x-axis. When negative examples vastly outnumber positive examples (a large pool of $TN$), a classifier can generate hundreds of false alarms while its False Positive Rate remains deceptively close to zero. Consequently, ROC AUC yields an artificially high score (e.g., 0.88 for Churn) even when the majority of positive predictions are incorrect.

**The Advantage of Precision-Recall AUC (PR AUC):** PR AUC ignores True Negatives entirely. It directly balances Precision ($\frac{TP}{TP + FP}$, the accuracy of flagged customers) against Recall ($\frac{TP}{TP + FN}$, the percentage of at-risk customers captured). A baseline random model on Churn achieves a PR AUC of only 0.1388 (its base rate); our Random Forest model achieves 0.4233—a **3.05× lift over baseline**—making it the optimal choice for cost-sensitive intervention.

### 2.3 Feature Space (11 Behavioral Attributes)

To prevent data leakage, each classifier trains on 10 features, explicitly withholding the raw source column that defines its threshold:

| Feature Name | Category | Description | Data Type |
| :--- | :--- | :--- | :--- |
| `Recency` | RFM | Days elapsed between the customer's last purchase and the snapshot reference date. | Integer (days) |
| `Frequency` | RFM | Total count of unique purchase invoice transactions placed by the customer. | Integer (count) |
| `Monetary` | RFM | Total cumulative spend (£) across all valid transactions. | Float (£) |
| `AvgBasketSize` | Basket | Average number of units purchased per invoice transaction. | Float (units) |
| `AvgOrderValue` | Basket | Average gross revenue (£) generated per transaction invoice. | Float (£) |
| `MaxOrderValue` | Basket | Value of the single largest transaction invoice recorded for this customer. | Float (£) |
| `PurchaseDaySpread` | Temporal | Total calendar days elapsed between first purchase and latest purchase. | Integer (days) |
| `AvgDaysBetweenOrders`| Temporal | Average cadence between repeat purchases ($\frac{\text{Spread}}{\text{Orders}}$). | Float (days) |
| `PreferredHour` | Temporal | Modal hour of day (0–23) when the customer places transactions. | Integer (hour) |
| `PreferredDayOfWeek` | Temporal | Modal day of week (0=Mon, 6=Sun) when transactions occur. | Integer (weekday) |
| `UniqueProducts` | Diversity | Total count of distinct `StockCode` items ordered across all visits. | Integer (SKU count) |

---

## 3. Unsupervised Clustering Model

### 3.1 Why Cluster on Probability Scores Instead of Raw Features?

Traditional RFM clustering applies K-Means directly to raw recency, frequency, and monetary values. This approach suffers from three major flaws:
1. **Severe Skew and Scaling Artifacts:** Monetary spend and order volumes exhibit heavy power-law distributions. A single wholesale customer spending £50,000 skews Euclidean distance metrics, isolating themselves into an unhelpful outlier cluster.
2. **Loss of Non-Linear Inter-relationships:** Raw distance functions treat all feature dimensions independently. They cannot capture learned relationships (e.g., how basket size and purchase spread together moderate churn probability).
3. **Incompatible Units:** Standardizing days, currency, and SKU counts does not give them equal business meaning. 

**Our Solution:** We cluster in **calibrated probability space** $[p_{\text{high}}, p_{\text{freq}}, p_{\text{churn}}]$. Every dimension is bounded on $[0, 1]$, carries the exact same mathematical unit of confidence, and represents the non-linear synthesis of all 11 behavioral features. A coordinate of `[0.95, 0.92, 0.05]` immediately defines a high-value, highly frequent, loyal champion regardless of whether their absolute basket size was 50 items or 500 items.

### 3.2 Algorithm Comparison: K-Means vs. Gaussian Mixture Models (GMM)

Clustering was evaluated across $K \in [2, 7]$ using Inertia Elbow analysis and Silhouette coefficients:

| Algorithm | Cluster Count ($K$) | Silhouette Score | Assignment Nature | Boundary Flexibility |
| :--- | :--- | :--- | :--- | :--- |
| **K-Means (selected)** | **6** | **0.9245** | **Hard / Rigid Partition** | **Spherical / Equal Variance** |
| Gaussian Mixture Model (GMM) | 6 | 0.883 | Soft / Probabilistic | Ellipsoidal / Unconstrained Covariance |

**Selection Decision:** K-Means was selected. The pre-set rule adopts GMM only when its silhouette is within 0.02 of K-Means; the gap is 0.0415 (0.9245 vs. 0.883), so the rule picks K-Means. The six K-Means clusters are then merged into four business segment names by a centroid rule in `src/clustering.py`. Silhouette is computed in a classifier-probability space where two of the three axes are almost binary, so the high absolute value mostly reflects well-separated labels, not unusually clean customer behavior.

---

## 4. Risk Factors & Failure Modes

Deploying these models in production introduces specific operational risks:

1. **Seasonal Holiday Distortions:** The dataset includes peak Q4 holiday trading (Christmas novelty purchasing). Shoppers who buy large gift orders once in November may be falsely classified as high-probability churners throughout spring, even though they are merely annual holiday buyers.
2. **Wholesale vs. Retail Conflation:** The retail store serves both B2B shop owners (purchasing 500 units per line item) and B2C consumers (purchasing 2 units). The models may blend these profiles if order volume spikes without category separation.
3. **Data Drift & Price Inflation:** The model was trained on 2010–2011 prices (£ Sterling). Applying these models to current transaction data without adjusting monetary percentiles for inflation and catalog expansion will lead to severe segment misclassification.
4. **Target Leakage Risk in Retraining:** Supervised labels are derived from quantile thresholds. If training data is updated dynamically without refitting threshold cutoffs on fixed temporal backtests, label distributions will drift.
