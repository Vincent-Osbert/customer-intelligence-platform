# Project Architectural & Analytical Decisions: Customer Intelligence Platform

This document logs the core architectural, statistical, and modeling decisions made throughout the project lifecycle. Each entry details what was selected, alternative approaches that were evaluated and rejected, and the technical and business rationale behind each choice.

---

### Decision 1: Treatment of Returns and Order Cancellations

* **What We Decided:** Explicitly identify and remove all transaction rows where the `Invoice` number begins with `'C'` (9,288 rows), as well as records where `Quantity <= 0` or `Price <= 0` (2,517 rows).
* **What We Rejected:** 
  1. Imputing negative quantities to zero.
  2. Keeping negative quantities as negative revenue deductions directly in customer totals.
  3. Attempting fuzzy string matching to pair return rows with earlier purchase invoices.
* **Why:** In retail analytics, merchandise returns are driven by fundamentally different business processes than purchase intent (e.g., damaged transit, wrong apparel size, customer buyer's remorse). Retaining return rows creates severe distortion:
  * A return transaction carries an `InvoiceDate`. If included, it artificially resets the customer's `Recency` to a fresh date, falsely indicating recent positive engagement.
  * Negative quantities deflate `Monetary` spend and basket statistics without reducing the recorded invoice frequency.
  * Fuzzy matching historical returns in this dataset is unreliable because return records often lack reference to the original invoice.
  * Return propensity is best treated as a separate post-purchase operational model. For customer lifetime segmentation, focusing on gross purchase behavior provides an uncorrupted signal.

---

### Decision 2: Handling of Missing Customer Identifiers

* **What We Decided:** Drop all 132,220 transaction rows where `Customer ID` is null (representing 24.9% of raw records).
* **What We Rejected:** 
  1. Imputing missing customer IDs with a single sentinel ID (e.g., `Customer ID = 99999` for "Guest Shopper").
  2. Assigning each missing row an independent synthetic customer ID.
  3. Using IP addresses or session cookies to impute identity (not present in dataset).
* **Why:** The primary mission of this platform is longitudinal customer relationship management (CRM) and lifetime segmentation.
  * If all 132,220 guest rows were assigned to a single guest identifier, that dummy ID would accumulate over 132,000 items and hundreds of thousands of pounds in revenue, becoming an impossible "super-whale" that distorts all distribution quantiles and clustering centroids.
  * If every guest row were assigned a unique synthetic ID, we would fabricate thousands of artificial one-time buyers, severely deflating customer retention rates and inflating churn scores.
  * While guest orders are valuable for aggregate inventory forecasting, they cannot support personalized retention actions. Filtering them leaves 397,884 verified customer transactions across 4,338 distinct profiles—statistically robust for machine learning.

---

### Decision 3: Optimization Metric for Classification Models

* **What We Decided:** Select Precision-Recall Area Under the Curve (PR AUC / Average Precision) via 5-fold stratified cross-validation as the definitive model selection metric.
* **What We Rejected:** 
  1. Classification Accuracy.
  2. Receiver Operating Characteristic Area Under Curve (ROC AUC).
  3. Standard F1-Score at an arbitrary 0.5 decision threshold.
* **Why:** Retail customer behavior labels are naturally imbalanced:
  * `IsChurning` has a positive class prevalence of only **13.9%**.
  * `IsHighValue` has a positive class prevalence of **25.0%**.
  * **Why not Accuracy:** A trivial dummy model that predicts "Not Churning" for every customer achieves 86.1% accuracy while failing to identify a single at-risk customer.
  * **Why not ROC AUC:** ROC AUC measures the False Positive Rate ($\frac{FP}{FP + TN}$). When True Negatives ($TN$) are abundant, hundreds of false positives barely budge the denominator, making ROC AUC look deceptively strong (e.g., 0.88) even when precision is low.
  * **Why PR AUC:** PR AUC concentrates strictly on the minority positive class. It evaluates precision against recall across all classification thresholds, penalizing false alarms without being inflated by abundant negative cases.

---

### Decision 4: Clustering on Prediction Probabilities vs. Raw RFM Features

* **What We Decided:** Pass the three calibrated prediction probability scores ($p_{\text{high}}, p_{\text{freq}}, p_{\text{churn}}$) into the clustering algorithm rather than raw RFM and behavioral metrics.
* **What We Rejected:** Running K-Means or hierarchical clustering directly on normalized raw features (Recency, Frequency, Monetary, Basket Size, Product Diversity).
* **Why:**
  1. **Mitigating Extreme Skew:** Raw retail monetary values and order counts follow power-law distributions. Even with standard scaling or logarithmic transforms, wholesale outliers distort Euclidean distances and dominate cluster assignments.
  2. **Equalized Semantic Scale:** Raw features live in disparate units (calendar days, GBP sterling, units of stock). Standardizing them statistically does not guarantee equal business importance. In contrast, probability scores are strictly bounded on $[0, 1]$ and share the identical unit of meaning: model confidence in a commercial behavior.
  3. **Encoding Non-Linear Interactions:** The supervised classifiers have already ingested all 11 behavioral features, learning complex non-linear interactions. Clustering in probability space clusters customers according to *inferred intent and risk profile* rather than raw surface statistics.

---

### Decision 5: Algorithm Selection — K-Means Chosen Over GMM

* **What We Decided:** Evaluate both K-Means and Gaussian Mixture Models (GMM) on the same probability vectors, and adopt GMM only if its silhouette score is within 0.02 of K-Means. Latest run (`reports/clustering_metrics.json`): K-Means = 0.9245, GMM = 0.883, gap 0.0415, so **K-Means is the production model**.
* **What We Rejected:** GMM as the default. Its soft memberships are attractive for borderline customers, but it did not meet the pre-set silhouette margin, so the rule was followed instead of overridden after seeing the result.
* **Why:** The margin was fixed before the comparison to avoid choosing an algorithm after the fact. Soft assignments remain a possible extension if borderline-customer targeting becomes a requirement.

---

### Decision 6: Temporal Snapshot Reference Date

* **What We Decided:** Set the reference observation date (`snapshot_date`) to `max(InvoiceDate) + 1 day`.
* **What We Rejected:** 
  1. Setting `snapshot_date = max(InvoiceDate)`.
  2. Using the current execution system timestamp (`datetime.now()`).
* **Why:**
  * If `snapshot_date` equals `max(InvoiceDate)`, any customer who placed an order on the final recorded day of the dataset receives a `Recency` value of exactly 0 days. Zero recency creates division-by-zero errors in frequency-to-recency ratio calculations and renders logarithmic transformations ($\ln(0)$) undefined. Adding 1 day guarantees that $\text{Recency} \ge 1$ for all accounts.
  * If we used the current system date (e.g., 2026), all customers would register recencies exceeding 5,000+ days. This would completely flatten historical variance and render churn modeling impossible.

---

### Decision 7: Customer Lifetime Value (CLV) Calculation Methodology

* **What We Decided:** Compute segment-level Estimated Customer Lifetime Value using a discounted cash flow formula:
  $$\text{CLV} = (\text{Monthly Revenue} \times \text{Gross Margin}) \times \left( \frac{\text{Retention}}{1 + \text{Discount Rate} - \text{Retention}} \right)$$
  with explicitly documented assumptions (35% gross margin, 5% monthly churn, 10% annual discount rate).
* **What We Rejected:** Implementing complex probabilistic buy-till-you-die models (such as BG/NBD + Gamma-Gamma) via external third-party libraries.
* **Why:**
  * **Executive Understandability:** In an operational business intelligence dashboard, C-suite executives and marketing managers must understand exactly how CLV numbers are derived. Closed-form discounted cash flow formulas can be verified with a calculator.
  * **Interactive Scenario Planning:** Dashboard users can adjust margin and retention assumptions dynamically to see sensitivity analyses in real time.
  * **Model Stability:** Probabilistic Gamma-Gamma models frequently suffer from numerical non-convergence or unrealistic infinite lifetime estimates on sparse segments with small customer counts. The simplified formula delivers consistent, directionally accurate financial benchmarks for portfolio evaluation.
