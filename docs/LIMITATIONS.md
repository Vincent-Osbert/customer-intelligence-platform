# Project Limitations: Customer Intelligence Platform

This document outlines the concrete analytical, statistical, and operational boundaries of the Customer Intelligence Platform. Rather than generic disclaimers, each entry specifies what this particular implementation cannot answer, why that limitation matters to commercial stakeholders, and the specific engineering or experimental work required to remediate it.

---

### 1. Complete Absence of Marketing Promotions, Discounts, and Campaign Tracking

* **What It Is:** The raw UCI Online Retail records capture only the net unit price billed per line item (`Price`) and quantity (`Quantity`). There are no records of promotional coupon codes, seasonal sales events, email click-through attributions, or PPC acquisition channels.
* **Why It Matters:** We cannot determine whether a customer's high purchase frequency was organic brand loyalty or an artificial byproduct of heavy discount subsidies. Similarly, price sensitivity cannot be isolated: if a customer stopped buying, we cannot verify whether they left due to product fatigue, price increases, or simply because promotional email sequences stopped.
* **What Would Fix It:** Integrate a marketing attribution table linking each transaction invoice to specific promotional campaign IDs, coupon codes, and affiliate referral tags. This would allow modeling promo-dependency as an additional behavioral feature.

---

### 2. Inherent Unreliability in Classifying One-Time Buyers

* **What It Is:** In our cleaned dataset of 4,338 customers, approximately 35% of accounts have placed exactly one transaction (`Frequency = 1`). For these customers, temporal metrics such as `PurchaseDaySpread` and `AvgDaysBetweenOrders` collapse to zero or undefined baselines.
* **Why It Matters:** Supervised churn classifiers rely heavily on interval cadence to predict churn. With only a single observation point, the model cannot distinguish between:
  1. A brand-new customer who just discovered the store yesterday and intends to become a loyal regular.
  2. A permanently departed customer who bought a single novelty gift 11 months ago and will never return.
  This causes one-time buyers to cluster predominantly into broad "Occasional Buyers" or uninformative neutral categories.
* **What Would Fix It:** Incorporate real-time clickstream data (website session duration, product pages viewed, cart additions, search queries) and email open events. Pre-purchase web engagement allows intent scoring even before a customer places their second order.

---

### 3. Subjective Heuristic Mapping for Cluster Naming

* **What It Is:** While K-Means identifies mathematical cluster boundaries in three-dimensional probability space, assigning commercial identities (e.g., "VIP Champions", "Moderate (Value)", "Occasional Buyers", "Dormant / Lost") is governed by a rule-based centroid mapping function in [`src/clustering.py`](src/clustering.py).
* **Why It Matters:** If the model discovers an unexpected 6th cluster with an unusual behavioral combination (for instance, customers with high churn probability and high purchase frequency, as seen in `Segment 4`), rule-based heuristics cannot assign an intuitive business name without human intervention. Centroid interpretation remains inherently subjective.
* **What Would Fix It:** Conduct cross-functional Delphi panels with sales, customer support, and marketing teams to establish standardized corporate segment taxonomies. Additionally, deploy automated LLM-assisted cluster profiling that analyzes full multi-attribute feature vectors to generate semantic descriptions with confidence intervals.

---

### 4. Simplified Deterministic Customer Lifetime Value (CLV) Modeling

* **What It Is:** The platform estimates CLV using an annualized discounted cash flow formula based on aggregate monthly revenue, fixed 35% gross margin, 5% monthly churn, and a 10% discount rate.
* **Why It Matters:** The calculation assumes constant purchasing rates and uniform churn risk across time. In reality, customer churn in non-contractual e-commerce is probabilistic and non-linear: purchase velocity follows a Poisson process, while customer drop-out follows a Pareto or Beta-Geometric distribution. Consequently, our CLV figures represent directional benchmarks rather than calibrated financial forecasts.
* **What Would Fix It:** Implement full probabilistic modeling using the Beta-Geometric / Negative Binomial Distribution (BG/NBD) model for transaction frequency paired with a Gamma-Gamma submodel for monetary spend (e.g., using Python's `lifetimes` or `pymc-marketing` packages), backtesting predictions against out-of-time holdout cohorts.

---

### 5. Absence of Controlled A/B Testing for Action Recommendations

* **What It Is:** The prescriptive recommendations assigned to each segment (such as offering 20% win-back incentives to "High Value at Risk" or suppressing "Dormant / Lost" accounts) are derived from industry best practices, not from randomized controlled field experiments.
* **Why It Matters:** We cannot prove causal lift. Offering a 20% discount to high-value customers might induce margin cannibalization if those customers intended to repurchase anyway. Conversely, ceasing communications to dormant users might forfeit organic reactivation opportunities.
* **What Would Fix It:** Establish an automated experimentation framework. Split each target segment into an 80% treatment group receiving the prescriptive intervention and a 20% holdout control group receiving baseline communications. Measure incremental gross margin and 90-day retention lift before scaling recommendations company-wide.

---

### 6. Temporal Obsolescence of Historical 2010–2011 Training Data

* **What It Is:** The underlying transactional records were collected between December 2010 and December 2011 from a UK retailer.
* **Why It Matters:** Consumer habits, e-commerce adoption, macroeconomic inflation, and mobile shopping behavior have fundamentally changed over the past 15 years. Nominal price levels (where £500 constituted a major wholesale order in 2011) do not reflect current purchasing power. Furthermore, payment preferences, shipping expectations (same-day delivery), and digital advertising dynamics are completely different today.
* **What Would Fix It:** Connect the data ingestion pipeline to a modern cloud data warehouse (Snowflake, BigQuery, or Amazon Redshift) storing contemporary Shopify, Stripe, or ERP transaction streams. Implement scheduled automated model retraining with data drift monitoring using tools like Evidently AI or Great Expectations.

### Labels Are Rules Over the Same Features

* **What It Is:** `IsHighValue`, `IsFrequent` and `IsChurning` are defined by quantile and recency rules on the same customer features the classifiers receive (for example, churn = Recency over 90 days among repeat buyers). The near-perfect PR-AUC of the first two classifiers (0.999 and 1.000) therefore shows the models recover a rule, not that they predict future behavior.
* **Why It Matters:** Only the churn model's PR-AUC (0.423) and the 3.05x lift over the 0.139 base rate should be read as model performance, and even those measure agreement with a recency rule rather than observed future churn. The segment silhouette inherits the same effect.
* **What Would Fix It:** Split the data by time, build features from the earlier window, and label churn from purchases in the later window.
