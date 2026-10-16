# Project Workflow: End-to-End Pipeline Architecture

This document outlines the sequential 15-step execution flow of the **Customer Intelligence Platform**, tracing the path from raw online transactional ingestion to real-time interactive business intelligence.

---

### Step 1: Automated Ingestion & Local Caching
* **What Happens:** The pipeline connects to the UCI Machine Learning Repository via the `ucimlrepo` API, pulls 541,909 raw transaction records, and caches them locally to eliminate redundant network overhead.
* **File:** [`src/data_ingestion.py`](src/data_ingestion.py)

### Step 2: Schema Validation & Column Normalization
* **What Happens:** Verifies mandatory transactional fields and renames raw headers to platform standards (`InvoiceNo` $\to$ `Invoice`, `UnitPrice` $\to$ `Price`, `CustomerID` $\to$ `Customer ID`).
* **File:** [`src/data_ingestion.py`](src/data_ingestion.py)

### Step 3: Cancellation & Return Removal
* **What Happens:** Identifies and filters out 9,288 return transactions prefixed with `'C'` to prevent negative quantities from corrupting customer spend totals and false recent activity dates.
* **File:** [`src/preprocessing.py`](src/preprocessing.py)

### Step 4: Outlier & Zero-Price Anomaly Cleaning
* **What Happens:** Strips 2,517 internal test transactions and administrative write-offs characterized by zero unit prices or non-positive quantities.
* **File:** [`src/preprocessing.py`](src/preprocessing.py)

### Step 5: Anonymous Guest Transaction Pruning
* **What Happens:** Drops 132,220 unidentifiable guest and walk-in purchases to preserve customer longitudinal tracking integrity, yielding 397,884 clean records across 4,338 customers.
* **File:** [`src/preprocessing.py`](src/preprocessing.py)

### Step 6: Customer-Level RFM Metric Aggregation
* **What Happens:** Computes Recency (days since last transaction relative to snapshot date), Frequency (count of distinct invoices), and Monetary value (total spend) for every individual customer.
* **File:** [`src/features.py`](src/features.py)

### Step 7: Behavioral, Temporal, and Basket Feature Engineering
* **What Happens:** Extracts 8 rich behavioral metrics per customer, including average basket size, order value spread, preferred purchase hour, day-of-week habits, and product diversity.
* **File:** [`src/features.py`](src/features.py)

### Step 8: Quantile-Based Classification Label Generation
* **What Happens:** Establishes binary supervision targets based on historical thresholds (`IsHighValue` at 75th percentile spend, `IsFrequent` at median order count, and `IsChurning` at >90 days dormancy for multi-buyers).
* **File:** [`src/features.py`](src/features.py)

### Step 9: Exploratory Data Analysis & Diagnostic Figure Generation
* **What Happens:** Analyzes missing values, revenue trends, RFM distributions, top country markets, order timing heatmaps, and Pareto revenue concentration curves, saving visual artifacts to disk.
* **File:** [`src/eda.py`](src/eda.py)

### Step 10: Multi-Model Stratified Cross-Validation
* **What Happens:** Benchmarks Logistic Regression, Random Forest, and XGBoost classifiers across 5 stratified folds using Precision-Recall AUC to identify the best model for each target while avoiding target leakage.
* **File:** [`src/classification.py`](src/classification.py)

### Step 11: Production Classifier Training & Probability Inference
* **What Happens:** Retrains champion models on full features, persists serialized pipelines to `models/`, and outputs a calibrated 3-dimensional behavioral probability vector for each customer.
* **File:** [`src/classification.py`](src/classification.py)

### Step 12: Cluster Number Optimization & Silhouette Profiling
* **What Happens:** Evaluates cluster solutions across $K \in [2, 7]$ via inertia elbow plots and silhouette coefficients to objectively select the optimal number of customer groups.
* **File:** [`src/clustering.py`](src/clustering.py)

### Step 13: Probability-Space Clustering & Heuristic Naming
* **What Happens:** Fits K-Means (and a GMM for comparison) on standardized probability vectors, keeps the algorithm that passes the silhouette rule, and assigns transparent commercial segment names based on centroid profiles.
* **File:** [`src/clustering.py`](src/clustering.py)

### Step 14: Financial Valuation & Prescriptive Segment Reporting
* **What Happens:** Computes segment revenue share, applies discounted cash flow Customer Lifetime Value (CLV) formulas, attaches marketing action recommendations, and exports a business-ready CSV report.
* **File:** [`src/business_metrics.py`](src/business_metrics.py)

### Step 15: Streaming Data Simulation & Interactive Dashboard Delivery
* **What Happens:** Generates synthetic daily transactional streams to simulate ongoing operations and serves a multi-tab Streamlit web application with automatic 30-second cache refreshing.
* **Files:** [`src/simulate_live.py`](src/simulate_live.py) & [`dashboard/app.py`](dashboard/app.py)
