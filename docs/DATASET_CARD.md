# Dataset Card: Online Retail (UCI ML Repository id=352)

## 1. Overview & Provenance

* **Dataset Name:** Online Retail
* **Repository:** UCI Machine Learning Repository (Dataset ID: 352)
* **Source URL:** [https://archive.ics.uci.edu/dataset/352/online+retail](https://archive.ics.uci.edu/dataset/352/online+retail)
* **Ingestion Method:** Automated via `ucimlrepo` library (`fetch_ucirepo(id=352)`) with automated offline caching.
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
* **Temporal Coverage:** 01/12/2010 to 09/12/2011 (December 1, 2010 through December 9, 2011)
* **Geographical Coverage:** United Kingdom (primary, >85% of volume), Germany, France, Netherlands, and 34 other international destinations.
* **Domain:** E-commerce / Retail / Wholesale Novelty & Giftware
* **Raw File Artifact:** `data/raw/online_retail.csv`
* **File SHA-256 Checksum:** `9F91016BA528F11E17199D8E1A048D8AFB87A4D987B2B31F59A8DC028E2C1E3C`

---

## 2. Dataset Description

The dataset contains actual, transactional records for a registered non-store online retailer based in the United Kingdom. The company mainly sells unique all-occasion giftware. A substantial portion of the customer base consists of small businesses, wholesalers, and independent gift shops buying in bulk, alongside individual retail consumers.

During the recorded period (2010–2011), the retailer recorded **541,909 individual order line items** across 4,372 unique customer accounts.

---

## 3. Schema & Column Descriptions

The raw dataset adheres to the schema below. During pipeline ingestion in [`src/data_ingestion.py`](src/data_ingestion.py), UCI column headers are normalized to internal standard names:

| Raw Column | Normalized Column | Data Type | Null Count (%) | Description | Example |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `InvoiceNo` | `Invoice` | String / Object | 0 (0.0%) | A 6-digit integral number uniquely assigned to each transaction. If the code starts with the letter `'C'`, it indicates a cancellation or return. | `536365`, `C536379` |
| `StockCode` | `StockCode` | String / Object | 0 (0.0%) | A 5- or 6-digit alphanumeric code uniquely assigned to each distinct product in inventory. | `85123A`, `71053` |
| `Description` | `Description` | String / Object | 1,454 (0.27%) | Nominal product name/title. Missing descriptions correspond almost exclusively to unassigned or damaged stock records. | `WHITE HANGING HEART T-LIGHT HOLDER` |
| `Quantity` | `Quantity` | Integer | 0 (0.0%) | The number of units of the item purchased per transaction. Negative values indicate returns/cancellations or inventory write-offs. | `6`, `-12` |
| `InvoiceDate` | `InvoiceDate` | String / Datetime | 0 (0.0%) | Day and time when the transaction was generated (format: `MM/DD/YYYY HH:MM`). | `12/1/2010 8:26` |
| `UnitPrice` | `Price` | Float | 0 (0.0%) | Unit price of the product in Sterling (£). | `2.55`, `18.00` |
| `CustomerID` | `Customer ID` | Float / Integer | 135,080 (24.9%) | A 5-digit integral number uniquely assigned to each registered customer account. | `17850`, `13047` |
| `Country` | `Country` | String / Categorical | 0 (0.0%) | Name of the country where the customer resides or where the shipment was billed. | `United Kingdom`, `Germany` |

---

## 4. Data Quality Checks & Cleaning Transformations

The cleaning pipeline in [`src/preprocessing.py`](src/preprocessing.py) performs rigorous quality gates in sequential order:

```
[Raw Data: 541,909 rows]
       │
       ▼  Step 1: Remove cancellations (Invoice starts with 'C')
[-9,288 rows removed]
       │
       ▼  Step 2: Remove invalid quantities/prices (Quantity <= 0 or Price <= 0)
[-2,517 rows removed]
       │
       ▼  Step 3: Drop records lacking Customer ID (cannot be linked to a customer)
[-132,220 rows dropped]
       │
       ▼  Step 4: Compute transaction Revenue (Quantity × Price)
[Clean Data: 397,884 rows, 4,338 unique customers]
```

### Audit Breakdown:
1. **Returns & Cancellations (`remove_returns`):** 
   * **Count Removed:** 9,288 rows.
   * **Rationale:** Invoices prefixed with `'C'` reflect returned merchandise. Including returns with negative quantities inflates recency dates while distorting total frequency and monetary spend. Returns represent a distinct analytical workflow (return propensity modeling) and are excluded from customer lifetime segmentation.
2. **Invalid Quantities and Zero Prices (`remove_invalid_quantities`):** 
   * **Count Removed:** 2,517 rows.
   * **Rationale:** Rows with `Price <= 0` or negative quantities without a cancellation code represent inventory adjustments, damaged goods write-offs, or administrative system testing by warehouse staff. These create false revenue deflations.
3. **Missing Customer Identifiers (`drop_missing_customers`):** 
   * **Count Dropped:** 132,220 rows (24.9% of raw records).
   * **Rationale:** Guest checkouts and walk-in purchases cannot be associated with any historical profile. Imputing random customer IDs would fabricate artificial customer behavior, while assigning all missing rows to a single dummy customer would distort clustering centroids.
4. **Calculated Columns:** 
   * `Revenue` is derived as $\text{Quantity} \times \text{Price}$ for every validated line item.

---

## 5. Dataset Limitations & Analytical Boundaries

While the UCI Online Retail dataset is widely recognized as a benchmark for e-commerce modeling, practitioners must account for several structural limitations:

1. **No Inventory or Out-of-Stock Data:** The dataset records completed purchases but provides no data on out-of-stock items, browse events, or abandoned shopping carts. We cannot determine whether a customer failed to buy because of low demand or lack of inventory.
2. **Absence of Marketing Promotions & Discounts:** Transaction records list the unit price charged, but do not capture whether an order was driven by an email campaign, a seasonal coupon code, or an affiliate discount.
3. **Zero Demographic Information:** The data contains no customer age, gender, household income, or business structure (individual hobbyist vs. registered wholesaler). Segmentation is purely behavioral.
4. **High Domestic Concentration:** Over 85% of transactions originate from the United Kingdom, which may cause regional purchasing patterns to drown out international consumer trends.
5. **Right-Censored Window:** The dataset ends abruptly on December 9, 2011, midway through the peak holiday shopping season, causing slight right-censorship for holiday recency tracking.
