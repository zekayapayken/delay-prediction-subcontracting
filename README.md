# Delay Prediction Pipeline (Subcontracted Production Orders)

This repository contains a Python pipeline for predicting **nonnegative delivery delay (in days)** for subcontracted production orders using tabular machine learning. The script builds an end-to-end workflow including **domain specific derived features**, **preprocessing**, **nested cross-validation**, **hyperparameter optimisation via RandomizedSearchCV**, **diagnostic plots**, **permutation importance**, and **risk profiling via K-means clustering**.

## 1) What the script does

Given a dataset of subcontracted orders, the script:

1. Constructs the regression target `Delay_Days_Positive` (if not already present) as:
   - `Delay_Days_Raw = Actual_LT - Promised_LT`
   - `Delay_Days_Positive = max(Delay_Days_Raw, 0)`

2. Engineers operationally meaningful features such as:
   - `LT_Tightness` (schedule pressure),
   - `Complexity_Score` (revision/prototype/test/steps),
   - `Workshop_Reliability` (skill + OTD − delay),
   - `Congestion_Index` (workload + historical delay signals),
   - `Reliability_Ratio`,
   - `Material_Criticality_Group` (Low/Medium/High).

3. Trains and evaluates four regressors:
   - **Ridge Regression**
   - **Decision Tree Regressor**
   - **SVR (RBF kernel)**
   - **Gradient Boosting Regressor**

4. Uses a **nested CV** setup:
   - **Outer KFold**: standard random splits (workshops can appear in both train/test).
   - **Outer GroupKFold**: workshop-held-out splits (entire `Workshop_ID` groups are separated).

5. Performs **hyperparameter search** inside each outer fold using **RandomizedSearchCV**:
   - Optimisation metric: `neg_root_mean_squared_error`
   - The best configuration from the inner CV is then evaluated on the outer held-out fold.

6. Produces:
   - Cross-validated metrics (RMSE, MAE, R2, adjusted R2, RMSLE, SMAPE, etc.)
   - Out-of-fold diagnostic plots (pred vs actual, residuals, decile error curve, workshop MAE bar)
   - Permutation importance plots (computed on **raw columns before one-hot encoding**)
   - K-means clustering outputs for risk profiling (including silhouette score selection for K)

## 2) Expected input data

Default input file:
- `synthetic_fason_orders.csv` (set by `DATA_PATH` in the script)

The dataset is expected to include (at minimum) the fields below. If `Delay_Days_Positive` is missing, the script requires:
- `Actual_LT`
- `Promised_LT`

Common columns used by the pipeline:
- `Order_ID` (dropped from modelling if present)
- `Workshop_ID` (used as GroupKFold key)
- `Product_Category`
- `Revision_Index`
- `Prototype_Flag`
- `Test_Required`
- `Expected_Steps`
- `Material_Criticality_Num`
- `Workshop_Skill`
- `Workshop_OTD_Rate`
- `Workshop_Avg_Delay`
- `Workshop_Delay_Var`
- `Workload_Level`
- `Downtime_Risk`
- `Quantity`
- `Urgency_Level`

## 3) How to run

### Option A — Run as a script (recommended)
1. Place `synthetic_fason_orders.csv` in the same folder as the script.
2. Run:
   ```bash
   python delay_prediction_pipeline.py
