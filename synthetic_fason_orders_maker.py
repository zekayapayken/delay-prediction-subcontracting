import numpy as np
import pandas as pd

# Ensure reproducibility
rng = np.random.default_rng(42)

# ==============================
# 1) Parameters
# ==============================
N_WORKSHOPS = 8              # Number of subcontractor workshops
ORDERS_PER_WORKSHOP = 150    # Number of orders per workshop

workshops = [f"W{str(i+1).zfill(2)}" for i in range(N_WORKSHOPS)]

# Product categories
product_categories = [
    "Electronic Board",
    "Power Distribution Module",
    "Communication Module",
    "RF Module",
    "Control Module",
    "LRU",
    "Harness"
]

material_criticalities = ["Low", "Medium", "High"]

# Urgency: 0=Normal, 1=Urgent, 2=Critical
urgency_levels = [0, 1, 2]

# ==============================
# 1.5) TUNING: Noise & Shock Settings
# ==============================
# Base micro noise
BASE_NOISE_STD = 0.10  

# Heteroskedastic noise range: sigma = HET_SIGMA_MIN ... HET_SIGMA_MAX
HET_SIGMA_MIN = 0.10   # Small uncertainty even for low-complexity orders
HET_SIGMA_MAX = 1.10   # Higher uncertainty for complex orders

# Rare shock probability and shock magnitude
SHOCK_PROB = 0.08        # Typically reasonable in the 5%–12% range
SHOCK_MEAN = 2.5         # Average additional delay caused by shocks (days)
SHOCK_STD = 0.9          # Shock variability
SHOCK_CLIP = (0.0, 6.0)  # Clipping range for shock duration (days)


# ==============================
# 2) Workshop-level parameters
# ==============================
workshop_params = []

for w in workshops:
    # Workshop skill level (1–5)
    skill = rng.integers(2, 6)      # 2–5

    # Historical on-time delivery rate
    otd_rate = rng.uniform(0.6, 0.98)

    # Average historical delay (days)
    avg_delay = rng.uniform(0, 7)

    # Delay variability
    delay_var = rng.uniform(0.5, 3.0)

    workshop_params.append({
        "Workshop_ID": w,
        "Workshop_Skill": skill,
        "Workshop_OTD_Rate": otd_rate,
        "Workshop_Avg_Delay": avg_delay,
        "Workshop_Delay_Var": delay_var,
    })

df_workshops = pd.DataFrame(workshop_params)


# ==============================
# 3) Order-level data generation (NONLINEAR)
# ==============================
rows = []
order_id_counter = 1

for _, wrow in df_workshops.iterrows():
    w_id = wrow["Workshop_ID"]
    w_skill = wrow["Workshop_Skill"]
    w_otd = wrow["Workshop_OTD_Rate"]
    w_avg_delay = wrow["Workshop_Avg_Delay"]
    w_delay_var = wrow["Workshop_Delay_Var"]

    # Base workload level representing typical utilisation of the workshop
    base_load = rng.uniform(0.4, 0.9)

    for _ in range(ORDERS_PER_WORKSHOP):
        order_id = f"O{str(order_id_counter).zfill(5)}"
        order_id_counter += 1

        # ---------- PRODUCT-RELATED FEATURES (proxy for complexity) ----------
        product_cat = rng.choice(
            product_categories,
            p=[0.25, 0.1, 0.15, 0.1, 0.2, 0.1, 0.1]
        )

        revision_index = rng.choice([1, 2, 3, 4], p=[0.5, 0.25, 0.15, 0.10])
        prototype_flag = rng.choice([0, 1], p=[0.7, 0.3])
        test_required = rng.choice([0, 1], p=[0.4, 0.6])

        if product_cat in ["Electronic Board", "Control Module"]:
            expected_steps = rng.integers(6, 11)
        elif product_cat in ["Power Distribution Module", "Communication Module", "RF Module"]:
            expected_steps = rng.integers(7, 13)
        elif product_cat == "LRU":
            expected_steps = rng.integers(8, 15)
        else:  # Harness
            expected_steps = rng.integers(3, 8)

        material_criticality = rng.choice(material_criticalities, p=[0.4, 0.4, 0.2])
        if material_criticality == "Low":
            mat_crit_num = 0
        elif material_criticality == "Medium":
            mat_crit_num = 1
        else:
            mat_crit_num = 2

        # ---------- WORKSHOP AND PERIOD-SPECIFIC FACTORS ----------
        workload = np.clip(base_load + rng.normal(0, 0.1), 0, 1)
        downtime_risk = rng.uniform(0, 1)

        # ---------- ORDER-RELATED FEATURES ----------
        quantity = rng.integers(1, 11)
        urgency = rng.choice(urgency_levels, p=[0.6, 0.3, 0.1])

        # Promised lead time (days)
        base_lt = 7
        if product_cat in ["Power Distribution Module", "Communication Module",
                           "RF Module", "Control Module", "LRU"]:
            base_lt += 5
        if prototype_flag == 1:
            base_lt += 7
        base_lt += int(revision_index / 2)

        if urgency == 2:
            base_lt -= 2
        elif urgency == 1:
            base_lt -= 1

        promised_lt = base_lt + rng.integers(-2, 3)
        promised_lt = max(3, promised_lt)

        delay_base = 0.0

        # Linear core effects
        delay_base += 1.2 * prototype_flag
        delay_base += 0.6 * (revision_index - 1)
        delay_base += 0.9 * test_required
        delay_base += 4.0 * max(workload - 0.7, 0)
        delay_base += -1.5 * (w_skill - 3)
        delay_base += -3.0 * (w_otd - 0.8)
        delay_base += 1.5 * mat_crit_num
        delay_base += 1.0 * (urgency == 2)
        delay_base += 0.5 * (urgency == 1)
        delay_base += 1.0 * (downtime_risk > 0.7)
        delay_base += 1.0 * (w_delay_var > 2.0)

        # Interaction terms
        delay_base += 2.0 * prototype_flag * (workload > 0.8)
        delay_base += 1.5 * test_required * (mat_crit_num == 2)
        delay_base += 1.0 * (revision_index >= 3) * (workload > 0.75)
        delay_base += 1.0 * (urgency == 2) * (w_skill <= 3)

        # Nonlinear workload congestion effect
        delay_base += 6.0 * np.clip(workload - 0.85, 0, None) ** 2

        # ---------- Heteroskedastic noise ----------
        # Difficulty score: approximately scaled to [0,1]
        difficulty = (
            0.35 * workload +
            0.20 * prototype_flag +
            0.15 * test_required +
            0.15 * (revision_index >= 3) +
            0.15 * (mat_crit_num == 2)
        )
        difficulty = float(np.clip(difficulty, 0, 1))

        sigma = HET_SIGMA_MIN + (HET_SIGMA_MAX - HET_SIGMA_MIN) * difficulty
        hetero_noise = rng.normal(0, sigma)

        # ---------- Rare shock events ----------
        shock_flag = rng.choice([0, 1], p=[1 - SHOCK_PROB, SHOCK_PROB])
        shock_days = 0.0
        if shock_flag == 1:
            shock_days = rng.normal(SHOCK_MEAN, SHOCK_STD)
            shock_days = float(np.clip(shock_days, SHOCK_CLIP[0], SHOCK_CLIP[1]))

        # ---------- Micro noise ----------
        micro_noise = rng.normal(0, BASE_NOISE_STD)

        delay_cont = delay_base + hetero_noise + shock_days + micro_noise

        # Clip extreme values
        delay_cont = np.clip(delay_cont, -2, 20)

        # Regression target: non-negative delay days
        delay_days_positive = max(delay_cont, 0)

        # Classification target
        late_flag = int(delay_days_positive > 0.5)

        # Realised lead time
        actual_lt = int(round(max(1, promised_lt + delay_days_positive)))

        rows.append({
            "Order_ID": order_id,
            "Workshop_ID": w_id,
            "Product_Category": product_cat,
            "Revision_Index": revision_index,
            "Prototype_Flag": prototype_flag,
            "Test_Required": test_required,
            "Expected_Steps": expected_steps,
            "Material_Criticality": material_criticality,
            "Material_Criticality_Num": mat_crit_num,
            "Workshop_Skill": w_skill,
            "Workshop_OTD_Rate": w_otd,
            "Workshop_Avg_Delay": w_avg_delay,
            "Workshop_Delay_Var": w_delay_var,
            "Workload_Level": workload,
            "Downtime_Risk": downtime_risk,
            "Quantity": quantity,
            "Urgency_Level": urgency,
            "Promised_LT": promised_lt,
            "Actual_LT": actual_lt,
            "LateFlag": late_flag,
            # Optional diagnostics for analysis
            "Difficulty_Score": difficulty,
            "Shock_Flag": shock_flag
        })

# ==============================
# 5) Create DataFrame
# ==============================
df_orders = pd.DataFrame(rows)

print("First 5 rows:")
print(df_orders.head(), "\n")
print("Total number of rows:", len(df_orders))
print("Late rate (LateFlag=1):", round(df_orders["LateFlag"].mean(), 3))

# Compute delay columns
df_orders["Delay_Days_Raw"] = df_orders["Actual_LT"] - df_orders["Promised_LT"]
df_orders["Delay_Days_Positive"] = df_orders["Delay_Days_Raw"].clip(lower=0)

print("\nDelay_Days_Positive statistics:")
print(df_orders["Delay_Days_Positive"].describe())

# ==============================
# 6) Write to CSV
# ==============================
output_file = "synthetic_fason_orders.csv"
df_orders.to_csv(output_file, index=False, encoding="utf-8")
print(f"\nCSV file saved: {output_file}")
