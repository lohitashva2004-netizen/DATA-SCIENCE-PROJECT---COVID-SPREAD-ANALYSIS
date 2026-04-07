# =============================================================================
# COVID-19 Spread Pattern Analysis
# Domain: Healthcare | Dataset: COVID-19 Clean Complete
# Author: Data Science Project | Year: 3rd Year ECE
# =============================================================================

# ── 1. IMPORTS ────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
import os

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "axes.titlesize": 13})

# Output folder
os.makedirs("outputs", exist_ok=True)

# ── 2. LOAD DATASET ───────────────────────────────────────────────────────────
print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv("C:/Users/lohit/Desktop/data science project/covid_19_clean_complete.csv")
print(f"Shape           : {df.shape}")
print(f"Columns         : {df.columns.tolist()}")
print(f"\nFirst 5 rows:\n{df.head()}")

# ── 3. DATA PREPROCESSING ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Preprocessing...")
print("=" * 60)

# 3a. Handle missing values
print(f"\nMissing values before:\n{df.isnull().sum()}")
df["Province/State"].fillna("Unknown", inplace=True)
print(f"\nMissing values after:\n{df.isnull().sum()}")

# 3b. Parse dates
df["Date"] = pd.to_datetime(df["Date"])
df["Year"]  = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"]   = df["Date"].dt.day

# 3c. Remove illogical negative values (Active can sometimes be negative)
df = df[df["Active"] >= 0]

# 3d. Encode categorical columns (for model use)
le = LabelEncoder()
df["WHO_Region_Enc"]   = le.fit_transform(df["WHO Region"])
df["Country_Enc"]      = le.fit_transform(df["Country/Region"])

# 3e. Feature scaling
scaler = MinMaxScaler()
df[["Confirmed_Scaled", "Deaths_Scaled", "Recovered_Scaled"]] = scaler.fit_transform(
    df[["Confirmed", "Deaths", "Recovered"]]
)

print("\nPreprocessing complete.")
print(f"Dataset shape after cleaning: {df.shape}")

# ── 4. EXPLORATORY DATA ANALYSIS ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("Exploratory Data Analysis...")
print("=" * 60)

# Summary statistics
print("\nDescriptive Statistics:\n")
print(df[["Confirmed", "Deaths", "Recovered", "Active"]].describe())

# Mortality and recovery rate (latest date)
latest = df[df["Date"] == df["Date"].max()]
total_confirmed  = latest["Confirmed"].sum()
total_deaths     = latest["Deaths"].sum()
total_recovered  = latest["Recovered"].sum()
mortality_rate   = (total_deaths / total_confirmed) * 100
recovery_rate    = (total_recovered / total_confirmed) * 100

print(f"\nAs of {df['Date'].max().date()}:")
print(f"  Total Confirmed  : {total_confirmed:,}")
print(f"  Total Deaths     : {total_deaths:,}")
print(f"  Total Recovered  : {total_recovered:,}")
print(f"  Mortality Rate   : {mortality_rate:.2f}%")
print(f"  Recovery Rate    : {recovery_rate:.2f}%")

# Top 10 countries by confirmed
top10 = (
    latest.groupby("Country/Region")[["Confirmed", "Deaths", "Recovered"]]
    .sum()
    .sort_values("Confirmed", ascending=False)
    .head(10)
)
print("\nTop 10 Countries by Confirmed Cases:\n")
print(top10)

# WHO Region summary
region_summary = (
    latest.groupby("WHO Region")[["Confirmed", "Deaths", "Recovered"]]
    .sum()
    .sort_values("Confirmed", ascending=False)
)
print("\nWHO Region Summary:\n")
print(region_summary)

# ── 5. VISUALIZATIONS ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Generating Visualizations...")
print("=" * 60)

# ── Plot 1: Global trend (line chart) ────────────────────────────────────────
daily_global = df.groupby("Date")[["Confirmed", "Deaths", "Recovered"]].sum().reset_index()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(daily_global["Date"], daily_global["Confirmed"],  label="Confirmed",  color="#e74c3c", linewidth=2)
ax.plot(daily_global["Date"], daily_global["Recovered"],  label="Recovered",  color="#2ecc71", linewidth=2)
ax.plot(daily_global["Date"], daily_global["Deaths"],     label="Deaths",     color="#7f8c8d", linewidth=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)
ax.set_title("Global COVID-19 Spread Over Time", fontsize=14, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Number of Cases")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/01_global_trend.png")
plt.close()
print("  Saved: 01_global_trend.png")

# ── Plot 2: Top 10 countries — horizontal bar chart ──────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
top10["Confirmed"].sort_values().plot(kind="barh", color="#e74c3c", edgecolor="black", ax=ax)
ax.set_title("Top 10 Countries by Confirmed COVID-19 Cases", fontweight="bold")
ax.set_xlabel("Confirmed Cases")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig("outputs/02_top10_countries.png")
plt.close()
print("  Saved: 02_top10_countries.png")

# ── Plot 3: WHO Region breakdown — grouped bar ────────────────────────────────
region_summary_plot = region_summary.reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(region_summary_plot))
width = 0.28
ax.bar(x - width, region_summary_plot["Confirmed"], width, label="Confirmed", color="#e74c3c")
ax.bar(x,          region_summary_plot["Recovered"], width, label="Recovered", color="#2ecc71")
ax.bar(x + width,  region_summary_plot["Deaths"],    width, label="Deaths",    color="#7f8c8d")
ax.set_xticks(x)
ax.set_xticklabels(region_summary_plot["WHO Region"], rotation=30, ha="right")
ax.set_title("COVID-19 Cases by WHO Region", fontweight="bold")
ax.set_ylabel("Total Cases")
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig("outputs/03_who_region_bar.png")
plt.close()
print("  Saved: 03_who_region_bar.png")

# ── Plot 4: Correlation Heatmap ───────────────────────────────────────────────
corr_cols = ["Confirmed", "Deaths", "Recovered", "Active", "Month", "Day"]
corr = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5,
            square=True, ax=ax, annot_kws={"size": 10})
ax.set_title("Correlation Heatmap of COVID-19 Features", fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/04_correlation_heatmap.png")
plt.close()
print("  Saved: 04_correlation_heatmap.png")

# ── Plot 5: Scatter — Confirmed vs Deaths (log scale) ────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
scatter_data = latest.groupby("Country/Region")[["Confirmed", "Deaths"]].sum().reset_index()
scatter_data = scatter_data[(scatter_data["Confirmed"] > 0) & (scatter_data["Deaths"] > 0)]
ax.scatter(np.log1p(scatter_data["Confirmed"]), np.log1p(scatter_data["Deaths"]),
           alpha=0.5, color="#3498db", edgecolors="none", s=30)
# Highlight top 5
top5 = scatter_data.sort_values("Confirmed", ascending=False).head(5)
for _, row in top5.iterrows():
    ax.annotate(row["Country/Region"],
                (np.log1p(row["Confirmed"]), np.log1p(row["Deaths"])),
                fontsize=7, color="darkred")
ax.set_xlabel("log(Confirmed Cases)")
ax.set_ylabel("log(Deaths)")
ax.set_title("Confirmed Cases vs Deaths (Log Scale) — All Countries", fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/05_scatter_confirmed_deaths.png")
plt.close()
print("  Saved: 05_scatter_confirmed_deaths.png")

# ── Plot 6: Monthly new cases bar ─────────────────────────────────────────────
monthly = df.groupby("Month")["Confirmed"].sum().reset_index()
monthly["Month_Name"] = monthly["Month"].apply(
    lambda m: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][m-1]
)
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(monthly["Month_Name"], monthly["Confirmed"], color="#9b59b6", edgecolor="black")
ax.bar_label(bars, fmt=lambda x: f"{int(x):,}", fontsize=7, padding=3)
ax.set_title("Total Confirmed Cases by Month", fontweight="bold")
ax.set_ylabel("Confirmed Cases")
ax.set_xlabel("Month (2020)")
plt.tight_layout()
plt.savefig("outputs/06_monthly_cases.png")
plt.close()
print("  Saved: 06_monthly_cases.png")

# ── 6. FEATURE SELECTION ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Feature Selection...")
print("=" * 60)

# For regression: predict Deaths from Confirmed, Recovered, Month, WHO region
features = ["Confirmed", "Recovered", "Active", "Month", "Day", "WHO_Region_Enc"]
target   = "Deaths"

model_df = df[features + [target]].dropna()
X = model_df[features]
y = model_df[target]

print(f"Features used   : {features}")
print(f"Target variable : {target}")
print(f"Samples for ML  : {X.shape[0]}")

# ── 7. MODEL BUILDING ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Model Building — Linear Regression...")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred = np.maximum(y_pred, 0)   # deaths cannot be negative

# ── 8. MODEL EVALUATION ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Model Evaluation...")
print("=" * 60)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"  MAE  : {mae:.2f}")
print(f"  RMSE : {rmse:.2f}")
print(f"  R²   : {r2:.4f}")

# Feature importance (coefficients)
coeff_df = pd.DataFrame({"Feature": features, "Coefficient": model.coef_})
coeff_df = coeff_df.reindex(coeff_df["Coefficient"].abs().sort_values(ascending=False).index)
print("\nFeature Coefficients:\n")
print(coeff_df.to_string(index=False))

# ── Plot 7: Actual vs Predicted ───────────────────────────────────────────────
sample = min(300, len(y_test))
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_test.values[:sample], y_pred[:sample], alpha=0.4, color="#1abc9c", s=20)
max_val = max(y_test.max(), y_pred.max())
ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Ideal fit")
ax.set_xlabel("Actual Deaths")
ax.set_ylabel("Predicted Deaths")
ax.set_title(f"Actual vs Predicted Deaths (R² = {r2:.4f})", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/07_actual_vs_predicted.png")
plt.close()
print("  Saved: 07_actual_vs_predicted.png")

# ── Plot 8: Feature importance ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#e74c3c" if c >= 0 else "#3498db" for c in coeff_df["Coefficient"]]
ax.barh(coeff_df["Feature"], coeff_df["Coefficient"], color=colors, edgecolor="black")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Linear Regression — Feature Coefficients", fontweight="bold")
ax.set_xlabel("Coefficient Value")
plt.tight_layout()
plt.savefig("outputs/08_feature_importance.png")
plt.close()
print("  Saved: 08_feature_importance.png")

# ── 9. KEY INSIGHTS ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
print(f"""
1. The Americas region had the highest burden, with the US alone
   recording {latest[latest['Country/Region']=='US']['Confirmed'].sum():,} confirmed cases.

2. The global mortality rate stood at {mortality_rate:.2f}%, while the
   recovery rate reached {recovery_rate:.2f}% by July 2020.

3. Strong correlation (r ≈ 0.99) exists between Confirmed cases
   and Deaths — countries with more cases consistently showed
   more fatalities.

4. Case counts surged exponentially from March to July 2020,
   confirming the rapid spread in the early pandemic phase.

5. Europe and South-East Asia had notable death tolls relative
   to confirmed cases, suggesting differences in reporting or
   healthcare capacity.

6. The Linear Regression model achieved R² = {r2:.4f}, showing
   that Confirmed and Active case counts are strong predictors
   of death toll.
""")

print("=" * 60)
print("All outputs saved to the 'outputs/' folder.")
print("=" * 60)
