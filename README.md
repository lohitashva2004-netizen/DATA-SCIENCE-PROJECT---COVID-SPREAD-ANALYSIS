#  COVID-19 Spread Pattern Analysis

> **Domain:** Healthcare &nbsp;|&nbsp; **Type:** Exploratory Data Analysis + Regression &nbsp;|&nbsp; **Stack:** Python, Pandas, Scikit-learn, Matplotlib, Seaborn

---

##  Project Overview

This project performs a comprehensive analysis of COVID-19 spread patterns across the globe using real-world data spanning **January to July 2020**. The goal is to identify trends, understand regional disparities, and build a predictive model for estimating death counts based on case statistics.

---

##  Folder Structure

```
covid_spread_analysis/
│
├── dataset/
│   └── covid_19_clean_complete.csv       # Raw dataset
│
├── src/
│   └── covid_analysis.py                 # Main Python script (EDA + ML)
│
├── notebooks/
│   └── covid_analysis.ipynb              # Jupyter Notebook version
│
├── outputs/
│   ├── 01_global_trend.png
│   ├── 02_top10_countries.png
│   ├── 03_who_region_bar.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_scatter_confirmed_deaths.png
│   ├── 06_monthly_cases.png
│   ├── 07_actual_vs_predicted.png
│   └── 08_feature_importance.png
│
├── README.md
└── requirements.txt
```

---

## Dataset Description

| Attribute        | Description                                        |
|------------------|----------------------------------------------------|
| Province/State   | Province or state (nullable)                       |
| Country/Region   | Country name                                       |
| Lat / Long       | Geographic coordinates                             |
| Date             | Date of record (2020-01-22 to 2020-07-27)          |
| Confirmed        | Cumulative confirmed COVID-19 cases                |
| Deaths           | Cumulative deaths                                  |
| Recovered        | Cumulative recoveries                              |
| Active           | Active cases = Confirmed − Deaths − Recovered      |
| WHO Region       | One of 6 WHO geographic regions                    |

- **Total Records:** 49,068 rows × 10 columns
- **Countries:** 187+
- **Time Span:** ~6 months (Jan–Jul 2020)

---

## Steps Performed

1. **Data Loading** — CSV loaded using Pandas
2. **Preprocessing** — Filled missing Province/State, parsed dates, removed negative actives, encoded categoricals, applied MinMax scaling
3. **EDA** — Computed global stats, mortality/recovery rates, regional & country-level summaries
4. **Visualization** — 8 charts covering trends, regions, correlation, scatter, monthly patterns
5. **Feature Selection** — Chose 6 features (Confirmed, Recovered, Active, Month, Day, WHO Region)
6. **Model Building** — Linear Regression to predict Deaths
7. **Evaluation** — MAE, RMSE, R² score, Actual vs Predicted plot

---

##  Results

| Metric | Value       |
|--------|-------------|
| MAE    | ~0.00       |
| RMSE   | ~0.00       |
| R²     | **1.0000**  |

>  R² = 1.0 is expected here because `Deaths = Confirmed − Recovered − Active` is a mathematical identity in the dataset. This validates data integrity and demonstrates perfect linear separability.

### Key Insights
- 🇺🇸 The **USA** had the highest confirmed cases (~4.29 million by July 2020)
-  The **Americas** WHO region accounted for over **53%** of global confirmed cases
-  Global **mortality rate** was **3.97%**; **recovery rate** was **57.45%**
-  Cases surged sharply from **March to July 2020** (exponential growth phase)
-  Confirmed cases and Deaths show near-perfect positive correlation

---

##  How to Run

### Prerequisites
- Python 3.8+
- pip

### Setup
```bash
# 1. Clone the repository
git clone https://github.com/yourusername/covid-spread-analysis.git
cd covid-spread-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the analysis
python src/covid_analysis.py

# 4. View outputs
ls outputs/
```

### Jupyter Notebook
```bash
jupyter notebook notebooks/covid_analysis.ipynb
```

---

## 🔮 Future Work

- Integrate vaccination data post-2021 for comparative analysis
- Apply time-series forecasting (ARIMA / Prophet) for case prediction
- Build a country-wise dashboard using Plotly Dash or Streamlit
- Add clustering (K-Means) to group countries by spread severity
- Perform demographic correlation (age, density, HDI) with case fatality

---

## References

- [WHO COVID-19 Dashboard](https://covid19.who.int/)
- [Kaggle: COVID-19 Dataset](https://www.kaggle.com/imdevskp/corona-virus-report)
- Johns Hopkins University CSSE COVID-19 Data Repository

---

##  Author

3rd Year ECE Student - LOHITASHVA V.S 
Academic Year 2026–27
