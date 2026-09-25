# Marketing A/B Testing: Ad vs. PSA Conversion Analysis

## Business Question
Does showing users a paid advertisement actually drive more conversions than showing them a Public Service Announcement (PSA) in the same slot — and is the difference large enough to justify continued ad spend?

## Dataset
- **Source:** [Marketing A/B Testing dataset — Kaggle](https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing)
- **Size:** 588,101 users
- **Columns:** `user id`, `test group` (ad / psa), `converted` (True/False), `total ads` seen, `most ads day`, `most ads hour`

## Methodology
1. **EDA** — duplicate checks, missing values, IQR-based outlier detection, skewness/kurtosis on continuous variables, confidence intervals.
2. **Chi-square tests of independence** — checked whether conversion is associated with day of week and time slot ads were seen.
3. **Independent samples t-test** — compared `total ads` seen between converted vs. non-converted users, with Levene's test to check the equal-variance assumption first (used Welch's t-test since variances were unequal), plus Cohen's d for effect size.
4. **A/B test (core analysis)**:
   - Stated H₀ (no difference in conversion rate between ad and PSA groups) and H₁ (a difference exists).
   - Set α = 0.05, power = 0.8.
   - Calculated required sample size per group using power analysis.
   - Drew a random subsample at that calculated size (rather than using the full 588K rows, to reflect a realistic experiment size).
   - Ran a two-proportion z-test on the subsample.
   - Computed the odds ratio with a 95% confidence interval.

## Key Results
- Conversion rate: **ad group 2.66%** vs. **PSA group 1.90%** (on the drawn subsample).
- Two-proportion z-test: **p = 0.0033** — statistically significant at α = 0.05.
- Odds ratio: **[fill in after verifying direction]**, 95% CI **[x, y]** — does not cross 1, confirming the effect is unlikely due to chance.
- On the full dataset, conversion is also associated with time slot and day of week (chi-square p < 0.05), though these effects are small in absolute size given the large sample.

## Recommendation
Ads show a statistically significant lift in conversion rate over PSAs. Given the sample size involved, this result is unlikely to be due to chance — but the absolute lift (~0.76 percentage points) should be weighed against the cost per ad impression before scaling spend further. A follow-up analysis incorporating ad cost and revenue per conversion would turn this statistical result into a clear ROI decision.

## Tools Used
Python, pandas, NumPy, SciPy, statsmodels, Matplotlib, Seaborn

## Live Demo
[Add your Streamlit link here once deployed]

## Notebook
See [`marketing_ab_test_analysis.ipynb`](./marketing_ab_test_analysis.ipynb) for the full analysis with code, statistical outputs, and visualizations.
