import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import chi2_contingency, ttest_ind, levene, norm
from statsmodels.stats.power import NormalIndPower
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Marketing A/B Test: Ad vs PSA", layout="wide")

st.title("Marketing A/B Testing: Ad vs. PSA Conversion Analysis")
st.write(
    "Does showing users a paid ad drive more conversions than a Public Service "
    "Announcement (PSA), and is the difference statistically significant?"
)

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("marketing_AB.csv")
    return df

df = load_data()

st.header("1. Dataset Overview")
st.write(f"Total users: **{len(df):,}**")
st.dataframe(df.head())

# ---------------------------------------------------------
# Conversion rates by test group
# ---------------------------------------------------------
st.header("2. Conversion Rate by Group")

conv_by_group = df.groupby("test group")["converted"].mean()
st.write(conv_by_group)

st.bar_chart(conv_by_group)

# ---------------------------------------------------------
# Chi-square: converted vs test group (full data)
# ---------------------------------------------------------
st.header("3. Chi-Square Test: Conversion vs. Test Group")

contingency_table_CT = pd.crosstab(df["converted"], df["test group"])
st.write("Contingency table (frequencies):")
st.dataframe(contingency_table_CT)

chi2, p_chi, dof, expected = chi2_contingency(contingency_table_CT)
st.write(f"Chi-square statistic: **{chi2:.2f}**")
st.write(f"p-value: **{p_chi:.6f}**")
if p_chi < 0.05:
    st.success("Statistically significant association between group and conversion (p < 0.05).")
else:
    st.info("No statistically significant association detected (p >= 0.05).")

# ---------------------------------------------------------
# Independent t-test: total ads seen, converted vs not
# ---------------------------------------------------------
st.header("4. Total Ads Seen: Converted vs. Non-Converted")

groupN = df[df["converted"] == False]["total ads"]
groupY = df[df["converted"] == True]["total ads"]

lev_stat, lev_p = levene(groupN, groupY)
st.write(f"Levene's test p-value: **{lev_p:.6f}**")
equal_var = lev_p >= 0.05
test_name = "standard" if equal_var else "Welch's"
assumption_text = "holds" if equal_var else "does NOT hold"
st.write(f"Equal variance assumption {assumption_text} — using {test_name} t-test.")

t_stat, p_ttest = ttest_ind(groupN, groupY, equal_var=equal_var)
st.write(f"t-statistic: **{t_stat:.2f}**, p-value: **{p_ttest:.6f}**")

pooled_std = np.sqrt(
    ((len(groupN) - 1) * np.std(groupN, ddof=1) ** 2
     + (len(groupY) - 1) * np.std(groupY, ddof=1) ** 2)
    / (len(groupN) + len(groupY) - 2)
)
cohen_d = (np.mean(groupN) - np.mean(groupY)) / pooled_std
st.write(f"Cohen's d (effect size): **{cohen_d:.2f}**")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(groupN, bins=30, alpha=0.6, label="Non-converted")
ax.hist(groupY, bins=30, alpha=0.6, label="Converted")
ax.set_xlabel("Total ads seen")
ax.set_ylabel("Count")
ax.legend()
st.pyplot(fig)

# ---------------------------------------------------------
# Power analysis / sample size
# ---------------------------------------------------------
st.header("5. Sample Size Calculation (Power Analysis)")

p1 = conv_by_group.get("psa", np.nan)
p2 = conv_by_group.get("ad", np.nan)

alpha = 0.05
power = 0.8
p_avg = (p1 + p2) / 2
effect_size = (p2 - p1) / np.sqrt(p_avg * (1 - p_avg))

analysis = NormalIndPower()
sample_size = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha, ratio=1)
sample_size = int(np.ceil(sample_size))

st.write(f"PSA conversion rate: **{p1:.4%}**")
st.write(f"Ad conversion rate: **{p2:.4%}**")
st.write(f"Required sample size per group (α=0.05, power=0.8): **{sample_size:,}**")

# ---------------------------------------------------------
# Subsample and run z-test
# ---------------------------------------------------------
st.header("6. Two-Proportion Z-Test on Calculated Sample Size")

def choose_random_sample(data, n, random_state=42):
    return data.sample(n=min(n, len(data)), random_state=random_state)

df_smpl = df.groupby("test group", group_keys=False)[df.columns].apply(
    lambda g: choose_random_sample(g, sample_size)
)

contingency_table_Z = pd.crosstab(df_smpl["test group"], df_smpl["converted"])
st.write("Subsample contingency table:")
st.dataframe(contingency_table_Z)

# counts: successes = converted True, n = group totals
success_ad = contingency_table_Z.loc["ad", True] if True in contingency_table_Z.columns else 0
success_psa = contingency_table_Z.loc["psa", True] if True in contingency_table_Z.columns else 0
n_ad = contingency_table_Z.loc["ad"].sum()
n_psa = contingency_table_Z.loc["psa"].sum()

p_ad = success_ad / n_ad
p_psa = success_psa / n_psa
p_pool = (success_ad + success_psa) / (n_ad + n_psa)
se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_ad + 1 / n_psa))
z_stat = (p_ad - p_psa) / se
p_value_z = 2 * (1 - norm.cdf(abs(z_stat)))

st.write(f"Ad group conversion rate (subsample): **{p_ad:.4%}**")
st.write(f"PSA group conversion rate (subsample): **{p_psa:.4%}**")
st.write(f"Z-statistic: **{z_stat:.4f}**")
st.write(f"p-value: **{p_value_z:.6f}**")

if p_value_z < 0.05:
    st.success("Statistically significant difference — reject H0. Conversion rates differ between groups.")
else:
    st.info("No statistically significant difference detected — fail to reject H0.")

fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.countplot(data=df_smpl, x="test group", hue="converted", ax=ax2)
ax2.set_title("Conversions in Subsample by Group")
st.pyplot(fig2)

# ---------------------------------------------------------
# Odds ratio with CI
# ---------------------------------------------------------
st.header("7. Odds Ratio with 95% Confidence Interval")

def odds_ratio_ci(table):
    # table: rows = groups (ad, psa), columns = [False, True]
    a = table.loc["ad", True]
    b = table.loc["ad", False]
    c = table.loc["psa", True]
    d = table.loc["psa", False]

    odds_ratio = (a / b) / (c / d)
    log_or = np.log(odds_ratio)
    se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)

    ci_low = np.exp(log_or - 1.96 * se_log_or)
    ci_high = np.exp(log_or + 1.96 * se_log_or)
    return odds_ratio, ci_low, ci_high

odds_ratio, ci_low, ci_high = odds_ratio_ci(contingency_table_Z)

st.write(f"Odds ratio (ad vs. psa): **{odds_ratio:.2f}**")
st.write(f"95% Confidence Interval: **[{ci_low:.2f}, {ci_high:.2f}]**")

if ci_low > 1 or ci_high < 1:
    st.success("The CI does not cross 1 — the odds ratio is statistically significant.")
else:
    st.info("The CI crosses 1 — the odds ratio is not statistically significant.")

fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.errorbar(1, odds_ratio,
             yerr=[[odds_ratio - ci_low], [ci_high - odds_ratio]],
             fmt="o", capsize=5, color="blue")
ax3.axhline(1, linestyle="--", color="red", label="No Effect (OR=1)")
ax3.set_title("Odds Ratio with 95% Confidence Interval")
ax3.set_ylabel("Odds Ratio")
ax3.set_xlim(0.5, 1.5)
ax3.set_xticks([])
ax3.legend()
st.pyplot(fig3)

# ---------------------------------------------------------
# Recommendation
# ---------------------------------------------------------
st.header("8. Recommendation")
st.write(
    "Based on the two-proportion z-test and odds ratio above, the difference in "
    "conversion rates between the ad and PSA groups is evaluated for statistical "
    "significance. Any business decision to scale ad spend further should weigh this "
    "result against the cost per impression and cost per conversion, which are not "
    "captured in this dataset."
)
