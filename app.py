import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])

st.set_page_config(page_title="Benford's Law Analysis Dashboard", layout="wide")
st.title("📊 Benford's Law & Statistical Analysis Dashboard")
st.markdown(
    "Upload a dataset, apply filters, and test numeric columns against Benford's Law using NumPy KS statistics."
)


st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV or Excel file", type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.success(
        f"Loaded **{uploaded_file.name}** ({df.shape[0]:,} rows, {df.shape[1]} columns)"
    )

    st.sidebar.header("2. Filter Data (Min. 3 Criteria)")
    filtered_df = df.copy()

    exclude_non_positive = st.sidebar.checkbox(
        "Criterion 1: Exclude Zeros & Negatives (Recommended)", value=True
    )

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        selected_cat_col = st.sidebar.selectbox(
            "Criterion 2: Select Categorical Filter Column",
            ["None"] + cat_cols,
        )
        if selected_cat_col != "None":
            unique_vals = filtered_df[selected_cat_col].dropna().unique().tolist()
            selected_vals = st.sidebar.multiselect(
                f"Filter values in '{selected_cat_col}'",
                options=unique_vals,
                default=unique_vals[:5] if len(unique_vals) > 5 else unique_vals,
            )
            if selected_vals:
                filtered_df = filtered_df[
                    filtered_df[selected_cat_col].isin(selected_vals)
                ]
    else:
        st.sidebar.info("Criterion 2: No categorical columns available.")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.error("No numeric columns found in the uploaded dataset.")
        st.stop()

    filter_num_col = st.sidebar.selectbox(
        "Criterion 3: Select Numeric Range Filter Column", num_cols
    )
    min_val = float(df[filter_num_col].min())
    max_val = float(df[filter_num_col].max())
    if min_val < max_val:
        selected_range = st.sidebar.slider(
            f"Range for '{filter_num_col}'",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val),
        )
        filtered_df = filtered_df[
            (filtered_df[filter_num_col] >= selected_range[0])
            & (filtered_df[filter_num_col] <= selected_range[1])
        ]

    max_rows = len(filtered_df)
    if max_rows > 10:
        row_limit = st.sidebar.number_input(
            "Criterion 4: Max Rows to Analyze",
            min_value=10,
            max_value=max_rows,
            value=max_rows,
        )
        filtered_df = filtered_df.head(int(row_limit))

    st.sidebar.header("3. Benford Target Column")
    target_col = st.sidebar.selectbox(
        "Select Column of Interest for Benford Analysis", num_cols
    )

    series = filtered_df[target_col].dropna()
    if exclude_non_positive:
        series = series[series > 0]
    else:
        series = series[series != 0].abs()

    if len(series) == 0:
        st.warning(
            "No valid positive numbers remain after filtering. Please adjust your filters."
        )
        st.stop()

    def get_leading_digits(val, n_digits=1):
        s = f"{abs(val):.10e}".replace(".", "")
        return int(s[:n_digits])

    first_digits = series.apply(lambda x: get_leading_digits(x, 1))
    two_digits = series.apply(lambda x: get_leading_digits(x, 2))

    def compute_benford_and_ks(observed_series, digits_range):
        n = len(observed_series)
        counts = (
            observed_series.value_counts().reindex(digits_range, fill_value=0)
        )
        actual_pct = (counts / n) * 100
        benford_prob = np.log10(1 + 1 / np.array(digits_range))
        benford_pct = benford_prob * 100
        diff_pct = actual_pct - benford_pct

 
        actual_cdf = np.cumsum(counts.values / n)
        benford_cdf = np.cumsum(benford_prob)

        ks_stat = float(np.max(np.abs(actual_cdf - benford_cdf)))
        crit_val_05 = 1.36 / np.sqrt(n)
        is_different = ks_stat > crit_val_05

        res_df = pd.DataFrame(
            {
                "Digit": digits_range,
                "Actual Count": counts.values,
                "Actual (%)": np.round(actual_pct.values, 4),
                "Benford (%)": np.round(benford_pct, 4),
                "Difference (%)": np.round(diff_pct.values, 4),
                "Actual CDF": np.round(actual_cdf, 4),
                "Benford CDF": np.round(benford_cdf, 4),
            }
        )
        return res_df, ks_stat, crit_val_05, is_different, n

    df_1st, ks_1st, crit_1st, diff_1st, n_1st = compute_benford_and_ks(
        first_digits, range(1, 10)
    )
    df_2nd, ks_2nd, crit_2nd, diff_2nd, n_2nd = compute_benford_and_ks(
        two_digits, range(10, 100)
    )

    st.subheader(f"📋 Descriptive & Statistical Results for `{target_col}`")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Descriptive Statistics (`.describe()`)**")
        desc_df = series.describe().to_frame(name=target_col)
        st.dataframe(desc_df, use_container_width=True)

    with col2:
        st.markdown("**1st-Digit KS Test (NumPy)**")
        st.metric("1st-Digit KS Statistic (D)", f"{ks_1st:.4f}")
        st.metric("Critical Value (α = 0.05)", f"{crit_1st:.4f}")
        if diff_1st:
            st.error(
                f"Statistically Different (D > {crit_1st:.4f}): Reject Benford's Law."
            )
        else:
            st.success(
                f"Conforms to Benford (D ≤ {crit_1st:.4f}): Not statistically different."
            )

    with col3:
        st.markdown("**First 2-Digits KS Test (NumPy)**")
        st.metric("2-Digit KS Statistic (D)", f"{ks_2nd:.4f}")
        st.metric("Critical Value (α = 0.05)", f"{crit_2nd:.4f}")
        if diff_2nd:
            st.error(
                f"Statistically Different (D > {crit_2nd:.4f}): Reject Benford's Law."
            )
        else:
            st.success(
                f"Conforms to Benford (D ≤ {crit_2nd:.4f}): Not statistically different."
            )

    st.subheader("📈 Visual Comparison: Actual vs. Benford Distribution")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    axes[0, 0].bar(
        df_1st["Digit"] - 0.18,
        df_1st["Actual (%)"],
        width=0.36,
        label="Actual (%)",
        color="#1f77b4",
    )
    axes[0, 0].bar(
        df_1st["Digit"] + 0.18,
        df_1st["Benford (%)"],
        width=0.36,
        label="Benford (%)",
        color="#ff7f0e",
    )
    axes[0, 0].set_title("1st Leading Digit: Actual vs. Benford (%)")
    axes[0, 0].set_xlabel("Leading Digit (1–9)")
    axes[0, 0].set_ylabel("Percentage (%)")
    axes[0, 0].set_xticks(range(1, 10))
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.5)

    axes[0, 1].step(
        df_1st["Digit"],
        df_1st["Actual CDF"],
        where="mid",
        label="Actual CDF",
        color="#1f77b4",
        linewidth=2,
    )
    axes[0, 1].step(
        df_1st["Digit"],
        df_1st["Benford CDF"],
        where="mid",
        label="Benford CDF",
        color="#ff7f0e",
        linestyle="--",
        linewidth=2,
    )
    axes[0, 1].set_title(f"1st Digit CDF Comparison (KS = {ks_1st:.4f})")
    axes[0, 1].set_xlabel("Leading Digit (1–9)")
    axes[0, 1].set_ylabel("Cumulative Probability")
    axes[0, 1].set_xticks(range(1, 10))
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    axes[1, 0].bar(
        df_2nd["Digit"],
        df_2nd["Actual (%)"],
        label="Actual (%)",
        color="#1f77b4",
        alpha=0.75,
    )
    axes[1, 0].plot(
        df_2nd["Digit"],
        df_2nd["Benford (%)"],
        label="Benford (%)",
        color="#ff7f0e",
        linewidth=2,
    )
    axes[1, 0].set_title("First 2 Leading Digits: Actual vs. Benford (%)")
    axes[1, 0].set_xlabel("First Two Digits (10–99)")
    axes[1, 0].set_ylabel("Percentage (%)")
    axes[1, 0].legend()
