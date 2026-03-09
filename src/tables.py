import pandas as pd


def make_side_by_side_table(agg):
    """
    Build a side-by-side comparison table like the screenshot:
    - columns for Procrustes (On/Off), CCA Corr. (On/Off), Accuracy (On/Off)
    - one row per target pair
    - plus average row at the bottom
    """
    df_on  = agg['ai_on']
    df_off = agg['ai_off']

    # Merge ON and OFF by (cat1,cat2)
    merged = pd.merge(
        df_on[['cat1','cat2','procrustes_disparity_mean','cca_correlation_mean','classification_accuracy_mean']],
        df_off[['cat1','cat2','procrustes_disparity_mean','cca_correlation_mean','classification_accuracy_mean']],
        on=['cat1','cat2'],
        suffixes=(' (On)',' (Off)')
    )

    # Rename columns for clarity
    merged = merged.rename(columns={
        'procrustes_disparity_mean (On)': 'Procrustes (On)',
        'procrustes_disparity_mean (Off)': 'Procrustes (Off)',
        'cca_correlation_mean (On)': 'CCA Corr. (On)',
        'cca_correlation_mean (Off)': 'CCA Corr. (Off)',
        'classification_accuracy_mean (On)': 'Accuracy (On)',
        'classification_accuracy_mean (Off)': 'Accuracy (Off)'
    })

    # Create a single Target Pair label
    merged['Target Pair'] = merged['cat1'] + " vs " + merged['cat2']
    merged = merged[['Target Pair','Procrustes (On)','Procrustes (Off)',
                     'CCA Corr. (On)','CCA Corr. (Off)',
                     'Accuracy (On)','Accuracy (Off)']]

    # Average row
    avg_row = pd.DataFrame({
        'Target Pair': ['AVERAGE'],
        'Procrustes (On)': [merged['Procrustes (On)'].mean()],
        'Procrustes (Off)': [merged['Procrustes (Off)'].mean()],
        'CCA Corr. (On)': [merged['CCA Corr. (On)'].mean()],
        'CCA Corr. (Off)': [merged['CCA Corr. (Off)'].mean()],
        'Accuracy (On)': [merged['Accuracy (On)'].mean()],
        'Accuracy (Off)': [merged['Accuracy (Off)'].mean()]
    })

    merged = pd.concat([merged, avg_row], ignore_index=True)
    return merged

def table_to_dict(df: pd.DataFrame, key_cols=('cat1','cat2'), decimals=3):
    """
    Convert a results DataFrame into a nested Python dict for easy debugging.

    - Keys: "cat1 vs cat2" (or just the single key if key_cols has length 1)
    - Values: dict of all remaining columns (rounded numerics)

    Example keys:
      "left vs right" for pairwise tables
      "straight" for same-target tables (use key_cols=('target',))
    """
    if df is None or df.empty:
        return {}

    df2 = df.copy()
    num_cols = df2.select_dtypes(include=['number']).columns
    df2[num_cols] = df2[num_cols].round(decimals)

    out = {}
    for _, row in df2.iterrows():
        if len(key_cols) == 2:
            key = f"{row[key_cols[0]]} vs {row[key_cols[1]]}"
        else:
            key = row[key_cols[0]]
        # store everything except the key columns
        value = {col: row[col] for col in df2.columns if col not in key_cols}
        out[key] = value
    return out

class SuccessRateSummary:
    def __init__(self):
        self.rows = []

    def add_entry(self, experiment, monkey, success_rate_ai_on, success_rate_ai_off,
                  variability_on, variability_off, slope_on, slope_off, statistic, p_value, p_slope_on, p_slope_off, p_var):
        mean_on = success_rate_ai_on.mean()
        std_on = success_rate_ai_on.std()
        mean_off = success_rate_ai_off.mean()
        std_off = success_rate_ai_off.std()

        self.rows.append({
            "Experiment": experiment,
            "Monkey": monkey,
            "AI ON Success (mean ± std)": f"{mean_on:.2f} ± {std_on:.2f}",
            "AI OFF Success (mean ± std)": f"{mean_off:.2f} ± {std_off:.2f}",
            "p-value + statistic (paired t-test)": f"{p_value:.4f}, {statistic:.4f}",
            "AI ON Variability (std + p-value)": f"{variability_on:.2f}, {p_var:.2f}",
            "AI OFF Variability (std)": f"{variability_off:.2f}",
            "AI ON Learning Slope": f"{slope_on:.3f}, {p_slope_on:.2f}",
            "AI OFF Learning Slope": f"{slope_off:.3f}, {p_slope_off:.2f}"
        })

    def finalize(self, save_path="f{experiment}_summary.csv", display=True):
        summary_df = pd.DataFrame(self.rows)
        summary_df.to_csv(save_path, index=False)
        if display:
            print("\n========== Summary Table ==========\n")
            print(summary_df.to_string(index=False))
        return summary_df