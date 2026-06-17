import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import os
from pathlib import Path

def run_egemaps_statistical_analysis(csv_path: str, output_path: str):
    """
    Performs a high-dimensional paired statistical analysis over the eGeMAPS 
    feature set. Includes automated normality diagnostics, adaptive test routing, 
    and Benjamini-Hochberg FDR correction.
    """
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Separate metadata headers from the acoustic features
    meta_cols = ["filename", "subject_id", "vowel", "condition"]
    feature_cols = [col for col in df.columns if col not in meta_cols]
    
# Drop the strict filename index layer to isolate comparative feature columns safely
    res_flat = df[df["condition"] == "resonant"].set_index(["subject_id", "vowel"])
    non_flat = df[df["condition"] == "non-resonant"].set_index(["subject_id", "vowel"])
    
    # Clean alignment: Intersect indices to drop any completely unpaired files
    common_idx = res_flat.index.intersection(non_flat.index)
    
    # FIX: Compute the difference directly within Pandas to guarantee safe internal index broadcasting
    # This automatically handles asymmetric row counts or missing file sets safely
    res_aligned = res_flat.loc[common_idx, feature_cols].astype(float)
    non_res_aligned = non_flat.loc[common_idx, feature_cols].astype(float)
    
    # Calculate the unified delta matrix plane
    delta_matrix = non_res_aligned - res_aligned
    
    analysis_records = []
    
    # Iterate through each eGeMAPS acoustic parameter
    for feature in feature_cols:
        feature_deltas = delta_matrix[feature].dropna().values
        
        res_vals = res_aligned[feature].dropna().values
        non_res_vals = non_res_aligned[feature].dropna().values
        min_length = min(len(res_vals), len(non_res_vals), len(feature_deltas))
        if min_length < 5 or np.std(feature_deltas) == 0:
            continue
        
        feature_deltas = feature_deltas[:min_length]
        # Omit missing rows dynamically to maintain strict pair integrity
        x_res = res_vals[:min_length]
        x_non = non_res_vals[:min_length]
            
        # Normality diagnostics on the within-pair differences
        _, shapiro_p = stats.shapiro(feature_deltas)
        is_normal = shapiro_p > 0.05
        
        # Adaptive inferential test routing
        if is_normal:
            stat, p_val = stats.ttest_rel(x_non, x_res)
            test_name = "Paired t-test"
        else:
            stat, p_val = stats.wilcoxon(x_non, x_res, zero_method='pratt')
            test_name = "Wilcoxon Signed-Rank"
            
        # Standardized Cohen's d calculation for paired designs
        mean_delta = np.mean(feature_deltas)
        std_delta = np.std(feature_deltas, ddof=1)
        cohen_d = float(mean_delta / std_delta) if std_delta != 0 else 0.0
        
        analysis_records.append({
            "Feature": feature,
            "Applied_Test": test_name,
            "Normality_p_value": round(shapiro_p, 5),
            "Is_Normal": is_normal,
            "Statistic": round(float(stat), 4),
            "Raw_p_value": float(p_val),
            "Cohen_d": round(cohen_d, 4)
        })
        
    analysis_df = pd.DataFrame(analysis_records)
    
    # Apply multiple testing correction (Benjamini-Hochberg procedure)
    if not analysis_df.empty:
        reject, corrected_p, _, _ = multipletests(
            analysis_df["Raw_p_value"].values, 
            alpha=0.05, 
            method="fdr_bh"
        )
        analysis_df["Corrected_p_value"] = corrected_p
        analysis_df["Significance"] = np.where(reject, "* Significant", "n.s.")
        
        # Sort features based on their raw significance levels
        analysis_df = analysis_df.sort_values(by="Raw_p_value").reset_index(drop=True)
    
    # Save the finalized report matrix
    analysis_df.to_csv(output_path, index=False)
    print(f"🚀 eGeMAPS statistical matrix successfully saved to: {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs_dir = os.path.join(base_dir, "outputs")
    csv_in = os.path.join(outputs_dir, "egemaps_features_master.csv")
    csv_out = os.path.join(outputs_dir, "egemaps_statistical_results.csv")
    
    Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
    run_egemaps_statistical_analysis(csv_in, csv_out)