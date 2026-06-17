import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from pathlib import Path

def run_feature_matrix_analysis(csv_path: str, dataset_tag: str, output_dir: Path):
    """
    Executes a strict paired-design statistical and visualization routine on a given 
    master feature CSV file. Dynamically separates metadata and computes paired statistics.
    """
    print(f"\n────────────────────────────────────────────────────────────")
    print(f"🚀 PROCESSING DATASET: [{dataset_tag.upper()}] -> {Path(csv_path).name}")
    print(f"────────────────────────────────────────────────────────────")
    
    # 1. Load data
    df = pd.read_csv(csv_path)
    
    # Explicitly enforce known common metadata headers
    meta_cols = ["filename", "singer_id", "condition", "vowel", "trial"]
    
    # Verify input integrity
    missing_meta = [m for m in meta_cols if m not in df.columns]
    if missing_meta:
        raise KeyError(f"❌ Missing critical metadata columns in file: {missing_meta}")
        
    # Isolate all remaining columns dynamically as the acoustic feature space
    feature_cols = [col for col in df.columns if col not in meta_cols]
    
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    print(f"📊 Dataset Grid: {df.shape[0]} rows loaded across {len(feature_cols)} features.")
    
    # 2. Replicate Index Alignment Matrix Pass
    # Isolate conditions and lock into a unique 3-tier tracking coordinate system
    res_flat = df[df["condition"] == "resonant"].set_index(["singer_id", "vowel", "trial"])
    non_flat = df[df["condition"] == "non-resonant"].set_index(["singer_id", "vowel", "trial"])
    
    # Set-theoretic intersection to discard any completely unpaired files cleanly
    common_idx = res_flat.index.intersection(non_flat.index)
    
    res_aligned = res_flat.loc[common_idx, feature_cols].astype(float)
    non_aligned = non_flat.loc[common_idx, feature_cols].astype(float)
    
    # Compute direct planar change mapping matrix
    delta_matrix = non_aligned - res_aligned
    
    analysis_records = []
    
    # 3. Statistical Inferential Sub-Engine Loop
    for feature in feature_cols:
        feature_deltas = delta_matrix[feature].dropna().values
        x_res = res_aligned[feature].dropna().values
        x_non = non_aligned[feature].dropna().values
        
        # Minimum sample size boundary guardrail
        min_length = min(len(x_res), len(x_non), len(feature_deltas))
        if min_length < 5 or np.std(feature_deltas) == 0:
            continue
            
        feature_deltas = feature_deltas[:min_length]
        x_res = x_res[:min_length]
        x_non = x_non[:min_length]
        
        # Normality distribution testing (Shapiro-Wilk)
        _, shapiro_p = stats.shapiro(feature_deltas)
        is_normal = shapiro_p > 0.05
        
        # Adaptive within-subject inference testing routing
        if is_normal:
            stat, p_val = stats.ttest_rel(x_non, x_res)
            test_name = "Paired t-test"
        else:
            stat, p_val = stats.wilcoxon(x_non, x_res, zero_method='pratt')
            test_name = "Wilcoxon Signed-Rank"
            
        # Mathematically rigorous Paired Cohen's d computation
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
            "Cohen_d": round(cohen_d, 4),
            "x_res_data": x_res,  # Cached arrays for visualization rendering pass
            "x_non_data": x_non
        })
        
    analysis_df = pd.DataFrame(analysis_records)
    
    if analysis_df.empty:
        print(f"⚠️ No analyzable features extracted for dataset: {dataset_tag}")
        return

    # 4. Global High-Dimensional FDR Adjustment (Benjamini-Hochberg)
    reject, corrected_p, _, _ = multipletests(
        analysis_df["Raw_p_value"].values, 
        alpha=0.05, 
        method="fdr_bh"
    )
    analysis_df["Corrected_p_value"] = corrected_p
    analysis_df["Significance"] = np.where(reject, "* Significant", "n.s.")
    
    # Sort findings by raw significance margins
    analysis_df = analysis_df.sort_values(by="Raw_p_value").reset_index(drop=True)
    
    # 5. Conditional Graphics Plotting Layer
    plots_folder = output_dir / "plots" / dataset_tag
    plots_folder.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="ticks", context="paper")
    plot_count = 0
    
    for idx, row in analysis_df.iterrows():
        p_val = row["Raw_p_value"]
        feature = row["Feature"]
        test_name = row["Applied_Test"]
        
        # Strict gate constraint rule check: p-value must be less than or equal to 0.05
        if p_val <= 0.05:
            # Reconstruct isolated subset data frame from data trace cache
            plot_df = pd.DataFrame({
                "condition": ["non-resonant"] * len(row["x_non_data"]) + ["resonant"] * len(row["x_res_data"]),
                "value": np.concatenate([row["x_non_data"], row["x_res_data"]])
            })
            
            fig, ax = plt.subplots(figsize=(6, 5))
            
            # Modern warning-free Boxplot
            sns.boxplot(
                data=plot_df, 
                x="condition", 
                y="value", 
                order=["non-resonant", "resonant"],
                hue="condition", 
                palette="Set2",
                legend=False,
                width=0.4,
                fliersize=0,
                ax=ax
            )
            
            # Modern warning-free Stripplot overlay
            sns.stripplot(
                data=plot_df, 
                x="condition", 
                y="value", 
                order=["non-resonant", "resonant"], 
                hue="condition",
                palette="dark:black", 
                alpha=0.4, 
                size=5, 
                legend=False,
                jitter=0.15,
                ax=ax
            )
            
            # Frame title labels
            ax.set_title(f"[{dataset_tag.upper()}] Distribution Analysis: {feature}\n({test_name} p={p_val:.5f})", fontsize=9, fontweight="bold", pad=10)
            ax.set_xlabel("Vocal Condition", fontsize=9, labelpad=6)
            ax.set_ylabel("Metric Calibration Units", fontsize=9, labelpad=6)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Non-Resonant Phonation", "Resonant Phonation"], fontsize=9)
            
            plt.tight_layout()
            
            # 6. Adaptive File Prefix Rule Controller Logic
            # Applies egemaps_ to openSMILE outputs, leaves custom feature names untouched
            clean_name = feature.replace(".", "_").replace("-", "_")
            if dataset_tag.lower() == "egemaps":
                export_filename = f"egemaps_{clean_name}.png"
            else:
                export_filename = f"{clean_name}.png"
                
            fig_save_path = plots_folder / export_filename
            plt.savefig(str(fig_save_path), dpi=300)
            plt.close(fig)
            plot_count += 1
            
    print(f"📈 Plot generation layer closed. Generated {plot_count} images inside: {plots_folder}")
    
    # Save the spreadsheet report matrix
    analysis_df = analysis_df.drop(columns=["x_res_data", "x_non_data"])
    report_path = output_dir / f"{dataset_tag}_statistical_report.csv"
    analysis_df.to_csv(str(report_path), index=False)
    print(f"✨ Successfully exported final report table to: {report_path}")

def run_consolidated_master_pipeline():
    """
    Main structural orchestration sequence mapping input matrix data directly 
    to the statistical evaluation engine.
    """
    # Define primary workspace paths
    output_workspace = Path(r"C:/Users/rfyuli/resonant-singing-voice-analysis/outputs")
    
    # Map input source coordinates
    custom_features_csv = output_workspace.parent / "outputs" / "singing_features_master.csv"
    egemaps_features_csv = output_workspace.parent / "outputs" / "egemaps_features_master.csv"
    
    # Task Pass 1: Execute analysis over your customized physics metrics
    if custom_features_csv.exists():
        run_feature_matrix_analysis(
            csv_path=str(custom_features_csv),
            dataset_tag="custom_covox",
            output_dir=output_workspace
        )
    else:
        print(f"⚠️ Could not locate custom features master file path at: {custom_features_csv}")
        
    # Task Pass 2: Execute analysis over the generic openSMILE baseline parameters
    if egemaps_features_csv.exists():
        run_feature_matrix_analysis(
            csv_path=str(egemaps_features_csv),
            dataset_tag="egemaps",
            output_dir=output_workspace
        )
    else:
        print(f"⚠️ Could not locate eGeMAPS master file path at: {egemaps_features_csv}")
        
    print("\n🏁 CONSOLIDATED MASTER PIPELINE EXECUTION COMPLETE.")

if __name__ == "__main__":
    run_consolidated_master_pipeline()