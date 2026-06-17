import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def generate_egemaps_boxplots(results_csv: str, master_csv: str, output_directory: str):
    """
    Identifies eGeMAPS features with an uncorrected raw p-value < 0.05
    and constructs standalone boxplots comparing phonation conditions.
    Adheres strictly to graphic presentation safety guardrails.
    """
    # Load dataset frames
    results_df = pd.read_csv(results_csv)
    master_df = pd.read_csv(master_csv)
    
    # Isolate features meeting the raw alpha threshold boundary
    target_features = results_df[results_df["Raw_p_value"] < 0.05]["Feature"].tolist()
    
    if not target_features:
        print("⚠️ No features found matching the raw p < 0.05 threshold criteria.")
        return
        
    # Establish clean output folder path layout
    out_path = Path(output_directory)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Found {len(target_features)} features with raw p < 0.05. Commencing visualization loop...")
    
    # Configure clean, high-contrast academic plotting layout style
    sns.set_theme(style="ticks", context="paper")
    
    for feature in target_features:
        if feature in master_df.columns:
            # Clear previous drawing axes safely without calling forbidden .figure allocations
            plt.clf()
            fig, ax = plt.subplots(figsize=(6, 5))
            
            # Construct boxplot matrix mapping the phonation groups
            # Uses a color palette to cleanly separate the categorical states
            sns.boxplot(
                x="condition", 
                y=feature, 
                order=["non-resonant", "resonant"], 
                data=master_df, 
                hue="condition",
                palette="Set2",
                legend=False,
                width=0.4,
                fliersize=4,
                ax=ax
            )
            sns.stripplot(
            data=master_df, 
            x="condition", 
            y=feature, 
            order=["non-resonant", "resonant"], 
            hue="condition",
            palette="dark:black", 
            alpha=0.4, 
            size=5,
            ax=ax
            )
            
            # Format display overlays using clean, non-overlapping labels
            ax.set_title(f"Acoustic Variation: {feature}\n(Raw p < 0.05)", fontsize=11, fontweight="bold", pad=10)
            
            ax.set_xlabel("Phonation Condition", fontsize=10, labelpad=8)
            ax.set_ylabel("Acoustic Scale Units", fontsize=10, labelpad=8)
            plt.tight_layout()
            # Sanitize file string formatting to prevent path parsing failures
            safe_name = feature.replace(".", "_").replace("-", "_")
            export_filename = f"egemaps_{safe_name}.png"
            export_path = out_path / export_filename
            
            # Export visual data asset
            plt.savefig(str(export_path), dpi=300)
            print(f"   ✓ Exported: {export_filename}")
            
    print(f"✨ Success! All visual asset plots saved to directory folder: {output_directory}")

if __name__ == "__main__":
    # Define absolute directory tracking variables
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs_dir = os.path.join(base_dir, "outputs")
    csv_results = f"{outputs_dir}\egemaps_statistical_results.csv"
    csv_master = f"{outputs_dir}\egemaps_features_master.csv"
    plots_out = f"{outputs_dir}\plots"
    
    generate_egemaps_boxplots(csv_results, csv_master, plots_out)