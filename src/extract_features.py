
# src/extract_features.py
import os
import glob
import csv
import numpy as np
import librosa
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s -%(message)s")

# Import self-define functions
from features import (
    compute_spectral_proximity,
    compute_formant_clustering,
    compute_micro_modulations
)
def main():
    # Resolve project root relative paths cleanly.
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, "data", "processed")
    outputs_dir = os.path.join(base_dir, "outputs")
    
    processed_files = glob.glob(os.path.join(processed_dir, "*.wav"))
    print(f"🚀 Found {len(processed_files)} processed segments. Initiating feature extraction...")
    
    features_master_list = []
    for file_path in processed_files:
        filename = os.path.basename(file_path)
        
        # ---- Metadata Parsing ----
        # Splitting filename string (e.g., M062_Mic_a_non-resonant_1st.wav)
        name_parts = filename.replace(".wav", "").split("_")
        singer_id = name_parts[0]
        condition = "resonant" if "resonant" in filename and "non-resonant" not in filename else "non-resonant"
        trial = name_parts[-1]
        vowel = name_parts[2]
        try:
            # Execute the modular features by passing the file path
            
            prox_data = compute_spectral_proximity(file_path)
            
            # Call our upgraded clustering function
            cluster_data = compute_formant_clustering(file_path)
            
            mod_data = compute_micro_modulations(file_path)
            
            features_master_list.append({
                "filename": filename,
                "singer_id": singer_id,
                "condition": condition,
                "vowel": vowel,
                "trial": trial,
                "feature_h1_f0_f1_abs_hz": round(prox_data["spectral_proximity_abs_hz"], 4) if not np.isnan(prox_data["spectral_proximity_abs_hz"]) else "NaN",
                "feature_h1_f0_f1_norm_ratio": round(prox_data["spectral_proximity_norm_ratio"], 4) if not np.isnan(prox_data["spectral_proximity_norm_ratio"]) else "NaN",

                # Map the stabilized, pure LTAS Hypothesis 2 feature trio
                "feature_h2_1_ltas_max_peak_db": round(cluster_data["h2_1_ltas_max_peak_db"], 4) if not np.isnan(cluster_data["h2_1_ltas_max_peak_db"]) else "NaN",
                "feature_h2_2_ltas_cluster_energy_ratio_db": round(cluster_data["h2_2_ltas_cluster_energy_ratio_db"], 4) if not np.isnan(cluster_data["h2_2_ltas_cluster_energy_ratio_db"]) else "NaN",
                "feature_h2_3_ltas_f3_f5_slope_db_per_khz": round(cluster_data["h2_3_ltas_f3_f5_slope_db_per_khz"], 4) if not np.isnan(cluster_data["h2_3_ltas_f3_f5_slope_db_per_khz"]) else "NaN",
                
                "feature_h3_1_f1_band_energy_db": round(mod_data["h3_1_f1_band_energy_db"], 4) if not np.isnan(mod_data["h3_1_f1_band_energy_db"]) else "NaN",
                "feature_h3_2_normalized_f1_energy": round(mod_data["h3_2_normalized_f1_energy"], 4) if not np.isnan(mod_data["h3_2_normalized_f1_energy"]) else "NaN",
                "feature_h3_3_cv_of_f1_energy": round(mod_data["h3_3_cv_of_f1_energy"], 4) if not np.isnan(mod_data["h3_3_cv_of_f1_energy"]) else "NaN",
                "feature_h3_4_q_factor_mean": round(mod_data["h3_4_q_factor_mean"], 4) if not np.isnan(mod_data["h3_4_q_factor_mean"]) else "NaN",
                "feature_h3_5_q_factor_std": round(mod_data["h3_5_q_factor_std"], 4) if not np.isnan(mod_data["h3_5_q_factor_std"]) else "NaN", # ◄ Map New Feature
                "feature_h3_6_peak_prominence_db": round(mod_data["h3_6_peak_prominence_db"], 4) if not np.isnan(mod_data["h3_6_peak_prominence_db"]) else "NaN",
                "feature_h3_7_time_domain_envelope_modulation_depth": round(mod_data["h3_7_time_domain_envelope_modulation_depth"], 4) if not np.isnan(mod_data["h3_7_time_domain_envelope_modulation_depth"]) else "NaN",
                "feature_h3_8_f1_trajectory_rms_roughness_hz": round(mod_data["h3_8_f1_trajectory_rms_roughness_hz"], 4) if not np.isnan(mod_data["h3_8_f1_trajectory_rms_roughness_hz"]) else "NaN"
                
                })
        except Exception as e:
            print(f"⚠️ Error extraction matrix row for {filename}: {str(e)}")
        
        # --- Write out final dataset to Excel-compatible CSV ---
        csv_path = os.path.join(outputs_dir, "singing_features_master.csv")
        csv_columns = ["filename", "singer_id", "condition", "vowel","trial", 
                       "feature_h1_f0_f1_abs_hz", "feature_h1_f0_f1_norm_ratio",
                       "feature_h2_1_ltas_max_peak_db", 
                       "feature_h2_2_ltas_cluster_energy_ratio_db", "feature_h2_3_ltas_f3_f5_slope_db_per_khz",
                       "feature_h3_1_f1_band_energy_db", "feature_h3_2_normalized_f1_energy",
                       "feature_h3_3_cv_of_f1_energy", "feature_h3_4_q_factor_mean", "feature_h3_5_q_factor_std",
                       "feature_h3_6_peak_prominence_db", "feature_h3_7_time_domain_envelope_modulation_depth",
                       "feature_h3_8_f1_trajectory_rms_roughness_hz"]
        
        with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
            writer.writeheader()
            for row in features_master_list:
                writer.writerow(row)
        
        print(f"\n📊 Master Feature Matrix successfully saved!")
        print(f"📂 Location: {csv_path}")
        
if __name__ == "__main__":
    main()