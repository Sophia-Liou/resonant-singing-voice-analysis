import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path

def generate_flat_spectral_plots(csv_path: str, audio_dir: str, output_dir: str):
    """
    Computes FFT spectral profiles and outputs a single flat folder of images.
    Explicitly enforces calibrated x and y tick marks to fix formatting bugs.
    """
    # 1. Establish flat output workspace directory
    plots_folder = Path(output_dir) / "spectral_curves_flat"
    plots_folder.mkdir(parents=True, exist_ok=True)
    
    # Load metadata sheet
    df = pd.read_csv(csv_path)
    df["condition"] = df["condition"].str.strip().str.lower()
    
    unique_singers = sorted(df["singer_id"].unique())
    unique_vowels = sorted(df["vowel"].unique())
    
    # Use standard white ticks style to guarantee visibility
    sns.set_theme(style="ticks", context="paper")
    
    print(f"🎵 Starting spectral pipeline. Exporting flat folder to: {plots_folder}")
    total_plots_rendered = 0
    
    # 2. Iterate systematically over combinations
    for singer in unique_singers:
        singer_df = df[df["singer_id"] == singer]
        
        for vowel in unique_vowels:
            pair_df = singer_df[singer_df["vowel"] == vowel]
            
            if pair_df.empty:
                continue
                
            # Allocate clean axis containers
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Remove top and right spines safely without trimming ticks off the bottom/left
            sns.despine(ax=ax, top=True, right=True, left=False, bottom=False, trim=False)
            
            trace_count = 0
            max_y_val = -100  # Track max dB to calibrate limits dynamically
            
            # 3. Processing Layer: Render the curve lines
            for idx, row in pair_df.iterrows():
                filename = row["filename"]
                condition = row["condition"]
                trial = row["trial"]
                
                audio_path = Path(audio_dir) / filename
                if not audio_path.exists():
                    audio_path = Path(audio_dir) / condition / filename
                    if not audio_path.exists():
                        continue
                        
                try:
                    # Load audio waveform at native sampling rate
                    y, sr = librosa.load(str(audio_path), sr=None)
                    
                    # Compute windowed Fast Fourier Transform
                    n_fft = 2048
                    stft_matrix = np.abs(librosa.stft(y, n_fft=n_fft))
                    
                    # Compress magnitude linear scale to decibels (dB)
                    mean_spectrum = librosa.amplitude_to_db(np.mean(stft_matrix, axis=1), ref=np.max)
                    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
                    
                    # Track absolute maximum amplitude seen in this file
                    if len(mean_spectrum) > 0:
                        max_y_val = max(max_y_val, np.max(mean_spectrum))
                    
                    # 4. Map condition strings directly to line configurations
                    if "non" in condition:
                        linestyle = "--"
                        label_text = f"Non-Resonant Baseline (Trial {trial})"
                        line_color = "#4c72b0" # Deep steel blue
                    else:
                        linestyle = "-"
                        label_text = f"Resonant Performance (Trial {trial})"
                        line_color = "#dd8452" # Coral orange
                        
                    ax.plot(
                        frequencies, 
                        mean_spectrum, 
                        color=line_color, 
                        linestyle=linestyle, 
                        linewidth=1.3, 
                        alpha=0.85,
                        label=label_text
                    )
                    trace_count += 1
                    
                except Exception as e:
                    print(f"   ⚠️ Skipping trace file {filename}: {str(e)}")
                    continue
            
            # 5. FIX: Explicit Axis Tick Calibration Engine
            if trace_count > 0:
                # Force rigid axes limits to match real data scales
                ax.set_xlim(0, 5000)
                ax.set_ylim(-80, 5) # Calibrated for standard normalized openSMILE dB ranges
                
                # Force major tick positions every 500 Hz on the x-axis
                ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
                # Force major tick positions every 10 dB on the y-axis
                yaxis_step = 10
                ax.yaxis.set_major_locator(ticker.MultipleLocator(yaxis_step))
                
                # Ensure the ticks are drawn pointing outward and are clearly visible
                ax.tick_params(axis='both', which='major', direction='out', length=5, width=1, colors='black', labelsize=9)
                
                # Apply high-contrast text labels
                ax.set_title(f"Spectral Profile Comparison | Subject: {singer} | Vowel: /{vowel}/", fontsize=11, fontweight="bold", pad=12)
                ax.set_xlabel("Frequency (Hz)", fontsize=10, labelpad=8)
                ax.set_ylabel("Spectral Magnitude (dB)", fontsize=10, labelpad=8)
                
                # Render clean background grid lines to assist visual measurement
                ax.grid(True, which='major', linestyle=':', linewidth=0.5, color='#e0e0e0', alpha=0.7)
                
                # Append internal legend block
                ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor="#ffffff", framealpha=0.9)
                plt.tight_layout()
                
                # 6. FIX: Save directly to a single flat folder (No nested singer folders)
                clean_filename = f"{singer}_spectral_atlas_vowel_{vowel}.png"
                save_filepath = plots_folder / clean_filename
                
                plt.savefig(str(save_filepath), dpi=250, bbox_inches="tight")
                total_plots_rendered += 1
                
            plt.close(fig) # Prevent memory leaks
            
    print(f"\n🏁 PIPELINE EXECUTION PASSED. Generated {total_plots_rendered} calibrated spectral curves inside a flat folder.")
    print(f"   📂 Output Workspace Destination: {plots_folder}")

if __name__ == "__main__":
    target_csv = r"C:\Users\rfyuli\resonant-singing-voice-analysis\outputs\singing_features_master.csv"
    raw_wav_dir = r"C:\Users\rfyuli\resonant-singing-voice-analysis\data\processed"
    output_workspace = r"C:\Users\rfyuli\resonant-singing-voice-analysis\outputs"
    
    generate_flat_spectral_plots(target_csv, raw_wav_dir, output_workspace)