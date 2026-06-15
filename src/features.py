import numpy as np
import parselmouth
from parselmouth.praat import call
from typing import Dict, List
import logging
from scipy.signal import medfilt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s -%(message)s")

def compute_spectral_proximity(file_path: str) -> Dict[str, float]:
    """
    Hypothesis 1: Tracks F0 and F1 to calculate the average distance.

    Args:
        file_path (str): -> Absolute or relative path to a .wav audio file.

    Returns:
        Dict[str, float]: -> A dictionary containing absolute and normalized metrics.
                          Values will be np.nan if tracking fails.
    """
    # Initialize your default dictionary layout
    results = {
        "spectral_proximity_abs_hz": np.nan,
        "spectral_proximity_norm_ratio": np.nan
    }
    try:
        sound = parselmouth.Sound(file_path)
        pitch = call(sound, "To Pitch", 0.0, 75.0, 600.0)
        formant_object = call(sound, "To Formant (burg)", 0.0, 5, 5000.0, 0.025, 50.0)
        num_formant_frames = call(formant_object, "Get number of frames")

        abs_distances = []
        norm_distances = []
        
        for frame in range(1, num_formant_frames + 1):
            current_time_sec = call(formant_object, "Get time from frame number", frame)
            f1 = call(formant_object, "Get value at time", 1, current_time_sec, "Hertz", "Linear")
            f0 = call(pitch, "Get value at time", current_time_sec, "Hertz", "Linear")
            
            if not np.isnan(f0) and not np.isnan(f1) and f0 > 0:
                abs_distances.append(f1 - f0)
                norm_distances.append((f1 - f0) / f0)
            
            # update values if valid frames were successfully calculated
            if len(abs_distances) > 0:
                results["spectral_proximity_abs_hz"] = float(np.mean(abs_distances))
                results["spectral_proximity_norm_ratio"] = float(np.mean(norm_distances))
                
            
    except Exception as e:
        print(f"⚠️ Tracking exception inside compute_spectral_proximity: {e}")
    
    return results

def compute_formant_clustering(file_path: str) -> Dict[str, float]:
    """
    Hypothesis 2 (Definitive LTAS Set): Extracts macro-resonance features 
    directly from Praat's stable Long-Term Average Spectrum (LTAS) object.
    Bypasses unstable framewise harmonic jitter completely.
    """
    results = {
        "h2_1_ltas_max_peak_db": np.nan,
        "h2_2_ltas_cluster_energy_ratio_db": np.nan,
        "h2_3_ltas_f3_f5_slope_db_per_khz": np.nan
    }
    
    try:
        sound = parselmouth.Sound(file_path)
        
        # 1. Generate a robust 20 Hz bandwidth LTAS object
        ltas = call(sound, "To Ltas", 100.0)
        
        # --- Feature 1: Max Peak in Cluster Zone ---
        ltas_max_peak = call(ltas, "Get maximum", 2000.0, 4000.0, "Parabolic")
        if not np.isnan(ltas_max_peak):
            results["h2_1_ltas_max_peak_db"] = float(ltas_max_peak)
            
        # --- Feature 2: Classic Integrated Cluster Energy Ratio ---
        # Query mean energies directly from the stable LTAS bins
        mean_low_zone = call(ltas, "Get mean", 0.0, 2000.0, "dB")
        mean_cluster_zone = call(ltas, "Get mean", 2000.0, 4000.0, "dB")
        
        if not np.isnan(mean_low_zone) and not np.isnan(mean_cluster_zone):
            # Difference in dB represents the log ratio of energy concentration
            results["h2_2_ltas_cluster_energy_ratio_db"] = float(mean_cluster_zone - mean_low_zone)
            
        # --- Feature 3: F3-F5 Spectral Slope Tilt ---
        # Uses a robust Burg formant tracker to locate global mean formant anchors
        formant_object = call(sound, "To Formant (burg)", 0.0, 5, 5000.0, 0.025, 50.0)
        
        f3_mean = call(formant_object, "Get mean", 3, 0.0, 0.0, "Hertz")
        f4_mean = call(formant_object, "Get mean", 4, 0.0, 0.0, "Hertz")
        
        if not np.isnan(f3_mean) and f3_mean > 0 and not np.isnan(f4_mean) and f4_mean > f3_mean:
            # Project steady F5 location based on structural F3-F4 distance
            f5_mean_projected = f4_mean + (f4_mean - f3_mean)
            
            # Query LTAS energy amplitudes at the formant anchor centers
            amp_f3 = call(ltas, "Get value at frequency", f3_mean, "Nearest")
            amp_f5 = call(ltas, "Get value at frequency", f5_mean_projected, "Nearest")
            
            if not np.isnan(amp_f3) and not np.isnan(amp_f5):
                db_delta = amp_f5 - amp_f3
                khz_delta = (f5_mean_projected - f3_mean) / 1000.0
                
                if khz_delta > 0:
                    results["h2_3_ltas_f3_f5_slope_db_per_khz"] = float(db_delta / khz_delta)
                    
    except Exception as e:
        print(f"⚠️ Exception inside compute_formant_clustering: {e}")
        
    return results

# def compute_micro_modulations(file_path: str) -> Dict[str, float]:
#     """
#     Hypothesis 3 (Publication-Grade Complete Set): Computes Energy, Resonance, 
#     and Temporal features centered around a dynamically adjusted, median-smoothed
#     F1 bandwidth window. Utilizes pure linear Power Spectral Density queries to 
#     eliminate unit mismatches and bandwidth accumulation biases.
#     """
#     results = {
#         "h3_1_f1_band_energy_db": np.nan,
#         "h3_2_normalized_f1_energy": np.nan,
#         "h3_3_cv_of_f1_energy": np.nan,
#         "h3_4_q_factor": np.nan,
#         "h3_5_peak_prominence_db": np.nan,
#         "h3_6_f1_modulation_power_4_8hz": np.nan,
#         "h3_7_f1_trajectory_roughness_hz_per_sec": np.nan
#     }
    
#     try:
#         sound = parselmouth.Sound(file_path)
        
#         # 1. Generate core tracking objects
#         pitch = call(sound, "To Pitch", 0.0, 75.0, 600.0)
#         formant_object = call(sound, "To Formant (burg)", 0.0, 5, 5000.0, 0.025, 50.0)
#         spectrogram = call(sound, "To Spectrogram", 0.005, 5000.0, 0.002, 20.0, "Gaussian")
#         # Convert the global spectrogram to a Matrix object to allow safe linear queries
#         spec_matrix = call(spectrogram, "To Matrix")
        
#         # Pull raw numeric arrays and geometric traits directly via Python attributes.
#         grid_data = spec_matrix.as_array()
#         num_rows, num_cols = grid_data.shape
#         dx = spec_matrix.dx
#         dy = spec_matrix.dy
#         x1 = spec_matrix.x1
#         y1 = spec_matrix.y1
        
#         num_frames = call(formant_object, "Get number of frames")
        
#         raw_times = []
#         raw_f1_freqs = []
#         raw_f1_bws = []
        
#         # Pass 1: Collect tracked values over the voiced timeline
#         for frame in range(1, num_frames + 1):
#             time = call(formant_object, "Get time from frame number", frame)
#             f0 = call(pitch, "Get value at time", time, "Hertz", "Linear")
            
#             if not np.isnan(f0) and f0 > 0:
#                 f1_freq = call(formant_object, "Get value at time", 1, time, "Hertz", "Linear")
#                 f1_bw = call(formant_object, "Get bandwidth at time", 1, time, "Hertz", "Linear")
                
#                 if not np.isnan(f1_freq) and f1_freq > 0:
#                     raw_times.append(time)
#                     raw_f1_freqs.append(f1_freq)
#                     raw_f1_bws.append(f1_bw if (not np.isnan(f1_bw) and f1_bw > 0) else np.nan)
                    
#         n_voiced_samples = len(raw_f1_freqs)
#         if n_voiced_samples <= 1:
#             return results

#         # --- Issue 4 Fixed: Local Linear Interpolation for Bandwidth Gaps ---
#         raw_f1_bws = np.array(raw_f1_bws)
#         nans = np.isnan(raw_f1_bws)
        
#         if np.all(nans):
#             # Fallback if the entire bandwidth track failed to resolve
#             filled_bws = np.full_like(raw_f1_bws, 150.0)
#         elif np.any(nans):
#             # Interpolate missing values based on adjacent timeline indices
#             x_indices = np.arange(len(raw_f1_bws))
#             raw_f1_bws[nans] = np.interp(x_indices[nans], x_indices[~nans], raw_f1_bws[~nans])
#             filled_bws = raw_f1_bws
#         else:
#             filled_bws = raw_f1_bws
            
#         # --- Issue 3 Fixed: Apply a 5-point median filter to the filled trace ---
#         smoothed_bws = medfilt(filled_bws, kernel_size=5 if len(filled_bws) >= 5 else 3)

#         # Pass 2: Feature extraction using linear power spectral density values
#         q_values = []
#         f1_prominences_linear = []
#         f1_band_linear_energies = []
#         total_frame_linear_energies = []
        
#         for i in range(n_voiced_samples):
#             time = raw_times[i]
#             f1_freq = raw_f1_freqs[i]
#             f1_bw = smoothed_bws[i]
            
#             # Issue 3 Fixed: Q-Factor now uses the consistent median-smoothed bandwidth track
#             q_values.append(f1_freq / f1_bw)
                
#             effective_half_width = np.clip(f1_bw / 2.0, 75.0, 200.0)
#             f1_low = max(0.0, f1_freq - effective_half_width)
#             f1_high = f1_freq + effective_half_width
            
#             # Map temporal/Spectral floats to raw array indices safely
#             col_idx = np.clip(int(round((time - x1) / dx)), 0, num_cols - 1)
#             row_low_idx = np.clip(int(round((f1_low - y1) / dy)), 0, num_rows -1)
#             row_high_idx = np.clip(int(round((f1_high - y1) / dy)) + 1, row_low_idx + 1, num_rows)
            
#             # Slice the raw array vector
#             band_slice = grid_data[row_low_idx:row_high_idx, col_idx]
            
#             total_high_idx = np.clip(int(round((5000.0 - y1) / dy)) + 1, 1, num_rows)
#             total_slice = grid_data[0:total_high_idx, col_idx]
            
#             band_density = np.mean(band_slice)
#             total_density = np.mean(total_slice)
            

            
#             if not np.isnan(band_density) and band_density > 0 and not np.isnan(total_density) and total_density > 0:
#                 f1_band_linear_energies.append(band_density * dy)
#                 total_frame_linear_energies.append(total_density * dy)
                
#                 # Unbiased prominence calculation via adjacent index indexing
#                 row_span = row_high_idx - row_low_idx
#                 n_low_idx_start = np.clip(row_low_idx - row_span, 0, num_rows -1)
#                 n_high_idx_end = np.clip(row_high_idx + row_span, row_high_idx + 1, num_rows)
                
                
#                 neighbor_low_density = np.mean(grid_data[n_low_idx_start:row_low_idx, col_idx])
        
#                 neighbor_high_density = np.mean(grid_data[row_high_idx:n_high_idx_end, col_idx])
                
#                 f1_peak_row_idx = np.clip(int(round((f1_freq - y1) / dy)), 0, num_rows - 1)
#                 peak_density = grid_data[f1_peak_row_idx, col_idx]
                
#                 if not np.isnan(neighbor_low_density) and not np.isnan(neighbor_high_density) and not np.isnan(peak_density):
#                     avg_neighbor_density = (neighbor_low_density + neighbor_high_density) / 2.0
                    
#                     if avg_neighbor_density > 0:
#                         # Store density-normalized prominence ratio
#                         f1_prominences_linear.append(peak_density / avg_neighbor_density)

#         # 5. Final Feature Compilation & Normalization Calculations
#         if len(f1_band_linear_energies) > 0:
#             # Convert linear energy density mean to dB for standardized scale reporting
#             results["h3_1_f1_band_energy_db"] = float(10 * np.log10(np.mean(f1_band_linear_energies)))
#             results["h3_3_cv_of_f1_energy"] = float(np.std(f1_band_linear_energies) / np.mean(f1_band_linear_energies))
            
#             sum_f1_linear = np.sum(f1_band_linear_energies)
#             sum_total_linear = np.sum(total_frame_linear_energies)
#             if sum_total_linear > 0:
#                 results["h3_2_normalized_f1_energy"] = float(10 * np.log10(sum_f1_linear / sum_total_linear))
            
#             # Feature 6: Modulation Power over the adaptive energy contour
#             n_samples = len(f1_band_linear_energies)
#             if n_samples > 4:
#                 dt = float(np.mean(np.diff(raw_times)))
#                 if dt > 0:
#                     ac_signal = np.array(f1_band_linear_energies) - np.mean(f1_band_linear_energies)
#                     fft_vals = np.abs(np.fft.rfft(ac_signal))
#                     fft_freqs = np.fft.rfftfreq(n_samples, d=dt)
                    
#                     fft_power = fft_vals ** 2
#                     tremor_indices = np.where((fft_freqs >= 4.0) & (fft_freqs <= 8.0))[0]
#                     total_mod_power = np.sum(fft_power[1:])
                    
#                     if total_mod_power > 0 and len(tremor_indices) > 0:
#                         results["h3_7_f1_modulation_power_4_8hz"] = float(np.sum(fft_power[tremor_indices]) / total_mod_power)

#         if len(q_values) > 0:
#             results["h3_4_q_factor"] = float(np.mean(q_values))
#             results["h3_5_q_factor"] = float(np.std(q_values))
            
#         if len(f1_prominences_linear) > 0:
#             results["h3_6_peak_prominence_db"] = float(10 * np.log10(np.mean(f1_prominences_linear)))
            
#         results["h3_8_f1_trajectory_roughness_hz_per_sec"] = float(np.mean(np.abs(np.diff(raw_f1_freqs)) / np.diff(raw_times)))
            
#     except Exception as e:
#         print(f"⚠️ Exception inside compute_micro_modulations: {e}")
        
#     return results

def compute_micro_modulations(file_path: str) -> Dict[str, float]:
    """
    Hypothesis 3 (8-Feature Set): Computes true integrated energy, 
    normalized energy ratios, band prominence, Q-factor central tendency,
    and now Q-factor variability (Std) across the voiced timeline.
    """
    results = {
        "h3_1_f1_band_energy_db": np.nan,
        "h3_2_normalized_f1_energy": np.nan,
        "h3_3_cv_of_f1_energy": np.nan,
        "h3_4_q_factor_mean": np.nan,
        "h3_5_q_factor_std": np.nan,  # ◄ New 8th Feature
        "h3_6_peak_prominence_db": np.nan,
        "h3_7_time_domain_envelope_modulation_depth": np.nan,
        "h3_8_f1_trajectory_rms_roughness_hz": np.nan
    }
    
    try:
        sound = parselmouth.Sound(file_path)
        pitch = call(sound, "To Pitch", 0.0, 75.0, 600.0)
        formant_object = call(sound, "To Formant (burg)", 0.0, 5, 5000.0, 0.025, 50.0)
        spectrogram = call(sound, "To Spectrogram", 0.005, 5000.0, 0.002, 20.0, "Gaussian")
        spec_matrix = call(spectrogram, "To Matrix")
        
        grid_data = spec_matrix.as_array()
        num_rows, num_cols = grid_data.shape
        dx, dy = spec_matrix.dx, spec_matrix.dy
        x1, y1 = spec_matrix.x1, spec_matrix.y1
        
        num_frames = call(formant_object, "Get number of frames")
        
        raw_times = []
        raw_f1_freqs = []
        raw_f1_bws = []
        
        for frame in range(1, num_frames + 1):
            time = call(formant_object, "Get time from frame number", frame)
            f0 = call(pitch, "Get value at time", time, "Hertz", "Linear")
            
            if not np.isnan(f0) and f0 > 0:
                f1_freq = call(formant_object, "Get value at time", 1, time, "Hertz", "Linear")
                f1_bw = call(formant_object, "Get bandwidth at time", 1, time, "Hertz", "Linear")
                if not np.isnan(f1_freq) and f1_freq > 0:
                    raw_times.append(time)
                    raw_f1_freqs.append(f1_freq)
                    raw_f1_bws.append(f1_bw if (not np.isnan(f1_bw) and f1_bw > 0) else np.nan)
                    
        n_voiced = len(raw_f1_freqs)
        if n_voiced <= 1:
            return results
            
        # Bandwidth linear interpolation track
        raw_f1_bws = np.array(raw_f1_bws)
        nans = np.isnan(raw_f1_bws)
        if np.all(nans):
            filled_bws = np.full_like(raw_f1_bws, 150.0)
        elif np.any(nans):
            x_indices = np.arange(len(raw_f1_bws))
            raw_f1_bws[nans] = np.interp(x_indices[nans], x_indices[~nans], raw_f1_bws[~nans])
            filled_bws = raw_f1_bws
        else:
            filled_bws = raw_f1_bws
            
        smoothed_bws = medfilt(filled_bws, kernel_size=5 if len(filled_bws) >= 5 else 3)
        
        q_values = []
        f1_prominences_linear = []
        f1_band_integrated_energies = []
        total_frame_integrated_energies = []
        
        for i in range(n_voiced):
            time = raw_times[i]
            f1_freq = raw_f1_freqs[i]
            f1_bw = smoothed_bws[i]
            
            q_values.append(f1_freq / f1_bw)
            
            effective_half_width = np.clip(f1_bw / 2.0, 75.0, 200.0)
            f1_low = max(0.0, f1_freq - effective_half_width)
            f1_high = f1_freq + effective_half_width
            
            col_idx = np.clip(int(round((time - x1) / dx)), 0, num_cols - 1)
            row_low_idx = np.clip(int(round((f1_low - y1) / dy)), 0, num_rows - 1)
            row_high_idx = np.clip(int(round((f1_high - y1) / dy)) + 1, row_low_idx + 1, num_rows)
            
            band_slice = grid_data[row_low_idx:row_high_idx, col_idx]
            total_high_idx = np.clip(int(round((5000.0 - y1) / dy)) + 1, 1, num_rows)
            total_slice = grid_data[0:total_high_idx, col_idx]
            
            band_bandwidth_hz = len(band_slice) * dy
            total_bandwidth_hz = len(total_slice) * dy
            
            f1_integrated_energy = np.mean(band_slice) * band_bandwidth_hz
            total_integrated_energy = np.mean(total_slice) * total_bandwidth_hz
            
            if f1_integrated_energy > 0 and total_integrated_energy > 0:
                f1_band_integrated_energies.append(f1_integrated_energy)
                total_frame_integrated_energies.append(total_integrated_energy)
                
                # Integrated Band-to-Band Prominence
                row_span = row_high_idx - row_low_idx
                n_low_start = np.clip(row_low_idx - row_span, 0, num_rows - 1)
                n_high_end = np.clip(row_high_idx + row_span, row_high_idx + 1, num_rows)
                
                neighbor_low_slice = grid_data[n_low_start:row_low_idx, col_idx]
                neighbor_high_slice = grid_data[row_high_idx:n_high_end, col_idx]
                
                low_neighbor_energy = np.mean(neighbor_low_slice) * (len(neighbor_low_slice) * dy)
                high_neighbor_energy = np.mean(neighbor_high_slice) * (len(neighbor_high_slice) * dy)
                
                avg_neighbor_energy = (low_neighbor_energy + high_neighbor_energy) / 2.0
                
                if avg_neighbor_energy > 0:
                    f1_prominences_linear.append(f1_integrated_energy / avg_neighbor_energy)
                        
        # 4. Compute Final Descriptive Matrix Summary
        if len(f1_band_integrated_energies) > 0:
            results["h3_1_f1_band_energy_db"] = float(10 * np.log10(np.mean(f1_band_integrated_energies)))
            results["h3_3_cv_of_f1_energy"] = float(np.std(f1_band_integrated_energies) / np.mean(f1_band_integrated_energies))
            
            sum_f1 = np.sum(f1_band_integrated_energies)
            sum_total = np.sum(total_frame_integrated_energies)
            if sum_total > 0:
                results["h3_2_normalized_f1_energy"] = float(10 * np.log10(sum_f1 / sum_total))
                
            max_envelope = np.max(f1_band_integrated_energies)
            min_envelope = np.min(f1_band_integrated_energies)
            if (max_envelope + min_envelope) > 0:
                results["h3_7_time_domain_envelope_modulation_depth"] = float((max_envelope - min_envelope) / (max_envelope + min_envelope))
                
        if len(q_values) > 0:
            # Map both mean tracking and standard deviation tracking metrics cleanly
            results["h3_4_q_factor_mean"] = float(np.mean(q_values))
            results["h3_5_q_factor_std"] = float(np.std(q_values))
            
        if len(f1_prominences_linear) > 0:
            results["h3_6_peak_prominence_db"] = float(10 * np.log10(np.mean(f1_prominences_linear)))
            
        if len(raw_f1_freqs) > 1:
            results["h3_8_f1_trajectory_rms_roughness_hz"] = float(np.sqrt(np.mean(np.diff(raw_f1_freqs) ** 2)))
            
    except Exception as e:
        print(f"⚠️ Exception inside compute_micro_modulations: {e}")
        
    return results