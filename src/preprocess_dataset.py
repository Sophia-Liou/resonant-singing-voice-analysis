import os
import glob
import csv
import numpy as np
import librosa
import soundfile as sf

def extract_stable_vowel_with_timestamps(y, sr, window_duration=0.3, frame_length=2048, hop_length=512):
    """
    Locates the peak vocal energy, skips the consonant transition, 
    and returns the audio segment along with its start and end sample indices.
    """
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    peak_frame = np.argmax(rms)
    peak_sample = librosa.frames_to_samples(peak_frame, hop_length=hop_length)
    
    # 30ms safety buffer
    safety_offset = int(0.03 * sr) 
    start_sample = peak_sample + safety_offset
    
    duration_samples = int(window_duration * sr)
    end_sample = start_sample + duration_samples
    
    # Fallback Boundary Guardrail
    if end_sample > len(y):
        end_sample = len(y)
        start_sample = max(0, end_sample - duration_samples)
        
    stable_segment = y[start_sample:end_sample]
    
    return stable_segment, start_sample, end_sample


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")
    outputs_dir = os.path.join(base_dir, "outputs")
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    raw_files = glob.glob(os.path.join(raw_dir, "*.wav"))
    print(f"🚀 Found {len(raw_files)} raw audio samples. Starting preprocessing...")
    
    # Initialize our log storage
    audit_log = []
    
    for file_path in raw_files:
        original_filename = os.path.basename(file_path)
        # In our pipeline, the new filename remains the same but lives in 'processed/'
        new_filename = original_filename 
        
        try:
            y, sr = librosa.load(file_path, sr=16000)
            y_norm = y / np.max(np.abs(y))
            
            # Extract segment and capture raw sample boundaries
            y_stable, start_samp, end_samp = extract_stable_vowel_with_timestamps(y_norm, sr)
            
            # Convert sample indices to absolute timestamps (seconds)
            start_time_sec = round(start_samp / sr, 4)
            end_time_sec = round(end_samp / sr, 4)
            
            # Export the audio snippet
            output_path = os.path.join(processed_dir, new_filename)
            sf.write(output_path, y_stable, sr, subtype='PCM_16')
            
            # Append rows to our log dictionary
            audit_log.append({
                "original_filename": original_filename,
                "new_filename": new_filename,
                "start_time_seconds": start_time_sec,
                "end_time_seconds": end_time_sec
            })
            
        except Exception as e:
            print(f"⚠️ Failed to process file {original_filename}: {str(e)}")
            
    # --- Export Audit Log to Excel-Compatible CSV ---
    csv_path = os.path.join(outputs_dir, "preprocessing_audit_log.csv")
    csv_columns = ["original_filename", "new_filename", "start_time_seconds", "end_time_seconds"]
    
    try:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
            writer.writeheader()
            for data in audit_log:
                writer.writerow(data)
        print(f"\n📊 Audit log successfully saved for Excel inspection!")
        print(f"📂 Location: {csv_path}")
    except Exception as e:
        print(f"⚠️ Failed to write audit log spreadsheet: {str(e)}")


if __name__ == "__main__":
    main()