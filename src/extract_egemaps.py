import os
import re
from pathlib import Path
import pandas as pd
import opensmile

def extract_egemaps_dataset(audio_dir: str, output_csv: str):
    """
    Automates eGeMAPS feature extraction across an entire folder of 300ms audio 
    snippets using openSMILE. Automatically parses filenames for CoVox metadata tags.
    """
    audio_path = Path(audio_dir)
    audio_files = list(audio_path.glob("*.wav"))
    
    if not audio_files:
        print(f"⚠️ No .wav files found inside: {audio_dir}")
        return
        
    print(f"📦 Found {len(audio_files)} samples. Initializing openSMILE eGeMAPS engine...")
    
    # Configure openSMILE engine for functional-level eGeMAPS outputs
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    
    master_records = []
    

    
    for file_path in audio_files:
        try:
            # 1. Run openSMILE Extraction
            # Returns a pandas DataFrame with multi-index (file, start_time, end_time)
            
            features_df = smile.process_file(file_path)
            
            # Convert single-row dataframe to a flat dictionary matrix
            feature_dict = features_df.iloc[0].to_dict()
            
            # 2. Parse Filename Metadata
            filename = os.path.basename(file_path)
            name_parts = filename.replace(".wav", "").split("_")
            singer_id = name_parts[0]
            vowel = name_parts[2]
            condition = "resonant" if "resonant" in filename and "non-resonant" not in filename else "non-resonant"
            trial = name_parts[-1]
            
            # 3. Assemble Record Frame Row
            meta_record = {
                "filename": filename,
                "singer_id": singer_id,
                "condition": condition,
                "vowel": vowel,
                "trial": trial,
            }
            
            # Merge metadata dictionary into openSMILE features dictionary
            meta_record.update(feature_dict)
            master_records.append(meta_record)
            
        except Exception as e:
            print(f"⚠️ Failed to extract openSMILE features for {file_path.name}: {e}")
            
    # 4. Export Master Matrix
    output_df = pd.DataFrame(master_records)
    output_df.to_csv(output_csv, index=False)
    print(f"✨ Success! Saved eGeMAPS Feature Matrix to: {output_csv}")

if __name__ == "__main__":

    # Resolve project root relative paths cleanly.
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, "data", "processed")
    outputs_dir = os.path.join(base_dir, "outputs", "egemaps_features_master.csv")


    
    extract_egemaps_dataset(processed_dir, outputs_dir)