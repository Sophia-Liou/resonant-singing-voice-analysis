# Resonant Singing Voice Analysis

An acoustic and data-driven investigation into the properties of resonant singing voices using traditional digital signal processing (DSP) and self-supervised learning (SSL) audio representations.

## 📊 Project Overview
This project evaluates three core hypotheses regarding vocal tract resonance in singing:
1. **Spectral Proximity**: Analyzing the $F_0 - F_1$ distance.
2. **Formant Clustering**: Quantifying energy concentration in the 2000–4000 Hz band.
3. **Micro-modulations**: Detecting rapid frequency and amplitude fluctuations in the low-pitch range.

## 📂 Data & OSF Repository
The accompanying dataset consists of 48 syllable singing audio recordings from 12 singers, partitioned into resonant and non-resonant pairs. 
* **Acoustic Data & DOI**: [Link to your OSF Project Here]
* *Note: Raw audio files are omitted from this repository via `.gitignore` to maintain participant privacy.*

## 🛠️ Installation & Requirements
```bash
pip install -r requirements.txt