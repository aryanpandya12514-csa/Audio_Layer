# 🎖️ Multi-Factor Ensemble Deepfake Audio Detector

An advanced, CPU-optimized **Audio Forensics & Synthetic Voice Detection Engine**. This system combines deep learning, spectral physics analysis, and biological pitch tremor tracking to detect deepfake audio in media files with high accuracy and explainable evidence.

---

## 📌 Overview

Detecting AI-generated audio is challenging when relying on a single detection method. **Multi-Factor Ensemble Deepfake Audio Detector** solves this by employing a 3-judge courtroom panel architecture, backed by dual emergency override mechanisms for localized temporal bursts and hard spectral cutoffs.

### 🏛️ The 3-Judge Panel

```
                              ┌────────────────────────┐
                              │    Input Media File    │
                              └───────────┬────────────┘
                                          │
                   ┌──────────────────────┼──────────────────────┐
                   │ (16 kHz Track)       │ (44.1 kHz Track)     │ (16 kHz Track)
                   ▼                      ▼                      ▼
         ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
         │ 🧠 Neural Judge  │   │ 📡 Physics Judge │   │ 🧬 Biology Judge │
         │  (Weight: 40%)   │   │  (Weight: 30%)   │   │  (Weight: 30%)   │
         └─────────┬────────┘   └─────────┬────────┘   └─────────┬────────┘
                   │                      │                      │
                   └──────────────────────┼──────────────────────┘
                                          │
                                          ▼
                               ┌─────────────────────┐
                               │  Weighted Ensemble  │
                               └──────────┬──────────┘
                                          │
                      ┌───────────────────┴───────────────────┐
                      ▼                                       ▼
         ┌─────────────────────────┐             ┌─────────────────────────┐
         │ 💥 Temporal Burst Check │             │ 📡 Physics Override Check│
         └────────────┬────────────┘             └────────────┬────────────┘
                      │                                       │
                      └───────────────────┬───────────────────┘
                                          ▼
                             ┌─────────────────────────┐
                             │  Final Verdict Dashboard│
                             └─────────────────────────┘
```

1. **🧠 Judge 1 — Neural Judge (40% Weight): "The Deep Learning Bloodhound"**
   - Utilizes fine-tuned **WavLM** embeddings (`WavLMFakeDetector`) to analyze latent micro-structural waveform features.
   - Slices audio into 3-second sliding windows with 1-second step overlaps to evaluate localized confidence.
2. **📡 Judge 2 — Physics Judge (30% Weight): "The Spectrogram Surgeon"**
   - Performs High-Resolution Short-Time Fourier Transform (STFT) power spectrum analysis at 44.1 kHz.
   - Sniffs out the **"8 kHz Cliff"**—a classic fingerprint of Text-to-Speech (TTS) models trained at 16 kHz sample rates, which lack energy above 8 kHz.
3. **🧬 Judge 3 — Biology Judge (30% Weight): "The Vocal Tremor Analyst"**
   - Uses the YIN algorithm to extract fundamental pitch ($F_0$) tracks from voiced audio frames.
   - Measures biological micro-irregularity: **Pitch Coefficient of Variation (CV)** and **Local Jitter**. Synthetic voices appear unnaturally smooth or robotic compared to biological vocal folds.

### ⚡ Dual Emergency Overrides
- **Temporal Burst Override**: Triggers a `DEEPFAKE` verdict if $\ge 25\%$ of individual sliding windows register high synthetic confidence ($\ge 70\%$), preventing long real intros from diluting localized deepfake snippets.
- **Physics Override**: Forces a `DEEPFAKE` verdict if the Physics Judge detects an extreme score ($\ge 0.98$) alongside an abrupt $8\text{ kHz}$ spectral drop-off ratio.

---

## ⚙️ Key Features

- 💻 **CPU-Optimized Efficiency**: Designed to execute comfortably on standard consumer CPUs (e.g., 12th Gen Intel Core i5) without requiring discrete GPU acceleration.
- 🎬 **Multi-Format Extraction**: Dual-rate FFmpeg audio extractor handles `.mp4`, `.mkv`, and `.avi` inputs, resampling automatically to 16 kHz and 44.1 kHz.
- 📊 **Visual Verdict Dashboard**: Formats clear, multi-judge diagnostic evidence, energy distributions, pitch statistics, and sub-vote scores in ASCII tables.
- 📁 **Batch & Single-File Processing**: Provides utilities for dataset pre-processing as well as single-file analysis pipelines.

---

## 📂 Project Structure

```
Audio_Layer/
├── Data/                       # Raw input video and extracted audio data
├── models/                     # Trained model checkpoints (e.g. Brain_V5_WavLM.pth)
├── src/
│   ├── audio_extraction.py     # FFmpeg-backed audio extraction utilities
│   ├── data_loader.py          # PyTorch Dataset & DataLoader implementations
│   └── model_classifier.py     # WavLMFakeDetector architecture based on WavLM
├── main.py                     # Standalone Audio Extraction Pre-Processor CLI
├── multi_factor_evaluator.py   # Multi-Factor Ensemble Detector & Evaluator Engine
├── run_full_pipeline.py        # Configurable entry-point script for running single-file analysis
├── trainer.py                  # PyTorch model training script
├── Roadmap                     # Project development roadmap & phase tracking
└── .gitignore                  # Git ignore rules for cached & dataset files
```

---

## 🚀 Getting Started

### Prerequisites

Make sure **FFmpeg** is installed on your system:

- **Linux (Ubuntu/Debian)**: `sudo apt update && sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Download binaries via [ffmpeg.org](https://ffmpeg.org/) and add them to system `PATH`.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Audio_Layer.git
   cd Audio_Layer
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install required Python packages:
   ```bash
   pip install torch torchaudio transformers librosa scipy numpy soundfile tqdm
   ```

---

## 💻 Usage

### 1. Run Ensemble Detection on a Video File

#### Option A: Quick Script (`run_full_pipeline.py`)
Edit the `VIDEO_PATH` variable inside `run_full_pipeline.py`, then run:
```bash
python run_full_pipeline.py
```

#### Option B: Direct CLI (`multi_factor_evaluator.py`)
Run evaluation on any video file directly:
```bash
python multi_factor_evaluator.py --video-path Data/Raw_video/sample_video.mp4
```

*Optional CLI Flags:*
- `--weights-path`: Path to custom model weights (default: `models/Brain_V5_WavLM.pth`).
- `--threshold`: Custom verdict decision threshold (default: `0.50`).
- `--no-neural`: Skip Neural Judge for ablation testing.
- `--no-physics`: Skip Physics Judge.
- `--no-biology`: Skip Biology Judge.

---

### 2. Audio Extraction Pre-Processor (`main.py`)

Extract clean `.wav` files from raw video clips.

- **Single File**:
  ```bash
  python main.py --file path/to/video.mp4 --output path/to/output.wav
  ```
- **Batch Processing**: Place raw videos in `Data/Raw_video/` and execute:
  ```bash
  python main.py
  ```

---

### 3. Model Training (`trainer.py`)

To train or fine-tune the WavLM classifier on your custom dataset:
```bash
python trainer.py
```

---

## 📊 Sample Output Dashboard

```text
================================================================================
                    🎖️  MULTI-FACTOR ENSEMBLE DEEPFAKE VERDICT                  
================================================================================
  Target : sample_video.mp4
  Weights: Neural 40%  │  Physics 30%  │  Biology 30%
--------------------------------------------------------------------------------

  JUDGE 1 ─ NEURAL  (Weight: 40%)
  "The Deep Learning Bloodhound"

  Score : 0.8421  [████████████████████░░░░]  DEEPFAKE ⚠️

  ┌─ Evidence ────────────────────────────────────────────────────┐
  │ Windows analysed : 8
  │ Fake sub-votes   : 7/8 (88% of windows)
  │ Sub-vote scores  : [0.8123, 0.8941, 0.9102, 0.8415, 0.7910]
  └───────────────────────────────────────────────────────────────┘

--------------------------------------------------------------------------------

  JUDGE 2 ─ PHYSICS  (Weight: 30%)
  "The Spectrogram Surgeon"

  Score : 0.9520  [█████████████████████░░░]  DEEPFAKE ⚠️

  ┌─ Evidence ────────────────────────────────────────────────────┐
  │ HF energy ratio (>8 kHz) : 0.120%
  │   (Real baseline ≥ 4.0% — below → suspicious)
  │ Low-band energy  (<8 kHz) : 42.10 dB
  │ High-band energy (>8 kHz) : 12.30 dB
  │ 8 kHz cliff ratio         : 0.0412
  │   → Hard cliff detected (synthetic-like)
  └───────────────────────────────────────────────────────────────┘

================================================================================
  🚨  FINAL VERDICT:  DEEPFAKE  (High Penalty)
      Score 0.8914 exceeds threshold 0.50 by +0.3914
================================================================================
```

---

## 🛠️ Roadmap & Future Enhancements

- [x] Phase 1: FFmpeg Audio Extraction & Resampling
- [x] Phase 2: Librosa Spectral Feature Analysis & 8 kHz Cliff Detection
- [x] Phase 3: YIN Fundamental Pitch Tracking & Biological Jitter Analysis
- [x] Phase 4: WavLM Deep Learning Feature Extractor & Classification Head
- [x] Phase 5: Multi-Factor Ensemble Voting Panel & Temporal Overrides
- [ ] Phase 6: Automatic Speech Recognition (Whisper ASR Integration)
- [ ] Phase 7: NLP Fact-Checking & Semantic Logic Consistency Verification

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
