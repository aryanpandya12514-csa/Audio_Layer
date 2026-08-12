"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       🎖️  MULTI-FACTOR ENSEMBLE DEEPFAKE AUDIO DETECTOR  v1.1              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Architecture: 3-Judge Weighted Panel + Temporal Burst Override              ║
║                                                                              ║
║  Think of this as a courtroom trial. The accused is the audio track.         ║
║  Three expert witnesses vote with weighted authority. A fourth rule acts     ║
║  as an Emergency Override — triggered by burst patterns of high-confidence   ║
║  synthetic windows even when the mean score stays below the threshold.       ║
║                                                                              ║
║  ┌────────────────────────────────────────────────────────────────────────┐ ║
║  │  Judge 1 ─ NEURAL   (40%): "The Deep Learning Bloodhound"             │ ║
║  │  Judge 2 ─ PHYSICS  (30%): "The Spectrogram Surgeon"                  │ ║
║  │  Judge 3 ─ BIOLOGY  (30%): "The Vocal Tremor Analyst"                 │ ║
║  │  Override ─ BURST        : "The Temporal Spike Detector"              │ ║
║  └────────────────────────────────────────────────────────────────────────┘ ║
║                                                                              ║
║  FINAL VERDICT:                                                              ║
║    Primary : Score = (Neural×0.40)+(Physics×0.30)+(Biology×0.30) > 0.50    ║
║    Override: burst_ratio ≥ 25% of windows AND neural_avg ≥ 0.20            ║
║    Either condition alone → 🚨 DEEPFAKE                                      ║
║                                                                              ║
║  HARDWARE:  CPU-only. Lenovo IdeaPad (12th Gen i5). No GPU needed.          ║
║                                                                              ║
║  USAGE:                                                                      ║
║    python multi_factor_evaluator.py --video-path Data/Raw_video/video_8.mp4 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Dependencies (install inside your venv if missing):
    pip install librosa soundfile
    (torch, scipy, numpy, transformers already required by existing pipeline)
"""

import argparse
import math
import os
import shutil
import tempfile

import numpy as np
import torch
from scipy import signal
from scipy.io import wavfile

# ── Project-local imports ──────────────────────────────────────────────────────
from src.audio_extraction import extract_audio_wav
from src.model_classifier import WavLMFakeDetector

# ── Suppress noisy logs ────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# WavLM was trained on 16 kHz audio → neural judge always works at 16 kHz
SR_NEURAL = 16_000

# Physics judge needs high sample-rate audio to *see* frequencies above 8 kHz.
# At 16 kHz, the Nyquist limit IS 8 kHz, so any TTS hard-cutoff is invisible.
# At 44.1 kHz, the ceiling is 22.05 kHz — the cliff becomes clearly visible.
SR_PHYSICS = 44_100

# Sliding window configuration for neural judge
WINDOW_SEC = 3          # Each window is 3 seconds of audio
STEP_SEC   = 1          # Step between windows is 1 second
WINDOW_SAMPLES = SR_NEURAL * WINDOW_SEC   # 48,000 samples per window
STEP_SAMPLES   = SR_NEURAL * STEP_SEC     # 16,000 samples per step

# ── Judge vote weights (must sum to 1.0) ──────────────────────────────────────
WEIGHT_NEURAL  = 0.40   # 40% — neural pattern recognition
WEIGHT_PHYSICS = 0.30   # 30% — spectral physics
WEIGHT_BIOLOGY = 0.30   # 30% — biological vocal variation

# ── Physics Judge calibration ─────────────────────────────────────────────────
# What fraction of total audio energy lives *above* 8 kHz in real speech?
# Real voice at 44.1 kHz SR: ~4–15%  (sibilants, fricatives, breath, harmonics)
# TTS model at 16 kHz SR:    ~0.0%   (Nyquist = 8 kHz → digital silence above)
# If the ratio is BELOW this baseline, the Physics Judge votes DEEPFAKE.
HF_CUTOFF_HZ       = 8_000.0   # The TTS hard ceiling frequency
HF_REAL_BASELINE   = 0.04      # 4% — anything less triggers suspicion

# ── Biology Judge calibration ─────────────────────────────────────────────────
# Human pitch (F0) contains natural micro-tremors from muscle tension.
# Coefficient of Variation (CV) = std / mean of the pitch track.
# Real voice:      CV ≈ 10–25%,  jitter ≈ 1–5%
# Synthetic voice: CV ≈ 2–6%,    jitter ≈ 0.1–0.5%
# If variation falls BELOW these baselines, Biology Judge votes DEEPFAKE.
FMIN_HZ              = 60.0    # Lowest human vocal fundamental (bass)
FMAX_HZ              = 400.0   # Highest human vocal fundamental (soprano)
CV_REAL_BASELINE     = 0.10    # 10% CV — below this is suspiciously smooth
JITTER_REAL_BASELINE = 0.010   # 1%  jitter — below this is robotically perfect

# ── Final verdict threshold ───────────────────────────────────────────────────
VERDICT_THRESHOLD = 0.50        # Cross this line → DEEPFAKE

# ── Temporal Burst Override calibration ──────────────────────────────────────
# Problem solved: a deepfake video with intermittent synthetic segments will
# spike WavLM scores in bursts but keep the *mean* below threshold, causing
# a false negative. This override catches that pattern independently.
#
# A window is a "high spike" if its deepfake score exceeds this:
BURST_HIGH_SPIKE_THRESHOLD = 0.70   # 70% — unmistakably synthetic window
#
# If this fraction of all windows are high spikes → override fires:
BURST_RATIO_THRESHOLD      = 0.25   # 25% — more than 1-in-4 windows spiking
#
# Safety guard: neural average must also be at least this (prevents false
# triggers on very short clips with 1 window happening to spike randomly):
BURST_NEURAL_AVG_MIN       = 0.20   # Neural avg ≥ 20% required for override


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO LOADING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_wav_as_float32(wav_path: str) -> tuple[np.ndarray, int]:
    """
    Read a WAV file and return a float32 mono array + sample rate.
    Handles both integer PCM (int16/int32) and float WAV files.
    """
    sr, audio = wavfile.read(wav_path)

    # Convert stereo → mono by averaging channels
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Convert integer PCM to [-1.0, 1.0] float32
    if np.issubdtype(audio.dtype, np.integer):
        max_val = float(np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / max_val
    else:
        audio = audio.astype(np.float32)

    return audio, sr


def extract_audio_two_rates(video_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract audio from video at TWO sample rates in one pass:
      - 16 kHz  → for Neural Judge + Biology Judge
      - 44.1 kHz → for Physics Judge (must see above-8kHz frequencies)

    Uses ffmpeg via the existing src.audio_extraction module.
    Both extractions land in a temp directory that is cleaned up automatically.
    """
    tmp_dir = tempfile.mkdtemp(prefix="mfe_audio_")
    try:
        wav_16k_path   = os.path.join(tmp_dir, "audio_16k.wav")
        wav_441k_path  = os.path.join(tmp_dir, "audio_44k.wav")

        print(f"   🎞️  Extracting 16 kHz audio (Neural + Biology)…")
        extract_audio_wav(video_path, wav_16k_path,  sample_rate=SR_NEURAL,  channels=1)

        print(f"   🎞️  Extracting 44.1 kHz audio (Physics)…")
        extract_audio_wav(video_path, wav_441k_path, sample_rate=SR_PHYSICS, channels=1)

        audio_16k,  _ = load_wav_as_float32(wav_16k_path)
        audio_44k,  _ = load_wav_as_float32(wav_441k_path)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return audio_16k, audio_44k


# ══════════════════════════════════════════════════════════════════════════════
# JUDGE 1 — THE NEURAL JUDGE  (Weight: 40%)
# "The Deep Learning Bloodhound"
# ══════════════════════════════════════════════════════════════════════════════
#
#  This judge spent years training on thousands of real and fake voices.
#  It recognises the *latent patterns* in WavLM's 768-dim embedding space —
#  statistical signatures that no human ear can hear but that WavLM can smell
#  like a bloodhound: phase incoherence, harmonic periodicity glitches,
#  formant transition smoothness, and more.
#
#  Voting process:
#    1. Pre-process audio (bandpass → peak-normalise)
#    2. Slice into 3-second windows (with 1-second overlap step)
#    3. Each window independently casts a sub-vote (deepfake probability)
#    4. The judge's final vote = mean of all sub-votes
# ─────────────────────────────────────────────────────────────────────────────

def _bandpass_filter(audio: np.ndarray, sr: int = SR_NEURAL) -> np.ndarray:
    """
    4th-order Butterworth bandpass (80 Hz – 7500 Hz).
    Removes AC-hum, microphone rumble (<80 Hz) and tape-hiss/aliasing (>7500 Hz)
    while leaving all vocal-tract markers intact for WavLM.
    """
    nyquist   = sr / 2.0
    low_norm  = max(0.0001, 80.0   / nyquist)
    high_norm = min(0.9999, 7500.0 / nyquist)
    high_norm = max(low_norm + 0.0001, high_norm)

    sos = signal.butter(4, [low_norm, high_norm], btype="band", output="sos")
    return signal.sosfilt(sos, audio).astype(np.float32)


def _peak_normalise(audio: np.ndarray, target: float = 0.90) -> np.ndarray:
    """
    Stretch audio so the absolute peak equals `target` (default 0.9).
    Equalises loud vs. quiet recordings before feeding the model.
    """
    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 1.0
    if peak <= 0.0:
        return audio.astype(np.float32)
    return np.clip((audio / peak) * target, -1.0, 1.0).astype(np.float32)


def _make_sliding_windows(audio: np.ndarray) -> list:
    """
    Slice the audio ribbon into 3-second windows with a 1-second step.
    Short clips (<3 s) are zero-padded to one full window.
    Each window is a separate sub-vote for the Neural Judge.
    """
    if len(audio) < WINDOW_SAMPLES:
        padded = np.pad(audio, (0, WINDOW_SAMPLES - len(audio)), mode="constant")
        return [padded.astype(np.float32)]

    windows = []
    starts  = list(range(0, len(audio) - WINDOW_SAMPLES + 1, STEP_SAMPLES))
    # Always include the final tail window so no audio is left unanalysed
    final_start = len(audio) - WINDOW_SAMPLES
    if not starts or starts[-1] != final_start:
        starts.append(final_start)

    for s in starts:
        chunk = audio[s : s + WINDOW_SAMPLES]
        if len(chunk) < WINDOW_SAMPLES:
            chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)), mode="constant")
        windows.append(chunk.astype(np.float32))

    return windows


def load_neural_model(weights_path: str) -> WavLMFakeDetector:
    """
    Load WavLMFakeDetector onto CPU only.
    map_location=cpu ensures no GPU calls are made — safe for IdeaPad.
    """
    model      = WavLMFakeDetector()
    checkpoint = torch.load(weights_path, map_location=torch.device("cpu"))

    # Handle training checkpoints that wrap the state_dict in a top-level dict
    state_dict = checkpoint.get("state_dict", checkpoint) \
                 if isinstance(checkpoint, dict) else checkpoint

    # Strip DataParallel "module." prefix if present
    if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def neural_judge(audio_16k: np.ndarray, model: WavLMFakeDetector) -> tuple:
    """
    ┌──────────────────────────────────────────────────────────────┐
    │  JUDGE 1 — NEURAL  (Weight: 40%)                            │
    │  "The Deep Learning Bloodhound"                              │
    └──────────────────────────────────────────────────────────────┘

    The Bloodhound sniffs the WavLM embedding space for synthetic patterns.
    It cannot be fooled by surface-level tricks (loudness, pitch shifting)
    because WavLM encodes *micro-structural* features of the waveform.

    Returns:
        penalty (float): 0.0 (confident REAL) → 1.0 (confident DEEPFAKE)
        meta    (dict):  diagnostic breakdown for the dashboard
    """
    # Pre-process: scalpel away noise bands, then equalise amplitude
    audio = _bandpass_filter(audio_16k, sr=SR_NEURAL)
    audio = _peak_normalise(audio)

    # Each window is an independent sub-vote
    windows = _make_sliding_windows(audio)
    scores  = []

    with torch.no_grad():
        for i, window in enumerate(windows, start=1):
            tensor = torch.from_numpy(window).unsqueeze(0)   # Shape: [1, T]
            prob   = float(model(tensor).squeeze().item())   # Sigmoid output ∈ [0,1]
            scores.append(prob)
            print(f"      Sub-vote {i:02d}/{len(windows):02d} │ Deepfake score: {prob:.4f}")

    penalty       = float(np.mean(scores)) if scores else 0.5
    fake_windows  = int(np.sum(np.array(scores) > 0.5))

    meta = {
        "windows"       : len(windows),
        "window_scores" : [round(s, 4) for s in scores],
        "fake_windows"  : fake_windows,
        "fake_ratio"    : round(fake_windows / len(scores), 4) if scores else 0.0,
    }
    return penalty, meta


# ══════════════════════════════════════════════════════════════════════════════
# JUDGE 2 — THE PHYSICS JUDGE  (Weight: 30%)
# "The Spectrogram Surgeon"
# ══════════════════════════════════════════════════════════════════════════════
#
#  This judge studies the STFT spectrogram like an X-ray — looking for
#  unnatural frequency patterns that reveal synthetic origin.
#
#  The Key Insight — The "8 kHz Cliff":
#    TTS models are almost universally trained/sampled at 16 kHz.
#    The Nyquist theorem says: max representable frequency = SR / 2 = 8 kHz.
#    So TTS audio has ZERO energy above 8 kHz. Not 'very little'. ZERO.
#
#    Real human speech, captured at 44.1 kHz, naturally contains energy
#    above 8 kHz from:
#      • Sibilant consonants (/s/, /z/, /sh/) — peaks at 4–14 kHz
#      • Voiceless fricatives (/f/, /th/)     — broadband 5–15 kHz
#      • Dental/alveolar clicks and breath    — up to 20 kHz
#      • Room tone and microphone characteristics
#
#  The Surgeon's scalpel: split the spectrum at 8 kHz.
#  If the high-frequency partition is *suspiciously silent*, vote DEEPFAKE.
# ─────────────────────────────────────────────────────────────────────────────

def physics_judge(audio_44k: np.ndarray, sr: int = SR_PHYSICS) -> tuple:
    """
    ┌──────────────────────────────────────────────────────────────┐
    │  JUDGE 2 — PHYSICS  (Weight: 30%)                           │
    │  "The Spectrogram Surgeon"                                   │
    └──────────────────────────────────────────────────────────────┘

    Investigates whether the 8 kHz frequency cliff exists — a forensic
    hallmark of TTS-synthesised audio. Uses STFT power spectral analysis.

    Returns:
        penalty (float): 0.0 (confident REAL) → 1.0 (confident DEEPFAKE)
        meta    (dict):  diagnostic breakdown for the dashboard
    """
    try:
        import librosa
    except ImportError:
        print("      ⚠️  librosa not installed. Physics Judge returns neutral 0.5.")
        print("          Install with: pip install librosa soundfile")
        return 0.5, {"error": "librosa not installed"}

    # ── Build the power spectrogram ──────────────────────────────────────────
    # n_fft=4096 gives ~10 Hz per bin at 44.1 kHz — surgical frequency resolution
    n_fft   = 4096
    D       = np.abs(librosa.stft(audio_44k, n_fft=n_fft))   # [freq_bins, time_frames]
    D_power = D ** 2    # Work in power domain: physically meaningful (W/Hz)

    # Map each frequency bin to its centre frequency in Hz
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)      # [freq_bins]

    # ── Split the courtroom evidence at the 8 kHz boundary ──────────────────
    low_mask  = freqs <= HF_CUTOFF_HZ   # 0 → 8 kHz  (TTS lives here)
    high_mask = freqs >  HF_CUTOFF_HZ  # 8 kHz → Nyquist (real voices have energy here)

    # Total power in each band, summed across all time frames
    low_energy  = float(np.sum(D_power[low_mask,  :]))
    high_energy = float(np.sum(D_power[high_mask, :]))
    total_energy = low_energy + high_energy

    if total_energy < 1e-10:
        # Silence — cannot make a judgment
        return 0.5, {"error": "Audio appears silent — unable to analyse spectrum"}

    # ── HF Ratio: The primary evidence metric ────────────────────────────────
    # What fraction of all audio energy lives *above* 8 kHz?
    # Real speech at 44.1 kHz: typically 4%–15%
    # TTS at 16 kHz:           approximately 0% (hard Nyquist cutoff)
    hf_ratio = high_energy / total_energy

    # ── Convert to penalty: low HF ratio → high penalty ─────────────────────
    # Linear mapping: [0, HF_REAL_BASELINE] → [1.0, 0.0]
    # Below the baseline → penalty rises toward 1.0 (deepfake)
    # At or above the baseline → penalty is 0.0 (real)
    penalty = max(0.0, min(1.0, 1.0 - (hf_ratio / HF_REAL_BASELINE)))

    # ── Cliff detector: energy drop ratio across the 8 kHz boundary ─────────
    # Compares energy in the 7–8 kHz band vs. 8–9 kHz band.
    # A sudden cliff (near-zero above 8 kHz) is a synthetic fingerprint.
    band_below_8k = (freqs >= 7000) & (freqs <= 8000)
    band_above_8k = (freqs >  8000) & (freqs <= 9000)
    e_below = float(np.sum(D_power[band_below_8k, :])) + 1e-12
    e_above = float(np.sum(D_power[band_above_8k, :])) + 1e-12
    cliff_ratio = e_above / e_below   # ≈1 = gradual rolloff (real); ≈0 = hard cliff (fake)

    meta = {
        "hf_ratio"          : round(hf_ratio,    6),
        "hf_percentage"     : f"{hf_ratio*100:.3f}%",
        "low_energy_db"     : round(10 * math.log10(low_energy  + 1e-12), 2),
        "high_energy_db"    : round(10 * math.log10(high_energy + 1e-12), 2),
        "cliff_ratio_8khz"  : round(cliff_ratio, 4),
        "cliff_verdict"     : "Gradual rolloff (real-like)" if cliff_ratio > 0.3
                              else "Hard cliff detected (synthetic-like)",
    }
    return penalty, meta


# ══════════════════════════════════════════════════════════════════════════════
# JUDGE 3 — THE BIOLOGY JUDGE  (Weight: 30%)
# "The Vocal Tremor Analyst"
# ══════════════════════════════════════════════════════════════════════════════
#
#  Humans are imperfect biological oscillators.
#  The larynx is a pair of wet, fleshy, air-driven membranes — not a tuning fork.
#  Every human voice carries measurable micro-variations:
#
#  JITTER  = Cycle-to-cycle variation in fundamental frequency (F0).
#    Real : ±0.5–2%   (involuntary muscle micro-tremors)
#    Fake : ±0.01–0.1% (mathematically smooth synthesiser path)
#
#  PITCH COEFFICIENT OF VARIATION (CV) = global pitch "wildness" (std / mean).
#    Real : ~10–25%  (natural melodic/emotional variation)
#    Fake : ~2–6%    (flat, affectless, robotic)
#
#  The Analyst's test: subpoena the pitch. If it's suspiciously monotone,
#  too smooth, or too perfectly on-curve — the voice is manufactured.
# ─────────────────────────────────────────────────────────────────────────────

def biology_judge(audio_16k: np.ndarray, sr: int = SR_NEURAL) -> tuple:
    """
    ┌──────────────────────────────────────────────────────────────┐
    │  JUDGE 3 — BIOLOGY  (Weight: 30%)                           │
    │  "The Vocal Tremor Analyst"                                  │
    └──────────────────────────────────────────────────────────────┘

    Extracts the pitch (F0) track with librosa YIN and measures two
    biological irregularity metrics:
      1. pitch_cv     — coefficient of variation of the full pitch track
      2. local_jitter — mean normalised frame-to-frame pitch change

    A real voice will score high on both. A synthetic voice will score
    suspiciously low — triggering a DEEPFAKE vote.

    Returns:
        penalty (float): 0.0 (confident REAL) → 1.0 (confident DEEPFAKE)
        meta    (dict):  diagnostic breakdown for the dashboard
    """
    try:
        import librosa
    except ImportError:
        print("      ⚠️  librosa not installed. Biology Judge returns neutral 0.5.")
        return 0.5, {"error": "librosa not installed"}

    # ── Extract frame-by-frame pitch with the YIN algorithm ─────────────────
    # YIN is robust, CPU-friendly, and accurate for speech-range F0 (60–400 Hz).
    # hop_length=256 gives a new pitch estimate every 16 ms at 16 kHz —
    # fine enough to capture micro-tremors between syllables.
    try:
        f0 = librosa.yin(
            audio_16k,
            fmin        = FMIN_HZ,
            fmax        = FMAX_HZ,
            sr          = sr,
            frame_length= 2048,
            hop_length  = 256,
        )
    except Exception as exc:
        print(f"      ⚠️  YIN pitch extraction failed: {exc}. Neutral 0.5 returned.")
        return 0.5, {"error": str(exc)}

    # ── Filter: keep only *voiced* frames ───────────────────────────────────
    # Unvoiced frames (silence, breath, plosives) push F0 to boundary values.
    # Retaining them would pollute the statistics with non-pitch data.
    VOICED_MIN = FMIN_HZ * 1.20   # 72 Hz — avoids unvoiced floor artefacts
    VOICED_MAX = FMAX_HZ * 0.95   # 380 Hz — avoids ceiling clipping artefacts
    voiced = f0[(f0 > VOICED_MIN) & (f0 < VOICED_MAX)]

    if len(voiced) < 20:
        # Fewer than 20 voiced frames → not enough evidence for a confident vote.
        # Return a neutral abstention so this judge doesn't swing the verdict.
        msg = f"Only {len(voiced)} voiced frames — insufficient for a confident vote."
        print(f"      ⚠️  {msg}")
        return 0.5, {"voiced_frames": len(voiced), "status": "abstain", "note": msg}

    # ── Metric 1: Pitch Coefficient of Variation (CV) ───────────────────────
    # CV = std / mean — a scale-free measure of overall pitch variability.
    # A whisper and a shout of the same phrase have the same CV if the
    # *relative* variation is equal — making CV speaker-agnostic.
    pitch_mean = float(np.mean(voiced))
    pitch_std  = float(np.std(voiced))
    pitch_cv   = pitch_std / pitch_mean   # e.g., 0.12 = 12% variation

    # ── Metric 2: Local Jitter (frame-to-frame pitch change) ─────────────────
    # Average absolute difference between consecutive F0 estimates, normalised
    # by the mean pitch. This directly models glottal cycle irregularity.
    f0_diffs     = np.abs(np.diff(voiced))
    local_jitter = float(np.mean(f0_diffs)) / pitch_mean

    # ── Convert each metric to a penalty score ───────────────────────────────
    # Linear mapping [0, baseline] → [1.0, 0.0]:
    #   At baseline or above → fully 'real': penalty = 0.0
    #   At or near zero      → fully 'fake': penalty = 1.0
    cv_penalty     = max(0.0, min(1.0, 1.0 - (pitch_cv     / CV_REAL_BASELINE)))
    jitter_penalty = max(0.0, min(1.0, 1.0 - (local_jitter / JITTER_REAL_BASELINE)))

    # ── Combine the two metrics into a single Biology vote ──────────────────
    # Equal weighting: both micro-tremor dimensions are equally important.
    penalty = (cv_penalty * 0.5) + (jitter_penalty * 0.5)

    meta = {
        "voiced_frames"  : len(voiced),
        "pitch_mean_hz"  : round(pitch_mean,    2),
        "pitch_std_hz"   : round(pitch_std,     2),
        "pitch_cv"       : round(pitch_cv,      6),
        "pitch_cv_pct"   : f"{pitch_cv*100:.2f}%",
        "local_jitter"   : round(local_jitter,  6),
        "local_jitter_pct": f"{local_jitter*100:.2f}%",
        "cv_penalty"     : round(cv_penalty,    4),
        "jitter_penalty" : round(jitter_penalty,4),
    }
    return penalty, meta


# ══════════════════════════════════════════════════════════════════════════════
# BURST TEMPORAL OVERRIDE
# "The Temporal Spike Detector"
# ══════════════════════════════════════════════════════════════════════════════
#
#  The three weighted judges produce a *mean* which can be diluted by a long
#  real preamble before the synthetic segment starts. Example: a 30-second
#  video where only seconds 15–25 are TTS. 10 spiking windows out of 30 means
#  the mean stays low even though 33% of windows clearly fired.
#
#  This override bypasses the weighted average entirely. It asks:
#  "Does the Neural Judge have a burst of high-confidence hits, even if the
#   average is hidden by dilution?"
#
#  It does NOT produce a separate penalty score — it is a binary override rule.
#  Think of it as the judge saying: "I don't care about the average —
#  I found conclusive evidence in one section of this recording."
# ─────────────────────────────────────────────────────────────────────────────

def burst_override_check(
    window_scores : list,
    neural_avg    : float,
) -> tuple:
    """
    ┌──────────────────────────────────────────────────────────────┐
    │  BURST OVERRIDE — "The Temporal Spike Detector"              │
    │  Binary rule, not a weighted vote.                           │
    └──────────────────────────────────────────────────────────────┘

    Fires DEEPFAKE if BOTH conditions are met:
      1. ≥ BURST_RATIO_THRESHOLD of windows scored > BURST_HIGH_SPIKE_THRESHOLD
      2. Neural average is ≥ BURST_NEURAL_AVG_MIN (sanity guard)

    Why it works for video_7:
      - 8 out of 31 windows (25.8%) scored > 0.70
      - Neural average = 0.3179 > 0.20
      - Both conditions met → DEEPFAKE override fires

    Why it does NOT fire for video_8 (real):
      - 0 out of 6 windows scored > 0.70
      - Burst ratio = 0% < 25% threshold
      - Override stays silent → verdict falls through to weighted score

    Returns:
        triggered   (bool):  True if override fires
        burst_ratio (float): Fraction of windows that spiked high
        spike_count (int):   Raw number of high-spike windows
        spike_scores(list):  The actual scores of spiking windows
    """
    if not window_scores:
        return False, 0.0, 0, []

    high_spikes = [s for s in window_scores if s > BURST_HIGH_SPIKE_THRESHOLD]
    burst_ratio = len(high_spikes) / len(window_scores)

    # Both guards must pass to avoid false positives on short clips
    triggered = (
        (burst_ratio  >= BURST_RATIO_THRESHOLD) and
        (neural_avg   >= BURST_NEURAL_AVG_MIN)
    )

    return triggered, burst_ratio, len(high_spikes), sorted(high_spikes, reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# VERDICT DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def _score_bar(score: float, width: int = 24) -> str:
    """Render a mini ASCII progress bar for a 0→1 score."""
    filled = round(score * width)
    empty  = width - filled
    return "█" * filled + "░" * empty


def _judge_verdict_label(score: float) -> str:
    """Convert a 0→1 judge score to a coloured string label."""
    if score >= 0.75:
        return "DEEPFAKE ⚠️"
    elif score >= 0.50:
        return "SUSPICIOUS ⚠️"
    elif score >= 0.25:
        return "BORDERLINE"
    else:
        return "REAL ✔"


def render_verdict_dashboard(
    video_path     : str,
    neural_score   : float, neural_meta   : dict,
    physics_score  : float, physics_meta  : dict,
    biology_score  : float, biology_meta  : dict,
    final_score    : float, is_deepfake   : bool,
    burst_triggered: bool  = False,
    burst_ratio    : float = 0.0,
    burst_count    : int   = 0,
    burst_scores   : list  = None,
):
    """Print the full multi-judge verdict dashboard to stdout."""
    W = 80   # Dashboard width

    def divider(char="═"): return char * W
    def row(text=""): return f"  {text}"

    print()
    print(divider("═"))
    print(f"  🎖️  MULTI-FACTOR ENSEMBLE DEEPFAKE VERDICT".center(W))
    print(divider("═"))
    print(row(f"  Target : {os.path.basename(video_path)}"))
    print(row(f"  Weights: Neural {WEIGHT_NEURAL:.0%}  │  Physics {WEIGHT_PHYSICS:.0%}  │  Biology {WEIGHT_BIOLOGY:.0%}"))
    print(divider("─"))

    # ── NEURAL JUDGE ─────────────────────────────────────────────────────────
    n_label = _judge_verdict_label(neural_score)
    print()
    print(row(f"  JUDGE 1 ─ NEURAL  (Weight: {WEIGHT_NEURAL:.0%})"))
    print(row(f"  \"The Deep Learning Bloodhound\""))
    print()
    print(row(f"  Score : {neural_score:.4f}  [{_score_bar(neural_score)}]  {n_label}"))
    print()
    print(row(f"  ┌─ Evidence ────────────────────────────────────────────────────┐"))
    print(row(f"  │ Windows analysed : {neural_meta.get('windows', '?')}"))
    print(row(f"  │ Fake sub-votes   : {neural_meta.get('fake_windows', '?')}/{neural_meta.get('windows', '?')} "
              f"({neural_meta.get('fake_ratio', 0):.0%} of windows)"))

    scores_str = str(neural_meta.get("window_scores", []))
    # Wrap long score lists to avoid overflowing the terminal width
    if len(scores_str) <= 56:
        print(row(f"  │ Sub-vote scores  : {scores_str}"))
    else:
        print(row(f"  │ Sub-vote scores  :"))
        chunks = [neural_meta["window_scores"][i:i+8]
                  for i in range(0, len(neural_meta["window_scores"]), 8)]
        for chunk in chunks:
            print(row(f"  │   {chunk}"))
    print(row(f"  └───────────────────────────────────────────────────────────────┘"))

    # ── PHYSICS JUDGE ────────────────────────────────────────────────────────
    p_label = _judge_verdict_label(physics_score)
    print()
    print(divider("─"))
    print()
    print(row(f"  JUDGE 2 ─ PHYSICS  (Weight: {WEIGHT_PHYSICS:.0%})"))
    print(row(f"  \"The Spectrogram Surgeon\""))
    print()
    print(row(f"  Score : {physics_score:.4f}  [{_score_bar(physics_score)}]  {p_label}"))
    print()

    if "error" in physics_meta:
        print(row(f"  ⚠️  {physics_meta['error']}"))
    else:
        print(row(f"  ┌─ Evidence ────────────────────────────────────────────────────┐"))
        print(row(f"  │ HF energy ratio (>8 kHz) : {physics_meta.get('hf_percentage','?')}"))
        print(row(f"  │   (Real baseline ≥ {HF_REAL_BASELINE*100:.1f}% — below → suspicious)"))
        print(row(f"  │ Low-band energy  (<8 kHz) : {physics_meta.get('low_energy_db','?')} dB"))
        print(row(f"  │ High-band energy (>8 kHz) : {physics_meta.get('high_energy_db','?')} dB"))
        print(row(f"  │ 8 kHz cliff ratio         : {physics_meta.get('cliff_ratio_8khz','?')}"))
        print(row(f"  │   → {physics_meta.get('cliff_verdict','?')}"))
        print(row(f"  └───────────────────────────────────────────────────────────────┘"))

    # ── BIOLOGY JUDGE ────────────────────────────────────────────────────────
    b_label = _judge_verdict_label(biology_score)
    print()
    print(divider("─"))
    print()
    print(row(f"  JUDGE 3 ─ BIOLOGY  (Weight: {WEIGHT_BIOLOGY:.0%})"))
    print(row(f"  \"The Vocal Tremor Analyst\""))
    print()
    print(row(f"  Score : {biology_score:.4f}  [{_score_bar(biology_score)}]  {b_label}"))
    print()

    if "error" in biology_meta:
        print(row(f"  ⚠️  {biology_meta['error']}"))
    elif biology_meta.get("status") == "abstain":
        print(row(f"  ⚠️  {biology_meta.get('note', 'Insufficient voiced frames — abstained.')}"))
    else:
        print(row(f"  ┌─ Evidence ────────────────────────────────────────────────────┐"))
        print(row(f"  │ Voiced frames analysed : {biology_meta.get('voiced_frames','?')}"))
        print(row(f"  │ Mean pitch (F0)         : {biology_meta.get('pitch_mean_hz','?')} Hz"))
        print(row(f"  │ Pitch std dev           : {biology_meta.get('pitch_std_hz','?')} Hz"))
        print(row(f"  │"))
        print(row(f"  │ Pitch CV   : {biology_meta.get('pitch_cv_pct','?')} "
                  f"(real baseline ≥ {CV_REAL_BASELINE*100:.0f}%)"))
        print(row(f"  │   → CV penalty    : {biology_meta.get('cv_penalty','?')}"))
        print(row(f"  │ Local Jitter : {biology_meta.get('local_jitter_pct','?')} "
                  f"(real baseline ≥ {JITTER_REAL_BASELINE*100:.1f}%)"))
        print(row(f"  │   → Jitter penalty: {biology_meta.get('jitter_penalty','?')}"))
        print(row(f"  └───────────────────────────────────────────────────────────────┘"))

    # ── WEIGHTED TALLY ───────────────────────────────────────────────────────
    print()
    print(divider("═"))
    print()
    print(row(f"  WEIGHTED VERDICT CALCULATION:"))
    print()
    n_contrib = neural_score  * WEIGHT_NEURAL
    p_contrib = physics_score * WEIGHT_PHYSICS
    b_contrib = biology_score * WEIGHT_BIOLOGY

    print(row(f"    Neural  {neural_score:.4f}  × {WEIGHT_NEURAL:.2f}  =  {n_contrib:.4f}"))
    print(row(f"    Physics {physics_score:.4f}  × {WEIGHT_PHYSICS:.2f}  =  {p_contrib:.4f}"))
    print(row(f"    Biology {biology_score:.4f}  × {WEIGHT_BIOLOGY:.2f}  =  {b_contrib:.4f}"))
    print(row(f"                            {'─'*12}"))
    print(row(f"    FINAL SCORE             {final_score:.4f}   (threshold: {VERDICT_THRESHOLD:.2f})"))
    print()

    # ── BURST OVERRIDE STATUS ────────────────────────────────────────────────
    print(divider("─"))
    print()
    print(row(f"  BURST TEMPORAL OVERRIDE  (independent of weighted score)"))
    print()
    burst_icon = "🚨 TRIGGERED" if burst_triggered else "✔ Silent"
    print(row(f"  Status        : {burst_icon}"))
    print(row(f"  High-spike windows (>{BURST_HIGH_SPIKE_THRESHOLD:.0%}) : "
              f"{burst_count} / {len(neural_meta.get('window_scores', []))} "
              f"({burst_ratio:.1%}) — threshold ≥ {BURST_RATIO_THRESHOLD:.0%}"))
    if burst_scores:
        top5 = (burst_scores or [])[:5]
        print(row(f"  Top spike scores  : {[round(s,4) for s in top5]}"))
    print(row(f"  Neural avg guard  : {neural_score:.4f} (need ≥ {BURST_NEURAL_AVG_MIN:.2f})"))
    if burst_triggered:
        print(row(f"  → Both conditions met — override fires regardless of weighted score"))
    print()

    # ── FINAL VERDICT ────────────────────────────────────────────────────────
    print(divider("═"))
    if is_deepfake:
        if burst_triggered and final_score <= VERDICT_THRESHOLD:
            # Override was the deciding factor
            print(f"  🚨  FINAL VERDICT:  DEEPFAKE  (High Penalty)".center(W))
            print(f"      Burst Override fired: {burst_count} windows ({burst_ratio:.1%}) > {BURST_HIGH_SPIKE_THRESHOLD:.0%}".center(W))
        else:
            margin = final_score - VERDICT_THRESHOLD
            print(f"  🚨  FINAL VERDICT:  DEEPFAKE  (High Penalty)".center(W))
            print(f"      Score {final_score:.4f} exceeds threshold {VERDICT_THRESHOLD:.2f} by +{margin:.4f}".center(W))
    else:
        margin = VERDICT_THRESHOLD - final_score
        print(f"  ✅  FINAL VERDICT:  REAL  (Low Penalty)".center(W))
        print(f"      Score {final_score:.4f} is {margin:.4f} below the {VERDICT_THRESHOLD:.2f} threshold".center(W))
    print(divider("═"))
    print()


# ══════════════════════════════════════════════════════════════════════════════
# CLI ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Factor Ensemble Deepfake Audio Detector — 3-Judge Voting Panel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video-path",
        type=str,
        required=True,
        help="Path to the input video/audio file to analyse.",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default="models/Brain_V5_WavLM.pth",
        help="Path to the WavLMFakeDetector checkpoint (.pth).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=VERDICT_THRESHOLD,
        help="Final ensemble score threshold above which DEEPFAKE is declared.",
    )
    parser.add_argument(
        "--no-neural",
        action="store_true",
        help="Skip Neural Judge (useful for fast ablation tests).",
    )
    parser.add_argument(
        "--no-physics",
        action="store_true",
        help="Skip Physics Judge.",
    )
    parser.add_argument(
        "--no-biology",
        action="store_true",
        help="Skip Biology Judge.",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    print()
    print("═" * 80)
    print("  🎖️  MULTI-FACTOR ENSEMBLE DEEPFAKE AUDIO DETECTOR  — initialising…")
    print("═" * 80)

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not os.path.isfile(args.video_path):
        print(f"\n  ❌ ERROR: Video file not found: {args.video_path}")
        raise SystemExit(1)

    if not os.path.isfile(args.weights_path) and not args.no_neural:
        print(f"\n  ❌ ERROR: WavLM weights not found: {args.weights_path}")
        raise SystemExit(1)

    # ── Load Neural model (CPU only) ──────────────────────────────────────────
    model = None
    if not args.no_neural:
        print("\n  🧠 Loading WavLM model onto CPU…")
        model = load_neural_model(args.weights_path)
        print("     Model ready.")

    # ── Extract audio at two sample rates ─────────────────────────────────────
    print("\n  🎞️  Extracting audio from video…")
    audio_16k, audio_44k = extract_audio_two_rates(args.video_path)
    print(f"     16 kHz track : {len(audio_16k)/SR_NEURAL:.2f}s  ({len(audio_16k):,} samples)")
    print(f"     44.1 kHz track: {len(audio_44k)/SR_PHYSICS:.2f}s  ({len(audio_44k):,} samples)")

    # ══════════════════════════════════════════════════════════════════════════
    # CONVENE THE PANEL — Each judge deliberates independently
    # ══════════════════════════════════════════════════════════════════════════

    # ── Judge 1: Neural ───────────────────────────────────────────────────────
    if not args.no_neural and model is not None:
        print("\n  ──────────────────────────────────────────────────────────────────")
        print("  🧠 JUDGE 1 — NEURAL  deliberating…")
        print("  ──────────────────────────────────────────────────────────────────")
        neural_score, neural_meta = neural_judge(audio_16k, model)
        print(f"     ➤ Neural vote: {neural_score:.4f}")
    else:
        print("\n  ⏩ Neural Judge skipped.")
        neural_score, neural_meta = 0.0, {"status": "skipped"}

    # ── Judge 2: Physics ──────────────────────────────────────────────────────
    if not args.no_physics:
        print("\n  ──────────────────────────────────────────────────────────────────")
        print("  📡 JUDGE 2 — PHYSICS  deliberating…")
        print("  ──────────────────────────────────────────────────────────────────")
        physics_score, physics_meta = physics_judge(audio_44k, sr=SR_PHYSICS)
        print(f"     ➤ Physics vote: {physics_score:.4f}  │  HF ratio: {physics_meta.get('hf_percentage','?')}")
    else:
        print("\n  ⏩ Physics Judge skipped.")
        physics_score, physics_meta = 0.0, {"status": "skipped"}

    # ── Judge 3: Biology ──────────────────────────────────────────────────────
    if not args.no_biology:
        print("\n  ──────────────────────────────────────────────────────────────────")
        print("  🧬 JUDGE 3 — BIOLOGY  deliberating…")
        print("  ──────────────────────────────────────────────────────────────────")
        biology_score, biology_meta = biology_judge(audio_16k, sr=SR_NEURAL)
        print(f"     ➤ Biology vote: {biology_score:.4f}  │  Pitch CV: {biology_meta.get('pitch_cv_pct','?')}")
    else:
        print("\n  ⏩ Biology Judge skipped.")
        biology_score, biology_meta = 0.0, {"status": "skipped"}

    # ══════════════════════════════════════════════════════════════════════════
    # TALLY THE VOTES
    # ══════════════════════════════════════════════════════════════════════════
    final_score = (
        (neural_score  * WEIGHT_NEURAL)  +
        (physics_score * WEIGHT_PHYSICS) +
        (biology_score * WEIGHT_BIOLOGY)
    )
    primary_deepfake = final_score > args.threshold

    # ── Burst Override check (uses neural window scores, no extra inference) ──
    print("\n  ──────────────────────────────────────────────────────────────────")
    print("  💥 BURST OVERRIDE  checking temporal spike pattern…")
    print("  ──────────────────────────────────────────────────────────────────")
    window_scores = neural_meta.get("window_scores", [])
    burst_triggered, burst_ratio, burst_count, burst_scores = burst_override_check(
        window_scores, neural_score
    )
    if burst_triggered:
        print(f"     ⚡ OVERRIDE FIRED: {burst_count} windows ({burst_ratio:.1%}) scored > "
              f"{BURST_HIGH_SPIKE_THRESHOLD:.0%} with neural avg {neural_score:.4f}")
    else:
        print(f"     ✔  No burst pattern: {burst_count} high-spike windows "
              f"({burst_ratio:.1%}) — threshold {BURST_RATIO_THRESHOLD:.0%}")

    # Final verdict: primary weighted score OR burst override
    is_deepfake = primary_deepfake or burst_triggered

    # ══════════════════════════════════════════════════════════════════════════
    # RENDER THE VERDICT DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════
    render_verdict_dashboard(
        video_path      = args.video_path,
        neural_score    = neural_score,    neural_meta    = neural_meta,
        physics_score   = physics_score,   physics_meta   = physics_meta,
        biology_score   = biology_score,   biology_meta   = biology_meta,
        final_score     = final_score,     is_deepfake    = is_deepfake,
        burst_triggered = burst_triggered, burst_ratio    = burst_ratio,
        burst_count     = burst_count,     burst_scores   = burst_scores,
    )


if __name__ == "__main__":
    main()
