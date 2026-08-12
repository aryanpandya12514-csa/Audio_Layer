from pathlib import Path

import multi_factor_evaluator as mfe

# ==========================================================
# EDIT ONLY THIS PATH, THEN SAVE + RUN THIS FILE
# ==========================================================
VIDEO_PATH = "/home/aryan-pandya/Audio_Layer/Data/Raw_video/video_3.mp4"

# Optional settings
WEIGHTS_PATH = "models/Brain_V5_WavLM.pth"
THRESHOLD = mfe.VERDICT_THRESHOLD
ENABLE_NEURAL = True
ENABLE_PHYSICS = True
ENABLE_BIOLOGY = True

# Guarded Physics Override:
# If Physics judge is extremely high AND 8kHz cliff looks synthetic-like,
# force final verdict to DEEPFAKE even when weighted score is diluted.
ENABLE_PHYSICS_OVERRIDE = True
PHYSICS_OVERRIDE_SCORE = 0.98
PHYSICS_OVERRIDE_MAX_CLIFF_RATIO = 0.80


def run(video_path: str):
    video_file = Path(video_path)
    if not video_file.exists() or not video_file.is_file():
        print(f"\n❌ ERROR: Video file not found: {video_file}")
        raise SystemExit(1)

    if ENABLE_NEURAL and not Path(WEIGHTS_PATH).is_file():
        print(f"\n❌ ERROR: WavLM weights not found: {WEIGHTS_PATH}")
        raise SystemExit(1)

    print("\n" + "═" * 80)
    print("  🎖️  MULTI-FACTOR ENSEMBLE DEEPFAKE AUDIO DETECTOR  — initialising…")
    print("═" * 80)

    model = None
    if ENABLE_NEURAL:
        print("\n  🧠 Loading WavLM model onto CPU…")
        model = mfe.load_neural_model(WEIGHTS_PATH)
        print("     Model ready.")

    print("\n  🎞️  Extracting audio from video…")
    audio_16k, audio_44k = mfe.extract_audio_two_rates(str(video_file))
    print(f"     16 kHz track : {len(audio_16k)/mfe.SR_NEURAL:.2f}s  ({len(audio_16k):,} samples)")
    print(f"     44.1 kHz track: {len(audio_44k)/mfe.SR_PHYSICS:.2f}s  ({len(audio_44k):,} samples)")

    if ENABLE_NEURAL and model is not None:
        print("\n  ──────────────────────────────────────────────────────────────────")
        print("  🧠 JUDGE 1 — NEURAL  deliberating…")
        print("  ──────────────────────────────────────────────────────────────────")
        neural_score, neural_meta = mfe.neural_judge(audio_16k, model)
        print(f"     ➤ Neural vote: {neural_score:.4f}")
    else:
        print("\n  ⏩ Neural Judge skipped.")
        neural_score, neural_meta = 0.0, {"status": "skipped", "window_scores": []}

    if ENABLE_PHYSICS:
        print("\n  ──────────────────────────────────────────────────────────────────")
        print("  📡 JUDGE 2 — PHYSICS  deliberating…")
        print("  ──────────────────────────────────────────────────────────────────")
        physics_score, physics_meta = mfe.physics_judge(audio_44k, sr=mfe.SR_PHYSICS)
        print(f"     ➤ Physics vote: {physics_score:.4f}  │  HF ratio: {physics_meta.get('hf_percentage', '?')}")
    else:
        print("\n  ⏩ Physics Judge skipped.")
        physics_score, physics_meta = 0.0, {"status": "skipped"}

    if ENABLE_BIOLOGY:
        print("\n  ──────────────────────────────────────────────────────────────────")
        print("  🧬 JUDGE 3 — BIOLOGY  deliberating…")
        print("  ──────────────────────────────────────────────────────────────────")
        biology_score, biology_meta = mfe.biology_judge(audio_16k, sr=mfe.SR_NEURAL)
        print(f"     ➤ Biology vote: {biology_score:.4f}  │  Pitch CV: {biology_meta.get('pitch_cv_pct', '?')}")
    else:
        print("\n  ⏩ Biology Judge skipped.")
        biology_score, biology_meta = 0.0, {"status": "skipped"}

    final_score = (
        (neural_score * mfe.WEIGHT_NEURAL)
        + (physics_score * mfe.WEIGHT_PHYSICS)
        + (biology_score * mfe.WEIGHT_BIOLOGY)
    )
    primary_deepfake = final_score > THRESHOLD

    cliff_ratio = float(physics_meta.get("cliff_ratio_8khz", 1.0)) if isinstance(physics_meta, dict) else 1.0
    physics_override = (
        ENABLE_PHYSICS_OVERRIDE
        and ENABLE_PHYSICS
        and physics_score >= PHYSICS_OVERRIDE_SCORE
        and cliff_ratio <= PHYSICS_OVERRIDE_MAX_CLIFF_RATIO
    )

    print("\n  ──────────────────────────────────────────────────────────────────")
    print("  💥 BURST OVERRIDE  checking temporal spike pattern…")
    print("  ──────────────────────────────────────────────────────────────────")
    window_scores = neural_meta.get("window_scores", [])
    burst_triggered, burst_ratio, burst_count, burst_scores = mfe.burst_override_check(
        window_scores, neural_score
    )

    if burst_triggered:
        print(
            f"     ⚡ OVERRIDE FIRED: {burst_count} windows ({burst_ratio:.1%}) scored > "
            f"{mfe.BURST_HIGH_SPIKE_THRESHOLD:.0%} with neural avg {neural_score:.4f}"
        )
    else:
        print(
            f"     ✔  No burst pattern: {burst_count} high-spike windows "
            f"({burst_ratio:.1%}) — threshold {mfe.BURST_RATIO_THRESHOLD:.0%}"
        )

    if physics_override:
        print(
            f"     ⚡ PHYSICS OVERRIDE FIRED: score={physics_score:.4f} (>= {PHYSICS_OVERRIDE_SCORE:.2f}), "
            f"cliff_ratio={cliff_ratio:.4f} (<= {PHYSICS_OVERRIDE_MAX_CLIFF_RATIO:.2f})"
        )

    is_deepfake = primary_deepfake or burst_triggered or physics_override

    mfe.render_verdict_dashboard(
        video_path=str(video_file),
        neural_score=neural_score,
        neural_meta=neural_meta,
        physics_score=physics_score,
        physics_meta=physics_meta,
        biology_score=biology_score,
        biology_meta=biology_meta,
        final_score=final_score,
        is_deepfake=is_deepfake,
        burst_triggered=burst_triggered,
        burst_ratio=burst_ratio,
        burst_count=burst_count,
        burst_scores=burst_scores,
    )


if __name__ == "__main__":
    run(VIDEO_PATH)
