import json
import shutil
import subprocess
from pathlib import Path


def _check_binary_available(binary_name: str):
    if shutil.which(binary_name) is None:
        raise RuntimeError(
            f"Required binary '{binary_name}' not found. Install FFmpeg on Ubuntu: "
            "sudo apt install ffmpeg"
        )


def has_valid_audio(video_file_path):
    """
    Fast stream-level audio validation using ffprobe.
    Returns True only when at least one audio stream is present.
    """
    try:
        _check_binary_available("ffprobe")
        video_path = Path(video_file_path)
        if not video_path.exists() or not video_path.is_file():
            return False

        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index,codec_type,duration",
            "-of",
            "json",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(result.stdout or "{}")
        streams = payload.get("streams", [])
        return len(streams) > 0
    except Exception:
        return False


def extract_audio_wav(
    video_file_path,
    output_audio_path,
    sample_rate=16000,
    channels=1,
    overwrite=True,
):
    """
    Extract mono 16kHz PCM WAV from a video using ffmpeg.

    Returns:
        str: Absolute output wav path.
    Raises:
        RuntimeError / FileNotFoundError / ValueError on extraction failures.
    """
    _check_binary_available("ffmpeg")

    video_path = Path(video_file_path)
    output_path = Path(output_audio_path)

    if not video_path.exists() or not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not has_valid_audio(str(video_path)):
        raise ValueError(f"No valid audio stream found in: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
    ]

    cmd.append("-y" if overwrite else "-n")
    cmd.append(str(output_path))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"FFmpeg extraction failed for '{video_path.name}': {stderr}")

    if not output_path.exists() or output_path.stat().st_size <= 44:
        raise RuntimeError(f"Extracted WAV appears invalid or empty: {output_path}")

    return str(output_path.resolve())


def audio_extraction(video_file_path, output_audio_path):
    """
    Backward-compatible wrapper used by older scripts.
    Returns True/False and avoids raising to preserve legacy behavior.
    """
    try:
        extract_audio_wav(video_file_path, output_audio_path)
        return True
    except Exception:
        return False
