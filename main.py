import os
import argparse
from pathlib import Path
from tqdm import tqdm
from src.audio_extraction import audio_extraction


def process_dataset(raw_video_folder, output_audio_folder):
    """
    Scans a directory for mp4, mkv, and avi files, and uses the
    audio_extraction module to strip out audio into clean .wav files
    inside the unified output folder.
    """
    # 1. Guarantee the output directory framework exists
    os.makedirs(output_audio_folder, exist_ok=True)

    # 2. Leverage Pathlib for efficient filesystem traversal
    raw_path = Path(raw_video_folder)

    # Scan for common video format extensions (case-agnostic logic)
    extensions = ['*.mp4', '*.mkv', '*.avi']
    video_files = []
    for ext in extensions:
        # We use rglob to ensure we catch files even if nested under subfolders
        video_files.extend(raw_path.rglob(ext))
        video_files.extend(raw_path.rglob(ext.upper()))

    print(f"Discovered {len(video_files)} video files in '{raw_video_folder}'")

    # 3. Iterate through all identified video pipelines and extract sequentially
    for video_file in tqdm(video_files, desc="Extracting Audio to .wav"):
        # Construct the target output string: e.g., 'sample1.mp4' -> 'sample1.wav'
        output_file_name = f"{video_file.stem}.wav"
        output_file_path = os.path.join(output_audio_folder, output_file_name)

        try:
            # 4. Interface with the underlying extraction module
            audio_extraction(str(video_file), output_file_path)

        except Exception as e:
            # Failsafe loop wrapper to prevent a single file from ruining the entire pipeline
            print(f"\n[!] Critical error processing {video_file.name}: {str(e)}")


def process_single_file(video_path_str, output_path_str=None):
    """
    Extracts audio from a single video file.

    Args:
        video_path_str: Path to the input video file (mp4/mkv/avi).
        output_path_str: Optional path for the output .wav file.
                         Defaults to the same directory as the input.
    """
    video_path = Path(video_path_str)

    if not video_path.is_file():
        print(f"[!] File not found: {video_path}")
        raise SystemExit(1)

    if video_path.suffix.lower() not in {".mp4", ".mkv", ".avi"}:
        print(f"[!] Unsupported format: '{video_path.suffix}'. Supported: mp4, mkv, avi.")
        raise SystemExit(1)

    # Resolve output path
    if output_path_str:
        output_file_path = output_path_str
        os.makedirs(Path(output_path_str).parent, exist_ok=True)
    else:
        output_file_path = str(video_path.with_suffix(".wav"))

    print(f"Input  : {video_path}")
    print(f"Output : {output_file_path}")

    try:
        audio_extraction(str(video_path), output_file_path)
        print(f"\n[+] Audio extracted successfully -> {output_file_path}")
    except Exception as e:
        print(f"\n[!] Critical error processing {video_path.name}: {e}")
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Audio Extraction Pre-Processor — extracts .wav audio from video files.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        metavar="VIDEO_PATH",
        help=(
            "Path to a single video file to process (mp4/mkv/avi).\n"
            "Example: python main.py --file /path/to/video.mp4\n"
            "If omitted, the full batch-directory pipeline runs instead."
        ),
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        metavar="OUTPUT_WAV",
        help=(
            "Destination .wav file path (only used with --file).\n"
            "Defaults to the same directory and stem as the input file."
        ),
    )
    args = parser.parse_args()

    print("==================================================")
    print("   Initializing Audio Extraction Pre-Processor    ")
    print("==================================================")

    if args.file:
        # ── Single-file mode ──────────────────────────────────────────────────
        process_single_file(args.file, args.output)
    else:
        # ── Batch-directory mode (original behaviour) ─────────────────────────
        RAW_VIDEOS_DIR = "Data/Raw_video"
        EXTRACTED_AUDIO_DIR = "Data/extracted_audio"
        process_dataset(RAW_VIDEOS_DIR, EXTRACTED_AUDIO_DIR)

    print("\nPipeline Sequence Complete.")