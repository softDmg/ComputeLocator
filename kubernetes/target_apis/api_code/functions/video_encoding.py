"""
Video Encoding (H.264) for Kubernetes API.

FFmpeg-based video encoding benchmark.
Uses local video file from resources folder.
"""
import subprocess
import shutil
import time
import platform
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple


# Video source configuration (local file in resources)
VIDEO_NAME = "video_to_encode.mp4"
RESOURCES_DIR = Path(__file__).parent.parent / "resources"

# Benchmark configuration
SEGMENT_DURATION = 4  # seconds
SEGMENT_START = 30  # start at 30s for interesting content
FFMPEG_PRESET = "medium"

# Cache directory (relative to this file's location)
CACHE_DIR = Path(__file__).parent.parent / "resources"


@dataclass
class EncodingProfile:
    """Encoding profile specification."""
    name: str
    resolution: str
    bitrate_kbps: int
    width: int
    height: int


# Resolution profiles from the article (integer keys for ML compatibility)
# 0 = HD-ready (720p), 1 = Full HD (1080p), 2 = Quad HD (1440p)
PROFILES: Dict[int, EncodingProfile] = {
    0: EncodingProfile(
        name="HD-ready",
        resolution="1280x720",
        bitrate_kbps=1500,
        width=1280,
        height=720
    ),
    1: EncodingProfile(
        name="Full HD",
        resolution="1920x1080",
        bitrate_kbps=3000,
        width=1920,
        height=1080
    ),
    2: EncodingProfile(
        name="Quad HD",
        resolution="2560x1440",
        bitrate_kbps=6500,
        width=2560,
        height=1440
    ),
}


def _check_ffmpeg() -> bool:
    """Check if FFmpeg with libx264 is available."""
    ffmpeg_path = shutil.which('ffmpeg')
    if not ffmpeg_path:
        return False

    result = subprocess.run(
        ['ffmpeg', '-encoders'],
        capture_output=True,
        text=True
    )
    return 'libx264' in result.stdout


def _extract_raw_segment(
    input_path: Path,
    output_path: Path,
    duration: int = SEGMENT_DURATION,
    start_time: int = SEGMENT_START
) -> float:
    """Extract a raw Y4M segment from the source video. Returns extraction time."""
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_time),
        '-i', str(input_path),
        '-t', str(duration),
        '-pix_fmt', 'yuv420p',
        '-an',
        str(output_path)
    ]

    extraction_start = time.perf_counter()

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to extract video segment: {e.stderr}")

    return time.perf_counter() - extraction_start


def _get_or_prepare_raw_video(force_extract: bool = False) -> Tuple[Path, float]:
    """
    Get source video from resources and extract raw segment (with caching).

    Returns:
        Tuple of (raw_video_path, extraction_time)
    """
    source_path = RESOURCES_DIR / VIDEO_NAME

    # Validate source video exists
    if not source_path.exists():
        raise RuntimeError(
            f"Source video not found: {source_path}. "
            f"Please place '{VIDEO_NAME}' in the resources folder."
        )

    raw_path = CACHE_DIR / f"segment_{SEGMENT_DURATION}s.y4m"

    extraction_time = 0.0

    # Extract raw segment if needed
    if not raw_path.exists() or force_extract:
        extraction_time = _extract_raw_segment(source_path, raw_path)

    return raw_path, extraction_time


def _run_encoding(profile: EncodingProfile, input_path: Path) -> Dict[str, Any]:
    """Run a single video encoding."""

    output_filename = f"{input_path.stem}_{profile.name.replace(' ', '_')}.mp4"
    output_path = CACHE_DIR / output_filename
    input_size = input_path.stat().st_size

    cmd = [
        'ffmpeg',
        '-y',
        '-i', str(input_path),
        '-c:v', 'libx264',
        '-preset', FFMPEG_PRESET,
        '-b:v', f'{profile.bitrate_kbps}k',
        '-maxrate', f'{profile.bitrate_kbps}k',
        '-bufsize', f'{profile.bitrate_kbps * 2}k',
        '-vf', f'scale={profile.width}:{profile.height}',
        '-an',
        '-threads', '0',
        str(output_path)
    ]

    start_time = time.perf_counter()

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        success = True
        error_message = None
    except subprocess.CalledProcessError as e:
        success = False
        error_message = f"FFmpeg error: {e.stderr}"

    execution_time = time.perf_counter() - start_time
    output_size = output_path.stat().st_size if output_path.exists() else 0

    return {
        "execution_time": execution_time,
        "input_size": input_size,
        "output_size": output_size,
        "profile_name": profile.name,
        "resolution": profile.resolution,
        "bitrate_kbps": profile.bitrate_kbps,
        "success": success,
        "error_message": error_message
    }


def video_encoding(profile: Optional[int] = None) -> Dict[str, Any]:
    """
    Run video encoding benchmark.

    Args:
        profile: Encoding profile (integer). Options: 0=720p, 1=1080p, 2=1440p.
                 Default: 1 (Full HD 1080p)

    Returns:
        Dictionary with results:
        - execution_time: Encoding time in seconds
        - input_size: Input file size in bytes
        - output_size: Output file size in bytes
        - profile_name: Name of encoding profile
        - resolution: Output resolution
        - bitrate_kbps: Target bitrate
        - compression_percent: Compression ratio as percentage
        - extraction_time: Time spent extracting raw segment (0 if cached)
        - success: Boolean indicating success
    """
    # Default profile (1 = Full HD)
    if profile is None:
        profile = 1

    # Validate profile
    if profile not in PROFILES:
        return {
            "success": False,
            "error_message": f"Unknown profile: {profile}. Available: 0 (720p), 1 (1080p), 2 (1440p)"
        }

    try:
        # Check FFmpeg
        if not _check_ffmpeg():
            system = platform.system()
            if system == 'Darwin':
                hint = "brew install ffmpeg"
            elif system == 'Linux':
                hint = "apt-get install ffmpeg"
            else:
                hint = "Download from https://ffmpeg.org"
            return {
                "success": False,
                "error_message": f"FFmpeg with libx264 not found. Install with: {hint}"
            }

        # Prepare raw video (extract segment if needed)
        raw_path, extraction_time = _get_or_prepare_raw_video()

        # Run encoding
        encoding_profile = PROFILES[profile]
        result = _run_encoding(encoding_profile, raw_path)

        # Add extraction time and compression ratio
        result["extraction_time"] = float(extraction_time)

        if result["success"] and result["input_size"] > 0:
            compression = (1 - result["output_size"] / result["input_size"]) * 100
            result["compression_percent"] = float(compression)
        else:
            result["compression_percent"] = 0.0

        # Ensure all numeric values are Python native types for JSON serialization
        result["execution_time"] = float(result["execution_time"])
        result["input_size"] = int(result["input_size"])
        result["output_size"] = int(result["output_size"])
        result["bitrate_kbps"] = int(result["bitrate_kbps"])

        return result

    except Exception as e:
        return {
            "success": False,
            "error_message": str(e)
        }
