"""
YouTube audio wrapper using yt-dlp for track search and audio retrieval.

No API credentials required. Requires ffmpeg to be installed on the system.
"""

import re
import subprocess
import tempfile
from pathlib import Path

import yt_dlp

# YouTube video ID pattern (11 characters, alphanumeric with - and _)
VIDEO_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{11}$")

# Patterns to extract video ID from various YouTube URL formats
URL_PATTERNS = [
    re.compile(r"(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})"),
    re.compile(r"youtube\.com/embed/([a-zA-Z0-9_-]{11})"),
    re.compile(r"youtube\.com/v/([a-zA-Z0-9_-]{11})"),
]


def _extract_video_id(query: str) -> str | None:
    """
    Extract YouTube video ID from a URL or validate if query is a video ID.

    Args:
        query: YouTube URL, video ID, or search query

    Returns:
        Video ID if found/valid, None otherwise
    """
    # Check if it's already a video ID
    if VIDEO_ID_PATTERN.match(query):
        return query

    # Try to extract from URL patterns
    for pattern in URL_PATTERNS:
        match = pattern.search(query)
        if match:
            return match.group(1)

    return None


def _parse_song_and_artist(title: str, channel: str) -> tuple[str, str]:
    """
    Parse song name and artist from YouTube video title and channel.

    Handles common title formats like:
    - "Artist - Song Name"
    - "Song Name by Artist"
    - "Artist: Song Name"
    - "Song Name | Artist"
    - "Song Name (Official Video)"

    Args:
        title: Video title
        channel: Channel name (fallback for artist)

    Returns:
        Tuple of (song_name, artist)
    """
    # Remove common suffixes
    suffixes_to_remove = [
        r"\s*\(Official\s*(Music\s*)?Video\)",
        r"\s*\(Official\s*Audio\)",
        r"\s*\(Lyric\s*Video\)",
        r"\s*\(Lyrics?\)",
        r"\s*\[Official\s*(Music\s*)?Video\]",
        r"\s*\[Official\s*Audio\]",
        r"\s*\[Lyric\s*Video\]",
        r"\s*\[Lyrics?\]",
        r"\s*\|\s*Official\s*Video",
        r"\s*-\s*Official\s*Video",
        r"\s*HD\s*$",
        r"\s*HQ\s*$",
        r"\s*4K\s*$",
    ]

    clean_title = title
    for suffix in suffixes_to_remove:
        clean_title = re.sub(suffix, "", clean_title, flags=re.IGNORECASE)

    clean_title = clean_title.strip()

    # Try different separator patterns
    # Maps separator to (pattern, swap_order) where swap_order=True means "Artist - Song" format
    separators = [
        (r"\s+-\s+", True),   # "Artist - Song" -> swap to (Song, Artist)
        (r"\s+by\s+", False),  # "Song by Artist" -> keep as (Song, Artist)
        (r"\s*:\s*", True),   # "Artist: Song" -> swap to (Song, Artist)
        (r"\s*\|\s*", False),  # "Song | Artist" -> keep as (Song, Artist)
    ]

    for pattern, swap_order in separators:
        if re.search(pattern, clean_title, re.IGNORECASE):
            parts = re.split(pattern, clean_title, maxsplit=1)
            if len(parts) == 2:
                part1, part2 = parts[0].strip(), parts[1].strip()
                return (part2, part1) if swap_order else (part1, part2)

    # No separator found - use full title as song name, channel as artist
    return clean_title, channel


def _normalize_popularity(view_count: int | None) -> int:
    """
    Normalize view count to a 0-100 popularity score.

    Uses logarithmic scaling since view counts vary enormously.

    Args:
        view_count: Number of views

    Returns:
        Popularity score 0-100
    """
    if not view_count or view_count <= 0:
        return 0

    import math

    # Log scale: 1K views = ~30, 1M views = ~60, 1B views = ~90
    log_views = math.log10(view_count)
    # Scale: 3 (1K) -> 30, 6 (1M) -> 60, 9 (1B) -> 90
    popularity = int(log_views * 10)
    return max(0, min(100, popularity))


def _extract_track_info(info_dict: dict) -> dict:
    """
    Extract standardized track info from yt-dlp info dict.

    Args:
        info_dict: yt-dlp extracted video info

    Returns:
        Standardized track info dict
    """
    video_id = info_dict.get("id", "")
    title = info_dict.get("title", "Unknown")
    channel = info_dict.get("channel", info_dict.get("uploader", "Unknown"))
    channel_id = info_dict.get("channel_id", info_dict.get("uploader_id"))

    song_name, artist = _parse_song_and_artist(title, channel)

    duration_seconds = info_dict.get("duration", 0) or 0
    view_count = info_dict.get("view_count", 0)

    return {
        "track_id": video_id,
        "name": song_name,
        "artist": artist,
        "artist_id": channel_id,
        "album": "",  # YouTube doesn't have albums
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        "genres": [],  # YouTube doesn't provide genres
        "popularity": _normalize_popularity(view_count),
        "duration_ms": int(duration_seconds * 1000),
    }


def _get_ydl_opts(extract_audio: bool = False) -> dict:
    """
    Get yt-dlp options.

    Args:
        extract_audio: Whether to extract audio

    Returns:
        yt-dlp options dict
    """
    opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
    }

    if extract_audio:
        opts.update(
            {
                "format": "bestaudio/best",
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }
                ],
            }
        )

    return opts


def search_track(query: str) -> dict | None:
    """
    Search for a track on YouTube.

    Args:
        query: Search query (song name, artist, YouTube URL, or video ID)

    Returns:
        Dict with track info or None if not found:
        {
            'track_id': str,         # YouTube video ID (11 chars)
            'name': str,             # Parsed song name
            'artist': str,           # Parsed artist or channel name
            'artist_id': str | None, # Channel ID
            'album': str,            # Empty (YouTube doesn't have albums)
            'video_url': str,        # Full YouTube URL
            'genres': list[str],     # Empty (YouTube doesn't provide genres)
            'popularity': int,       # View count normalized 0-100
            'duration_ms': int,      # Duration in milliseconds
        }
    """
    # Check if query is a video ID or URL
    video_id = _extract_video_id(query)

    ydl_opts = _get_ydl_opts()

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            if video_id:
                # Fetch specific video
                url = f"https://www.youtube.com/watch?v={video_id}"
                info = ydl.extract_info(url, download=False)
            else:
                # Search YouTube
                search_url = f"ytsearch1:{query}"
                info = ydl.extract_info(search_url, download=False)
                if info and "entries" in info:
                    entries = info.get("entries", [])
                    if not entries:
                        return None
                    info = entries[0]

            if not info:
                return None

            return _extract_track_info(info)

    except yt_dlp.DownloadError:
        return None
    except Exception:
        return None


def get_audio(video_url_or_id: str, timeout: int = 120) -> bytes:
    """
    Download full audio from YouTube video.

    Args:
        video_url_or_id: YouTube URL or video ID
        timeout: Download timeout in seconds

    Returns:
        Audio data as bytes (MP3 format)

    Raises:
        ValueError: If URL/ID is invalid or empty
        RuntimeError: If download fails or ffmpeg is not installed
    """
    if not video_url_or_id:
        raise ValueError("Video URL or ID cannot be empty")

    # Extract video ID if URL was provided, otherwise use as-is
    video_id = _extract_video_id(video_url_or_id)
    url = f"https://www.youtube.com/watch?v={video_id}" if video_id else video_url_or_id

    # Check if ffmpeg is available
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
            timeout=5,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        raise RuntimeError(
            "ffmpeg is required but not found. "
            "Install it with: brew install ffmpeg (macOS), "
            "apt install ffmpeg (Ubuntu), or download from ffmpeg.org (Windows)"
        ) from None

    # Download to temp file then read into memory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_template = str(Path(tmpdir) / "audio.%(ext)s")

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "socket_timeout": timeout,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # Find the downloaded file
            audio_file = Path(tmpdir) / "audio.mp3"
            if not audio_file.exists():
                raise RuntimeError(f"Audio file not found after download: {audio_file}")

            return audio_file.read_bytes()

        except yt_dlp.DownloadError as e:
            raise RuntimeError(f"Failed to download audio: {e}") from e


def trim_audio(audio_bytes: bytes, start_seconds: float, end_seconds: float) -> bytes:
    """
    Trim audio to a specific time interval using ffmpeg.

    Args:
        audio_bytes: Raw audio data as bytes (MP3 format)
        start_seconds: Start time in seconds
        end_seconds: End time in seconds

    Returns:
        Trimmed audio data as bytes (MP3 format)

    Raises:
        ValueError: If start >= end or times are negative
        RuntimeError: If ffmpeg fails
    """
    if start_seconds < 0 or end_seconds < 0:
        raise ValueError("Start and end times must be non-negative")
    if start_seconds >= end_seconds:
        raise ValueError("Start time must be less than end time")

    duration = end_seconds - start_seconds

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "input.mp3"
        output_file = Path(tmpdir) / "output.mp3"

        input_file.write_bytes(audio_bytes)

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i", str(input_file),
                    "-ss", str(start_seconds),
                    "-t", str(duration),
                    "-c", "copy",
                    str(output_file),
                ],
                capture_output=True,
                check=True,
                timeout=30,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed to trim audio: {e.stderr.decode()}") from e
        except subprocess.TimeoutExpired:
            raise RuntimeError("ffmpeg timed out while trimming audio") from None

        if not output_file.exists():
            raise RuntimeError("Trimmed audio file not created")

        return output_file.read_bytes()


def get_related_videos(video_id: str, limit: int = 20) -> list[dict]:
    """
    Get related videos for a given video.

    Note: YouTube's related videos API is not directly accessible via yt-dlp,
    so this uses search with the video title as a workaround.

    Args:
        video_id: YouTube video ID
        limit: Maximum number of results

    Returns:
        List of track info dicts
    """
    # First get the original video info
    original = search_track(video_id)
    if not original:
        return []

    # Search for similar content
    search_query = f"{original['artist']} {original['name']}"
    return search_candidates(search_query, original["artist"], limit=limit)


def search_candidates(
    seed_query: str,
    seed_artist: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """
    Search for candidate tracks similar to the seed.

    Uses YouTube search to find similar tracks based on query and artist.

    Args:
        seed_query: Search query (song name or general query)
        seed_artist: Optional artist name to include in search
        limit: Maximum number of candidates to return

    Returns:
        List of track dicts with track info (same format as search_track)

    Raises:
        ValueError: If seed_query is empty
    """
    if not seed_query:
        raise ValueError("seed_query must be provided")

    # Build search query
    if seed_artist:
        search_query = f"{seed_artist} {seed_query} music"
    else:
        search_query = f"{seed_query} music"

    ydl_opts = _get_ydl_opts()

    candidates = []
    seen_ids: set[str] = set()

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Search YouTube for multiple results
            search_url = f"ytsearch{limit}:{search_query}"
            info = ydl.extract_info(search_url, download=False)

            if not info or "entries" not in info:
                return []

            for entry in info.get("entries", []):
                if not entry:
                    continue

                video_id = entry.get("id", "")

                # Skip duplicates
                if video_id in seen_ids:
                    continue
                seen_ids.add(video_id)

                # Skip very short videos (likely not full songs)
                duration = entry.get("duration", 0) or 0
                if duration < 60:  # Less than 1 minute
                    continue

                # Skip very long videos (likely compilations/mixes)
                if duration > 600:  # More than 10 minutes
                    continue

                track_info = _extract_track_info(entry)
                candidates.append(track_info)

                if len(candidates) >= limit:
                    break

    except yt_dlp.DownloadError:
        pass
    except Exception:
        pass

    return candidates[:limit]
