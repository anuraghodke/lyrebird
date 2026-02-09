"""
Diverse candidate track discovery.

Orchestrates multiple music discovery sources (MusicBrainz, YouTube
multi-query) to build a wide, deduplicated pool of candidate tracks
for similarity comparison.
"""

import logging
import re

from src.musicbrainz_client import find_related_recordings
from src.youtube_client import (
    MAX_TRACK_DURATION_S,
    MIN_TRACK_DURATION_S,
    search_candidates,
    search_track,
)

logger = logging.getLogger(__name__)


def _normalize_key(artist: str, track: str) -> str:
    """
    Build a dedup key from artist and track name.

    Lowercases, strips, and collapses whitespace so that
    "Queen" / "Bohemian Rhapsody" and "queen"/"bohemian rhapsody"
    map to the same key.
    """
    artist_norm = re.sub(r"\s+", " ", artist.strip().lower())
    track_norm = re.sub(r"\s+", " ", track.strip().lower())
    return f"{artist_norm}:::{track_norm}"


# Words that indicate a cover, remix, or reupload of the same song
_COVER_REMIX_PATTERN = re.compile(
    r"\b(cover|remix|lofi|acoustic|live|piano|karaoke|instrumental|"
    r"version|edit|bootleg|flip|rework|mashup|ska punk|punk)\b",
    re.IGNORECASE,
)


def _normalize_track_name(name: str) -> str:
    """
    Reduce a track name to its core title for same-song detection.

    Strips parenthetical/bracket suffixes (e.g. "[Lofi Fruits Release]",
    "(Piano Cover)"), common cover/remix keywords, and normalises whitespace.

    Examples:
        "Redbone [Lofi Fruits Release]"  -> "redbone"
        "Redbone (SKA PUNK COVER)"       -> "redbone"
        "RedBone"                         -> "redbone"
    """
    # Remove anything inside (...) or [...]
    cleaned = re.sub(r"\([^)]*\)", "", name)
    cleaned = re.sub(r"\[[^\]]*\]", "", cleaned)
    # Remove cover/remix keywords
    cleaned = _COVER_REMIX_PATTERN.sub("", cleaned)
    # Remove stray punctuation left over, collapse whitespace
    cleaned = re.sub(r"['\"\-–—]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def _resolve_to_youtube(
    track_pairs: list[tuple[str, str]],
    seen_ids: set[str],
    seen_keys: set[str],
    seen_titles: set[str],
    source_label: str,
) -> list[dict]:
    """
    Resolve (artist, track) pairs to YouTube videos via search_track().

    Skips duplicates (by video ID, normalized artist+track key, and
    normalized title) and filters by duration.

    Args:
        track_pairs: List of (artist, track) tuples to resolve.
        seen_ids: Set of already-seen YouTube video IDs (mutated in place).
        seen_keys: Set of already-seen normalized keys (mutated in place).
        seen_titles: Set of already-seen normalized track titles (mutated in place).
        source_label: Label for logging (e.g. "musicbrainz").

    Returns:
        List of track info dicts that passed dedup and duration filters.
    """
    results = []
    for artist, track in track_pairs:
        key = _normalize_key(artist, track)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        title_key = _normalize_track_name(track)
        if title_key in seen_titles:
            logger.debug("Skipping same-song variant: %s - %s", artist, track)
            continue

        query = f"{artist} - {track}"
        info = search_track(query)
        if not info:
            continue

        vid = info["track_id"]
        if vid in seen_ids:
            continue

        duration_s = info["duration_ms"] / 1000
        if duration_s < MIN_TRACK_DURATION_S or duration_s > MAX_TRACK_DURATION_S:
            continue

        seen_ids.add(vid)
        seen_titles.add(title_key)
        results.append(info)
        logger.info("[%s] %s - %s", source_label, artist, track)

    return results


def get_diverse_candidates(ref_track: dict, limit: int = 30) -> list[dict]:
    """
    Gather candidate tracks from multiple discovery sources.

    Sources (in order):
        1. MusicBrainz related recordings (covers, remixes — resolved to YouTube)
        2. Smart YouTube multi-query (varied search strategies)
        3. Fallback: original "{artist} similar music" YouTube search

    Each source is wrapped in try/except so a single failure never
    crashes the pipeline.

    Args:
        ref_track: Reference track dict (from search_track()).
        limit: Target number of candidates.

    Returns:
        List of track info dicts, deduplicated, excluding the reference.
    """
    artist = ref_track.get("artist", "")
    track_name = ref_track.get("name", "")
    ref_id = ref_track.get("track_id", "")

    seen_ids: set[str] = {ref_id}  # exclude reference track from results
    seen_keys: set[str] = {_normalize_key(artist, track_name)}
    seen_titles: set[str] = {_normalize_track_name(track_name)}
    candidates: list[dict] = []

    # ------------------------------------------------------------------
    # Source 1: MusicBrainz related recordings
    # ------------------------------------------------------------------
    try:
        mb_pairs = find_related_recordings(artist, track_name, limit=10)
        if mb_pairs:
            logger.info("MusicBrainz returned %d related recordings", len(mb_pairs))
            resolved = _resolve_to_youtube(
                mb_pairs, seen_ids, seen_keys, seen_titles, "musicbrainz"
            )
            candidates.extend(resolved)
    except Exception as e:
        logger.debug("MusicBrainz related recordings failed: %s", e)

    if len(candidates) >= limit:
        return candidates[:limit]

    # ------------------------------------------------------------------
    # Source 2: Smart YouTube multi-query
    # ------------------------------------------------------------------
    yt_queries = [
        f"{artist} similar music",
        f"songs like {track_name}",
        f"{artist} type beat",
        f"music like {artist} {track_name}",
    ]

    for yq in yt_queries:
        if len(candidates) >= limit:
            break
        try:
            yt_results = search_candidates(seed_query=yq, limit=8)
            for track_info in yt_results:
                vid = track_info["track_id"]
                if vid in seen_ids:
                    continue
                key = _normalize_key(track_info["artist"], track_info["name"])
                if key in seen_keys:
                    continue
                title_key = _normalize_track_name(track_info["name"])
                if title_key in seen_titles:
                    logger.debug("Skipping same-song variant: %s", track_info["name"])
                    continue
                seen_ids.add(vid)
                seen_keys.add(key)
                seen_titles.add(title_key)
                candidates.append(track_info)
                logger.info("[yt-multi] %s - %s", track_info["artist"], track_info["name"])
        except Exception:
            continue

    if len(candidates) >= limit:
        return candidates[:limit]

    # ------------------------------------------------------------------
    # Source 3: Fallback — original broad YouTube search
    # ------------------------------------------------------------------
    remaining = limit - len(candidates)
    if remaining > 0:
        try:
            fallback = search_candidates(
                seed_query=f"{artist} similar music", limit=remaining + 10
            )
            for track_info in fallback:
                vid = track_info["track_id"]
                if vid in seen_ids:
                    continue
                key = _normalize_key(track_info["artist"], track_info["name"])
                if key in seen_keys:
                    continue
                title_key = _normalize_track_name(track_info["name"])
                if title_key in seen_titles:
                    continue
                seen_ids.add(vid)
                seen_keys.add(key)
                seen_titles.add(title_key)
                candidates.append(track_info)
                logger.info("[yt-fallback] %s - %s", track_info["artist"], track_info["name"])
                if len(candidates) >= limit:
                    break
        except Exception as e:
            logger.debug("YouTube fallback search failed: %s", e)

    return candidates[:limit]
