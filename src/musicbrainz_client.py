"""
MusicBrainz API client for finding alternate recordings.

Searches for covers, remixes, and alternate versions of tracks.
No API key required — only a descriptive User-Agent header.
Rate-limited to 1 request per 1.1 seconds per MusicBrainz policy.
"""

import logging
import time

import requests

logger = logging.getLogger(__name__)

MUSICBRAINZ_API_URL = "https://musicbrainz.org/ws/2"
USER_AGENT = "Lyrebird/0.1.0 (https://github.com/lyrebird)"

_last_request_time: float = 0.0


def _rate_limited_get(url: str, params: dict) -> dict | None:
    """
    Make a rate-limited GET request to MusicBrainz.

    Enforces at least 1.1 seconds between requests to comply with
    MusicBrainz rate-limit policy.

    Args:
        url: Full URL to request.
        params: Query parameters.

    Returns:
        Parsed JSON response, or None on any failure.
    """
    global _last_request_time

    elapsed = time.monotonic() - _last_request_time
    if elapsed < 1.1:
        time.sleep(1.1 - elapsed)

    params = {**params, "fmt": "json"}
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}

    try:
        _last_request_time = time.monotonic()
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.debug("MusicBrainz request failed: %s", e)
        return None


def find_related_recordings(artist: str, track: str, limit: int = 10) -> list[tuple[str, str]]:
    """
    Search MusicBrainz for alternate versions, covers, or remixes of a track.

    Args:
        artist: Original artist name.
        track: Original track name.
        limit: Maximum number of results.

    Returns:
        List of (artist, track) tuples for related recordings.
    """
    # Search for recordings matching the track name but not the original artist
    query = f'recording:"{track}" NOT artist:"{artist}"'
    data = _rate_limited_get(
        f"{MUSICBRAINZ_API_URL}/recording",
        {"query": query, "limit": limit},
    )
    if not data:
        return []

    results = []
    for rec in data.get("recordings", []):
        rec_title = rec.get("title")
        artist_credit = rec.get("artist-credit", [])
        if not rec_title or not artist_credit:
            continue
        rec_artist = artist_credit[0].get("name", "")
        if rec_artist and rec_title:
            results.append((rec_artist, rec_title))

    return results[:limit]


def find_recordings_by_related_artists(
    related_artists: list[str], limit_per_artist: int = 3
) -> list[tuple[str, str]]:
    """
    Find top recordings by related artists from MusicBrainz.

    Args:
        related_artists: List of artist names to search.
        limit_per_artist: Max recordings to fetch per artist.

    Returns:
        List of (artist, track) tuples.
    """
    results = []
    for artist in related_artists:
        data = _rate_limited_get(
            f"{MUSICBRAINZ_API_URL}/recording",
            {"query": f'artist:"{artist}"', "limit": limit_per_artist},
        )
        if not data:
            continue

        for rec in data.get("recordings", []):
            rec_title = rec.get("title")
            artist_credit = rec.get("artist-credit", [])
            if not rec_title or not artist_credit:
                continue
            rec_artist = artist_credit[0].get("name", "")
            if rec_artist and rec_title:
                results.append((rec_artist, rec_title))

    return results
