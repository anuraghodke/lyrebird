"""
Spotify API wrapper for authentication, track search, and preview retrieval.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def get_spotify_credentials() -> tuple[str, str]:
    """
    Get Spotify credentials from environment variables.

    Returns:
        Tuple of (client_id, client_secret)

    Raises:
        ValueError: If credentials are not set in environment
    """
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise ValueError(
            "Spotify credentials not found. "
            "Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env file"
        )

    return client_id, client_secret


def authenticate_spotify():
    """
    Authenticate with Spotify API using client credentials flow.

    Returns:
        Authenticated Spotipy client
    """
    # Placeholder - will be implemented in Step 2
    raise NotImplementedError("Will be implemented in Step 2: Spotify Integration")


def search_track(query: str) -> dict:
    """
    Search for a track on Spotify.

    Args:
        query: Search query (song name, artist, or track ID)

    Returns:
        Dict with track_id, preview_url, and metadata
    """
    # Placeholder - will be implemented in Step 2
    raise NotImplementedError("Will be implemented in Step 2: Spotify Integration")


def get_preview_audio(url: str) -> bytes:
    """
    Download preview audio from Spotify URL.

    Args:
        url: Spotify preview URL

    Returns:
        Audio data as bytes (in-memory, no caching)
    """
    # Placeholder - will be implemented in Step 2
    raise NotImplementedError("Will be implemented in Step 2: Spotify Integration")


def search_candidates(genre: str, artist: str, limit: int = 100) -> list[dict]:
    """
    Search for candidate tracks similar to the reference.

    Args:
        genre: Genre to search within
        artist: Reference artist for recommendations
        limit: Maximum number of candidates to return

    Returns:
        List of track dicts with track_id, preview_url, and metadata
    """
    # Placeholder - will be implemented in Step 2
    raise NotImplementedError("Will be implemented in Step 2: Spotify Integration")
