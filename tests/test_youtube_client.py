"""
Tests for YouTube Integration using yt-dlp.

All tests use mocks to avoid requiring actual YouTube API calls.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.youtube_client import (
    _extract_track_info,
    _extract_video_id,
    _normalize_popularity,
    _parse_song_and_artist,
    get_audio,
    get_related_videos,
    search_candidates,
    search_track,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_video_info():
    """Sample yt-dlp video info response."""
    return {
        "id": "dQw4w9WgXcQ",
        "title": "Rick Astley - Never Gonna Give You Up (Official Music Video)",
        "channel": "Rick Astley",
        "channel_id": "UCuAXFkgsw1L7xaCfnd5JJOw",
        "uploader": "Rick Astley",
        "uploader_id": "@RickAstley",
        "duration": 213,
        "view_count": 1500000000,
    }


@pytest.fixture
def mock_search_results(mock_video_info):
    """Sample yt-dlp search results response."""
    return {
        "entries": [mock_video_info],
    }


@pytest.fixture
def mock_multiple_search_results():
    """Sample yt-dlp search results with multiple entries."""
    return {
        "entries": [
            {
                "id": "video1",
                "title": "Artist 1 - Song 1",
                "channel": "Artist 1",
                "channel_id": "channel1",
                "duration": 180,
                "view_count": 1000000,
            },
            {
                "id": "video2",
                "title": "Artist 2 - Song 2",
                "channel": "Artist 2",
                "channel_id": "channel2",
                "duration": 200,
                "view_count": 500000,
            },
            {
                "id": "video3",
                "title": "Short Video",
                "channel": "Channel",
                "channel_id": "channel3",
                "duration": 30,  # Too short, should be filtered
                "view_count": 100000,
            },
            {
                "id": "video4",
                "title": "Long Compilation",
                "channel": "Mix Channel",
                "channel_id": "channel4",
                "duration": 3600,  # Too long, should be filtered
                "view_count": 200000,
            },
        ],
    }


# ============================================================================
# Test: _extract_video_id
# ============================================================================


class TestExtractVideoId:
    """Tests for _extract_video_id helper function."""

    def test_extracts_video_id_from_standard_url(self):
        """Should extract ID from standard YouTube URL."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert _extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extracts_video_id_from_short_url(self):
        """Should extract ID from youtu.be URL."""
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert _extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extracts_video_id_from_embed_url(self):
        """Should extract ID from embed URL."""
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        assert _extract_video_id(url) == "dQw4w9WgXcQ"

    def test_validates_video_id_directly(self):
        """Should return ID if query is already a valid video ID."""
        video_id = "dQw4w9WgXcQ"
        assert _extract_video_id(video_id) == video_id

    def test_returns_none_for_search_query(self):
        """Should return None for regular search queries."""
        assert _extract_video_id("never gonna give you up") is None

    def test_returns_none_for_invalid_id(self):
        """Should return None for invalid video IDs."""
        assert _extract_video_id("tooshort") is None
        assert _extract_video_id("this_is_way_too_long_to_be_a_video_id") is None


# ============================================================================
# Test: _parse_song_and_artist
# ============================================================================


class TestParseSongAndArtist:
    """Tests for _parse_song_and_artist helper function."""

    def test_parses_dash_format(self):
        """Should parse 'Artist - Song' format."""
        song, artist = _parse_song_and_artist("Queen - Bohemian Rhapsody", "Queen")
        assert song == "Bohemian Rhapsody"
        assert artist == "Queen"

    def test_parses_by_format(self):
        """Should parse 'Song by Artist' format."""
        song, artist = _parse_song_and_artist(
            "Bohemian Rhapsody by Queen", "QueenVEVO"
        )
        assert song == "Bohemian Rhapsody"
        assert artist == "Queen"

    def test_removes_official_video_suffix(self):
        """Should remove (Official Video) suffix."""
        song, artist = _parse_song_and_artist(
            "Queen - Bohemian Rhapsody (Official Music Video)", "Queen"
        )
        assert song == "Bohemian Rhapsody"
        assert artist == "Queen"

    def test_removes_lyrics_suffix(self):
        """Should remove (Lyrics) suffix."""
        song, artist = _parse_song_and_artist(
            "Queen - Bohemian Rhapsody (Lyrics)", "Queen"
        )
        assert song == "Bohemian Rhapsody"
        assert artist == "Queen"

    def test_falls_back_to_channel(self):
        """Should use channel as artist when no separator found."""
        song, artist = _parse_song_and_artist("Bohemian Rhapsody", "Queen")
        assert song == "Bohemian Rhapsody"
        assert artist == "Queen"


# ============================================================================
# Test: _normalize_popularity
# ============================================================================


class TestNormalizePopularity:
    """Tests for _normalize_popularity helper function."""

    def test_returns_zero_for_none(self):
        """Should return 0 for None view count."""
        assert _normalize_popularity(None) == 0

    def test_returns_zero_for_zero_views(self):
        """Should return 0 for zero views."""
        assert _normalize_popularity(0) == 0

    def test_returns_30_for_1k_views(self):
        """Should return ~30 for 1K views."""
        assert _normalize_popularity(1000) == 30

    def test_returns_60_for_1m_views(self):
        """Should return ~60 for 1M views."""
        assert _normalize_popularity(1000000) == 60

    def test_returns_90_for_1b_views(self):
        """Should return ~90 for 1B views."""
        assert _normalize_popularity(1000000000) == 90

    def test_caps_at_100(self):
        """Should not exceed 100."""
        # 10 trillion views would be log10 = 13, * 10 = 130
        assert _normalize_popularity(10_000_000_000_000) == 100


# ============================================================================
# Test: _extract_track_info
# ============================================================================


class TestExtractTrackInfo:
    """Tests for _extract_track_info helper function."""

    def test_extracts_all_fields(self, mock_video_info):
        """Should extract all expected fields from video info."""
        result = _extract_track_info(mock_video_info)

        assert result["track_id"] == "dQw4w9WgXcQ"
        assert result["name"] == "Never Gonna Give You Up"
        assert result["artist"] == "Rick Astley"
        assert result["artist_id"] == "UCuAXFkgsw1L7xaCfnd5JJOw"
        assert result["album"] == ""
        assert result["video_url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert result["genres"] == []
        assert result["duration_ms"] == 213000

    def test_handles_missing_fields(self):
        """Should handle missing fields gracefully."""
        info = {
            "id": "test123",
            "title": "Test Song",
        }
        result = _extract_track_info(info)

        assert result["track_id"] == "test123"
        assert result["name"] == "Test Song"
        assert result["artist"] == "Unknown"
        assert result["artist_id"] is None
        assert result["duration_ms"] == 0
        assert result["popularity"] == 0


# ============================================================================
# Test: search_track
# ============================================================================


class TestSearchTrack:
    """Tests for search_track function."""

    @patch("src.youtube_client.yt_dlp.YoutubeDL")
    def test_search_by_name(self, mock_ydl_class, mock_search_results):
        """Should search and return track info when searching by name."""
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = mock_search_results

        result = search_track("never gonna give you up")

        assert result is not None
        assert result["track_id"] == "dQw4w9WgXcQ"
        assert "Never Gonna Give You Up" in result["name"]

    @patch("src.youtube_client.yt_dlp.YoutubeDL")
    def test_search_by_video_id(self, mock_ydl_class, mock_video_info):
        """Should fetch video directly when query is a video ID."""
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = mock_video_info

        result = search_track("dQw4w9WgXcQ")

        assert result is not None
        assert result["track_id"] == "dQw4w9WgXcQ"
        # Should call with full URL, not search
        mock_ydl.extract_info.assert_called_once_with(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ", download=False
        )

    @patch("src.youtube_client.yt_dlp.YoutubeDL")
    def test_search_by_url(self, mock_ydl_class, mock_video_info):
        """Should fetch video directly when query is a URL."""
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = mock_video_info

        result = search_track("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        assert result is not None
        assert result["track_id"] == "dQw4w9WgXcQ"

    @patch("src.youtube_client.yt_dlp.YoutubeDL")
    def test_returns_none_when_not_found(self, mock_ydl_class):
        """Should return None when no results found."""
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = {"entries": []}

        result = search_track("nonexistent video xyz123456")

        assert result is None

    @patch("src.youtube_client.yt_dlp.YoutubeDL")
    def test_handles_download_error(self, mock_ydl_class):
        """Should return None on download error."""
        import yt_dlp

        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.side_effect = yt_dlp.DownloadError("Not found")

        result = search_track("invalid_video")

        assert result is None


# ============================================================================
# Test: get_audio
# ============================================================================


class TestGetAudio:
    """Tests for get_audio function."""

    def test_raises_on_empty_url(self):
        """Should raise ValueError for empty URL."""
        with pytest.raises(ValueError, match="Video URL or ID cannot be empty"):
            get_audio("")

    def test_raises_on_none_url(self):
        """Should raise ValueError for None URL."""
        with pytest.raises(ValueError, match="Video URL or ID cannot be empty"):
            get_audio(None)

    @patch("src.youtube_client.subprocess.run")
    def test_raises_when_ffmpeg_missing(self, mock_run):
        """Should raise RuntimeError when ffmpeg is not installed."""
        mock_run.side_effect = FileNotFoundError()

        with pytest.raises(RuntimeError, match="ffmpeg is required"):
            get_audio("dQw4w9WgXcQ")

    @patch("src.youtube_client.subprocess.run")
    @patch("src.youtube_client.yt_dlp.YoutubeDL")
    @patch("src.youtube_client.Path")
    def test_downloads_and_returns_bytes(self, mock_path, mock_ydl_class, mock_run):
        """Should download audio and return bytes."""
        # Mock ffmpeg check
        mock_run.return_value = MagicMock()

        # Mock yt-dlp
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl

        # Mock file reading
        mock_audio_file = MagicMock()
        mock_audio_file.exists.return_value = True
        mock_audio_file.read_bytes.return_value = b"fake audio data"
        mock_path.return_value.__truediv__.return_value = mock_audio_file

        result = get_audio("dQw4w9WgXcQ")

        assert result == b"fake audio data"

    @patch("src.youtube_client.subprocess.run")
    @patch("src.youtube_client.yt_dlp.YoutubeDL")
    def test_raises_on_download_error(self, mock_ydl_class, mock_run):
        """Should raise RuntimeError on download failure."""
        import yt_dlp

        mock_run.return_value = MagicMock()
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        mock_ydl.download.side_effect = yt_dlp.DownloadError("Download failed")

        with pytest.raises(RuntimeError, match="Failed to download"):
            get_audio("invalid_video_id")


# ============================================================================
# Test: search_candidates
# ============================================================================


class TestSearchCandidates:
    """Tests for search_candidates function."""

    @patch("src.youtube_client.yt_dlp.YoutubeDL")
    def test_search_with_query(self, mock_ydl_class, mock_multiple_search_results):
        """Should search using query."""
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = mock_multiple_search_results

        result = search_candidates("rock music")

        # Should filter out videos < 60s and > 600s
        assert len(result) == 2
        assert result[0]["track_id"] == "video1"
        assert result[1]["track_id"] == "video2"

    @patch("src.youtube_client.yt_dlp.YoutubeDL")
    def test_search_with_artist(self, mock_ydl_class, mock_multiple_search_results):
        """Should include artist in search query."""
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = mock_multiple_search_results

        search_candidates("rock music", seed_artist="Queen")

        # Verify search was called with artist included
        call_args = mock_ydl.extract_info.call_args
        assert "Queen" in call_args[0][0]

    def test_raises_without_query(self):
        """Should raise ValueError when no query is provided."""
        with pytest.raises(ValueError, match="seed_query must be provided"):
            search_candidates("")

    @patch("src.youtube_client.yt_dlp.YoutubeDL")
    def test_respects_limit(self, mock_ydl_class):
        """Should respect the limit parameter."""
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl

        # Create many results
        entries = [
            {
                "id": f"video{i}",
                "title": f"Song {i}",
                "channel": f"Artist {i}",
                "duration": 180,
                "view_count": 100000,
            }
            for i in range(20)
        ]
        mock_ydl.extract_info.return_value = {"entries": entries}

        result = search_candidates("rock music", limit=5)

        assert len(result) == 5

    @patch("src.youtube_client.yt_dlp.YoutubeDL")
    def test_handles_empty_results(self, mock_ydl_class):
        """Should return empty list when no results found."""
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = {"entries": []}

        result = search_candidates("nonexistent query")

        assert result == []

    @patch("src.youtube_client.yt_dlp.YoutubeDL")
    def test_filters_short_videos(self, mock_ydl_class):
        """Should filter out videos shorter than 60 seconds."""
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = {
            "entries": [
                {
                    "id": "short",
                    "title": "Short Video",
                    "channel": "Channel",
                    "duration": 30,
                    "view_count": 1000,
                },
                {
                    "id": "normal",
                    "title": "Normal Video",
                    "channel": "Channel",
                    "duration": 180,
                    "view_count": 1000,
                },
            ]
        }

        result = search_candidates("test")

        assert len(result) == 1
        assert result[0]["track_id"] == "normal"

    @patch("src.youtube_client.yt_dlp.YoutubeDL")
    def test_filters_long_videos(self, mock_ydl_class):
        """Should filter out videos longer than 600 seconds."""
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = {
            "entries": [
                {
                    "id": "long",
                    "title": "Long Compilation",
                    "channel": "Channel",
                    "duration": 3600,
                    "view_count": 1000,
                },
                {
                    "id": "normal",
                    "title": "Normal Video",
                    "channel": "Channel",
                    "duration": 180,
                    "view_count": 1000,
                },
            ]
        }

        result = search_candidates("test")

        assert len(result) == 1
        assert result[0]["track_id"] == "normal"

    @patch("src.youtube_client.yt_dlp.YoutubeDL")
    def test_deduplicates_results(self, mock_ydl_class):
        """Should remove duplicate video IDs."""
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = {
            "entries": [
                {
                    "id": "same_id",
                    "title": "Video 1",
                    "channel": "Channel",
                    "duration": 180,
                    "view_count": 1000,
                },
                {
                    "id": "same_id",
                    "title": "Video 1 Copy",
                    "channel": "Channel",
                    "duration": 180,
                    "view_count": 1000,
                },
            ]
        }

        result = search_candidates("test")

        assert len(result) == 1


# ============================================================================
# Test: get_related_videos
# ============================================================================


class TestGetRelatedVideos:
    """Tests for get_related_videos function."""

    @patch("src.youtube_client.search_candidates")
    @patch("src.youtube_client.search_track")
    def test_gets_related_videos(self, mock_search_track, mock_search_candidates):
        """Should get related videos based on original video info."""
        mock_search_track.return_value = {
            "track_id": "original",
            "name": "Original Song",
            "artist": "Original Artist",
        }
        mock_search_candidates.return_value = [
            {"track_id": "related1"},
            {"track_id": "related2"},
        ]

        result = get_related_videos("original", limit=10)

        assert len(result) == 2
        mock_search_candidates.assert_called_once_with(
            "Original Artist Original Song", "Original Artist", limit=10
        )

    @patch("src.youtube_client.search_track")
    def test_returns_empty_when_video_not_found(self, mock_search_track):
        """Should return empty list when original video not found."""
        mock_search_track.return_value = None

        result = get_related_videos("invalid_id")

        assert result == []


# ============================================================================
# Integration-style tests (still mocked, but testing component interaction)
# ============================================================================


class TestYouTubeClientIntegration:
    """Integration-style tests for the YouTube client module."""

    @patch("src.youtube_client.yt_dlp.YoutubeDL")
    def test_full_workflow_search_and_get_candidates(self, mock_ydl_class):
        """Test typical workflow: search track, then get candidates."""
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl

        # First search returns the seed track
        seed_info = {
            "entries": [
                {
                    "id": "seed_id",
                    "title": "Queen - Bohemian Rhapsody",
                    "channel": "Queen Official",
                    "channel_id": "queen_channel",
                    "duration": 355,
                    "view_count": 1_500_000_000,
                }
            ]
        }

        # Candidates search returns related tracks
        candidates_info = {
            "entries": [
                {
                    "id": "candidate1",
                    "title": "Queen - Don't Stop Me Now",
                    "channel": "Queen Official",
                    "duration": 209,
                    "view_count": 800_000_000,
                },
                {
                    "id": "candidate2",
                    "title": "Queen - We Will Rock You",
                    "channel": "Queen Official",
                    "duration": 122,
                    "view_count": 600_000_000,
                },
            ]
        }

        mock_ydl.extract_info.side_effect = [seed_info, candidates_info]

        # Search for seed track
        seed_track = search_track("Bohemian Rhapsody")
        assert seed_track is not None
        assert seed_track["track_id"] == "seed_id"
        assert seed_track["name"] == "Bohemian Rhapsody"
        assert seed_track["artist"] == "Queen"

        # Get candidates based on seed
        candidates = search_candidates(
            seed_query=seed_track["name"],
            seed_artist=seed_track["artist"],
            limit=10,
        )

        assert len(candidates) == 2
        assert candidates[0]["track_id"] == "candidate1"
        assert candidates[1]["track_id"] == "candidate2"
