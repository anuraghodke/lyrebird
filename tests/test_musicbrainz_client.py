"""
Tests for MusicBrainz client.

All tests use mocks to avoid actual API calls and rate-limit delays.
"""

from unittest.mock import patch

from src.musicbrainz_client import (
    find_recordings_by_related_artists,
    find_related_recordings,
)

# ============================================================================
# Test: find_related_recordings
# ============================================================================


class TestFindRelatedRecordings:
    """Tests for find_related_recordings function."""

    @patch("src.musicbrainz_client._rate_limited_get")
    def test_returns_related_recordings(self, mock_get):
        """Should return (artist, track) tuples from MusicBrainz results."""
        mock_get.return_value = {
            "recordings": [
                {
                    "title": "Bohemian Rhapsody",
                    "artist-credit": [{"name": "The Muppets"}],
                },
                {
                    "title": "Bohemian Rhapsody (Live)",
                    "artist-credit": [{"name": "Pentatonix"}],
                },
            ]
        }

        result = find_related_recordings("Queen", "Bohemian Rhapsody", limit=5)

        assert len(result) == 2
        assert result[0] == ("The Muppets", "Bohemian Rhapsody")
        assert result[1] == ("Pentatonix", "Bohemian Rhapsody (Live)")

    @patch("src.musicbrainz_client._rate_limited_get")
    def test_returns_empty_on_no_results(self, mock_get):
        """Should return empty list when no recordings found."""
        mock_get.return_value = {"recordings": []}

        result = find_related_recordings("Unknown", "Unknown Track")

        assert result == []

    @patch("src.musicbrainz_client._rate_limited_get")
    def test_returns_empty_on_api_failure(self, mock_get):
        """Should return empty list when API call fails."""
        mock_get.return_value = None

        result = find_related_recordings("Queen", "Bohemian Rhapsody")

        assert result == []

    @patch("src.musicbrainz_client._rate_limited_get")
    def test_respects_limit(self, mock_get):
        """Should respect the limit parameter."""
        mock_get.return_value = {
            "recordings": [
                {"title": f"Song {i}", "artist-credit": [{"name": f"Artist {i}"}]}
                for i in range(10)
            ]
        }

        result = find_related_recordings("Queen", "Bohemian Rhapsody", limit=3)

        assert len(result) == 3

    @patch("src.musicbrainz_client._rate_limited_get")
    def test_skips_entries_without_artist_credit(self, mock_get):
        """Should skip recordings missing artist-credit."""
        mock_get.return_value = {
            "recordings": [
                {"title": "No Artist"},
                {
                    "title": "Has Artist",
                    "artist-credit": [{"name": "Some Artist"}],
                },
            ]
        }

        result = find_related_recordings("Queen", "Bohemian Rhapsody")

        assert len(result) == 1
        assert result[0] == ("Some Artist", "Has Artist")


# ============================================================================
# Test: find_recordings_by_related_artists
# ============================================================================


class TestFindRecordingsByRelatedArtists:
    """Tests for find_recordings_by_related_artists function."""

    @patch("src.musicbrainz_client._rate_limited_get")
    def test_returns_recordings_from_multiple_artists(self, mock_get):
        """Should query each artist and aggregate results."""
        mock_get.side_effect = [
            {
                "recordings": [
                    {"title": "Song A", "artist-credit": [{"name": "Artist X"}]},
                ]
            },
            {
                "recordings": [
                    {"title": "Song B", "artist-credit": [{"name": "Artist Y"}]},
                ]
            },
        ]

        result = find_recordings_by_related_artists(["Artist X", "Artist Y"])

        assert len(result) == 2
        assert result[0] == ("Artist X", "Song A")
        assert result[1] == ("Artist Y", "Song B")

    @patch("src.musicbrainz_client._rate_limited_get")
    def test_handles_api_failure_for_one_artist(self, mock_get):
        """Should continue if one artist request fails."""
        mock_get.side_effect = [
            None,  # first artist fails
            {
                "recordings": [
                    {"title": "Song B", "artist-credit": [{"name": "Artist Y"}]},
                ]
            },
        ]

        result = find_recordings_by_related_artists(["Artist X", "Artist Y"])

        assert len(result) == 1
        assert result[0] == ("Artist Y", "Song B")

    @patch("src.musicbrainz_client._rate_limited_get")
    def test_returns_empty_for_empty_input(self, mock_get):
        """Should return empty list for no artists."""
        result = find_recordings_by_related_artists([])

        assert result == []
        mock_get.assert_not_called()
