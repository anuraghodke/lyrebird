"""
Tests for diverse candidate finder.

All tests use mocks to avoid actual API calls.
"""

from unittest.mock import patch

from src.candidate_finder import (
    _normalize_key,
    _normalize_track_name,
    _resolve_to_youtube,
    get_diverse_candidates,
)

# ============================================================================
# Helpers
# ============================================================================


def _make_track(track_id, name="Song", artist="Artist", duration_ms=200_000):
    """Create a minimal track info dict for testing."""
    return {
        "track_id": track_id,
        "name": name,
        "artist": artist,
        "artist_id": "channel_id",
        "album": "",
        "video_url": f"https://www.youtube.com/watch?v={track_id}",
        "genres": [],
        "popularity": 50,
        "duration_ms": duration_ms,
    }


# ============================================================================
# Test: _normalize_key
# ============================================================================


class TestNormalizeKey:
    """Tests for _normalize_key helper."""

    def test_lowercases_and_strips(self):
        assert _normalize_key("  Queen  ", "  Bohemian Rhapsody  ") == "queen:::bohemian rhapsody"

    def test_collapses_whitespace(self):
        assert _normalize_key("The   Beatles", "Hey   Jude") == "the beatles:::hey jude"

    def test_identical_inputs_match(self):
        k1 = _normalize_key("Queen", "Bohemian Rhapsody")
        k2 = _normalize_key("queen", "bohemian rhapsody")
        assert k1 == k2

    def test_different_inputs_differ(self):
        k1 = _normalize_key("Queen", "Bohemian Rhapsody")
        k2 = _normalize_key("Queen", "We Will Rock You")
        assert k1 != k2


# ============================================================================
# Test: _normalize_track_name
# ============================================================================


class TestNormalizeTrackName:
    """Tests for _normalize_track_name helper."""

    def test_strips_bracket_suffix(self):
        assert _normalize_track_name("Redbone [Lofi Fruits Release]") == "redbone"

    def test_strips_paren_suffix(self):
        assert _normalize_track_name("Redbone (SKA PUNK COVER)") == "redbone"

    def test_case_insensitive(self):
        assert _normalize_track_name("RedBone") == "redbone"

    def test_strips_cover_keyword(self):
        assert _normalize_track_name("Redbone Cover") == "redbone"

    def test_strips_remix_keyword(self):
        assert _normalize_track_name("Redbone Remix") == "redbone"

    def test_strips_acoustic_keyword(self):
        assert _normalize_track_name("Redbone Acoustic") == "redbone"

    def test_strips_live_keyword(self):
        assert _normalize_track_name("Bohemian Rhapsody Live") == "bohemian rhapsody"

    def test_preserves_different_songs(self):
        a = _normalize_track_name("Redbone")
        b = _normalize_track_name("Bohemian Rhapsody")
        assert a != b

    def test_matches_same_song_variants(self):
        variants = [
            "Redbone",
            "Redbone [Lofi Fruits Release]",
            "RedBone",
            "Redbone (Piano Cover)",
            "Redbone - Acoustic Version",
        ]
        keys = {_normalize_track_name(v) for v in variants}
        assert len(keys) == 1


# ============================================================================
# Test: _resolve_to_youtube
# ============================================================================


class TestResolveToYoutube:
    """Tests for _resolve_to_youtube helper."""

    @patch("src.candidate_finder.search_track")
    def test_resolves_pairs_to_tracks(self, mock_search):
        """Should resolve (artist, track) pairs to YouTube track info."""
        mock_search.return_value = _make_track("vid1", "Song 1", "Artist 1")

        seen_ids = set()
        seen_keys = set()
        seen_titles = set()
        result = _resolve_to_youtube(
            [("Artist 1", "Song 1")], seen_ids, seen_keys, seen_titles, "test"
        )

        assert len(result) == 1
        assert result[0]["track_id"] == "vid1"
        assert "vid1" in seen_ids

    @patch("src.candidate_finder.search_track")
    def test_skips_duplicate_keys(self, mock_search):
        """Should skip tracks whose normalized key is already seen."""
        seen_keys = {_normalize_key("Artist 1", "Song 1")}
        result = _resolve_to_youtube(
            [("Artist 1", "Song 1")], set(), seen_keys, set(), "test"
        )

        assert len(result) == 0
        mock_search.assert_not_called()

    @patch("src.candidate_finder.search_track")
    def test_skips_duplicate_ids(self, mock_search):
        """Should skip tracks whose video ID is already seen."""
        mock_search.return_value = _make_track("vid1")

        seen_ids = {"vid1"}
        result = _resolve_to_youtube(
            [("Artist", "Song")], seen_ids, set(), set(), "test"
        )

        assert len(result) == 0

    @patch("src.candidate_finder.search_track")
    def test_skips_same_title_variant(self, mock_search):
        """Should skip tracks that are covers/remixes of an already-seen title."""
        seen_titles = {_normalize_track_name("Redbone")}
        result = _resolve_to_youtube(
            [("Some Cover Artist", "Redbone (Lofi Remix)")], set(), set(), seen_titles, "test"
        )

        assert len(result) == 0
        mock_search.assert_not_called()

    @patch("src.candidate_finder.search_track")
    def test_filters_short_duration(self, mock_search):
        """Should filter out tracks shorter than MIN_TRACK_DURATION_S."""
        mock_search.return_value = _make_track("vid1", duration_ms=30_000)  # 30s

        result = _resolve_to_youtube(
            [("Artist", "Song")], set(), set(), set(), "test"
        )

        assert len(result) == 0

    @patch("src.candidate_finder.search_track")
    def test_filters_long_duration(self, mock_search):
        """Should filter out tracks longer than MAX_TRACK_DURATION_S."""
        mock_search.return_value = _make_track("vid1", duration_ms=700_000)  # 700s

        result = _resolve_to_youtube(
            [("Artist", "Song")], set(), set(), set(), "test"
        )

        assert len(result) == 0

    @patch("src.candidate_finder.search_track")
    def test_skips_not_found(self, mock_search):
        """Should skip when search_track returns None."""
        mock_search.return_value = None

        result = _resolve_to_youtube(
            [("Artist", "Song")], set(), set(), set(), "test"
        )

        assert len(result) == 0


# ============================================================================
# Test: get_diverse_candidates
# ============================================================================


class TestGetDiverseCandidates:
    """Tests for get_diverse_candidates orchestrator."""

    @patch("src.candidate_finder.search_candidates")
    @patch("src.candidate_finder.find_related_recordings")
    @patch("src.candidate_finder.search_track")
    def test_combines_multiple_sources(self, mock_search_track, mock_mb, mock_yt):
        """Should combine MusicBrainz and YouTube results."""
        ref = _make_track("ref_id", "Original Song", "Original Artist")

        # MusicBrainz returns one pair with a different title
        mock_mb.return_value = [("Cover Artist", "Different Song")]
        mock_search_track.return_value = _make_track("mb_vid", "Different Song", "Cover Artist")

        # YouTube returns additional candidates
        mock_yt.return_value = [
            _make_track("yt_vid1", "Similar Song 1", "Artist A"),
            _make_track("yt_vid2", "Similar Song 2", "Artist B"),
        ]

        result = get_diverse_candidates(ref, limit=10)

        assert len(result) >= 1  # at least the MusicBrainz result
        track_ids = [t["track_id"] for t in result]
        assert "ref_id" not in track_ids  # reference excluded

    @patch("src.candidate_finder.search_candidates")
    @patch("src.candidate_finder.find_related_recordings")
    def test_excludes_reference_track(self, mock_mb, mock_yt):
        """Should exclude the reference track from results."""
        ref = _make_track("ref_id", "Song", "Artist")
        mock_mb.return_value = []
        mock_yt.return_value = [
            _make_track("ref_id", "Song", "Artist"),  # same as ref
            _make_track("other_id", "Other Song", "Other Artist"),
        ]

        result = get_diverse_candidates(ref, limit=10)

        track_ids = [t["track_id"] for t in result]
        assert "ref_id" not in track_ids

    @patch("src.candidate_finder.search_candidates")
    @patch("src.candidate_finder.find_related_recordings")
    def test_respects_limit(self, mock_mb, mock_yt):
        """Should not return more than limit candidates."""
        ref = _make_track("ref_id", "Song", "Artist")
        mock_mb.return_value = []
        mock_yt.return_value = [
            _make_track(f"vid_{i}", f"Unique Title {i}", f"Artist {i}")
            for i in range(20)
        ]

        result = get_diverse_candidates(ref, limit=5)

        assert len(result) <= 5

    @patch("src.candidate_finder.search_candidates")
    @patch("src.candidate_finder.find_related_recordings")
    def test_survives_musicbrainz_failure(self, mock_mb, mock_yt):
        """Should still return YouTube results if MusicBrainz fails."""
        ref = _make_track("ref_id", "Song", "Artist")
        mock_mb.side_effect = Exception("MusicBrainz down")
        mock_yt.return_value = [
            _make_track("yt_vid", "YT Song", "YT Artist"),
        ]

        result = get_diverse_candidates(ref, limit=10)

        assert len(result) >= 1

    @patch("src.candidate_finder.search_candidates")
    @patch("src.candidate_finder.find_related_recordings")
    def test_returns_empty_when_all_fail(self, mock_mb, mock_yt):
        """Should return empty list when all sources fail."""
        ref = _make_track("ref_id", "Song", "Artist")
        mock_mb.side_effect = Exception("fail")
        mock_yt.return_value = []

        result = get_diverse_candidates(ref, limit=10)

        assert result == []

    @patch("src.candidate_finder.search_candidates")
    @patch("src.candidate_finder.find_related_recordings")
    def test_deduplicates_across_sources(self, mock_mb, mock_yt):
        """Should deduplicate candidates across sources by video ID."""
        ref = _make_track("ref_id", "Song", "Artist")
        mock_mb.return_value = []
        # Return the same track from multiple YouTube queries
        same_track = _make_track("dup_vid", "Dup Song", "Dup Artist")
        mock_yt.return_value = [same_track, same_track]

        result = get_diverse_candidates(ref, limit=10)

        ids = [t["track_id"] for t in result]
        assert ids.count("dup_vid") == 1

    @patch("src.candidate_finder.search_candidates")
    @patch("src.candidate_finder.find_related_recordings")
    def test_filters_same_song_covers(self, mock_mb, mock_yt):
        """Should filter out covers/remixes that are the same song."""
        ref = _make_track("ref_id", "Redbone", "Childish Gambino")
        mock_mb.return_value = []
        mock_yt.return_value = [
            _make_track("vid1", "Redbone [Lofi Fruits Release]", "Lofi Artist"),
            _make_track("vid2", "Redbone (Piano Cover)", "Piano Guy"),
            _make_track("vid3", "Totally Different Song", "Other Artist"),
        ]

        result = get_diverse_candidates(ref, limit=10)

        names = [t["name"] for t in result]
        assert "Totally Different Song" in names
        # The Redbone variants should be filtered by title dedup
        assert len([n for n in names if "redbone" in n.lower()]) == 0
