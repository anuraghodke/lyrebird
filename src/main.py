"""
CLI entry point for the music similarity tool.

Usage:
    lyrebird search "Bohemian Rhapsody" --type melody
    lyrebird search "Uptown Funk" --type rhythm
    lyrebird search "https://youtube.com/watch?v=..." --type both
"""

import sys

import click
import numpy as np
from dotenv import load_dotenv

from src.audio_analyzer import AudioAnalysisError, analyze_track, load_audio_from_bytes
from src.code_generator import (
    delete_function,
    generate_comparison_function,
    list_saved_functions,
    save_function,
)
from src.explainer import generate_reasoning
from src.sandbox import compile_and_execute, load_function_from_file, run_comparison
from src.similarity import find_similar
from src.youtube_client import get_audio, search_candidates, search_track, trim_audio

load_dotenv()


def _parse_time(time_str: str) -> float:
    """
    Parse a time string to seconds.

    Accepts formats:
    - "3:50" -> 230 seconds
    - "230" -> 230 seconds
    - "1:05:30" -> 3930 seconds

    Args:
        time_str: Time string to parse

    Returns:
        Time in seconds as float
    """
    time_str = time_str.strip()
    parts = time_str.split(":")

    if len(parts) == 1:
        return float(parts[0])
    elif len(parts) == 2:
        minutes, seconds = parts
        return float(minutes) * 60 + float(seconds)
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    else:
        raise ValueError(f"Invalid time format: {time_str}")


def _parse_interval(interval_str: str) -> tuple[float, float]:
    """
    Parse an interval string like "[3:50, 5:00]" or "[230, 300]".

    Args:
        interval_str: Interval string in format "[start, end]"

    Returns:
        Tuple of (start_seconds, end_seconds)
    """
    # Remove brackets and whitespace
    cleaned = interval_str.strip().strip("[]")
    parts = cleaned.split(",")

    if len(parts) != 2:
        raise ValueError(
            f"Invalid interval format: {interval_str}. "
            "Expected format: [start, end] e.g., [3:50, 5:00] or [230, 300]"
        )

    start = _parse_time(parts[0])
    end = _parse_time(parts[1])

    if start >= end:
        raise ValueError(f"Start time ({start}s) must be less than end time ({end}s)")

    return start, end


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """CLI Music Similarity Tool - Find similar songs based on audio features."""
    pass


@cli.command()
@click.argument("query")
@click.option(
    "--type",
    "feature_type",
    type=click.Choice(["melody", "rhythm", "both"]),
    default="both",
    help="Type of similarity to search for",
)
@click.option(
    "--limit",
    default=5,
    type=int,
    help="Number of similar tracks to return",
)
@click.option(
    "--threshold",
    default=0.0,
    type=float,
    help="Minimum similarity threshold (0.0-1.0)",
)
@click.option(
    "--candidates",
    default=20,
    type=int,
    help="Number of candidate tracks to analyze",
)
@click.option(
    "--interval",
    default=None,
    type=str,
    help="Time interval to analyze, e.g., '[3:50, 5:00]' or '[230, 300]'",
)
@click.option(
    "--nl",
    default=None,
    type=str,
    help="Natural language description of comparison criteria (uses Claude to generate function)",
)
@click.option(
    "--fn",
    default=None,
    type=str,
    help="Name of a saved comparison function to use",
)
def search(
    query: str,
    feature_type: str,
    limit: int,
    threshold: float,
    candidates: int,
    interval: str | None,
    nl: str | None,
    fn: str | None,
):
    """Search for songs similar to QUERY based on audio features.

    QUERY can be a song name, artist + song, YouTube URL, or video ID.

    Examples:
        lyrebird search "Never Gonna Give You Up"
        lyrebird search "Rick Astley - Never Gonna Give You Up"
        lyrebird search "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        lyrebird search dQw4w9WgXcQ --type melody --limit 3
        lyrebird search "Redbone" --interval "[3:50, 5:00]"
        lyrebird search "Bohemian Rhapsody" --nl "same intervals between notes"
        lyrebird search "Yesterday" --fn interval_based_comparison
    """
    # Validate --nl and --fn are not both provided
    if nl and fn:
        click.echo("Error: Cannot use both --nl and --fn options together.", err=True)
        sys.exit(1)

    # Parse interval if provided
    parsed_interval = None
    if interval:
        try:
            parsed_interval = _parse_interval(interval)
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

    try:
        _run_search(query, feature_type, limit, threshold, candidates, parsed_interval, nl, fn)
    except KeyboardInterrupt:
        click.echo("\n\nSearch cancelled.")
        sys.exit(1)
    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        sys.exit(1)


def _run_search(
    query: str,
    feature_type: str,
    limit: int,
    threshold: float,
    num_candidates: int,
    interval: tuple[float, float] | None = None,
    nl_prompt: str | None = None,
    fn_name: str | None = None,
):
    """Run the similarity search pipeline."""
    # Step 1: Find reference track
    click.echo(f"Searching for: {query}")
    ref_track = search_track(query)

    if not ref_track:
        click.echo("Could not find track. Try a different search query.", err=True)
        sys.exit(1)

    click.echo(f"Found: {ref_track['name']} by {ref_track['artist']}")
    click.echo(f"       {ref_track['video_url']}")

    # Step 2: Download reference track (before generating function to avoid wasted API calls)
    if interval:
        start, end = interval
        click.echo(f"\nDownloading reference track (interval {start:.0f}s - {end:.0f}s)...")
    else:
        click.echo("\nDownloading reference track...")

    try:
        ref_audio = get_audio(ref_track["track_id"])
    except Exception as e:
        click.echo(f"Failed to download audio: {e}", err=True)
        sys.exit(1)

    # Trim audio if interval specified
    if interval:
        try:
            ref_audio = trim_audio(ref_audio, interval[0], interval[1])
        except (ValueError, RuntimeError) as e:
            click.echo(f"Failed to trim audio: {e}", err=True)
            sys.exit(1)

    # Step 3: Generate or load comparison function (after successful download)
    comparison_func = None
    generated_fn_name = None

    if nl_prompt:
        click.echo(f"\nGenerating comparison function for: {nl_prompt}")
        try:
            generated_fn_name, function_code = generate_comparison_function(nl_prompt)
            click.echo(f"Generated function: {generated_fn_name}")

            # Save the function
            save_function(generated_fn_name, function_code, nl_prompt)
            click.echo(f"Saved to generated_functions.py")

            # Compile for use
            comparison_func = compile_and_execute(function_code, generated_fn_name)
        except Exception as e:
            click.echo(f"Failed to generate comparison function: {e}", err=True)
            sys.exit(1)

    elif fn_name:
        click.echo(f"\nLoading saved function: {fn_name}")
        try:
            comparison_func = load_function_from_file(fn_name)
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

    # Step 4: Analyze reference track
    click.echo("\nAnalyzing reference track...")
    try:
        ref_features = analyze_track(ref_audio, feature_type)
    except AudioAnalysisError as e:
        click.echo(f"Failed to analyze audio: {e}", err=True)
        sys.exit(1)
    except ImportError:
        click.echo(
            "Essentia is required for audio analysis.\n"
            "Install it with: pip install 'lyrebird[audio]'",
            err=True,
        )
        sys.exit(1)

    click.echo(f"Extracted {len(ref_features)} audio features ({feature_type})")

    # Step 3: Find candidate tracks
    click.echo(f"\nSearching for {num_candidates} candidate tracks...")

    # Use artist name to find similar music
    candidate_tracks = search_candidates(
        seed_query=f"{ref_track['artist']} similar music",
        limit=num_candidates,
    )

    if not candidate_tracks:
        click.echo("No candidate tracks found.", err=True)
        sys.exit(1)

    # Filter out the reference track itself
    candidate_tracks = [
        c for c in candidate_tracks if c["track_id"] != ref_track["track_id"]
    ]

    click.echo(f"Found {len(candidate_tracks)} candidates")

    # Step 4: Analyze candidates
    click.echo("\nAnalyzing candidates (this may take a while)...")
    candidate_features = []
    candidate_audio_data = []  # Store raw audio for custom comparison
    valid_candidates = []

    for i, candidate in enumerate(candidate_tracks):
        prefix = f"  [{i + 1}/{len(candidate_tracks)}]"
        name = candidate['name'][:50]
        try:
            audio = get_audio(candidate["track_id"])

            if comparison_func:
                # Store raw audio for custom comparison
                audio_array = load_audio_from_bytes(audio)
                candidate_audio_data.append(audio_array)
            else:
                # Extract features for default comparison
                features = analyze_track(audio, feature_type)
                candidate_features.append(features)

            valid_candidates.append(candidate)
            click.echo(f"{prefix} ✓ {name}")
        except Exception as e:
            click.echo(f"{prefix} ✗ {name} ({e})")
            continue

    if not valid_candidates:
        click.echo("No candidates could be analyzed.", err=True)
        sys.exit(1)

    click.echo(f"\nSuccessfully analyzed {len(valid_candidates)} tracks")

    # Step 5: Find similar tracks
    click.echo("\nComputing similarity...")

    if comparison_func:
        # Use custom comparison function
        ref_audio_array = load_audio_from_bytes(ref_audio)
        results = []
        for i, cand_audio in enumerate(candidate_audio_data):
            candidate_name = valid_candidates[i]['name'][:50]
            try:
                score = run_comparison(comparison_func, ref_audio_array, cand_audio)
                if score >= threshold:
                    results.append((i, score))
            except RuntimeError as e:
                click.echo(f"Warning: Comparison failed for '{candidate_name}': {e}", err=True)
                continue

        # Sort by score descending and limit
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:limit]
    else:
        # Use default cosine similarity
        results = find_similar(
            ref_features,
            np.array(candidate_features),
            k=min(limit, len(valid_candidates)),
            threshold=threshold,
        )

    if not results:
        click.echo(f"No tracks found with similarity >= {threshold}")
        sys.exit(0)

    # Step 6: Display results with explanations
    click.echo("\n" + "=" * 60)
    click.echo(f"Similar tracks to: {ref_track['name']}")
    if comparison_func:
        fn_display = generated_fn_name or fn_name
        click.echo(f"Using comparison: {fn_display}")
    click.echo("=" * 60)

    for rank, (idx, score) in enumerate(results, 1):
        candidate = valid_candidates[idx]
        click.echo(f"\n{rank}. {candidate['name']}")
        click.echo(f"   Artist: {candidate['artist']}")
        click.echo(f"   Similarity: {score:.1%}")
        click.echo(f"   URL: {candidate['video_url']}")

        # Generate explanations (only for default comparison)
        if not comparison_func:
            explanations = generate_reasoning(
                ref_features, candidate_features[idx], feature_type
            )
            if explanations:
                click.echo("   Why similar:")
                for explanation in explanations:
                    click.echo(f"     • {explanation}")

    click.echo("\n" + "=" * 60)


@cli.group()
def functions():
    """Manage saved comparison functions."""
    pass


@functions.command("list")
def functions_list():
    """List all saved comparison functions."""
    saved = list_saved_functions()

    if not saved:
        click.echo("No saved functions found.")
        click.echo("Generate one with: lyrebird search \"Song\" --nl \"your comparison criteria\"")
        return

    click.echo(f"Saved comparison functions ({len(saved)}):\n")
    for fn in saved:
        click.echo(f"  {fn['name']}")
        click.echo(f"    Prompt: \"{fn['prompt']}\"")
        click.echo(f"    Created: {fn['timestamp']}")
        click.echo()


@functions.command("delete")
@click.argument("name")
def functions_delete(name: str):
    """Delete a saved comparison function by NAME."""
    if delete_function(name):
        click.echo(f"Deleted function: {name}")
    else:
        click.echo(f"Function not found: {name}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
