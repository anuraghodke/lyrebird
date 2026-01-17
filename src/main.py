"""
CLI entry point for the music similarity tool.

Usage:
    python main.py search "Bohemian Rhapsody" --type melody
    python main.py search "Uptown Funk" --type rhythm
    python main.py search "track_id" --type both
"""

import click
from dotenv import load_dotenv

load_dotenv()


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
    default=10,
    type=int,
    help="Number of similar tracks to return",
)
@click.option(
    "--threshold",
    default=0.8,
    type=float,
    help="Minimum similarity threshold (0.0-1.0)",
)
def search(query: str, feature_type: str, limit: int, threshold: float):
    """Search for songs similar to QUERY based on audio features."""
    # Placeholder - will be implemented in later steps
    click.echo(f"Searching for songs similar to: {query}")
    click.echo(f"Feature type: {feature_type}")
    click.echo(f"Limit: {limit}")
    click.echo(f"Threshold: {threshold}")
    click.echo("\nNot yet implemented - coming in later steps!")


if __name__ == "__main__":
    cli()
