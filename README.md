# Lyrebird

CLI tool that finds similar songs by analyzing audio features (melody and rhythm).

## Install

```bash
pip install -e ".[audio]"
```

## Usage

```bash
lyrebird search <QUERY> [OPTIONS]
```

Query can be a song name, `"Artist - Song"`, a YouTube URL, or a video ID.

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--type` | `melody`, `rhythm`, or `both` | `both` |
| `--limit N` | Number of results | `5` |
| `--threshold N` | Minimum similarity (0.0-1.0) | `0.0` |
| `--candidates N` | Candidate tracks to analyze | `20` |
| `--interval "[start, end]"` | Time interval to analyze, e.g. `"[3:50, 5:00]"` | full track |

### Examples

```bash
lyrebird search "Posthumous Forgiveness - Tame Impala"
lyrebird search "Redbone" --type melody --limit 3
lyrebird search "https://www.youtube.com/watch?v=k49I5m1J6Is" --interval "[1:00, 2:30]"
```
