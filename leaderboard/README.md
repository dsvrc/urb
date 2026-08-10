# URB Leaderboard

This folder contains a static leaderboard generator for URB experiment results.

## Generate the page

From repository root:

```bash
python3 leaderboard/generate_leaderboard.py
```

Default output:

- `docs/leaderboard/index.html`
- `docs/leaderboard/urb.png`

## Options

- `--results-dir`: source directory with run folders (default: `results`)
- `--output-dir`: output site directory (default: `docs/leaderboard`)
- `--repo-url`: optional GitHub URL prefix for experiment links, e.g. `https://github.com/COeXISTENCE-PROJECT/URB/tree/main/`
: if omitted, the generator infers it from `title_link_url` in `leaderboard_strings.json`
- `--local-link-prefix`: relative fallback prefix for links when `--repo-url` is not provided (default: `..`)

## Included behavior

- Indexes all valid run folders with `exp_config.json` and `metrics/BenchmarkMetrics.csv`
- Sortable metric columns
- Filtering/grouping by experiment type, env config, task config, and network
- Collapsing reruns (same config, different seeds/IDs) into averaged rows
- Metric descriptions as column tooltips
- CSV export of the visible table
