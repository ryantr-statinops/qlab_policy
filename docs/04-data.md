# 04 — Data

Source of truth: `src/qlab_rl/data/provider.py`, `scripts/fetch_data.py`.

- Source: Yahoo Finance chart API
  (`https://query1.finance.yahoo.com/v8/finance/chart/{ticker}`), stdlib
  `urllib` with a browser `USER_AGENT`.
- Columns: `date, open, high, low, close, volume` (`FIELDS`).
- Cache: `data/raw/<TICKER>.csv` (gitignored). `YahooCsvProvider.load()`
  serves from cache; `load(refresh=True)` re-downloads.
- `fetch_data.py`: `--ticker`, `--start`, `--end`, `--cache-dir`,
  `--refresh`. Prints bar count, date range, first/last close, and a sanity
  check (`close <= 0` or `high < low` = malformed).
- Source is intentionally pluggable — `DataProvider` is a Protocol, so
  `vnstock` can be swapped in later without touching env or policy code.
