# Daily functional and tile sweeps

```bash
python tests/daily/run_daily.py --suite ZW-810E
python tests/daily/run_daily.py --suite ZW-M890P
python tests/daily/run_daily.py --suite ZW-M890P --skip-checkin
python tests/daily/run_daily.py --suite ZW-M890P --list-daily-scripts
```

This entry runs the board functional list with parallel compilation and serial accuracy, then executes the full test and tile-sweep commands from the board-specific daily list (`daily_810e.list` / `daily_890p.list`). Daily-only logic (sweep lists, per-command timeouts and failure-output detection) lives here; shared scheduling and JUnit reporting are imported from `tests/checkin/run_checkin.py`. Tile scans retain their existing 32 worker processes. Results share one JUnit report; raw script results are under `checkin-logs/`. `DG_USE_CUTE=0` is excluded.

Functional cases from `tests/checkin/checkin_*.list` are included by default. With `--skip-checkin`, the checkin list is neither read nor executed; only the selected board's daily scripts run, with JUnit reporting and failure logs retained. This option requires `--suite` and cannot be combined with `--caselist` or `--phase compile`. `--list` prints only functional cases, so its output is empty when checkin is skipped; use `--list-daily-scripts` to inspect daily commands. `--compile-workers` and `--timeout` apply only to functional cases; daily scripts retain their per-command timeout settings.

The default `tests/checkin/run_checkin.py` and DeepGEMM CI do not run these sweeps. This entry preserves full sweep access while daily caselists/perf scheduling are developed. Cold shape sweeps may exceed the original per-script timeout limits; those failures remain visible.
