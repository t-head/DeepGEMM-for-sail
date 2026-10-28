# Daily functional and tile sweeps

```bash
python tests/daily/run_daily.py --suite OAM-810E
python tests/daily/run_daily.py --suite ZW-M890P
python tests/daily/run_daily.py --suite ZW-M890P --list-daily-scripts
```

This opt-in entry runs the board functional list with parallel compilation and serial accuracy, then preserves the original full shape loops and both explicit tile scans from `.github/workflows/scripts/testcase.json`. Tile scans retain their existing 32 worker processes. The original runner retains command timeouts and failure-output checks. Results share one JUnit report; raw script results are under `checkin-logs/`. `DG_USE_CUTE=0` is excluded.

The default `tests/checkin/run_checkin.py` and DeepGEMM CI do not run these sweeps. This entry preserves full sweep access while daily caselists/perf scheduling are developed. Cold shape sweeps may exceed the original per-script timeout limits; those failures remain visible.
