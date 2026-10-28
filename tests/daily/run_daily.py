"""Run functional cases followed by the original full shape and tile sweeps."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'checkin'))
from run_checkin import main


if __name__ == '__main__':
    sys.exit(main(['--include-daily-scripts', *sys.argv[1:]]))
