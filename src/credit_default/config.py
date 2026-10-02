from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"

# Target: loan charged off at any point over its 36-month term (lifetime default)
TERM_MONTHS = 36

# Out-of-time split (only vintages whose 36-month loans fully matured by end of 2018)
TRAIN_START = "2012-01-01"
TRAIN_END   = "2014-12-31"
TEST_START  = "2015-01-01"
TEST_END    = "2015-12-31"

RANDOM_SEED = 42