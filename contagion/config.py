"""Configuration constants for contagion analysis."""

import os
from datetime import datetime

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
EXCEL_PATH = os.path.join(PROJECT_DIR, "lbnl_ix_queue_data_file_thru2024_v2.xlsx")
SHEET_NAME = "03. Complete Queue Data"
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")

# Excel date epoch (Excel uses 1899-12-30 as day 0)
EXCEL_EPOCH = datetime(1899, 12, 30)

# End-of-study date for censoring
END_OF_STUDY = datetime(2024, 12, 31)

# Calendar origin for Cox model time axis
COX_ORIGIN = datetime(1995, 1, 1)

# Tier 2 entities with reasonable wd_date coverage
TIER2_ENTITIES = ["PJM", "ERCOT", "CAISO", "SOCO", "ISO-NE", "PSCo"]

# Random seed
RANDOM_SEED = 42

# Permutation test
N_PERMUTATIONS = 1000

# POI depth bins for dose-response
DEPTH_BINS = [(2, 2), (3, 4), (5, 9), (10, float("inf"))]
DEPTH_LABELS = ["2", "3-4", "5-9", "10+"]
