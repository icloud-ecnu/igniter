#!/usr/bin/env python3

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-l",
    "--log-clear",
    action="store_true",
    help="If true, clear log either. "
)
FLAGES = parser.parse_args()
log_clear = FLAGES.log_clear

os.system("rm -rf ./perf_data/*")

if log_clear:
    os.system("rm -rf *.log")
