#!/usr/bin/env python3
"""
Unified training script that replaces run_sft.py and run_rft.py.
Supports model aliases and automatic alias mapping for trained models.
"""

from utils.training import main

if __name__ == "__main__":
    main()