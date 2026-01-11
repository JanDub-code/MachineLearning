#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spusti pouze finalni evaluaci a generovani grafu.
Predpoklada, ze modely jsou jiz natrénovane.
"""

import sys
sys.path.insert(0, '.')

from run_150_pipeline import final_evaluation

if __name__ == "__main__":
    print("Spoustim finalni evaluaci a generovani premium vizualizaci...")
    report = final_evaluation()
    if report:
        print("\nHOTOVO! Grafy ulozeny do data/150_tickers/figures/")
    else:
        print("\nCHYBA: Evaluace selhala.")
