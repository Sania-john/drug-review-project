#!/usr/bin/env python3
# Mapper for job (a): emits  condition||drug <TAB> rating
import sys

for line in sys.stdin:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 4:
        continue
    drug, cond, rating = parts[0], parts[1], parts[2]
    print(f"{cond}||{drug}\t{rating}")
