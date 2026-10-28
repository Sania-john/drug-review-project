#!/usr/bin/env python3
# Reducer for job (b): sums the 1s for each term (input is sorted by key)
import sys

current = None
total = 0

for line in sys.stdin:
    key, value = line.rstrip("\n").split("\t")
    if key != current:
        if current is not None:
            print(f"{current}\t{total}")
        current = key
        total = 0
    total += int(value)

if current is not None:
    print(f"{current}\t{total}")
