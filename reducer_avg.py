#!/usr/bin/env python3
# Reducer for job (a): input is sorted by key, output is
# condition <TAB> drug <TAB> avg_rating <TAB> review_count
import sys

current = None
total = 0.0
count = 0


def flush():
    if current is not None and count > 0:
        cond, drug = current.split("||", 1)
        print(f"{cond}\t{drug}\t{total / count:.2f}\t{count}")


for line in sys.stdin:
    key, value = line.rstrip("\n").split("\t")
    if key != current:
        flush()
        current = key
        total = 0.0
        count = 0
    total += float(value)
    count += 1

flush()
