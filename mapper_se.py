#!/usr/bin/env python3
# Mapper for job (b): emits  term <TAB> 1  for each side-effect term in a review,
# skipping terms that appear in the drug's own condition
import sys

TERMS = ["nausea", "headache", "dizziness", "fatigue", "insomnia", "drowsiness", "rash",
         "weight gain", "dry mouth", "anxiety", "vomiting", "diarrhea", "constipation",
         "sweating", "depression", "pain", "itching", "cramps"]

for line in sys.stdin:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 4:
        continue
    cond, text = parts[1], parts[3]
    for term in TERMS:
        if term in text and term not in cond:
            print(f"{term}\t1")
