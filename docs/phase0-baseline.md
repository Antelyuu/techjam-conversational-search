# Phase 0 Baseline Checkpoint

Branch: `phase/0-baseline-foundation`

Date: 2026-08-29

## Purpose

Establish a reproducible starting point before changing agent behavior.

## Environment

- Python: 3.14.2
- SQLite FTS5: available
- Catalogue: `data/catalog.jsonl`
- Catalogue rows: 50,000
- Unique `parent_asin` values: 50,000
- Duplicate `parent_asin` rows: 0
- Catalogue SHA-256: `da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67`

The SHA-256 value is the checksum of the local decompressed file. It should be compared with the organizer's published checksum when that file is available.

## Reproduction commands

From the repository root:

```bash
python3 -m unittest discover -s tests -v
python3 -m evaluator.local_evaluator
```

The evaluator writes `results.json`, which is intentionally ignored by Git.

## Test result

```text
Ran 3 tests in 0.006s
OK
```

## Published baseline reproduction

The local run matches `docs/baseline_results.json`:

| Metric | Result |
|---|---:|
| Sessions | 200 |
| Hit Rate@10 | 0.125000 |
| MRR | 0.068034 |
| MTTC | 9.810000 |
| Efficiency | 0.119000 |
| Recommended TechnicalScore | 0.106710 |
| Prompt tokens | 0 |
| Completion tokens | 0 |

## Scenario breakdown

| Scenario | Sessions | Hit Rate@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| Buying | 80 | 0.237500 | 0.126508 | 8.625000 |
| Browsing | 80 | 0.025000 | 0.004514 | 10.750000 |
| Intent Override | 30 | 0.133333 | 0.104167 | 10.066667 |
| Boundary | 10 | 0.000000 | 0.000000 | 11.000000 |

## Public data audit

- Public sessions: 200
- Scenario counts: 80 Buying, 80 Browsing, 30 Intent Override, 10 Boundary
- Exact unique anonymized profiles: 125
- Repeated aggregate profiles are expected; they do not identify the same real customer.

Catalogue field observations:

| Field | Present rows | Missing rows |
|---|---:|---:|
| `title` | 49,998 | 2 |
| `features` | 44,781 | 5,219 |
| `description` | 26,113 | 23,887 |
| `price` | 10,527 total non-empty, 10,410 numeric | 39,473 empty |
| `store` | 49,686 | 314 |

Numeric prices range from `$0.00` to `$4,119.00`; the numeric-price median is `$22.88`.

Common category values include `Shoes` (11,810 rows), `Jewelry` (5,127), `Accessories` (4,117), and `Athletic` (2,323). The broad `Clothing, Shoes & Jewelry` category appears in 49,990 rows.

## What this means for Phase 1

1. The starter is a valid baseline, not a broken setup.
2. Browsing and Boundary sessions are the largest immediate weaknesses.
3. Intent Override needs state handling before retrieval changes can be evaluated fairly.
4. Budget filtering must use a tri-state policy:
   - known price over budget: exclude when the budget is truly hard;
   - known price within budget: retain;
   - missing price: retain but do not claim verified budget compliance.
5. Product text should tolerate missing descriptions and features.

## Phase 0 exit status

Complete. The next branch should be created from this checkpoint and should focus on session state plus tests for accumulation and intent override.
