# E13 (P8 continued) — when to commit, not what to rank

**Branch**: `phase/8-semantic-evidence`
**Base**: `cee33e5`
**Date**: 2026-09-01
**Mission**: find a *mechanism* for the paraphrase gap, after E12 established that
five tuning levers were all already at their optimum.

**Result: +0.0204 mean across six paraphrase probes, at exactly zero public
cost.** The change is a deletion. Nothing about ranking moved.

| probe | before | after | Δ |
|---|---|---|---|
| **public (the submission)** | 0.945297 / 1.000 | **0.945297 / 1.000** | **0** |
| L2 synonyms | 0.847762 / 0.975 | **0.866816** / 0.960 | +0.01905 |
| L3 information loss | 0.790816 / 0.905 | 0.806093 / 0.910 | +0.01528 |
| `--paraphrase-category` | 0.798361 / 0.910 | 0.819643 / 0.910 | +0.02128 |
| held-out lexicon | 0.903642 / 0.990 | 0.922377 / 0.990 | +0.01874 |
| structural L2 | 0.891342 / 0.985 | **0.915919 / 0.990** | +0.02458 |
| structural L3 | 0.895402 / 0.985 | 0.918979 / 0.980 | +0.02358 |

---

## The diagnosis: it was never a matching problem

E12 closed the ranking levers. So this phase measured *where the loss is* before
proposing anything.

**Headroom, re-measured at the current configuration** (E10's figures predate
semantic evidence):

| | current | perfect-reranker oracle |
|---|---|---|
| composite | 0.847762 | 0.990300 |
| session pool recall | 1.000 | 1.000 |
| HitRate share | 0.4875 | 0.5000 |
| **MRR share** | **0.196062** | **0.300000** |
| efficiency share | 0.1642 | 0.1903 |

The reachable gap is 0.1425 and **73% of it is MRR**. Only 5 sessions of 200
miss entirely. The agent finds the target; it ranks it badly.

**Where the target actually sits** at the turn the agent commits, L2:

```
rank  1: 111 (55.5%)     rank  6:   5     
rank  2:  13             rank  7:   9     
rank  3:  11             rank  8:  10     
rank  4:   9             rank  9:  11     
rank  5:  11             rank 10:   5      never found: 5
```

**That tail is flat, and flat is the whole finding.** A reranker that is nearly
right produces a decaying tail. A near-uniform spread over 2–10 is what you get
when the ordering carries no signal at all — the features have gone quiet and
the ranking is effectively arbitrary among candidates it cannot separate.

**Why they are quiet**, counted over 553 scored turns at L2:

* **38.0%** of turns have **zero** disclosures — every evidence feature scores
  0.0 for every candidate, so the order is retrieval plus category alone.
* **89.3%** of turns have **zero exactly-owned** disclosures, so `slot_evidence`
  (weight 16) is dark.

And the agent commits early anyway: **119 of 200 sessions first show the target
at turn 2.**

---

## The mechanism: a robustness clause that was costing the thing it protected

`shortlist.shortlist_size` carried this, tested *before* the expand backstop:

```python
if disclosed > 0 and live_disclosures <= 0:
    # the agent cannot read this distribution and stops trying
    return top_k          # all ten
```

The reasoning is sound: if the evidence cannot be measured, do not withhold. The
remedy is backwards. **The evaluator freezes `best_rank` the first turn the
target appears.** Returning ten at the moment the evidence is weakest does not
buy a hit — it *locks in* a rank drawn from the flat distribution above, and
forfeits every later turn in which more disclosures would have sharpened it.

Narrowing instead defers the commitment. It is implicitly a confidence filter:
commit only when the top candidate is worth standing behind, otherwise ask
again. `EXPAND_TURN = 5` remains the backstop, so the agent cannot withhold
forever.

### The clause is inert on the public set, and this is a count

| | fires | of turns | rate |
|---|---|---|---|
| verbatim (L0) | **0** | 563 | **0.00%** |
| paraphrased (L2) | 284 | 553 | 51.36% |

180 of the 284 are on turn 2. **No value of this constant can move the public
score**, and every arm below scores exactly 0.945297 — measured, not inferred.

---

## The sweep

`k` is what the clause returns; `k=0` disables it entirely and falls through to
the ordinary narrowing path. Deltas against the shipped always-ten:

| k | L2 | L3 | cat | heldout | struct L2 | struct L3 | **mean** | **worst** |
|---|---|---|---|---|---|---|---|---|
| 5 | +0.0132 | +0.0198 | −0.0006 | +0.0071 | +0.0111 | +0.0028 | +0.0089 | −0.0006 |
| 3 | +0.0130 | **+0.0395** | +0.0114 | +0.0124 | +0.0163 | +0.0128 | +0.0176 | +0.0114 |
| 2 | +0.0185 | +0.0254 | +0.0127 | **+0.0207** | +0.0239 | +0.0181 | +0.0199 | +0.0127 |
| **0** | +0.0191 | +0.0153 | **+0.0213** | +0.0187 | **+0.0246** | **+0.0236** | **+0.0204** | **+0.0153** |

**0 wins the mean and the worst case**, and it is the only arm that is a
deletion rather than a constant — there is nothing left in it to overfit.

**k=0 and k=1 measured identical on all seven probes.** They can differ only
when `consistent == 1`, which never occurs under paraphrase. 0 is preferred
because it preserves that escape hatch for a regime where it might.

### The sign pattern, which is why this is believed

E12's rejected lever gained on the probes sharing its tuned vocabulary and lost
on the independent one. This is the reverse: the gain is **largest on the two
structural probes** (+0.0246, +0.0236), which were built by a different
mechanism and which nothing here was tuned against, and structural L2's HitRate
*rises* 0.985 → 0.990. A change that helps most where it was least fitted is a
mechanism, not a fit.

---

## What it costs, stated plainly

* **L2 HitRate falls 0.975 → 0.960** — three sessions that previously scraped a
  hit at rank 8–10 now surface none. This is a real loss and it is the price of
  refusing to commit on noise. It is outweighed roughly four to one by the MRR
  gain (0.653540 → 0.796054), which is why the composite rises.
* **MTTC rises 2.79 → 3.60 at L2.** Already priced into the composite through
  the efficiency term. The public MTTC is unchanged at 2.815.
* **The most aggressive arm carries the most model risk.** If a private set
  paraphrases far less than any of these three probes, k=0 withholds where the
  old clause would have collected a cheap hit. The public set is exactly that
  case, and there the clause never fires — but that is evidence about quoting
  customers, not about mildly-paraphrasing ones. `SHOPPING_AGENT_PARAPHRASE_SHORTLIST=10`
  restores the old behaviour in one step.

---

## Rules of the house — one added

**A sweep in which every arm returns the same number is a broken sweep, not a
flat response.** The first pilot here ran four values of `k` as
`( VAR=$k cmd ) &` in a zsh loop; every subshell resolved `$k` late, to the
loop's final value, so all four arms silently ran the same configuration and
returned identical scores. It was caught only because the "k=10 control" failed
to reproduce a baseline measured ten minutes earlier. **Always put a known
control in the sweep and check it reproduces**, and pass swept values as
explicit process arguments rather than closing over a loop variable.

---

## Open items

1. **The clarification policy is now the largest untouched lever.** 38% of
   scored turns still have zero disclosures. Asking the attribute that most
   splits the current pool would attack MRR and MTTC together, and would
   partially refund the 0.8 turns this change spends.
2. **The confidence-margin form was designed and not built.** Gating the commit
   on `top1 − top2` of the semantic score, rather than on a fixed size, is the
   principled version of what k=0 does bluntly. It is now a smaller prize than
   before — k=0 already takes most of the available gain — but it may recover
   the three lost hits.
3. **Browsing is the weak scenario** (rank-1 48.8%, MRR 0.6005, n=80) against
   Boundary's 80% / 0.8375. It discloses least, and nothing here addressed that
   asymmetry directly.
4. **Nothing is merged to `main`**, and the `phase/7` divergence is still open.
