# E15 (P8) — recognising a category when the customer talks normally

**Branch**: `main`
**Base**: `a35478e`
**Date**: 2026-09-01
**Mission**: make the agent usable by a customer who opens a conversation the way people
actually do, rather than the way the simulator does.

**Result: every conversational opener tested now resolves to a real category, and all six
scored probes are unchanged to six decimals.** The change is purely additive — it fires
only where the agent previously failed, and the benchmark contains no such openers.

---

## The gap

`slots.stated_category` asks *"does this message begin with a phrasing I know, and what
follows it?"*. It recognises nine lead-ins: `I'm looking for`, `I am looking for`,
`I'm searching for`, `I'm shopping for`, `I'm after`, `I'd like`, `looking for`,
`I want`, `I need`.

The simulator only ever uses those, so the public set never exercises the alternative. A
real customer does:

| opener | category found, before |
|---|---|
| `I'm looking for Shoes Loafers & Slip-Ons.` | ✅ |
| `Hey, do you have any Shoes Loafers & Slip-Ons?` | ❌ |
| `Can you help me find shoes loafers and slip-ons?` | ❌ |
| `Show me some loafers & slip-ons shoes` | ❌ |
| `I'm in the market for shoes loafers & slip-ons` | ❌ |

A missing category is not a crash — it stands the E9 filter down and retrieval widens from
a median **184 rows to all 50,000**. Measured cost of that fallback on the public set is
about 0.003 (`SHOPPING_AGENT_CATFILTER=0` scores 0.942229), but that figure is taken with
every other mechanism working; on a genuinely conversational opener the practical cost is
larger, because every later mechanism then works against 270× more impostors.

**A second failure mode, found while fixing the first.** The lead-in patterns do not only
return *nothing* — they return *junk*. `I need a gift for my wife` matches `I need` and
hands back `a gift for my wife`; `I need some loafers & slip-ons shoes` hands back
`some loafers & slip-ons shoes`, which resolves to no category because of the stray
`some`. The second case names a category perfectly well and was still being thrown away.

---

## The mechanism

Ask the other question: **does this message name a category I know?** — against the
catalogue's own 1,115 coarse categories, matching on tokens rather than position.

Two properties of the vocabulary make this safe without a similarity threshold to tune:

| | |
|---|---|
| distinct coarse categories | 1,115 |
| median token count | 4 |
| **single-token categories** | **0** |

Because no category is a single word, requiring **every** token of a category to appear in
the message is a strict test. `loafers` alone does not match `Shoes Loafers & Slip-Ons`;
it takes the whole name. Token-set matching also makes word order irrelevant, the same
insensitivity `canonical_category` already relies on for reworded categories.

**Most-specific-wins** is what keeps the twenty all-generic categories (`Men Shoes`,
`Women Jewelry`, `Novelty Men`) safe to leave in the vocabulary: they are selected only
when nothing longer matches.

**Returns None rather than a best guess.** The category is a *hard restriction* on
retrieval, so a wrong one costs the session outright while a missing one merely widens the
search. `got any loafers?` names only part of a category and is deliberately left
unresolved.

### Where the decision lives

The matcher is a pure function in `slots.py`; the *decision* is in the agent, because only
the agent holds the catalogue vocabulary. Its contract is one line:

> If the lead-in produced something the catalogue actually has, return it unchanged.

That is every opener on the public set, which is why the scored number cannot move — by
construction, before any measurement. Only when the stated category is `None`, or names
nothing real, is the free-text matcher consulted; and if it too finds nothing, the
original value is preserved, so the old behaviour survives exactly.

---

## Results

| opener | before | after |
|---|---|---|
| `I'm looking for Shoes Loafers & Slip-Ons.` | ✅ | ✅ *(unchanged)* |
| `Hey, do you have any Shoes Loafers & Slip-Ons?` | ❌ | ✅ |
| `Can you help me find shoes loafers and slip-ons?` | ❌ | ✅ |
| `Show me some loafers & slip-ons shoes` | ❌ | ✅ |
| `I'm in the market for shoes loafers & slip-ons` | ❌ | ✅ |
| `I need some loafers & slip-ons shoes` *(junk before)* | ❌ | ✅ |
| `Could you show me women dresses please?` | ❌ | ✅ |
| `hiya, after some accessories belts` | ❌ | ✅ |
| `I need a gift for my wife` | ❌ | ❌ *(correct — names nothing)* |
| `got any loafers?` | ❌ | ❌ *(correct — partial name)* |

### Every scored probe is unchanged

| probe | before | after |
|---|---|---|
| public | 0.945297 / 1.000 / 0.938657 / 2.815 | **identical** |
| L0 control | 0.945297, 200/200 openers parsed | **identical** |
| L2 synonyms | 0.875897 / 0.965 | **identical** |
| L3 / structural L2 | 0.936003 / 0.995 | **identical** |
| `--paraphrase-category` | 0.811172 / 0.905 | **identical** |
| `--reword-category` | 0.875897 / 0.965 | **identical** |

`--paraphrase-category` is unchanged for a legible reason rather than by luck: it
substitutes the category's own *words*, so a matcher keyed on catalogue tokens cannot find
them either. This change helps openers that are *phrased* differently, not ones whose
vocabulary has been replaced.

**Cost**: index built once at 0.84 ms over 1,115 categories; 0.010 ms per opener, once per
session. 237 tests.

---

## What this does not do

This closes conversational *phrasing*. It does not close conversational *inference*:

> *"I've got a wedding next month and need something that won't destroy my feet."*

No category is named, the constraints are implied, and resolving it needs world knowledge.
Our extraction is regex over a known vocabulary and the embedding feature matches meaning
without parsing intent. Closing that gap means putting a language model in the loop, which
would cost the offline, deterministic, zero-token, no-network properties that are among
this submission's strongest claims — and would still need this path as the fallback for
when network access is disabled.

The honest description of the agent after this change is that it **handles natural
phrasing, and does not attempt open-ended dialogue.**

---

## Open items

1. **Answer-side filler is still carried as evidence.** Any reply is absorbed whatever its
   phrasing, but an unrecognised framing clause ("what's really important for this item is
   that…") rides along as noise and dilutes the length-weighted features. Measured cost
   unknown; the framing-stripper could be generalised beyond the `is:` form.
2. **A partial category name is refused.** `got any loafers?` could plausibly resolve by
   *longest partial* match instead of requiring every token. That trades false negatives
   for false positives on a hard retrieval filter, and was not attempted.
3. The E13+E14 ranking code review is still outstanding.
