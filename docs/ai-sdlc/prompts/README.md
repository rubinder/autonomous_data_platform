# Prompts

The prompt artifacts for this build are not paraphrased here — they are in the
repository verbatim, under
`.superpowers/sdd/2026-08-09-autonomous-lakehouse/`:

| Artifact | What it is |
|---|---|
| `task-N-brief.md` | The dispatch prompt for task N: files to create, interfaces produced and consumed, the test to write first, the expected failure, the implementation steps, and the commit message. 20 of these. |
| `task-N-report.md` | What the implementer reported back: deviations from the brief with reasoning, verification transcripts, and concerns raised for later tasks. |
| `review-<base>..<head>.diff` | The exact diff each reviewer was given. |
| `progress.md` | The append-only ledger: every dispatch, review verdict, measured number, deferral, and override. |

This page records the *structure* those prompts share, because the structure is
the reusable part.

## The three prompt shapes

**1. Task brief (implementer).** Written before any code, as part of the plan.
Always contains, in this order:

1. the exact files to create and to modify;
2. the interfaces this task *produces* and the ones it *consumes* from earlier
   tasks — this is what makes twenty independently-contexted subagents compose;
3. **the test, written out in full, first** — with the expected failure message
   before implementation exists;
4. the implementation steps, often with draft code;
5. the verification commands to run;
6. the commit message, including *why* the design is what it is.

Draft code in a brief is a starting point, not an instruction. Implementers were
expected to deviate when the draft was wrong, and to say so in the report —
which is how the defects in `decisions/0002`–`0004` surfaced.

**2. Review prompt.** Given the diff, the brief, and one standing instruction:
**assume the report is wrong; re-run the commands and re-measure the numbers.**
Reviews return a verdict (`clean` / `changes requested`), findings graded
Important or Minor, and — crucially — the measurements that back each finding.
"This looks wrong" is not a finding; "I ran the plan's version and measured
800/800 nulls where the shipped version has 0/800" is.

**3. Fix prompt.** Sent to the *original implementer*, resumed with its context
(so it does not re-derive what it already knows), listing only the findings to
address and any explicitly out-of-scope areas. Followed by a **scoped re-review
of the fix alone**, because fixes introduce bugs — Task 14's fix introduced a
crash on malformed dates, and Task 17's first fix left SQL comments and quoted
identifiers still corruptible.

## Instructions that changed the outcome

A few lines from real dispatches, kept because they are the ones that mattered:

- *"Do not tune toward beating the baseline — target calibration, not a win. If
  all six now read 'beats', STOP and investigate; the canary is live again."*
  (Task 13 fix.) The result stayed 0 of 6.
- *"A path that has never executed under test is where silent row loss hides."*
  (Task 8, overriding a reviewer's deferral of the quarantine test.)
- *"Do not write a SQL parser."* (Task 17 fix round 2 — scoping a fix that could
  easily have grown into one.)
- *"CI must never gate on any model-skill metric."* (Tasks 19 and 20. At reduced
  scale ~59% of training rows have every spend feature NULL, so any skill metric
  computed there is noise, and a CI job that fails when the model loses teaches
  people to ignore CI.)

## What is deliberately not here

No prompt-engineering tricks, no persona preambles, no "you are a world-class
data engineer". The leverage in this build came from the *shape* of the loop —
fresh context per task, written tests before implementation, an adversarial
reviewer that re-measures, and a separate scoped review of every fix — not from
the wording of any individual prompt.
