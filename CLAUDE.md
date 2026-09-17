# CodewithK — conventions

A learning repo. One directory per language (`rust/`, `python/`), one directory
per artifact inside it. Each artifact is a working component of real
infrastructure, not a tutorial exercise.

## Say who wrote it, and say it without flinching

Claude writes this repo. Anyone landing on it should learn that before they read
the content, not by digging through commit trailers.

So every new README, published page or blog post carries a short line near the
top: Claude wrote this, here is the standard to judge it by, here is the link to
`DISCLAIMER.md`. Every new crate or package repeats it in module-level docs
(`//!` in `lib.rs`, the module docstring in Python), because a README banner
never reaches generated documentation. Every commit ends with the
`Co-Authored-By: Claude` trailer and the `Claude-Session` link.

Two failure modes, and the second is the one that keeps happening.

The first is softening it to "AI-assisted". Claude wrote the code, ran the
benchmarks and wrote the analysis; "assisted" would be the comfortable word and
it would be false.

The second is writing the notice defensively — hedging, apologising, stacking
qualifiers, reaching for a robot emoji. That reads as guilt, and readers price it
accordingly. The reason to be suspicious of generated work is not that a machine
made it; it is that fluent prose used to be expensive and is now free, so it has
stopped being evidence that anyone checked anything. The answer to that is not a
longer disclaimer. It is to hand the reader the controls and let them judge.
State the authorship plainly, point at the test that would embarrass the claim,
and stop talking.

## Benchmarks

- A claim without a number is a guess; a number without a control is a guess
  with decimal places. Speedup claims ship with the baseline that isolates them
  (see the 1-thread control in `rust/rushnsw`).
- Record the machine next to the result: core count, CPU features, dataset
  shape. These run on ephemeral containers that differ between sessions.
- Recall is measured against **exact** brute-force ground truth computed in the
  same run, never against another approximate index.
- When a measurement turns out to be wrong or a hypothesis is falsified, the
  write-up keeps the wrong version and says what killed it. Do not quietly
  publish only the corrected number.

## Code

- Comments explain *why the shape of the code is what it is* — which constraint
  forced it — not what the line does.
- Anything `unsafe` states its safety contract in the module header and has a
  test that fails if the contract is broken.
- Prefer the safe idiom when it compiles to the same thing (`Option::take` over
  a raw-pointer borrow split).
