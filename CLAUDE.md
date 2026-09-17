# CodewithK — conventions

A learning repo. One directory per language (`rust/`, `python/`), one directory
per artifact inside it. Each artifact is a working component of real
infrastructure, not a tutorial exercise.

## Attribution is mandatory on anything published

This repo is written, published and maintained by Claude. That has to be visible
to anyone who lands on it, not buried in commit metadata.

**Every new README, published page, blog post, or top-level document gets the
provenance banner**, near the top where it is read before the content:

```markdown
> 🤖 **Written, published and maintained by Claude (Anthropic)**, in Claude Code
> sessions. See [DISCLAIMER.md](DISCLAIMER.md) before relying on anything here.
```

Adjust the relative path to `DISCLAIMER.md` for the file's depth. For a longer
document, use the fuller paragraph form already in `rust/rushnsw/README.md`.

**Every new crate or package** also carries the provenance note in its
crate-level docs (`//!` in `lib.rs`, module docstring in Python) so it survives
into generated documentation, where a README banner does not reach.

**Every commit** ends with `Co-Authored-By: Claude ...` and the `Claude-Session`
link.

Do not soften this into "AI-assisted". Claude wrote the code, ran the
benchmarks, and wrote the analysis; the banner says that plainly.

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
