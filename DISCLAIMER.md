# Disclaimer

**This repository is written, published, and maintained by Claude (Anthropic),
working in Claude Code sessions with Rishabh Arora.**

Every crate, benchmark, test, and README here was produced inside a session and
committed from it. Commits carry `Co-Authored-By: Claude` and a `Claude-Session`
link back to the session that produced them.

What that means for you as a reader:

- **The numbers are real but narrow.** Benchmarks were run on the ephemeral
  cloud container the session happened to get — core count, cache sizes, and
  CPU features are recorded next to each result, and nothing was averaged across
  machines. Reproduce before you rely on it.
- **This is a learning repo, not a library.** It is built to be read and
  understood, not depended on. There is no stability guarantee, no semver, no
  release process.
- **Mistakes are left visible on purpose.** Where a first measurement was wrong
  or a hypothesis was falsified, the write-up says so rather than quietly
  presenting the corrected version. That record is part of the point.
- **Verify anything load-bearing.** Claude wrote the analysis as well as the
  code. Where a claim would affect a real decision, the repo tries to show the
  control or the test that backs it — check that, not the prose.
