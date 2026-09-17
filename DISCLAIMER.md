# Who wrote this

Claude wrote this repository. Not "with AI assistance" — Claude designed the
crates, wrote the AVX-512 intrinsics by hand, ran the benchmarks, found its own
bugs, and wrote the sentence you are reading. Rishabh decided what was worth
building and when a result was worth believing.

That admission usually arrives as an apology. This one isn't.

You have good reason to be suspicious, and it is worth being precise about why.
The problem with generated code was never that a machine produced it. The
problem is that fluent prose and clean-looking code used to be expensive, so
they worked as a proxy for someone having thought hard. They are cheap now. The
proxy broke. A README full of confident numbers costs nothing to produce and
tells you nothing about whether anyone checked them.

So don't read the prose as evidence. Read the controls.

Here is the one that matters most in this repo. The first parallel-build
measurement came back at 4.25x on four cores — 106% parallel efficiency. That is
not a good result; it is a broken measurement wearing a good result's clothes.
Two explanations were plausible: the thread scheduler was changing insertion
order so early inserts saw a cheaper, sparser graph, or the rewrite had
accidentally dropped an allocation the single-threaded version was still paying.
The second was fixed. Nothing moved. So a one-thread run of the *concurrent* code
was added as a control — same locks, same code path, one core. It came in within
2% of the sequential build, which killed the first explanation too: the speedup
was real parallelism and the extra 6% was timer noise on a small sample.

The first hypothesis was mine and it was wrong. It is still in the README,
alongside the control that disproved it, because a repository that only shows
you the measurements that worked is telling you a story rather than showing you
its work.

That is the standard to hold this to. Every speedup here ships with the baseline
that isolates it. Every recall number is measured against exact brute-force
ground truth computed in the same process, never against another approximation.
The integer kernels are tested for bit-exact equality against a scalar
reference rather than "close enough", because integer arithmetic lets you demand
that and float arithmetic doesn't. Where the code is `unsafe`, the safety
argument is written out in the module header and there is a test that fails when
the argument stops holding.

Now the parts you genuinely should not trust.

The benchmarks ran on whatever ephemeral cloud container the session happened to
get. Four cores, one machine, one run, no averaging. The scaling numbers look
clean up to four threads and say nothing about thirty-two, where lock contention
would change the shape of the answer. Reproduce anything you intend to rely on.

This is a place to learn things in public, not a library. There is no semver, no
stability guarantee, no release process, and the API will change whenever
changing it teaches something.

And Claude wrote the analysis as well as the code, which is exactly the situation
where an author grades their own work. The defence against that is not this
document. It is that the claims are small, specific, and falsifiable, and the
test that would embarrass them is checked in next to them. Run `cargo test`.
Run the benchmark on your own machine. If a number here is wrong, the repo is
built so you can prove it.
