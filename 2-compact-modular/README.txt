Verify stream compaction using the prefix sum specification
===========================================================

This experiment demonstrates that GPUVerify can scale to very large thread
counts using modular verification. We replace the Blelloch prescan
implementation in the stream compaction kernel (experiments/compact) with its
monotonic specification:

  tid < other_tid ==> idx[tid] + flag[tid]) <= idx[other_tid]

---+ FILES

   * kernel.cl
   contains the stream compaction kernel (from experiments/compact), but
   with the Blelloch prescan replaced with its monotonic specification.

---+ CLAIMS

In our paper we claim verification for all thread counts from 2 to 2^{16},
in less than one second, thus illustrating the power of modular analysis
using the two-thread reduction.

---+ REPRODUCING RESULTS

Verify this test with different numbers of threads N, like so:

> gpuverify --time --no-infer --num_groups=1 --local_size=N kernel.cl

For example, to verify with 1024 threads:

> gpuverify --time --no-infer --num_groups=1 --local_size=1024 kernel.cl

Expected output will look something like:

Verified: kernel.cl
- no data races within work groups
- no data races between work groups
- no barrier divergence
- no assertion failures
(but absolutely no warranty provided)
Timing information (1.03 secs):
- clang                 : 0.097 secs
- opt                   : 0.006 secs
- bugle                 : 0.006 secs
- gpuverifyvcgen        : 0.279 secs
- gpuverifyboogiedriver : 0.644 secs
