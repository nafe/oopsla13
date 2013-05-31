Verifying prefix sum specifications (Kogge-Stone)
=================================================

This experiment demonstrates that GPUVerify with Barrier Invariants can verify
the Kogge Stone prefix sum algorithm using our source-level abstract
interpretation.

---+ FILES

   * kernel.cl
   contains the Kogge Stone prefix sum with abstract interpretation described in
   "Other prefix sum algorithms" Section 4.2
   
---+ CLAIMS

In our paper, Figure 10 shows verification scaling to very large thread counts
for the Kogge Stone algorithm using our source-level abstract interpretation.

---+ REPRODUCING RESULTS

Verify this test with different numbers of threads N, like so:

> gpuverify --time --no-infer --num_groups=1 --local_size=N kernel.cl

For example, to verify with 1024 threads:

> gpuverify --time --no-infer --num_groups=1 --local_size=1024 kernel.cl

Expected output will look something like:

GPUVerify kernel analyser finished with 1 verified, 0 errors
Verified: kernel.cl
- no data races within work groups
- no data races between work groups
- no barrier divergence
- no assertion failures
(but absolutely no warranty provided)
Timing information (16.43 secs):
- clang                 :  0.091 secs
- opt                   :  0.010 secs
- bugle                 :  0.007 secs
- gpuverifyvcgen        :  0.452 secs
- gpuverifyboogiedriver : 15.871 secs
