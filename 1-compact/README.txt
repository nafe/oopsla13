Adversarial abstraction is too coarse for data-dependent kernels
================================================================

This experiment demonstrates that the adversarial abstraction is too coarse to
prove race-freedom of the stream compaction example, given as Figure 2 in the
paper.

---+ FILES

   * kernel.cl
   contains the stream compaction kernel as given in Figure 2

---+ CLAIMS

In our paper we claim the adversarial abstraction is too coarse for
data-dependent kernels, such as stream compaction.

---+ REPRODUCING RESULTS

Verify this result, like so:

> gpuverify --no-infer --num_groups=1 --local_size=2 kernel.cl

Expected output will look something like:

GPUVerify kernel analyser finished with 0 verified, 5 errors

It will report a possible write-write race.
