"Barrier Invariants: a Shared State Abstraction for the Analysis of
Data-Dependent GPU Kernels": Additional Supporting Materials
===================================================================

This directory gives the code and scripts necessary to repeat all experiments
given in our paper.

---+ FILES

Each directory contains code and a README.txt outlining how to run the experiment.

   * 1-compact
   Demonstrates that the adversarial abstraction is too coarse for the data-dependent 
   stream compaction kernel, given in Figure 2.
     
   * 2-compact-modular
   Demonstrates that GPUVerify can verify stream compaction using modular
   verification where we replace the prescan implementation with its monotonic
   specification.

   * 3-blelloch
   Demonstrates that GPUVerify can verify the Blelloch algorithm for its
   monotonic specification using Barrier Invariants.

   * 4-kogge-stone
   Demonstrates that GPUVerify can verify the Kogge Stone algorithm for its
   prefix sum specification using Barrier Invariants and source-level abstract
   interpretation.

   * 5-brent-kung
   Demonstrates that GPUVerify can verify the Brent Kung algorithm for its
   monotonic specification using Barrier Invariants.
