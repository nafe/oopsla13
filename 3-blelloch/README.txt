Verifying prefix sum specifications (Blelloch algorithm)
========================================================

This experiment demonstrates that GPUVerify with Barrier Invariants can verify
the monotonic specification of the Blelloch algorithm. We take the Blelloch
prescan implementation in the stream compaction kernel (experiments/compact)
and verify it as a separate procedure.

---+ FILES

   * kernel.cl
   contains the Blelloch prescan, as given in Figure 2

   * specs/*.h
   contain the barrier invariants given in Section 4.1, for specific N (the
   number of elements)

   * axioms/*.bpl
   associativity axioms with triggers discussed in "Scaling verification" in
   Section 5, used when verifying the algorithm using an abstract operator

---+ CLAIMS

In our paper we present two figures for the Blelloch algorithm.

   * Figure 9a gives verification results *of the whole algorithm* for different
     numbers of threads using different concrete operators (add, max and
     bitwise-or) over different bitwidths (bv32, bv16 and bv8).

   * Figure 10a gives verification results *of the upsweep and downsweep parts
     of the algorithm* for different numbers of threads using different concrete
     operators (add, max and bitwise-or) *and an abstract operator* over
     different bitwidths (bv32, bv16 and bv8).

The parameters we vary are:

   * operator          = add,max,or,abstract

   * bitwidth          = 32,16,8

   * number of threads = 2,4,8,...,128

---+ REPRODUCING RESULTS

As discussed in "Verification Strategy" of Section 5, we verified the algorithm
in a staged fashion as we found this gave us better performance than verifying
all properties in a single shot.

A single datapoint of either graph can be verified by running the following
script:

> ../scripts/staged.py --op=[add|max|or|abstract] --width=[32|16|8] nthreads

For example, to check verification for 4 threads using 32-bit bitwise-or:

> ../scripts/staged.py --op=or --width=32 4

Expected output will look something like:
# stage,                  clang,  opt,    bugle,  vcgen,  boogie, total,  verified
0004-or-race-biacc-uint,  0.110,  0.016,  0.025,  0.824,  6.188,  7.162,  PASS
0004-or-upsweep-uint,     0.110,  0.025,  0.013,  0.425,  2.617,  3.190,  PASS
0004-or-downsweep-uint,   0.109,  0.017,  0.023,  0.529,  3.961,  4.639,  PASS
0004-or-endspec-uint,     0.106,  0.014,  0.014,  0.519,  3.085,  3.739,  PASS

There are 4 rows in the output:

   * The 1st row gives the time to verify race-freedom of the whole algorithm

   * The 2nd row gives the time to verify the upsweep barrier invariant

   * The 3nd row gives the time to verify the downsweep barrier invariant

   * The 4nd row gives the time to verify the monotonic specification barrier invariant

If you use the flag "--op=abstract" then there will only be 3 rows because we do
not have appropriate axioms for dealing with associativity (see "Scaling
verification" of Section 5)

Each row consists of the time to process the kernel:

   * clang,opt,bugle and vcgen are the GPUVerify frontend

   * boogie is the GPUVerify backend which uses Z3 to check the verification
     condition

   * The total time for each row is given by 'total'

The times presented in Fig 9a is the sum of the four row totals.

The times presented in Fig 10a is the sum of the row totals excluding the
last, endspec row.
