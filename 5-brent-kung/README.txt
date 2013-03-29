Verifying prefix sum specifications (Brent Kung algorithm)
==========================================================

This experiment demonstrates that GPUVerify with Barrier Invariants can verify
the monotonic specification of the Brent Kung algorithm. 

---+ FILES

   * kernel.cl
   contains the Brent Kung prescan

   * specs/*.h
   contain the barrier invariants for specific N (the number of elements)

   * axioms/*.bpl
   associativity axioms with triggers discussed in "Scaling verification" in
   Section 5, used when verifying the algorithm using an abstract operator

---+ CLAIMS

In our paper we present two figures for the Blelloch algorithm.

   * Figure 9b gives verification results *of the whole algorithm* for different
     numbers of threads using different concrete operators (add, max and
     bitwise-or) over different bitwidths (bv32, bv16 and bv8).

   * Figure 10b gives verification results *of the upsweep and downsweep parts
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
0004-or-race-biacc-uint,  0.112,  0.016,  0.023,  0.821,  4.145,  5.117,  PASS
0004-or-upsweep-uint,     0.102,  0.015,  0.017,  0.539,  4.260,  4.933,  PASS
0004-or-downsweep-uint,   0.142,  0.027,  0.041,  0.838,  6.345,  7.393,  PASS
0004-or-endspec-uint,     0.122,  0.024,  0.030,  0.752,  5.611,  6.538,  PASS

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

The times presented in Fig 9b is the sum of the four row totals.

The times presented in Fig 10b is the sum of the row totals excluding the
last, endspec row.
