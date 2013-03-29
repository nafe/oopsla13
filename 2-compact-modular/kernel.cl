#define N __LOCAL_SIZE_0
#define tid get_local_id(0)
#define other_tid __other_int(tid)

__kernel void compact(__local uint*out, __local uint*in, __local uint *flag, __local uint *idx) {

  uint t = get_local_id(0);

  // (i) test each element with predicate p
  // flag = 1 if keeping element, 0 otherwise
  flag[t] = ((in[t] & 1) == 0) ? 1 : 0;

  // (ii) compute indices for compaction
  // using the specification of a prescan
  // -- precondition (vacuously true since flag[t] has type uint)
  // __assert(0 <= flag[tid]);
  // -- postcondition
  __assume(__implies(tid < other_tid, (idx[tid] + flag[tid]) <= idx[other_tid]));
  __assume(__implies(tid < other_tid, __add_noovfl(idx[tid], flag[tid])));

  // (iii) compaction
  if (flag[t]) out[idx[t]] = in[t];
}
