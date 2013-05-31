/*
 * Kogge Stone inclusive prefix sum in OpenCL
 */

#define N __LOCAL_SIZE_0
#define tid get_local_id(0)
#define other_tid __other_int(tid)

#define sweep(t,offset) \
  ((ghostsum_lower[t] <= ghostsum_upper[t]) & \
   __implies(t <  offset, (ghostsum_lower[t] == 0) & (ghostsum_upper[t] == t)) & \
   __implies(t >= offset, (ghostsum_lower[t] == t - offset + 1) & (ghostsum_upper[t] == t)))

#define isthreadid(t) (0 <= t & t < N)

__kernel void scan(__global int *input, __global int *output) {
  __local int sum[N];
  __local int ghostsum_lower[N];
  __local int ghostsum_upper[N];

  sum[tid] = input[tid];
  ghostsum_lower[tid] = tid;
  ghostsum_upper[tid] = tid;

  __assert(__accessed(input, tid));

  __barrier_invariant(sweep(tid,1), tid, tid-1);
  barrier(CLK_LOCAL_MEM_FENCE);

  int temp;
  int ghosttemp_lower;
  int ghosttemp_upper;

  for (int offset = 1;
        __invariant(__no_read(output)), __invariant(__no_write(output)),
        __invariant(__no_read(sum)), __invariant(__no_write(sum)),
        __invariant(__no_read(ghostsum_lower)), __invariant(__no_write(ghostsum_lower)),
        __invariant(__no_read(ghostsum_upper)), __invariant(__no_write(ghostsum_upper)),
        __invariant(0 <= offset),
        __invariant(__is_pow2(offset)),
        __invariant(offset <= N),
        __invariant(sweep(tid,offset)),
        __invariant(__implies(isthreadid(tid-offset), sweep(tid-offset,offset))),
      offset < N;
      offset *= 2) 
  {
    if (tid >= offset)
    {
      temp = sum[tid-offset];
      ghosttemp_lower = ghostsum_lower[tid-offset];
      ghosttemp_upper = ghostsum_upper[tid-offset];
    }

    __read_permission(ghostsum_lower[tid]);
    __read_permission(ghostsum_upper[tid]);
    __barrier_invariant(sweep(tid,offset), tid, tid-offset, other_tid);
    __barrier_invariant(__implies(tid >= offset, ghosttemp_lower == ghostsum_lower[tid-offset]), tid);
    __barrier_invariant(__implies(tid >= offset, ghosttemp_upper == ghostsum_upper[tid-offset]), tid);
    __barrier_invariant(__implies(tid >= offset, temp == sum[tid-offset]), tid);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid >= offset)
    {
      // concretely
      sum[tid] = __add_noovfl_int(sum[tid], temp);
      // abstractly, adding adjacent intervals
      __assert(ghosttemp_lower <= ghosttemp_upper);
      __assert(ghostsum_lower[tid] <= ghostsum_upper[tid]);
      __assert(ghosttemp_upper + 1 == ghostsum_lower[tid]);
      ghostsum_lower[tid] = ghosttemp_lower;
    }

    __read_permission(ghostsum_lower[tid]);
    __read_permission(ghostsum_upper[tid]);
    __barrier_invariant(sweep(tid,2*offset), tid, tid-(2*offset));
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  output[tid] = sum[tid];
  __assert(__accessed(output, tid));

#ifdef FORCE_FAIL
  __assert(false);
#endif
}
