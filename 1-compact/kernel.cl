#define N __LOCAL_SIZE_0
#define tid get_local_id(0)

__kernel void compact(__global int*out, __global int*in,
  __local unsigned *flag,
  __local unsigned *idx) {

  // (i) test each element with predicate p
  // flag = 1 if keeping element
  //        0 otherwise
  flag[tid] = ((in[tid] & 1) == 0) ? 1 : 0;

  // (ii) compute indexes for scatter
  //      using an exclusive prefix sum
  barrier(CLK_LOCAL_MEM_FENCE);
  if (tid < N/2) {
    idx[2*tid]   = flag[2*tid];
    idx[2*tid+1] = flag[2*tid+1];
  }
  // (a) upsweep
  int offset = 1;
  for (unsigned d = N/2; d > 0; d /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      idx[bi] += idx[ai];
    }
    offset *= 2;
  }
  // (b) downsweep
  if (tid == 0) idx[N-1] = 0;
  for (unsigned d = 1; d < N; d *= 2) {
    offset /= 2;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      int temp = idx[ai];
      idx[ai] = idx[bi];
      idx[bi] += temp;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // end of exclusive prefix sum of flag into idx

  // (iii) scatter
  if (flag[tid]) out[idx[tid]] = in[tid];
}
