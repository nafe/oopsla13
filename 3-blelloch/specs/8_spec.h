// ---------------------------------------------------------------------------
// AXIOMS    
// ---------------------------------------------------------------------------
#if defined(__CUDA_ARCH__)
#if (defined(INC_ENDSPEC) && defined(SPEC_ELEMENTWISE)) || defined(INC_CONVERT)
__axiom(blockDim.x == 8);
#else
__axiom(blockDim.x == 4);
#endif
__axiom(gridDim.x == 1);
#elif defined(__OPENCL_VERSION__)
#if (defined(INC_ENDSPEC) && defined(SPEC_ELEMENTWISE)) || defined(INC_CONVERT)
__axiom(get_local_size(0) == 8);
#else
__axiom(get_local_size(0) == 4);
#endif
__axiom(get_num_groups(0) == 1);
#else
  #error Not using CUDA or OpenCL?
#endif

// ---------------------------------------------------------------------------
// PARAMETERS
// ---------------------------------------------------------------------------
#if N != 8
  #error This specification is only valid when N=8
#endif

#if dwidth == 8
  #define __ite_dtype(b,x,y) __ite_unsigned_char(b,x,y)
#elif dwidth == 16
  #define __ite_dtype(b,x,y) __ite_unsigned_short(b,x,y)
#elif dwidth == 32
  #define __ite_dtype(b,x,y) __ite_unsigned_int(b,x,y)
#elif dwidth == 64
  #define __ite_dtype(b,x,y) __ite_unsigned_long(b,x,y)
#else
  #error dwidth must be defined
#endif

#if rwidth == 8
  #define __binop_add_raddf(x,y) __add_unsigned_char(x,y)
  #define __ite_rtype(b,x,y) __ite_unsigned_char(b,x,y)
#elif rwidth == 16
  #define __binop_add_raddf(x,y) __add_unsigned_short(x,y)
  #define __ite_rtype(b,x,y) __ite_unsigned_short(b,x,y)
#elif rwidth == 32
  #define __binop_add_raddf(x,y) __add_unsigned_int(x,y)
  #define __ite_rtype(b,x,y) __ite_unsigned_int(b,x,y)
#elif rwidth == 64
  #define __binop_add_raddf(x,y) __add_unsigned_long(x,y)
  #define __ite_rtype(b,x,y) __ite_unsigned_long(b,x,y)
#else
  #error rwidth must be defined
#endif

#if defined(BINOP_ADD)
  #define raddf(x,y) __binop_add_raddf(x,y)
  #define raddf_primed(x,y) __binop_add_raddf(x,y)
  #define ridentity 0
#elif defined(BINOP_OR)
  #define raddf(x,y) (x | y)
  #define raddf_primed(x,y) (x | y)
  #define ridentity 0
#elif defined(BINOP_MAX)
  #define raddf(x,y) __ite_rtype(x < y, y, x)
  #define raddf_primed(x,y) __ite_rtype(x < y, y, x)
  #define ridentity 0
#elif defined(BINOP_ABSTRACT)
  DECLARE_UF_BINARY(A, rtype, rtype, rtype);
  DECLARE_UF_BINARY(A1, rtype, rtype, rtype);
  #define raddf(x,y) A(x,y)
  #define raddf_primed(x,y) A1(x,y)
  #define ridentity 0
#else
  #error BINOP_ADD|BINOP_OR|BINOP_MAX|BINOP_ABSTRACT must be defined
#endif

// ---------------------------------------------------------------------------
// HELPERS
// ---------------------------------------------------------------------------
#define __non_temporal(x) \
  __non_temporal_loads_begin(), x, __non_temporal_loads_end()

#define div2(x) (x >> 1)
#define iseven(x) ((x & 1) == 0)
#define isone(bit,x) (((x >> bit) & 1) == 1)
#define modpow2(x,y) (x & (y-1))
#define mul2(x) (x << 1)
#define mul2add1(x) (mul2(x) | 1)
#define pow2(bit) (1 << bit)

#if defined(__CUDA_ARCH__)
  #define tid threadIdx.x
#elif defined(__OPENCL_VERSION__)
  #define tid get_local_id(0)
#else
  #error Not using CUDA or OpenCL?
#endif
#define other_tid __other_int(tid)

#define isvertex(x,offset) ((offset == 0) | (modpow2(x+1,offset) == 0))
#define stopped(x,offset) isvertex(x+offset, mul2(offset))
#define left(x,offset) (x - div2(offset))
#define iselement(x)  ((0 < x) & (x < 8))
#define isthreadid(t) ((0 < t) & (t < 4))

#define ai_idx(offset,tid) ((offset * mul2add1(tid)) - 1)
#define bi_idx(offset,tid) ((offset * (mul2(tid)+2)) - 1)

// ---------------------------------------------------------------------------
// UPSWEEP INVARIANTS
// ---------------------------------------------------------------------------
#define upsweep_core(offset,result,len,x) \
  result[x] == raddf(raddf(raddf(__ite_rtype((offset >= 8) & isvertex(x,8), result[x-4], ridentity),__ite_rtype((offset >= 4) & isvertex(x,4), result[x-2], ridentity)),__ite_rtype((offset >= 2) & isvertex(x,2), result[x-1], ridentity)),len[x])

#if defined(FORCE_NOOVFL) || (defined(INC_ENDSPEC) && defined(BINOP_ADD))
#define upsweep_nooverflow(offset,result,len,x) \
  (__implies((((offset == 1) & isvertex(x,offset)) | ((1 < offset) & stopped(x,1))), __add_noovfl(len[x])) & \
  __implies((((offset == 2) & isvertex(x,offset)) | ((2 < offset) & stopped(x,2))), __add_noovfl(len[x], result[x-1])) & \
  __implies((((offset == 4) & isvertex(x,offset)) | ((4 < offset) & stopped(x,4))), __add_noovfl(len[x], result[x-1], result[x-2])) & \
  __implies((((offset == 8) & isvertex(x,offset)) | ((8 < offset) & stopped(x,8))), __add_noovfl(len[x], result[x-1], result[x-2], result[x-4])))

#define upsweep(offset,result,len,x) \
  (upsweep_core(offset,result,len,x) & upsweep_nooverflow(offset,result,len,x))
#else
#define upsweep(offset,result,len,x) \
  upsweep_core(offset,result,len,x)
#endif

#define upsweep_barrier(tid,offset,result,len) \
  (__implies((tid < 4) & (offset >= 1), upsweep(offset,result,len,ai_idx(1,tid))) & \
  __implies((tid < 2) & (offset >= 4), upsweep(offset,result,len,ai_idx(2,tid))) & \
  __implies((tid < 1) & (offset >= 8), upsweep(offset,result,len,ai_idx(4,tid))) & \
  __implies((tid < 4) & (offset <= 2), upsweep(offset,result,len,bi_idx(1,tid))) & \
  __implies((tid < 2) & (offset == 4), upsweep(offset,result,len,bi_idx(2,tid))) & \
  __implies((tid < 1) & (offset == 8), upsweep(offset,result,len,bi_idx(4,tid))))

#define upsweep_d_offset \
  ((d == 4 & offset == 1) | (d == 2 & offset == 2) | (d == 1 & offset == 4) | (d == 0 & offset == 8))

#define upsweep_permissions(offset,result,len,x) \
  {ghostread2 = ghostread4 = ghostread8 = false; \
  if ((((offset == 2) && isvertex(x,offset)) || ((2 < offset) && stopped(x,2)))) ghostread2 = true; \
  if ((((offset == 4) && isvertex(x,offset)) || ((4 < offset) && stopped(x,4)))) ghostread2 = ghostread4 = true; \
  if ((((offset == 8) && isvertex(x,offset)) || ((8 < offset) && stopped(x,8)))) ghostread2 = ghostread4 = ghostread8 = true; \
  __read_permission(len[x]) \
  __read_permission(result[x]) \
  if (ghostread2) __read_permission(result[left(x,2)]) \
  if (ghostread4) __read_permission(result[left(x,4)]) \
  if (ghostread8) __read_permission(result[left(x,8)])}

#define upsweep_barrier_permissions(tid,offset,result,len) \
  {bool ghostread2, ghostread4, ghostread8; \
  if ((tid < 4) && (offset >= 1)) upsweep_permissions(offset,result,len,ai_idx(1,tid)) \
  if ((tid < 2) && (offset >= 4)) upsweep_permissions(offset,result,len,ai_idx(2,tid)) \
  if ((tid < 1) && (offset >= 8)) upsweep_permissions(offset,result,len,ai_idx(4,tid)) \
  if ((tid < 4) && (offset <= 2)) upsweep_permissions(offset,result,len,bi_idx(1,tid)) \
  if ((tid < 2) && (offset == 4)) upsweep_permissions(offset,result,len,bi_idx(2,tid)) \
  if ((tid < 1) && (offset == 8)) upsweep_permissions(offset,result,len,bi_idx(4,tid))}

// ---------------------------------------------------------------------------
// DOWNSWEEP INVARIANTS
// ---------------------------------------------------------------------------
#define sum_pow2_zeroes(bit,x) \
  (__ite_dtype((0 < bit) & !isone(0,x), pow2(0), 0) + \
__ite_dtype((1 < bit) & !isone(1,x), pow2(1), 0))

#define term(ghostsum,bit,x) \
  __ite_rtype(!isone(bit,x), 0, ghostsum[x + sum_pow2_zeroes(bit,x) - pow2(bit)])

#define downsweep_core(offset,result,ghostsum,x) \
  (result[x] == __ite_rtype(isvertex(x,mul2(offset)), raddf(raddf(__ite_rtype((offset <= 2), term(ghostsum,2,x), ridentity),__ite_rtype((offset <= 1), term(ghostsum,1,x), ridentity)),__ite_rtype((offset <= 0), term(ghostsum,0,x), ridentity)), ghostsum[x]))

#if defined(FORCE_NOOVFL) || (defined(INC_ENDSPEC) && defined(BINOP_ADD))
#define downsweep_nooverflow(offset,result,ghostsum,x) \
  (__implies(isvertex(x,mul2(offset)), __add_noovfl(__ite_rtype((offset <= 2), term(ghostsum,2,x), ridentity), __ite_rtype((offset <= 1), term(ghostsum,1,x), ridentity), __ite_rtype((offset <= 0), term(ghostsum,0,x), ridentity))))

#define downsweep(offset,result,ghostsum,x) \
  (downsweep_core(offset,result,ghostsum,x) & downsweep_nooverflow(offset,result,ghostsum,x))
#else
#define downsweep(offset,result,ghostsum,x) \
  downsweep_core(offset,result,ghostsum,x)
#endif

#define downsweep_barrier(tid,offset,result,ghostsum) \
  (__implies((tid < 1) & (offset >= 2), downsweep(offset,result,ghostsum,ai_idx(4,tid))) & \
  __implies((tid < 2) & (offset >= 1), downsweep(offset,result,ghostsum,ai_idx(2,tid))) & \
  __implies((tid < 4) & (offset >= 0), downsweep(offset,result,ghostsum,ai_idx(1,tid))) & \
  __implies((tid < 1) & (offset >= 2), downsweep(offset,result,ghostsum,bi_idx(4,tid))) & \
  __implies((tid < 2) & (offset == 1), downsweep(offset,result,ghostsum,bi_idx(2,tid))) & \
  __implies((tid < 4) & (offset == 0), downsweep(offset,result,ghostsum,bi_idx(1,tid))))

#define downsweep_d_offset \
  ((d == 1 & offset == 8) | (d == 2 & offset == 4) | (d == 4 & offset == 2) | (d == 8 & offset == 1))

#define downsweep_permissions(offset,result,ghostsum,x) \
  {__read_permission(result[x]) \
  __read_permission(ghostsum[x + sum_pow2_zeroes(0,x) - pow2(0)]) \
  __read_permission(ghostsum[x + sum_pow2_zeroes(1,x) - pow2(1)]) \
  __read_permission(ghostsum[x + sum_pow2_zeroes(2,x) - pow2(2)])}

#define downsweep_barrier_permissions(tid,offset,result,ghostsum) \
  {if ((tid < 1) && (offset >= 2)) downsweep_permissions(offset,result,ghostsum,ai_idx(4,tid)) \
  if ((tid < 2) && (offset >= 1)) downsweep_permissions(offset,result,ghostsum,ai_idx(2,tid)) \
  if ((tid < 4) && (offset >= 0)) downsweep_permissions(offset,result,ghostsum,ai_idx(1,tid)) \
  if ((tid < 1) && (offset >= 2)) downsweep_permissions(offset,result,ghostsum,bi_idx(4,tid)) \
  if ((tid < 2) && (offset == 1)) downsweep_permissions(offset,result,ghostsum,bi_idx(2,tid)) \
  if ((tid < 4) && (offset == 0)) downsweep_permissions(offset,result,ghostsum,bi_idx(1,tid))}

// ---------------------------------------------------------------------------
// END SPECIFICATION
// ---------------------------------------------------------------------------
#define x2t(x) __ite_dtype(iseven(x), div2(x), div2((x-1)))

#define final_upsweep_barrier(tid,result,len) \
  (__implies((tid < 4), upsweep(/*offset=*/N,result,len,ai_idx(1,tid))) & \
  __implies((tid < 2), upsweep(/*offset=*/N,result,len,ai_idx(2,tid))) & \
  __implies((tid < 1), upsweep(/*offset=*/N,result,len,ai_idx(4,tid))) & \
  __implies((tid < 1), upsweep(/*offset=*/N,result,len,bi_idx(4,tid))))

#define final_downsweep_barrier(tid,result,len) \
  (__implies((tid < 4), downsweep(/*offset=*/0,result,ghostsum,ai_idx(1,tid))) & \
  __implies((tid < 4), downsweep(/*offset=*/0,result,ghostsum,bi_idx(1,tid))))

#if defined(SPEC_THREADWISE)
#define upsweep_instantiation \
  tid, div2(tid), other_tid
#elif defined(SPEC_ELEMENTWISE)
#define upsweep_instantiation \
  x2t(tid), x2t(div2(tid)), x2t(other_tid)
#endif

// ---------------------------------------------------------------------------
// CONVERT SPECIFICATION
// ---------------------------------------------------------------------------
#define threadspec(s,t) \
 ((raddf(result[2*s], len[2*s]) == result[2*s+1]) & \
  (raddf(result[2*t], len[2*t]) == result[2*t+1]) & \
  __implies(s < t, raddf(result[2*s+1], len[2*s+1]) <= result[2*t]))
