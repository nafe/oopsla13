// ---------------------------------------------------------------------------
// AXIOMS    
// ---------------------------------------------------------------------------
#if defined(__CUDA_ARCH__)
#if defined(INC_ENDSPEC) && defined(SPEC_ELEMENTWISE)
__axiom(blockDim.x == 64);
#else
__axiom(blockDim.x == 32);
#endif
__axiom(gridDim.x == 1);
#elif defined(__OPENCL_VERSION__)
#if defined(INC_ENDSPEC) && defined(SPEC_ELEMENTWISE)
__axiom(get_local_size(0) == 64);
#else
__axiom(get_local_size(0) == 32);
#endif
__axiom(get_num_groups(0) == 1);
#else
  #error Not using CUDA or OpenCL?
#endif

// ---------------------------------------------------------------------------
// PARAMETERS
// ---------------------------------------------------------------------------
#if N != 64
  #error This specification is only valid when N=64
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
#define updated(x,offset) ((offset < x) & isvertex((x-offset), mul2(offset)))
#define iselement(x)  ((0 < x) & (x < 64))
#define isthreadid(t) ((0 < t) & (t < 32))

#define ai_idx(offset,tid) ((offset * mul2add1(tid)) - 1)
#define bi_idx(offset,tid) ((offset * (mul2(tid)+2)) - 1)

#define lf_ai_idx(offset,tid) ((offset * (tid + 1)) - 1)
#define lf_bi_idx(offset,tid) (lf_ai_idx(offset,tid) + div2(offset))

#define lf_ai_tid(tid) __ite_dtype(tid == 0, 0, __ite_dtype(iseven(tid), (div2(tid)-1), (div2((tid-1)))))
#define lf_bi_tid(tid) (tid+1)

// ---------------------------------------------------------------------------
// UPSWEEP INVARIANTS
// ---------------------------------------------------------------------------
#define upsweep_core(offset,result,len,x) \
  (__implies((((offset == 1) & isvertex(x,offset)) | ((1 < offset) & stopped(x,1))), result[x] == len[x]) & \
  __implies((((offset == 2) & isvertex(x,offset)) | ((2 < offset) & stopped(x,2))), result[x] == raddf(result[left(x,2)],len[x])) & \
  __implies((((offset == 4) & isvertex(x,offset)) | ((4 < offset) & stopped(x,4))), result[x] == raddf(raddf(result[left(x,4)],result[left(x,2)]),len[x])) & \
  __implies((((offset == 8) & isvertex(x,offset)) | ((8 < offset) & stopped(x,8))), result[x] == raddf(raddf(raddf(result[left(x,8)],result[left(x,4)]),result[left(x,2)]),len[x])) & \
  __implies((((offset == 16) & isvertex(x,offset)) | ((16 < offset) & stopped(x,16))), result[x] == raddf(raddf(raddf(raddf(result[left(x,16)],result[left(x,8)]),result[left(x,4)]),result[left(x,2)]),len[x])) & \
  __implies((((offset == 32) & isvertex(x,offset)) | ((32 < offset) & stopped(x,32))), result[x] == raddf(raddf(raddf(raddf(raddf(result[left(x,32)],result[left(x,16)]),result[left(x,8)]),result[left(x,4)]),result[left(x,2)]),len[x])) & \
  __implies((((offset == 64) & isvertex(x,offset)) | ((64 < offset) & stopped(x,64))), result[x] == raddf(raddf(raddf(raddf(raddf(raddf(result[left(x,64)],result[left(x,32)]),result[left(x,16)]),result[left(x,8)]),result[left(x,4)]),result[left(x,2)]),len[x])))

#if defined(INC_ENDSPEC) && defined(BINOP_ADD)
#define upsweep_nooverflow(offset,result,len,x) \
  (__implies((((offset == 1) & isvertex(x,offset)) | ((1 < offset) & stopped(x,1))), __add_noovfl(len[x])) & \
  __implies((((offset == 2) & isvertex(x,offset)) | ((2 < offset) & stopped(x,2))), __add_noovfl(len[x], result[left(x,2)])) & \
  __implies((((offset == 4) & isvertex(x,offset)) | ((4 < offset) & stopped(x,4))), __add_noovfl(len[x], result[left(x,2)], result[left(x,4)])) & \
  __implies((((offset == 8) & isvertex(x,offset)) | ((8 < offset) & stopped(x,8))), __add_noovfl(len[x], result[left(x,2)], result[left(x,4)], result[left(x,8)])) & \
  __implies((((offset == 16) & isvertex(x,offset)) | ((16 < offset) & stopped(x,16))), __add_noovfl(len[x], result[left(x,2)], result[left(x,4)], result[left(x,8)], result[left(x,16)])) & \
  __implies((((offset == 32) & isvertex(x,offset)) | ((32 < offset) & stopped(x,32))), __add_noovfl(len[x], result[left(x,2)], result[left(x,4)], result[left(x,8)], result[left(x,16)], result[left(x,32)])) & \
  __implies((((offset == 64) & isvertex(x,offset)) | ((64 < offset) & stopped(x,64))), __add_noovfl(len[x], result[left(x,2)], result[left(x,4)], result[left(x,8)], result[left(x,16)], result[left(x,32)], result[left(x,64)])))

#define upsweep(offset,result,len,x) \
  (upsweep_core(offset,result,len,x) & upsweep_nooverflow(offset,result,len,x))
#else
#define upsweep(offset,result,len,x) \
  upsweep_core(offset,result,len,x)
#endif

#define upsweep_barrier(tid,offset,result,len) \
  (__implies((tid < 32) & (offset >= 1), upsweep(offset,result,len,ai_idx(1,tid))) & \
  __implies((tid < 16) & (offset >= 4), upsweep(offset,result,len,ai_idx(2,tid))) & \
  __implies((tid < 8) & (offset >= 8), upsweep(offset,result,len,ai_idx(4,tid))) & \
  __implies((tid < 4) & (offset >= 16), upsweep(offset,result,len,ai_idx(8,tid))) & \
  __implies((tid < 2) & (offset >= 32), upsweep(offset,result,len,ai_idx(16,tid))) & \
  __implies((tid < 1) & (offset >= 64), upsweep(offset,result,len,ai_idx(32,tid))) & \
  __implies((tid < 32) & (offset <= 2), upsweep(offset,result,len,bi_idx(1,tid))) & \
  __implies((tid < 16) & (offset == 4), upsweep(offset,result,len,bi_idx(2,tid))) & \
  __implies((tid < 8) & (offset == 8), upsweep(offset,result,len,bi_idx(4,tid))) & \
  __implies((tid < 4) & (offset == 16), upsweep(offset,result,len,bi_idx(8,tid))) & \
  __implies((tid < 2) & (offset == 32), upsweep(offset,result,len,bi_idx(16,tid))) & \
  __implies((tid < 1) & (offset == 64), upsweep(offset,result,len,bi_idx(32,tid))))

#define upsweep_d_offset \
  ((d == 32 & offset == 1) | (d == 16 & offset == 2) | (d == 8 & offset == 4) | (d == 4 & offset == 8) | (d == 2 & offset == 16) | (d == 1 & offset == 32) | (d == 0 & offset == 64))

#define upsweep_permissions(offset,result,len,x) \
  {ghostread2 = ghostread4 = ghostread8 = ghostread16 = ghostread32 = ghostread64 = false; \
  if ((((offset == 2) && isvertex(x,offset)) || ((2 < offset) && stopped(x,2)))) ghostread2 = true; \
  if ((((offset == 4) && isvertex(x,offset)) || ((4 < offset) && stopped(x,4)))) ghostread2 = ghostread4 = true; \
  if ((((offset == 8) && isvertex(x,offset)) || ((8 < offset) && stopped(x,8)))) ghostread2 = ghostread4 = ghostread8 = true; \
  if ((((offset == 16) && isvertex(x,offset)) || ((16 < offset) && stopped(x,16)))) ghostread2 = ghostread4 = ghostread8 = ghostread16 = true; \
  if ((((offset == 32) && isvertex(x,offset)) || ((32 < offset) && stopped(x,32)))) ghostread2 = ghostread4 = ghostread8 = ghostread16 = ghostread32 = true; \
  if ((((offset == 64) && isvertex(x,offset)) || ((64 < offset) && stopped(x,64)))) ghostread2 = ghostread4 = ghostread8 = ghostread16 = ghostread32 = ghostread64 = true; \
  __read_permission(len[x]) \
  __read_permission(result[x]) \
  if (ghostread2) __read_permission(result[left(x,2)]) \
  if (ghostread4) __read_permission(result[left(x,4)]) \
  if (ghostread8) __read_permission(result[left(x,8)]) \
  if (ghostread16) __read_permission(result[left(x,16)]) \
  if (ghostread32) __read_permission(result[left(x,32)]) \
  if (ghostread64) __read_permission(result[left(x,64)])}

#define upsweep_barrier_permissions(tid,offset,result,len) \
  {bool ghostread2, ghostread4, ghostread8, ghostread16, ghostread32, ghostread64; \
  if ((tid < 32) && (offset >= 1)) upsweep_permissions(offset,result,len,ai_idx(1,tid)) \
  if ((tid < 16) && (offset >= 4)) upsweep_permissions(offset,result,len,ai_idx(2,tid)) \
  if ((tid < 8) && (offset >= 8)) upsweep_permissions(offset,result,len,ai_idx(4,tid)) \
  if ((tid < 4) && (offset >= 16)) upsweep_permissions(offset,result,len,ai_idx(8,tid)) \
  if ((tid < 2) && (offset >= 32)) upsweep_permissions(offset,result,len,ai_idx(16,tid)) \
  if ((tid < 1) && (offset >= 64)) upsweep_permissions(offset,result,len,ai_idx(32,tid)) \
  if ((tid < 32) && (offset <= 2)) upsweep_permissions(offset,result,len,bi_idx(1,tid)) \
  if ((tid < 16) && (offset == 4)) upsweep_permissions(offset,result,len,bi_idx(2,tid)) \
  if ((tid < 8) && (offset == 8)) upsweep_permissions(offset,result,len,bi_idx(4,tid)) \
  if ((tid < 4) && (offset == 16)) upsweep_permissions(offset,result,len,bi_idx(8,tid)) \
  if ((tid < 2) && (offset == 32)) upsweep_permissions(offset,result,len,bi_idx(16,tid)) \
  if ((tid < 1) && (offset == 64)) upsweep_permissions(offset,result,len,bi_idx(32,tid))}

// ---------------------------------------------------------------------------
// DOWNSWEEP INVARIANTS
// ---------------------------------------------------------------------------
#define ilog2(x) \
  (__ite_dtype(x == 1, 0, (__ite_dtype((2 <= x) & (x < 4), 1, (__ite_dtype((4 <= x) & (x < 8), 2, (__ite_dtype((8 <= x) & (x < 16), 3, (__ite_dtype((16 <= x) & (x < 32), 4, 5))))))))))

#define term(x,i) \
  (x - (__ite_dtype((0 <= i) & isone(0, (x+1)), 1, 0) + __ite_dtype((1 <= i) & isone(1, (x+1)), 2, 0) + __ite_dtype((2 <= i) & isone(2, (x+1)), 4, 0) + __ite_dtype((3 <= i) & isone(3, (x+1)), 8, 0) + __ite_dtype((4 <= i) & isone(4, (x+1)), 16, 0)))

#define downsweep_summation(result,ghostsum,x) \
  (result[x] == (raddf(raddf(raddf(raddf(raddf(__ite_rtype(isone(4,(x+1)) & (4 < ilog2(x+1)), ghostsum[term(x,4)], 0),__ite_rtype(isone(3,(x+1)) & (3 < ilog2(x+1)), ghostsum[term(x,3)], 0)),__ite_rtype(isone(2,(x+1)) & (2 < ilog2(x+1)), ghostsum[term(x,2)], 0)),__ite_rtype(isone(1,(x+1)) & (1 < ilog2(x+1)), ghostsum[term(x,1)], 0)),__ite_rtype(isone(0,(x+1)) & (0 < ilog2(x+1)), ghostsum[term(x,0)], 0)),ghostsum[x])))

#define downsweep_core(offset,result,ghostsum,x) \
  (__implies(((offset == 64) | \
            (!updated(x,16) & ((offset == 32) | \
            (!updated(x,8) & ((offset == 16) | \
            (!updated(x,4) & ((offset == 8) | \
            (!updated(x,2) & ((offset == 4) | \
            (!updated(x,1) & ((offset == 2)))))))))))), (result[x] == ghostsum[x])) & \
   __implies(((offset <= 32) & updated(x,16)) | ((offset <= 16) & updated(x,8)) | ((offset <= 8) & updated(x,4)) | ((offset <= 4) & updated(x,2)) | ((offset <= 2) & updated(x,1)), downsweep_summation(result,ghostsum,x)))

#if defined(INC_ENDSPEC) && defined(BINOP_ADD)
#define downsweep_nooverflow(offset,result,ghostsum,x) \
  __implies(((offset <= 32) & updated(x,16)) | ((offset <= 16) & updated(x,8)) | ((offset <= 8) & updated(x,4)) | ((offset <= 4) & updated(x,2)) | ((offset <= 2) & updated(x,1)), __add_noovfl(__ite_rtype(isone(4,(x+1)) & (4 < ilog2(x+1)), ghostsum[term(x,4)], 0), __ite_rtype(isone(3,(x+1)) & (3 < ilog2(x+1)), ghostsum[term(x,3)], 0), __ite_rtype(isone(2,(x+1)) & (2 < ilog2(x+1)), ghostsum[term(x,2)], 0), __ite_rtype(isone(1,(x+1)) & (1 < ilog2(x+1)), ghostsum[term(x,1)], 0), __ite_rtype(isone(0,(x+1)) & (0 < ilog2(x+1)), ghostsum[term(x,0)], 0), ghostsum[x]))

#define downsweep(offset,result,ghostsum,x) \
  downsweep_core(offset,result,ghostsum,x) & downsweep_nooverflow(offset,result,ghostsum,x)
#else
#define downsweep(offset,result,ghostsum,x) \
  downsweep_core(offset,result,ghostsum,x)
#endif

#define downsweep_barrier(tid,offset,result,ghostsum) \
  (__implies((tid < 32) & (offset > 2), downsweep(offset,result,ghostsum,ai_idx(1,tid))) & \
   __implies((tid < 16) & (offset > 4), downsweep(offset,result,ghostsum,ai_idx(2,tid))) & \
   __implies((tid < 8) & (offset > 8), downsweep(offset,result,ghostsum,ai_idx(4,tid))) & \
   __implies((tid < 4) & (offset > 16), downsweep(offset,result,ghostsum,ai_idx(8,tid))) & \
   __implies((tid < 2) & (offset > 32), downsweep(offset,result,ghostsum,ai_idx(16,tid))) & \
   __implies((tid < 1),                downsweep(offset,result,ghostsum,ai_idx(div2(offset),tid))) & \
   __implies((tid < 1),                downsweep(offset,result,ghostsum,bi_idx(32,tid))) & \
   __implies((tid < 1) & (offset == 32), downsweep(offset,result,ghostsum,lf_ai_idx(32,tid))) & \
   __implies((tid < 3) & (offset == 16), downsweep(offset,result,ghostsum,lf_ai_idx(16,tid))) & \
   __implies((tid < 7) & (offset == 8), downsweep(offset,result,ghostsum,lf_ai_idx(8,tid))) & \
   __implies((tid < 15) & (offset == 4), downsweep(offset,result,ghostsum,lf_ai_idx(4,tid))) & \
   __implies((tid < 31) & (offset == 2), downsweep(offset,result,ghostsum,lf_ai_idx(2,tid))) & \
   __implies((tid < 1) & (offset == 32), downsweep(offset,result,ghostsum,lf_bi_idx(32,tid))) & \
   __implies((tid < 3) & (offset == 16), downsweep(offset,result,ghostsum,lf_bi_idx(16,tid))) & \
   __implies((tid < 7) & (offset == 8), downsweep(offset,result,ghostsum,lf_bi_idx(8,tid))) & \
   __implies((tid < 15) & (offset == 4), downsweep(offset,result,ghostsum,lf_bi_idx(4,tid))) & \
   __implies((tid < 31) & (offset == 2), downsweep(offset,result,ghostsum,lf_bi_idx(2,tid))))

#define downsweep_d_offset \
  (((d == 2) & (offset == 64)) | ((d == 4) & (offset == 32)) | ((d == 8) & (offset == 16)) | ((d == 16) & (offset == 8)) | ((d == 32) & (offset == 4)) | ((d == 64) & (offset == 2)))

#define downsweep_permissions(offset,result,ghostsum,x) \
  {__read_permission(result[x]) \
  __read_permission(ghostsum[term(x,0)]) \
  __read_permission(ghostsum[term(x,1)]) \
  __read_permission(ghostsum[term(x,2)]) \
  __read_permission(ghostsum[term(x,3)]) \
  __read_permission(ghostsum[term(x,4)])}

#define downsweep_barrier_permissions(tid,offset,result,ghostsum) \
  {if ((tid < 32) & (offset > 2)) downsweep_permissions(offset,result,ghostsum,ai_idx(1,tid)) \
   if ((tid < 16) & (offset > 4)) downsweep_permissions(offset,result,ghostsum,ai_idx(2,tid)) \
   if ((tid < 8) & (offset > 8)) downsweep_permissions(offset,result,ghostsum,ai_idx(4,tid)) \
   if ((tid < 4) & (offset > 16)) downsweep_permissions(offset,result,ghostsum,ai_idx(8,tid)) \
   if ((tid < 2) & (offset > 32)) downsweep_permissions(offset,result,ghostsum,ai_idx(16,tid)) \
   if (tid < 1)                   downsweep_permissions(offset,result,ghostsum,ai_idx(div2(offset),tid)) \
   if (tid < 1)                   downsweep_permissions(offset,result,ghostsum,bi_idx(32,tid)) \
   if ((tid < 1) & (offset == 32)) downsweep_permissions(offset,result,ghostsum,lf_ai_idx(32,tid)) \
   if ((tid < 3) & (offset == 16)) downsweep_permissions(offset,result,ghostsum,lf_ai_idx(16,tid)) \
   if ((tid < 7) & (offset == 8)) downsweep_permissions(offset,result,ghostsum,lf_ai_idx(8,tid)) \
   if ((tid < 15) & (offset == 4)) downsweep_permissions(offset,result,ghostsum,lf_ai_idx(4,tid)) \
   if ((tid < 31) & (offset == 2)) downsweep_permissions(offset,result,ghostsum,lf_ai_idx(2,tid)) \
   if ((tid < 1) & (offset == 32)) downsweep_permissions(offset,result,ghostsum,lf_bi_idx(32,tid)) \
   if ((tid < 3) & (offset == 16)) downsweep_permissions(offset,result,ghostsum,lf_bi_idx(16,tid)) \
   if ((tid < 7) & (offset == 8)) downsweep_permissions(offset,result,ghostsum,lf_bi_idx(8,tid)) \
   if ((tid < 15) & (offset == 4)) downsweep_permissions(offset,result,ghostsum,lf_bi_idx(4,tid)) \
   if ((tid < 31) & (offset == 2)) downsweep_permissions(offset,result,ghostsum,lf_bi_idx(2,tid))}

// ---------------------------------------------------------------------------
// END SPECIFICATION
// ---------------------------------------------------------------------------
#define x2t(x) \
  __ite_dtype((x == 0) | (x == (N-1)), 0, \
    __ite_dtype(iseven(x), (div2(x)-1), div2((x-1))))

#define final_upsweep_barrier(tid,result,len) \
  (__implies((tid < 32), upsweep(/*offset=*/N,result,len,ai_idx(1,tid))) & \
   __implies((tid < 16), upsweep(/*offset=*/N,result,len,ai_idx(2,tid))) & \
   __implies((tid < 8), upsweep(/*offset=*/N,result,len,ai_idx(4,tid))) & \
   __implies((tid < 4), upsweep(/*offset=*/N,result,len,ai_idx(8,tid))) & \
   __implies((tid < 2), upsweep(/*offset=*/N,result,len,ai_idx(16,tid))) & \
   __implies((tid < 1), upsweep(/*offset=*/N,result,len,ai_idx(32,tid))) & \
   __implies((tid < 1), upsweep(/*offset=*/N,result,len,bi_idx(32,tid))))

#define final_downsweep_barrier(tid,result,ghostsum) \
  (__implies((tid < 1), downsweep(/*offset=*/2,result,ghostsum,ai_idx(1,tid))) & \
   __implies((tid < 1), downsweep(/*offset=*/2,result,ghostsum,bi_idx(32,tid))) & \
   __implies((tid < 31), downsweep(/*offset=*/2,result,ghostsum,lf_ai_idx(2,tid))) & \
   __implies((tid < 31), downsweep(/*offset=*/2,result,ghostsum,lf_bi_idx(2,tid))))

#if defined(SPEC_THREADWISE)
#define upsweep_instantiation \
  tid, (tid+1), div2((tid+1)), div2(div2((tid+1))), div2(div2(div2((tid+1)))), div2(div2(div2(div2((tid+1))))), other_tid, (other_tid+1), div2((other_tid+1)), div2(div2((other_tid+1))), div2(div2(div2((other_tid+1)))), div2(div2(div2(div2((other_tid+1)))))

#define downsweep_instantiation \
  tid, other_tid

#elif defined(SPEC_ELEMENTWISE)
#define upsweep_instantiation \
  x2t(tid), (x2t(tid)+1), div2((x2t(tid)+1)), div2(div2((x2t(tid)+1))), div2(div2(div2((x2t(tid)+1)))), div2(div2(div2(div2((x2t(tid)+1))))), x2t(other_tid), (x2t(other_tid)+1), div2((x2t(other_tid)+1)), div2(div2((x2t(other_tid)+1))), div2(div2(div2((x2t(other_tid)+1)))), div2(div2(div2(div2((x2t(other_tid)+1)))))

#define downsweep_instantiation \
  x2t(tid), x2t(other_tid)
#endif
