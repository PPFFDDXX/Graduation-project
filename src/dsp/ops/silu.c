#include <math.h>

#include "dsp/hvx_internal.h"
#include "dsp/hvx_inverse.h"
#include "dsp/hvx_math.h"
#include "dsp/hvx_utils.h"
#include "dsp/op_parallel.h"

#define PREFETCH_SIZE   (8 * 1024)
#define PREFETCH_N_VECS (PREFETCH_SIZE / VLEN)

static inline void hvx_silu_f32_inner(float *restrict dst, const float *restrict src, int ne0) {
  const int n_vecs = ne0 / 32;

  const HVX_Vector *pv_in  = (const HVX_Vector *) src;
  HVX_Vector       *pv_out = (HVX_Vector *) dst;
  const HVX_Vector  v_one  = hvx_vec_splat_f32(1.0f);
  const HVX_Vector  v_log2e = hvx_vec_splat_f32(1.4426950408889634f);

  for (int i = 0; i < n_vecs; ++i) {
    if (i % PREFETCH_N_VECS == 0) {
      int prefetch_idx = i + PREFETCH_N_VECS;
      if (prefetch_idx < n_vecs) {
        int prefetch_n_vecs = Q6_R_min_RR(n_vecs - prefetch_idx, PREFETCH_N_VECS);
        l2fetch(pv_in + prefetch_idx, VLEN, VLEN, prefetch_n_vecs, 0);
      }
    }

    HVX_Vector v_x       = *pv_in++;
    HVX_Vector v_negx    = hvx_vec_neg_f32(v_x);
    HVX_Vector v_exp2_arg = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_negx, v_log2e));
    HVX_Vector v_exp     = hvx_my_exp2_vsf(v_exp2_arg);
    HVX_Vector v_denom   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_one, v_exp));
    HVX_Vector v_sigmoid = hvx_vec_inverse_f32_guard(v_denom);
    *pv_out++            = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_x, v_sigmoid));
  }

  const int n_done = n_vecs * 32;
  for (int i = n_done; i < ne0; ++i) {
    dst[i] = src[i] / (1.0f + expf(-src[i]));
  }
}

typedef struct {
  float       *dst;
  const float *src;
  int          ne0;
} silu_parallel_ctx_t;

static void hvx_silu_f32_rows_fn(void *ctx, int row_begin, int row_end) {
  silu_parallel_ctx_t *p = (silu_parallel_ctx_t *) ctx;
  for (int j = row_begin; j < row_end; ++j) {
    float       *out_row = p->dst + j * p->ne0;
    const float *in_row  = p->src + j * p->ne0;
    hvx_silu_f32_inner(out_row, in_row, p->ne0);
  }
}

int hvx_silu_f32(float *restrict dst, const float *restrict src, int ne0, int ne1) {
  if (!dst || !src || !ne0 || !ne1) {
    return -1;
  }
  if (!is_aligned(dst, VLEN) || !is_aligned(src, VLEN)) {
    return -1;
  }

  silu_parallel_ctx_t ctx = {
    .dst = dst,
    .src = src,
    .ne0 = ne0,
  };

  // Small ne1 does not benefit much from worker dispatch.
  const int min_rows_per_task = 4;
  if (op_parallel_for_rows(ne1, min_rows_per_task, hvx_silu_f32_rows_fn, &ctx) == 0) {
    return 0;
  }

  for (int j = 0; j < ne1; ++j) {
    float       *out_row = dst + j * ne0;
    const float *in_row  = src + j * ne0;
    hvx_silu_f32_inner(out_row, in_row, ne0);
  }

  return 0;
}
