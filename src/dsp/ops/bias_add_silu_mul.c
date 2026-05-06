#include <math.h>

#include "dsp/hvx_internal.h"
#include "dsp/hvx_inverse.h"
#include "dsp/hvx_math.h"
#include "dsp/hvx_utils.h"
#include "dsp/op_parallel.h"

#define PREFETCH_SIZE   (8 * 1024)
#define PREFETCH_N_VECS (PREFETCH_SIZE / VLEN)

static inline HVX_Vector hvx_silu_from_x(HVX_Vector v_x, HVX_Vector v_one, HVX_Vector v_log2e) {
  HVX_Vector v_negx     = hvx_vec_neg_f32(v_x);
  HVX_Vector v_exp2_arg = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_negx, v_log2e));
  HVX_Vector v_exp      = hvx_my_exp2_vsf(v_exp2_arg);
  HVX_Vector v_denom    = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_one, v_exp));
  HVX_Vector v_sigmoid  = hvx_vec_inverse_f32_guard(v_denom);
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_x, v_sigmoid));
}

static inline void hvx_bias_add_silu_mul_f32_inner(float *restrict dst, const float *restrict src,
                                                    const float *restrict bias, const float *restrict mul, int ne0) {
  const int n_vecs = ne0 / 32;

  const HVX_Vector *pv_in_src  = (const HVX_Vector *) src;
  const HVX_Vector *pv_in_bias = (const HVX_Vector *) bias;
  const HVX_Vector *pv_in_mul  = (const HVX_Vector *) mul;
  HVX_Vector       *pv_out     = (HVX_Vector *) dst;

  const HVX_Vector v_one   = hvx_vec_splat_f32(1.0f);
  const HVX_Vector v_log2e = hvx_vec_splat_f32(1.4426950408889634f);

  for (int i = 0; i < n_vecs; ++i) {
    if (i % PREFETCH_N_VECS == 0) {
      int prefetch_idx = i + PREFETCH_N_VECS;
      if (prefetch_idx < n_vecs) {
        int prefetch_n_vecs = Q6_R_min_RR(n_vecs - prefetch_idx, PREFETCH_N_VECS);
        l2fetch(pv_in_src + prefetch_idx, VLEN, VLEN, prefetch_n_vecs, 0);
        l2fetch(pv_in_bias + prefetch_idx, VLEN, VLEN, prefetch_n_vecs, 0);
        l2fetch(pv_in_mul + prefetch_idx, VLEN, VLEN, prefetch_n_vecs, 0);
      }
    }

    HVX_Vector v_x    = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(*pv_in_src++, *pv_in_bias++));
    HVX_Vector v_silu = hvx_silu_from_x(v_x, v_one, v_log2e);
    *pv_out++         = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_silu, *pv_in_mul++));
  }

  const int n_done = n_vecs * 32;
  for (int i = n_done; i < ne0; ++i) {
    float x = src[i] + bias[i];
    dst[i]  = (x / (1.0f + expf(-x))) * mul[i];
  }
}

static inline void hvx_bias_add_silu_mul_f32_inner_2rows(float *restrict dst0, float *restrict dst1,
                                                          const float *restrict src0, const float *restrict src1,
                                                          const float *restrict bias, const float *restrict mul0,
                                                          const float *restrict mul1, int ne0) {
  const int n_vecs = ne0 / 32;

  const HVX_Vector *pv_src0 = (const HVX_Vector *) src0;
  const HVX_Vector *pv_src1 = (const HVX_Vector *) src1;
  const HVX_Vector *pv_bias = (const HVX_Vector *) bias;
  const HVX_Vector *pv_mul0 = (const HVX_Vector *) mul0;
  const HVX_Vector *pv_mul1 = (const HVX_Vector *) mul1;
  HVX_Vector       *pv_dst0 = (HVX_Vector *) dst0;
  HVX_Vector       *pv_dst1 = (HVX_Vector *) dst1;

  const HVX_Vector v_one   = hvx_vec_splat_f32(1.0f);
  const HVX_Vector v_log2e = hvx_vec_splat_f32(1.4426950408889634f);

  for (int i = 0; i < n_vecs; ++i) {
    if (i % PREFETCH_N_VECS == 0) {
      int prefetch_idx = i + PREFETCH_N_VECS;
      if (prefetch_idx < n_vecs) {
        int prefetch_n_vecs = Q6_R_min_RR(n_vecs - prefetch_idx, PREFETCH_N_VECS);
        l2fetch(pv_src0 + prefetch_idx, VLEN, VLEN, prefetch_n_vecs, 0);
        l2fetch(pv_src1 + prefetch_idx, VLEN, VLEN, prefetch_n_vecs, 0);
        l2fetch(pv_bias + prefetch_idx, VLEN, VLEN, prefetch_n_vecs, 0);
        l2fetch(pv_mul0 + prefetch_idx, VLEN, VLEN, prefetch_n_vecs, 0);
        l2fetch(pv_mul1 + prefetch_idx, VLEN, VLEN, prefetch_n_vecs, 0);
      }
    }

    HVX_Vector v_b  = *pv_bias++;
    HVX_Vector v_x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(*pv_src0++, v_b));
    HVX_Vector v_x1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(*pv_src1++, v_b));

    HVX_Vector v_silu0 = hvx_silu_from_x(v_x0, v_one, v_log2e);
    HVX_Vector v_silu1 = hvx_silu_from_x(v_x1, v_one, v_log2e);

    *pv_dst0++ = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_silu0, *pv_mul0++));
    *pv_dst1++ = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_silu1, *pv_mul1++));
  }

  const int n_done = n_vecs * 32;
  for (int i = n_done; i < ne0; ++i) {
    float x0 = src0[i] + bias[i];
    float x1 = src1[i] + bias[i];
    dst0[i]  = (x0 / (1.0f + expf(-x0))) * mul0[i];
    dst1[i]  = (x1 / (1.0f + expf(-x1))) * mul1[i];
  }
}

static inline void hvx_bias_add_silu_mul_f32_inner_4rows(
    float *restrict dst0, float *restrict dst1, float *restrict dst2, float *restrict dst3, const float *restrict src0,
    const float *restrict src1, const float *restrict src2, const float *restrict src3, const float *restrict bias,
    const float *restrict mul0, const float *restrict mul1, const float *restrict mul2, const float *restrict mul3,
    int ne0) {
  const int n_vecs = ne0 / 32;

  const HVX_Vector *pv_src0 = (const HVX_Vector *) src0;
  const HVX_Vector *pv_src1 = (const HVX_Vector *) src1;
  const HVX_Vector *pv_src2 = (const HVX_Vector *) src2;
  const HVX_Vector *pv_src3 = (const HVX_Vector *) src3;
  const HVX_Vector *pv_bias = (const HVX_Vector *) bias;
  const HVX_Vector *pv_mul0 = (const HVX_Vector *) mul0;
  const HVX_Vector *pv_mul1 = (const HVX_Vector *) mul1;
  const HVX_Vector *pv_mul2 = (const HVX_Vector *) mul2;
  const HVX_Vector *pv_mul3 = (const HVX_Vector *) mul3;
  HVX_Vector       *pv_dst0 = (HVX_Vector *) dst0;
  HVX_Vector       *pv_dst1 = (HVX_Vector *) dst1;
  HVX_Vector       *pv_dst2 = (HVX_Vector *) dst2;
  HVX_Vector       *pv_dst3 = (HVX_Vector *) dst3;

  const HVX_Vector v_one   = hvx_vec_splat_f32(1.0f);
  const HVX_Vector v_log2e = hvx_vec_splat_f32(1.4426950408889634f);

  for (int i = 0; i < n_vecs; ++i) {
    if (i % PREFETCH_N_VECS == 0) {
      int prefetch_idx = i + PREFETCH_N_VECS;
      if (prefetch_idx < n_vecs) {
        int prefetch_n_vecs = Q6_R_min_RR(n_vecs - prefetch_idx, PREFETCH_N_VECS);
        l2fetch(pv_src0 + prefetch_idx, VLEN, VLEN, prefetch_n_vecs, 0);
        l2fetch(pv_src1 + prefetch_idx, VLEN, VLEN, prefetch_n_vecs, 0);
        l2fetch(pv_src2 + prefetch_idx, VLEN, VLEN, prefetch_n_vecs, 0);
        l2fetch(pv_src3 + prefetch_idx, VLEN, VLEN, prefetch_n_vecs, 0);
        l2fetch(pv_bias + prefetch_idx, VLEN, VLEN, prefetch_n_vecs, 0);
        l2fetch(pv_mul0 + prefetch_idx, VLEN, VLEN, prefetch_n_vecs, 0);
        l2fetch(pv_mul1 + prefetch_idx, VLEN, VLEN, prefetch_n_vecs, 0);
        l2fetch(pv_mul2 + prefetch_idx, VLEN, VLEN, prefetch_n_vecs, 0);
        l2fetch(pv_mul3 + prefetch_idx, VLEN, VLEN, prefetch_n_vecs, 0);
      }
    }

    HVX_Vector v_b  = *pv_bias++;
    HVX_Vector v_x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(*pv_src0++, v_b));
    HVX_Vector v_x1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(*pv_src1++, v_b));
    HVX_Vector v_x2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(*pv_src2++, v_b));
    HVX_Vector v_x3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(*pv_src3++, v_b));

    HVX_Vector v_silu0 = hvx_silu_from_x(v_x0, v_one, v_log2e);
    HVX_Vector v_silu1 = hvx_silu_from_x(v_x1, v_one, v_log2e);
    HVX_Vector v_silu2 = hvx_silu_from_x(v_x2, v_one, v_log2e);
    HVX_Vector v_silu3 = hvx_silu_from_x(v_x3, v_one, v_log2e);

    *pv_dst0++ = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_silu0, *pv_mul0++));
    *pv_dst1++ = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_silu1, *pv_mul1++));
    *pv_dst2++ = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_silu2, *pv_mul2++));
    *pv_dst3++ = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_silu3, *pv_mul3++));
  }

  const int n_done = n_vecs * 32;
  for (int i = n_done; i < ne0; ++i) {
    float x0 = src0[i] + bias[i];
    float x1 = src1[i] + bias[i];
    float x2 = src2[i] + bias[i];
    float x3 = src3[i] + bias[i];
    dst0[i]  = (x0 / (1.0f + expf(-x0))) * mul0[i];
    dst1[i]  = (x1 / (1.0f + expf(-x1))) * mul1[i];
    dst2[i]  = (x2 / (1.0f + expf(-x2))) * mul2[i];
    dst3[i]  = (x3 / (1.0f + expf(-x3))) * mul3[i];
  }
}

typedef struct {
  float       *dst;
  const float *src;
  const float *bias;
  const float *mul;
  int          ne0;
} bias_add_silu_mul_parallel_ctx_t;

static void hvx_bias_add_silu_mul_f32_rows_fn(void *ctx, int row_begin, int row_end) {
  bias_add_silu_mul_parallel_ctx_t *p = (bias_add_silu_mul_parallel_ctx_t *) ctx;
  const int prefer_4rows = (p->ne0 >= 1024);
  int j = row_begin;
  if (prefer_4rows) {
    for (; j + 3 < row_end; j += 4) {
      float       *out_row0 = p->dst + (j + 0) * p->ne0;
      float       *out_row1 = p->dst + (j + 1) * p->ne0;
      float       *out_row2 = p->dst + (j + 2) * p->ne0;
      float       *out_row3 = p->dst + (j + 3) * p->ne0;
      const float *in_row0  = p->src + (j + 0) * p->ne0;
      const float *in_row1  = p->src + (j + 1) * p->ne0;
      const float *in_row2  = p->src + (j + 2) * p->ne0;
      const float *in_row3  = p->src + (j + 3) * p->ne0;
      const float *mul_row0 = p->mul + (j + 0) * p->ne0;
      const float *mul_row1 = p->mul + (j + 1) * p->ne0;
      const float *mul_row2 = p->mul + (j + 2) * p->ne0;
      const float *mul_row3 = p->mul + (j + 3) * p->ne0;
      hvx_bias_add_silu_mul_f32_inner_4rows(out_row0, out_row1, out_row2, out_row3, in_row0, in_row1, in_row2, in_row3,
                                             p->bias, mul_row0, mul_row1, mul_row2, mul_row3, p->ne0);
    }
  }

  for (; j + 1 < row_end; j += 2) {
    float       *out_row0 = p->dst + (j + 0) * p->ne0;
    float       *out_row1 = p->dst + (j + 1) * p->ne0;
    const float *in_row0  = p->src + (j + 0) * p->ne0;
    const float *in_row1  = p->src + (j + 1) * p->ne0;
    const float *mul_row0 = p->mul + (j + 0) * p->ne0;
    const float *mul_row1 = p->mul + (j + 1) * p->ne0;
    hvx_bias_add_silu_mul_f32_inner_2rows(out_row0, out_row1, in_row0, in_row1, p->bias, mul_row0, mul_row1, p->ne0);
  }

  for (; j < row_end; ++j) {
    float       *out_row = p->dst + j * p->ne0;
    const float *in_row  = p->src + j * p->ne0;
    const float *mul_row = p->mul + j * p->ne0;
    hvx_bias_add_silu_mul_f32_inner(out_row, in_row, p->bias, mul_row, p->ne0);
  }
}

int hvx_bias_add_silu_mul_f32(float *restrict dst, const float *restrict src, const float *restrict bias,
                              const float *restrict mul, int ne0, int ne1) {
  if (!dst || !src || !bias || !mul || !ne0 || !ne1) {
    return -1;
  }
  if (!is_aligned(dst, VLEN) || !is_aligned(src, VLEN) || !is_aligned(bias, VLEN) || !is_aligned(mul, VLEN)) {
    return -1;
  }

  bias_add_silu_mul_parallel_ctx_t ctx = {
    .dst  = dst,
    .src  = src,
    .bias = bias,
    .mul  = mul,
    .ne0  = ne0,
  };

  // Same granularity choice as silu: avoid dispatch overhead on tiny ne1.
  const int min_rows_per_task = 4;
  if (op_parallel_for_rows(ne1, min_rows_per_task, hvx_bias_add_silu_mul_f32_rows_fn, &ctx) == 0) {
    return 0;
  }

  for (int j = 0; j < ne1; ++j) {
    float       *out_row = dst + j * ne0;
    const float *in_row  = src + j * ne0;
    const float *mul_row = mul + j * ne0;
    hvx_bias_add_silu_mul_f32_inner(out_row, in_row, bias, mul_row, ne0);
  }

  return 0;
}
