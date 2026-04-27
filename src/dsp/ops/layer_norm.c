#include <math.h>

#include "dsp/hvx_internal.h"
#include "dsp/op_parallel.h"

#define PREFETCH_SIZE   (8 * 1024)
#define PREFETCH_N_VECS (PREFETCH_SIZE / VLEN)

static inline void hvx_layer_norm_f32_inner(float *restrict dst, const float *restrict src, int ne0) {
  const float eps = 1e-5f;

  const int n_vecs   = ne0 / 32;
  const int n_done   = n_vecs * 32;

  const HVX_Vector *pv_in = (const HVX_Vector *) src;
  const HVX_Vector  v_zero = Q6_V_vzero();

  HVX_Vector v_sum_qf32   = Q6_V_vzero();
  HVX_Vector v_sumsq_qf32 = Q6_V_vzero();

  for (int i = 0; i < n_vecs; ++i) {
    if (i % PREFETCH_N_VECS == 0) {
      int prefetch_idx = i + PREFETCH_N_VECS;
      if (prefetch_idx < n_vecs) {
        int prefetch_n_vecs = Q6_R_min_RR(n_vecs - prefetch_idx, PREFETCH_N_VECS);
        l2fetch(pv_in + prefetch_idx, VLEN, VLEN, prefetch_n_vecs, 0);
      }
    }

    HVX_Vector v_x = *pv_in++;
    v_sum_qf32     = Q6_Vqf32_vadd_Vqf32Vsf(v_sum_qf32, v_x);
    v_sumsq_qf32   = Q6_Vqf32_vadd_Vqf32Vqf32(v_sumsq_qf32, Q6_Vqf32_vmpy_VsfVsf(v_x, v_x));
  }

  float tail_sum   = 0.0f;
  float tail_sumsq = 0.0f;
  for (int i = n_done; i < ne0; ++i) {
    float x = src[i];
    tail_sum += x;
    tail_sumsq += x * x;
  }

  for (int s = 64; s >= 4; s >>= 1) {
    v_sum_qf32   = Q6_Vqf32_vadd_Vqf32Vqf32(v_sum_qf32, Q6_V_vlalign_VVR(v_sum_qf32, v_zero, s));
    v_sumsq_qf32 = Q6_Vqf32_vadd_Vqf32Vqf32(v_sumsq_qf32, Q6_V_vlalign_VVR(v_sumsq_qf32, v_zero, s));
  }

  float tmp[32] __attribute__((aligned(VLEN)));

  vmem(tmp) = Q6_Vsf_equals_Vqf32(v_sum_qf32);
  float sum = tmp[31] + tail_sum;

  vmem(tmp) = Q6_Vsf_equals_Vqf32(v_sumsq_qf32);
  float sumsq = tmp[31] + tail_sumsq;

  float mean    = sum / ne0;
  float var     = sumsq / ne0 - mean * mean;
  float inv_std = 1.0f / sqrtf(var > 0.0f ? (var + eps) : eps);

  const HVX_Vector v_mean    = Q6_V_vsplat_R(*(int32_t *) &mean);
  const HVX_Vector v_inv_std = Q6_V_vsplat_R(*(int32_t *) &inv_std);

  pv_in                  = (const HVX_Vector *) src;
  HVX_Vector *pv_out     = (HVX_Vector *) dst;
  for (int i = 0; i < n_vecs; ++i) {
    HVX_Vector v_x      = *pv_in++;
    HVX_Vector v_center = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v_x, v_mean));
    *pv_out++           = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_center, v_inv_std));
  }

  for (int i = n_done; i < ne0; ++i) {
    dst[i] = (src[i] - mean) * inv_std;
  }
}

typedef struct {
  float       *dst;
  const float *src;
  int          ne0;
} layer_norm_parallel_ctx_t;

static void hvx_layer_norm_f32_rows_fn(void *ctx, int row_begin, int row_end) {
  layer_norm_parallel_ctx_t *p = (layer_norm_parallel_ctx_t *) ctx;
  for (int j = row_begin; j < row_end; ++j) {
    float       *out_row = p->dst + j * p->ne0;
    const float *in_row  = p->src + j * p->ne0;
    hvx_layer_norm_f32_inner(out_row, in_row, p->ne0);
  }
}

int hvx_layer_norm_f32(float *restrict dst, const float *restrict src, int ne0, int ne1) {
  if (!dst || !src || !ne0 || !ne1) {
    return -1;
  }
  if (!is_aligned(dst, VLEN) || !is_aligned(src, VLEN)) {
    return -1;
  }

  layer_norm_parallel_ctx_t ctx = {
    .dst = dst,
    .src = src,
    .ne0 = ne0,
  };

  // LayerNorm per-row work is heavier than simple unary ops.
  const int min_rows_per_task = 2;
  if (op_parallel_for_rows(ne1, min_rows_per_task, hvx_layer_norm_f32_rows_fn, &ctx) == 0) {
    return 0;
  }

  for (int j = 0; j < ne1; ++j) {
    float       *out_row = dst + j * ne0;
    const float *in_row  = src + j * ne0;
    hvx_layer_norm_f32_inner(out_row, in_row, ne0);
  }

  return 0;
}
