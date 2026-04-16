#include <math.h>

#include "dsp/hvx_internal.h"
#include "dsp/hvx_math.h"
#include "dsp/hvx_utils.h"

#define PREFETCH_SIZE   (8 * 1024)
#define PREFETCH_N_VECS (PREFETCH_SIZE / VLEN)

static inline void hvx_softmax_f32_inner(float *restrict dst, const float *restrict src, int ne0) {
  const int n_vecs = ne0 / 32;
  const int n_done = n_vecs * 32;

  const HVX_Vector v_zero    = Q6_V_vzero();
  const HVX_Vector v_neg_inf = Q6_V_vsplat_R(0xFF800000);
  const HVX_Vector v_log2e   = hvx_vec_splat_f32(1.4426950408889634f);

  HVX_Vector       v_row_max = v_neg_inf;
  const HVX_Vector *pv_in    = (const HVX_Vector *) src;
  for (int i = 0; i < n_vecs; ++i) {
    if (i % PREFETCH_N_VECS == 0) {
      int prefetch_idx = i + PREFETCH_N_VECS;
      if (prefetch_idx < n_vecs) {
        int prefetch_n_vecs = Q6_R_min_RR(n_vecs - prefetch_idx, PREFETCH_N_VECS);
        l2fetch(pv_in + prefetch_idx, VLEN, VLEN, prefetch_n_vecs, 0);
      }
    }

    v_row_max = Q6_Vsf_vmax_VsfVsf(v_row_max, *pv_in++);//找出最大值
  }

  for (int s = 64; s >= 4; s >>= 1) {
    v_row_max = Q6_Vsf_vmax_VsfVsf(v_row_max, Q6_V_vlalign_VVR(v_row_max, v_neg_inf, s));//找出最大值
  }

  float tmp[32] __attribute__((aligned(VLEN)));
  vmem(tmp) = v_row_max;

  float row_max = tmp[31];
  for (int i = n_done; i < ne0; ++i) {
    row_max = fmaxf(row_max, src[i]);
  }

  // 计算exp(x - max)并求和
  const HVX_Vector v_row_max_sf = hvx_vec_splat_f32(row_max);
  HVX_Vector       v_sum_qf32   = v_zero;
  float            tail_sum     = 0.0f;

  pv_in                  = (const HVX_Vector *) src;
  HVX_Vector *pv_out_exp = (HVX_Vector *) dst;

  for (int i = 0; i < n_vecs; ++i) {
    if (i % PREFETCH_N_VECS == 0) {
      int prefetch_idx = i + PREFETCH_N_VECS;
      if (prefetch_idx < n_vecs) {
        int prefetch_n_vecs = Q6_R_min_RR(n_vecs - prefetch_idx, PREFETCH_N_VECS);
        l2fetch(pv_in + prefetch_idx, VLEN, VLEN, prefetch_n_vecs, 0);
      }
    }

    HVX_Vector v_shift = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(*pv_in++, v_row_max_sf));
    HVX_Vector v_exp2_arg = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_shift, v_log2e));
    HVX_Vector v_exp      = hvx_my_exp2_vsf(v_exp2_arg);

    *pv_out_exp++ = v_exp;
    v_sum_qf32    = Q6_Vqf32_vadd_Vqf32Vsf(v_sum_qf32, v_exp);
  }

  for (int i = n_done; i < ne0; ++i) {
    float v = expf(src[i] - row_max);
    dst[i]  = v;
    tail_sum += v;
  }

  for (int s = 64; s >= 4; s >>= 1) {
    v_sum_qf32 = Q6_Vqf32_vadd_Vqf32Vqf32(v_sum_qf32, Q6_V_vlalign_VVR(v_sum_qf32, v_zero, s));
  }

  vmem(tmp) = Q6_Vsf_equals_Vqf32(v_sum_qf32);
  const float sum = tmp[31] + tail_sum;

  float inv_sum = 0.0f;
  if (sum > 0.0f && isfinite(sum)) {
    inv_sum = 1.0f / sum;
  }

  //完成归一化
  const HVX_Vector v_inv_sum = hvx_vec_splat_f32(inv_sum);
  HVX_Vector       *pv_out   = (HVX_Vector *) dst;
  for (int i = 0; i < n_vecs; ++i) {
    *pv_out = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(*pv_out, v_inv_sum));
    ++pv_out;
  }

  for (int i = n_done; i < ne0; ++i) {
    dst[i] *= inv_sum;
  }
}

int hvx_softmax_f32(float *restrict dst, const float *restrict src, int ne0, int ne1) {
  if (!dst || !src || !ne0 || !ne1) {
    return -1;
  }
  if (!is_aligned(dst, VLEN) || !is_aligned(src, VLEN)) {
    return -1;
  }

  for (int j = 0; j < ne1; ++j) {
    float       *out_row = dst + j * ne0;
    const float *in_row  = src + j * ne0;
    hvx_softmax_f32_inner(out_row, in_row, ne0);
  }

  return 0;
}
