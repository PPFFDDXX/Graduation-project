#include <math.h>

#include "dsp/hvx_internal.h"
#include "dsp/hvx_inverse.h"
#include "dsp/hvx_math.h"
#include "dsp/hvx_utils.h"

#define PREFETCH_SIZE   (8 * 1024)
#define PREFETCH_N_VECS (PREFETCH_SIZE / VLEN)

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
        l2fetch(pv_in_mul + prefetch_idx, VLEN, VLEN, prefetch_n_vecs, 0);
      }
    }

    HVX_Vector v_x        = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(*pv_in_src++, *pv_in_bias++));
    HVX_Vector v_negx     = hvx_vec_neg_f32(v_x);
    HVX_Vector v_exp2_arg = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_negx, v_log2e));
    HVX_Vector v_exp      = hvx_my_exp2_vsf(v_exp2_arg);
    HVX_Vector v_denom    = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_one, v_exp));
    HVX_Vector v_sigmoid  = hvx_vec_inverse_f32_guard(v_denom);
    HVX_Vector v_silu     = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_x, v_sigmoid));
    *pv_out++             = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_silu, *pv_in_mul++));
  }

  const int n_done = n_vecs * 32;
  for (int i = n_done; i < ne0; ++i) {
    float x = src[i] + bias[i];
    dst[i]  = (x / (1.0f + expf(-x))) * mul[i];
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

  for (int j = 0; j < ne1; ++j) {
    float       *out_row = dst + j * ne0;
    const float *in_row  = src + j * ne0;
    const float *mul_row = mul + j * ne0;
    hvx_bias_add_silu_mul_f32_inner(out_row, in_row, bias, mul_row, ne0);
  }

  return 0;
}

