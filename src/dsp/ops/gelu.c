#include <math.h>

#include "dsp/hvx_internal.h"
#include "dsp/hvx_utils.h"

#define PREFETCH_SIZE   (8 * 1024)
#define PREFETCH_N_VECS (PREFETCH_SIZE / VLEN)

static inline void hvx_gelu_f32_inner(float *restrict dst, const float *restrict src, int ne0) {
  const int n_vecs = ne0 / 32;

  const HVX_Vector *pv_in   = (const HVX_Vector *) src;
  HVX_Vector       *pv_out  = (HVX_Vector *) dst;
  const HVX_Vector  v_half  = hvx_vec_splat_f32(0.5f);
  const HVX_Vector  v_one   = hvx_vec_splat_f32(1.0f);

  for (int i = 0; i < n_vecs; ++i) {
    if (i % PREFETCH_N_VECS == 0) {
      int prefetch_idx = i + PREFETCH_N_VECS;
      if (prefetch_idx < n_vecs) {
        int prefetch_n_vecs = Q6_R_min_RR(n_vecs - prefetch_idx, PREFETCH_N_VECS);
        l2fetch(pv_in + prefetch_idx, VLEN, VLEN, prefetch_n_vecs, 0);
      }
    }

    HVX_Vector v_x    = *pv_in++;
    HVX_Vector v_tanh = hvx_vec_tanh_gelu_f32(v_x);
    HVX_Vector v_sum  = hvx_vec_add_f32(v_one, v_tanh);
    HVX_Vector v_mul  = hvx_vec_mul_f32(v_x, v_sum);
    *pv_out++         = hvx_vec_mul_f32(v_half, v_mul);
  }

  const int n_done = n_vecs * 32;
  for (int i = n_done; i < ne0; ++i) {
    float x = src[i];
    dst[i]  = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
  }
}

int hvx_gelu_f32(float *restrict dst, const float *restrict src, int ne0, int ne1) {
  if (!dst || !src || !ne0 || !ne1) {
    return -1;
  }
  if (!is_aligned(dst, VLEN) || !is_aligned(src, VLEN)) {
    return -1;
  }

  for (int j = 0; j < ne1; ++j) {
    float       *out_row = dst + j * ne0;
    const float *in_row  = src + j * ne0;
    hvx_gelu_f32_inner(out_row, in_row, ne0);
  }

  return 0;
}
