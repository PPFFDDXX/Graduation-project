#include "dsp/hvx_internal.h"
#include "dsp/hvx_inverse.h"

#define PREFETCH_SIZE   (8 * 1024)
#define PREFETCH_N_VECS (PREFETCH_SIZE / VLEN)

static inline void hvx_div_f32_inner(float *restrict dst, const float *restrict src0, const float *restrict src1,
                                     int ne0) {
  const int n_vecs = ne0 / 32;

  const HVX_Vector *pv_in0 = (const HVX_Vector *) src0;
  const HVX_Vector *pv_in1 = (const HVX_Vector *) src1;
  HVX_Vector       *pv_out = (HVX_Vector *) dst;

  for (int i = 0; i < n_vecs; ++i) {
    if (i % PREFETCH_N_VECS == 0) {
      int prefetch_idx = i + PREFETCH_N_VECS;
      if (prefetch_idx < n_vecs) {
        int prefetch_n_vecs = Q6_R_min_RR(n_vecs - prefetch_idx, PREFETCH_N_VECS);
        l2fetch(pv_in0 + PREFETCH_N_VECS, VLEN, VLEN, prefetch_n_vecs, 0);
        l2fetch(pv_in1 + PREFETCH_N_VECS, VLEN, VLEN, prefetch_n_vecs, 0);
      }
    }

    HVX_Vector v_x0     = *pv_in0++;
    HVX_Vector v_inv_x1 = hvx_vec_inverse_f32_guard(*pv_in1++);
    *pv_out++           = Q6_Vqf32_vmpy_VsfVsf(v_x0, v_inv_x1);
  }

  const int n_done = n_vecs * 32;
  for (int i = n_done; i < ne0; ++i) {
    dst[i] = src0[i] / src1[i];
  }  // 处理未对齐内容
}

int hvx_div_f32(float *restrict dst, const float *restrict src0, const float *restrict src1, int ne0, int ne1) {
  if (!dst || !src0 || !src1 || !ne0 || !ne1) {
    return -1;
  }
  if (!is_aligned(dst, VLEN) || !is_aligned(src0, VLEN) || !is_aligned(src1, VLEN)) {
    return -1;
  }

  for (int j = 0; j < ne1; ++j) {
    float       *out_row  = dst + j * ne0;
    const float *src0_row = src0 + j * ne0;
    const float *src1_row = src1 + j * ne0;
    hvx_div_f32_inner(out_row, src0_row, src1_row, ne0);
  }

  return 0;
}
