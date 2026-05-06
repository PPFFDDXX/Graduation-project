#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dsp/hvx_internal.h"
#include "dsp/hvx_utils.h"
#include "dsp/op_parallel.h"

#define PREFETCH_SIZE   (8 * 1024)
#define PREFETCH_N_VECS (PREFETCH_SIZE / VLEN)

static inline void rope_build_step_trig(float *step_cos, float *step_sin, int half, float theta_base) {
  for (int pair_idx = 0; pair_idx < half; ++pair_idx) {
    const float theta   = powf(theta_base, -((float) pair_idx / (float) half));
    step_cos[pair_idx]  = cosf(theta);
    step_sin[pair_idx]  = sinf(theta);
  }
}

static inline void rope_update_phase_hvx(float *restrict cur_cos, float *restrict cur_sin, const float *restrict step_cos,
                                         const float *restrict step_sin, int half) {
  int i = 0;
  for (; i + 31 < half; i += 32) {
    HVX_Vector v_cur_c  = vmemu(cur_cos + i);
    HVX_Vector v_cur_s  = vmemu(cur_sin + i);
    HVX_Vector v_step_c = vmemu(step_cos + i);
    HVX_Vector v_step_s = vmemu(step_sin + i);

    HVX_Vector v_cc = Q6_Vqf32_vmpy_VsfVsf(v_cur_c, v_step_c);
    HVX_Vector v_ss = Q6_Vqf32_vmpy_VsfVsf(v_cur_s, v_step_s);
    HVX_Vector v_sc = Q6_Vqf32_vmpy_VsfVsf(v_cur_s, v_step_c);
    HVX_Vector v_cs = Q6_Vqf32_vmpy_VsfVsf(v_cur_c, v_step_s);

    HVX_Vector v_next_c = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(v_cc, v_ss));
    HVX_Vector v_next_s = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v_sc, v_cs));

    vmemu(cur_cos + i) = v_next_c;
    vmemu(cur_sin + i) = v_next_s;
  }

  for (; i < half; ++i) {
    const float c = cur_cos[i];
    const float s = cur_sin[i];
    cur_cos[i]    = c * step_cos[i] - s * step_sin[i];
    cur_sin[i]    = s * step_cos[i] + c * step_sin[i];
  }
}

static inline void hvx_rope_scale_add_f32_inner(float *restrict dst, const float *restrict src, const float *restrict add,
                                                 int ne0, const float *row_cos, const float *row_sin, float scale) {
  const HVX_Vector v_scale = hvx_vec_splat_f32(scale);
  int              i       = 0;
  const int        n_vecs  = ne0 / 32;

  for (int vi = 0; vi < n_vecs; ++vi, i += 32) {
    if (vi % PREFETCH_N_VECS == 0) {
      int prefetch_idx = vi + PREFETCH_N_VECS;
      if (prefetch_idx < n_vecs) {
        int prefetch_n_vecs = Q6_R_min_RR(n_vecs - prefetch_idx, PREFETCH_N_VECS);
        l2fetch((const HVX_Vector *) (src + prefetch_idx * 32), VLEN, VLEN, prefetch_n_vecs, 0);
        l2fetch((const HVX_Vector *) (add + prefetch_idx * 32), VLEN, VLEN, prefetch_n_vecs, 0);
      }
    }

    _Alignas(VLEN) float x_swap[32];
    _Alignas(VLEN) float c_dup[32];
    _Alignas(VLEN) float s_mix[32];

    for (int t = 0; t < 32; ++t) {
      const int idx      = i + t;
      const int pair_idx = idx >> 1;
      const int odd      = idx & 1;
      const float c      = row_cos[pair_idx];
      const float s      = row_sin[pair_idx];

      x_swap[t] = src[idx ^ 1];
      c_dup[t]  = c;
      s_mix[t]  = odd ? s : -s;
    }

    const HVX_Vector vx    = vmemu(src + i);
    const HVX_Vector vxs   = vmem(x_swap);
    const HVX_Vector vc    = vmem(c_dup);
    const HVX_Vector vsmix = vmem(s_mix);
    const HVX_Vector vadd  = vmemu(add + i);

    HVX_Vector vy0_qf32    = Q6_Vqf32_vmpy_VsfVsf(vx, vc);
    HVX_Vector vy1_qf32    = Q6_Vqf32_vmpy_VsfVsf(vxs, vsmix);
    HVX_Vector vy_qf32     = Q6_Vqf32_vadd_Vqf32Vqf32(vy0_qf32, vy1_qf32);
    HVX_Vector vys_qf32    = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(vy_qf32), v_scale);
    HVX_Vector vout_sf     = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(vys_qf32), vadd));
    vmemu(dst + i)         = vout_sf;
  }

  for (; i + 1 < ne0; i += 2) {
    const int   pair_idx = i / 2;
    const float c        = row_cos[pair_idx];
    const float s        = row_sin[pair_idx];

    const float x0 = src[i + 0];
    const float x1 = src[i + 1];
    dst[i + 0]     = (x0 * c - x1 * s) * scale + add[i + 0];
    dst[i + 1]     = (x0 * s + x1 * c) * scale + add[i + 1];
  }

  if (ne0 & 1) {
    dst[ne0 - 1] = src[ne0 - 1] * scale + add[ne0 - 1];
  }
}

typedef struct {
  float       *dst;
  const float *src;
  const float *add;
  const float *cos_table;
  const float *sin_table;
  int          ne0;
  int          half;
  float        scale;
} rope_scale_add_parallel_ctx_t;

static void hvx_rope_scale_add_f32_rows_fn(void *ctx, int row_begin, int row_end) {
  rope_scale_add_parallel_ctx_t *p = (rope_scale_add_parallel_ctx_t *) ctx;

  for (int j = row_begin; j < row_end; ++j) {
    float       *out_row  = p->dst + j * p->ne0;
    const float *in_row   = p->src + j * p->ne0;
    const float *add_row  = p->add + j * p->ne0;
    const float *row_cos  = p->cos_table + (size_t) j * p->half;
    const float *row_sin  = p->sin_table + (size_t) j * p->half;
    hvx_rope_scale_add_f32_inner(out_row, in_row, add_row, p->ne0, row_cos, row_sin, p->scale);
  }
}

int hvx_rope_scale_add_f32(float *restrict dst, const float *restrict src, const float *restrict add, int ne0, int ne1,
                           float scale) {
  const float theta_base = 10000.0f;
  const int   half       = ne0 / 2;

  if (!dst || !src || !add || !ne0 || !ne1) {
    return -1;
  }
  if (ne0 < 2) {
    return -1;
  }
  if (!is_aligned(dst, VLEN) || !is_aligned(src, VLEN) || !is_aligned(add, VLEN)) {
    return -1;
  }

  float *step_cos = NULL;
  float *step_sin = NULL;
  float *cur_cos  = NULL;
  float *cur_sin  = NULL;
  float *cos_table = NULL;
  float *sin_table = NULL;

  if (posix_memalign((void **) &step_cos, VLEN, (size_t) half * sizeof(float)) != 0 ||
      posix_memalign((void **) &step_sin, VLEN, (size_t) half * sizeof(float)) != 0 ||
      posix_memalign((void **) &cur_cos, VLEN, (size_t) half * sizeof(float)) != 0 ||
      posix_memalign((void **) &cur_sin, VLEN, (size_t) half * sizeof(float)) != 0) {
    free(step_cos);
    free(step_sin);
    free(cur_cos);
    free(cur_sin);
    return -1;
  }

  rope_build_step_trig(step_cos, step_sin, half, theta_base);
  for (int i = 0; i < half; ++i) {
    cur_cos[i] = 1.0f;
    cur_sin[i] = 0.0f;
  }

  const size_t table_elems = (size_t) half * (size_t) ne1;
  if (posix_memalign((void **) &cos_table, VLEN, table_elems * sizeof(float)) == 0 &&
      posix_memalign((void **) &sin_table, VLEN, table_elems * sizeof(float)) == 0) {
    for (int j = 0; j < ne1; ++j) {
      float *row_cos = cos_table + (size_t) j * half;
      float *row_sin = sin_table + (size_t) j * half;
      memcpy(row_cos, cur_cos, (size_t) half * sizeof(float));
      memcpy(row_sin, cur_sin, (size_t) half * sizeof(float));
      rope_update_phase_hvx(cur_cos, cur_sin, step_cos, step_sin, half);
    }
  } else {
    free(cos_table);
    free(sin_table);
    cos_table = NULL;
    sin_table = NULL;
  }

  // Fallback if table allocation failed: single-thread recurrence (old behavior).
  if (!cos_table || !sin_table) {
    for (int i = 0; i < half; ++i) {
      cur_cos[i] = 1.0f;
      cur_sin[i] = 0.0f;
    }
    for (int j = 0; j < ne1; ++j) {
      float       *out_row = dst + j * ne0;
      const float *in_row  = src + j * ne0;
      const float *add_row = add + j * ne0;
      hvx_rope_scale_add_f32_inner(out_row, in_row, add_row, ne0, cur_cos, cur_sin, scale);
      rope_update_phase_hvx(cur_cos, cur_sin, step_cos, step_sin, half);
    }

    free(step_cos);
    free(step_sin);
    free(cur_cos);
    free(cur_sin);
    return 0;
  }

  rope_scale_add_parallel_ctx_t ctx = {
    .dst       = dst,
    .src       = src,
    .add       = add,
    .cos_table = cos_table,
    .sin_table = sin_table,
    .ne0       = ne0,
    .half      = half,
    .scale     = scale,
  };

  const int min_rows_per_task = 16;
  (void) op_parallel_for_rows(ne1, min_rows_per_task, hvx_rope_scale_add_f32_rows_fn, &ctx);

  free(step_cos);
  free(step_sin);
  free(cur_cos);
  free(cur_sin);
  free(cos_table);
  free(sin_table);
  return 0;
}
