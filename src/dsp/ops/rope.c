#include <math.h>
#include <stdlib.h>
#include "dsp/hvx_internal.h"

//生成 RoPE 的 逆频率表 inv_freq
static inline void rope_build_inv_freq(float *inv_freq, int half, float theta_base) {
  for (int pair_idx = 0; pair_idx < half; ++pair_idx) {
    inv_freq[pair_idx] = powf(theta_base, -((float) pair_idx / (float) half));
  }
}
//根据 inv_freq，生成每个通道对对应的：cos和sin
static inline void rope_build_step_trig(float *step_cos, float *step_sin, const float *inv_freq, int half) {
  for (int pair_idx = 0; pair_idx < half; ++pair_idx) {
    const float theta   = inv_freq[pair_idx];
    step_cos[pair_idx]  = cosf(theta);
    step_sin[pair_idx]  = sinf(theta);
  }
}
//维护并更新相位
// (cur_c, cur_s) <- (cur_c, cur_s) * exp(i * theta)
// cur_c' = cur_c * step_c - cur_s * step_s
// cur_s' = cur_s * step_c + cur_c * step_s
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

static inline void hvx_rope_f32_inner(float *restrict dst, const float *restrict src, int ne0, const float *row_cos,
                                      const float *row_sin) {
  int i = 0;
  for (; i + 31 < ne0; i += 32) {
    _Alignas(VLEN) float x_swap[32];
    _Alignas(VLEN) float c_dup[32];
    _Alignas(VLEN) float s_mix[32];

    // Process 16 RoPE pairs per HVX chunk (32 float lanes).
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

    HVX_Vector vy0_qf32 = Q6_Vqf32_vmpy_VsfVsf(vx, vc);
    HVX_Vector vy1_qf32 = Q6_Vqf32_vmpy_VsfVsf(vxs, vsmix);
    HVX_Vector vy_sf    = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(vy0_qf32, vy1_qf32));
    vmemu(dst + i)      = vy_sf;
  }

  for (; i + 1 < ne0; i += 2) {
    const int   pair_idx = i / 2;
    const float c        = row_cos[pair_idx];
    const float s        = row_sin[pair_idx];

    const float x0 = src[i + 0];
    const float x1 = src[i + 1];
    dst[i + 0]     = x0 * c - x1 * s;
    dst[i + 1]     = x0 * s + x1 * c;
  }

  if (ne0 & 1) {
    dst[ne0 - 1] = src[ne0 - 1];
  }
}

int hvx_rope_f32(float *restrict dst, const float *restrict src, int ne0, int ne1) {
  const float theta_base = 10000.0f;
  const int   half       = ne0 / 2;

  if (!dst || !src || !ne0 || !ne1) {
    return -1;
  }
  if (ne0 < 2) {
    return -1;
  }
  if (!is_aligned(dst, VLEN) || !is_aligned(src, VLEN)) {
    return -1;
  }

  float *inv_freq = NULL;
  float *step_cos = NULL;
  float *step_sin = NULL;
  float *cur_cos  = NULL;
  float *cur_sin  = NULL;

  if (posix_memalign((void **) &inv_freq, VLEN, (size_t) half * sizeof(float)) != 0 ||
      posix_memalign((void **) &step_cos, VLEN, (size_t) half * sizeof(float)) != 0 ||
      posix_memalign((void **) &step_sin, VLEN, (size_t) half * sizeof(float)) != 0 ||
      posix_memalign((void **) &cur_cos, VLEN, (size_t) half * sizeof(float)) != 0 ||
      posix_memalign((void **) &cur_sin, VLEN, (size_t) half * sizeof(float)) != 0) {
    free(inv_freq);
    free(step_cos);
    free(step_sin);
    free(cur_cos);
    free(cur_sin);
    return -1;
  }

  rope_build_inv_freq(inv_freq, half, theta_base);
  rope_build_step_trig(step_cos, step_sin, inv_freq, half);
  for (int i = 0; i < half; ++i) {
    cur_cos[i] = 1.0f;
    cur_sin[i] = 0.0f;
  }

  for (int j = 0; j < ne1; ++j) {
    float       *out_row = dst + j * ne0;
    const float *in_row  = src + j * ne0;
    hvx_rope_f32_inner(out_row, in_row, ne0, cur_cos, cur_sin);
    rope_update_phase_hvx(cur_cos, cur_sin, step_cos, step_sin, half);
  }

  free(inv_freq);
  free(step_cos);
  free(step_sin);
  free(cur_cos);
  free(cur_sin);
  return 0;
}
