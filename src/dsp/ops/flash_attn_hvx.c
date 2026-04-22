#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "dsp/hvx_convert.h"
#include "dsp/hvx_internal.h"
#include "dsp/utils.h"

#define FLASH_ATTN_HVX_BLOCK_SIZE 64

static inline HVX_VectorPair hvx_vec_mpyacc_f32_f16(HVX_VectorPair acc, HVX_Vector x_hf, HVX_Vector y_hf) {
  HVX_VectorPair m = Q6_Wqf32_vmpy_VhfVhf(x_hf, y_hf);
  HVX_Vector     a0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(m), Q6_V_lo_W(acc)));
  HVX_Vector     a1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(m), Q6_V_hi_W(acc)));
  return Q6_W_vcombine_VV(a1, a0);
}

static inline HVX_Vector hvx_vec_f32_to_f16(HVX_Vector v0_sf, HVX_Vector v1_sf) {
  HVX_Vector q0_qf32 = Q6_Vqf32_vadd_VsfVsf(v0_sf, Q6_V_vzero());
  HVX_Vector q1_qf32 = Q6_Vqf32_vadd_VsfVsf(v1_sf, Q6_V_vzero());
  HVX_Vector v_hf    = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(q1_qf32, q0_qf32));
  return Q6_Vh_vdeal_Vh(v_hf);
}

static inline void hvx_copy_f16_f32_row(__fp16 *restrict dst_hf, const float *restrict src_sf, int d) {
  int i = 0;
  for (; i + 63 < d; i += 64) {
    HVX_Vector v0_sf = vmemu(src_sf + i);
    HVX_Vector v1_sf = vmemu(src_sf + i + 32);
    vmemu(dst_hf + i) = hvx_vec_f32_to_f16(v0_sf, v1_sf);
  }
  for (; i < d; ++i) {
    dst_hf[i] = (__fp16) src_sf[i];
  }
}

static inline float hvx_dot_f16_f16_aa(const __fp16 *restrict x_hf, const __fp16 *restrict y_hf, int n, float s) {
  const HVX_Vector *restrict vx = (const HVX_Vector *) x_hf;
  const HVX_Vector *restrict vy = (const HVX_Vector *) y_hf;

  const int nvec = n / 64;
  const int nloe = n % 64;

  HVX_VectorPair rsum_p = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());

  int i = 0;
  for (; i < nvec; ++i) {
    rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, vx[i], vy[i]);
  }

  if (nloe) {
    HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
    HVX_Vector     x     = Q6_V_vand_QV(bmask, vx[i]);
    HVX_Vector     y     = Q6_V_vand_QV(bmask, vy[i]);
    rsum_p               = hvx_vec_mpyacc_f32_f16(rsum_p, x, y);
  }

  HVX_Vector rsum_sf = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(rsum_p), Q6_V_hi_W(rsum_p)));

  // Horizontal reduce 32-lane float vector.
  for (int shift = 64; shift >= 4; shift >>= 1) {
    rsum_sf = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(rsum_sf, Q6_V_vlalign_VVR(rsum_sf, Q6_V_vzero(), shift)));
  }
  _Alignas(VLEN) float tmp[32];
  vmem(tmp) = rsum_sf;
  return tmp[31] * s;
}

static inline void hvx_mad_f32_f16_aa(float *restrict y_sf, const __fp16 *restrict x_hf, float s, int n) {
  const HVX_Vector *restrict vx = (const HVX_Vector *) x_hf;
  HVX_VectorPair   *restrict vy = (HVX_VectorPair *) y_sf;
  HVX_Vector       *restrict vyv = (HVX_Vector *) y_sf;

  const int nvec = n / 64;
  int       nloe = n % 64;

  union {
    __fp16   f;
    uint16_t u;
  } su = { .f = (__fp16) s };
  const HVX_Vector S = Q6_Vh_vsplat_R(su.u);

  int i = 0;
  for (; i < nvec; ++i) {
    vy[i] = hvx_vec_mpyacc_f32_f16(vy[i], Q6_Vh_vshuff_Vh(vx[i]), S);
  }

  if (nloe) {
    HVX_VectorPair acc = vy[i];
    acc                = hvx_vec_mpyacc_f32_f16(acc, Q6_Vh_vshuff_Vh(vx[i]), S);

    HVX_Vector lo = Q6_V_lo_W(acc);
    HVX_Vector hi = Q6_V_hi_W(acc);
    i             = 2 * i;

    if (nloe >= 32) {
      vyv[i++] = lo;
      nloe -= 32;
      lo = hi;
    }
    if (nloe) {
      _Alignas(VLEN) float tmp[32];
      vmem(tmp) = lo;
      for (int t = 0; t < nloe; ++t) {
        y_sf[i * 32 + t] = tmp[t];
      }
    }
  }
}

static inline void hvx_vec_scale_f32(float *restrict y, int d, float s) {
  union {
    float    f;
    uint32_t u;
  } u = { .f = s };
  const HVX_Vector v_s_sf = Q6_V_vsplat_R((int) u.u);

  int i = 0;
  for (; i + 31 < d; i += 32) {
    HVX_Vector v_y_sf = vmemu(y + i);
    v_y_sf            = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_y_sf, v_s_sf));
    vmemu(y + i)      = v_y_sf;
  }

  for (; i < d; ++i) {
    y[i] *= s;
  }
}

static inline void hvx_vec_zero_f32(float *restrict y, int d) {
  const HVX_Vector v_zero = Q6_V_vzero();

  int i = 0;
  for (; i + 31 < d; i += 32) {
    vmemu(y + i) = v_zero;
  }
  for (; i < d; ++i) {
    y[i] = 0.0f;
  }
}

int simple_flash_attn_hvx_f32(float *restrict O, const float *restrict Q, const __fp16 *restrict K,
                              const __fp16 *restrict V, const __fp16 *restrict mask, int qo_len, int kv_len,
                              int n_heads, int n_kv_heads, int head_dim) {
  if (!O || !Q || !K || !V) {
    return -1;
  }
  if (qo_len <= 0 || kv_len <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0) {
    return -1;
  }
  if ((n_heads % n_kv_heads) != 0) {
    return -1;
  }

  const int   gqa_factor = n_heads / n_kv_heads;
  const float qk_scale    = 1.0f / sqrtf((float) head_dim) * 1.4426950408889634f;
  const size_t kv_pad_len = align_up((size_t) kv_len, 64);
  const int head_dim_pad  = (int) align_up((size_t) head_dim, 64);

  __fp16 *q_row_hf = NULL;
  if (posix_memalign((void **) &q_row_hf, VLEN, (size_t) head_dim_pad * sizeof(__fp16)) != 0) {
    return -1;
  }

  for (int h = 0; h < n_heads; ++h) {
    const int h_kv = h / gqa_factor;

    for (int iq = 0; iq < qo_len; ++iq) {
      float *restrict o_row       = O + ((size_t) iq * n_heads + h) * head_dim;
      const float *restrict q_row = Q + ((size_t) iq * n_heads + h) * head_dim;
      hvx_copy_f16_f32_row(q_row_hf, q_row, head_dim);

      hvx_vec_zero_f32(o_row, head_dim);

      float m = -INFINITY;
      float l = 0.0f;

      for (int jb = 0; jb < kv_len; jb += FLASH_ATTN_HVX_BLOCK_SIZE) {
        const int block_size = Q6_R_min_RR(FLASH_ATTN_HVX_BLOCK_SIZE, kv_len - jb);

        const __fp16 *k_base = K + ((size_t) jb * n_kv_heads + h_kv) * head_dim;
        const __fp16 *v_base = V + ((size_t) jb * n_kv_heads + h_kv) * head_dim;

        l2fetch(k_base, n_kv_heads * head_dim * (int) sizeof(__fp16), head_dim * (int) sizeof(__fp16), block_size,
                0);
        l2fetch(v_base, n_kv_heads * head_dim * (int) sizeof(__fp16), head_dim * (int) sizeof(__fp16), block_size,
                0);

        float scores[FLASH_ATTN_HVX_BLOCK_SIZE];
        float block_max = -INFINITY;

        for (int t = 0; t < block_size; ++t) {
          const __fp16 *k_row = k_base + (size_t) t * n_kv_heads * head_dim;
          float score          = hvx_dot_f16_f16_aa(q_row_hf, k_row, head_dim, qk_scale);

          if (mask && (float) mask[(size_t) iq * kv_pad_len + (jb + t)] < -128.0f) {
            score = -INFINITY;
          }

          scores[t] = score;
          if (score > block_max) {
            block_max = score;
          }
        }

        const float m_new = (block_max > m) ? block_max : m;

        if (isfinite(m_new)) {
          float scale_old = 0.0f;
          if (isfinite(m)) {
            scale_old = exp2f(m - m_new);
          }

          if (scale_old > 0.0f) {
            hvx_vec_scale_f32(o_row, head_dim, scale_old);
            l *= scale_old;
          } else {
            hvx_vec_zero_f32(o_row, head_dim);
            l = 0.0f;
          }

          for (int t = 0; t < block_size; ++t) {
            if (!isfinite(scores[t])) {
              continue;
            }

            const float p = exp2f(scores[t] - m_new);
            l += p;

            const __fp16 *v_row = v_base + (size_t) t * n_kv_heads * head_dim;
            hvx_mad_f32_f16_aa(o_row, v_row, p, head_dim);
          }

          m = m_new;
        }
      }

      if (l > 0.0f) {
        hvx_vec_scale_f32(o_row, head_dim, 1.0f / l);
      } else {
        hvx_vec_zero_f32(o_row, head_dim);
      }
    }
  }

  free(q_row_hf);
  return 0;
}
