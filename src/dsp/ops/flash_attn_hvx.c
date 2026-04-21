#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "dsp/hvx_convert.h"
#include "dsp/hvx_internal.h"
#include "dsp/utils.h"

#define FLASH_ATTN_HVX_BLOCK_SIZE 64

static inline float hvx_dot_qf32_khf(const float *restrict q_row, const __fp16 *restrict k_row, int d) {
  HVX_Vector v_sum_qf32 = Q6_V_vzero();

  int i = 0;
  for (; i + 31 < d; i += 32) {
    _Alignas(VLEN) float k_chunk[32];
    for (int t = 0; t < 32; ++t) {
      k_chunk[t] = (float) k_row[i + t];
    }

    const HVX_Vector v_q_sf = vmemu(q_row + i);
    const HVX_Vector v_k_sf = vmem(k_chunk);
    const HVX_Vector v_mul_qf32 = Q6_Vqf32_vmpy_VsfVsf(v_q_sf, v_k_sf);
    v_sum_qf32 = Q6_Vqf32_vadd_Vqf32Vqf32(v_sum_qf32, v_mul_qf32);
  }

  _Alignas(VLEN) float partial[32];
  vmem(partial) = Q6_Vsf_equals_Vqf32(v_sum_qf32);

  float sum = 0.0f;
  for (int t = 0; t < 32; ++t) {
    sum += partial[t];
  }

  for (; i < d; ++i) {
    sum += q_row[i] * (float) k_row[i];
  }

  return sum;
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

static inline void hvx_vec_axpy_vhf(float *restrict y, const __fp16 *restrict x_hf, int d, float a) {
  union {
    float    f;
    uint32_t u;
  } u = { .f = a };
  const HVX_Vector v_a_sf = Q6_V_vsplat_R((int) u.u);

  int i = 0;
  for (; i + 31 < d; i += 32) {
    _Alignas(VLEN) float x_chunk[32];
    for (int t = 0; t < 32; ++t) {
      x_chunk[t] = (float) x_hf[i + t];
    }

    HVX_Vector v_y_sf = vmemu(y + i);
    HVX_Vector v_x_sf = vmem(x_chunk);
    HVX_Vector v_ax_sf = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_x_sf, v_a_sf));
    v_y_sf = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_y_sf, v_ax_sf));
    vmemu(y + i) = v_y_sf;
  }

  for (; i < d; ++i) {
    y[i] += a * (float) x_hf[i];
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
  const float qk_scale   = 1.0f / sqrtf((float) head_dim) * 1.4426950408889634f;
  const size_t kv_pad_len = align_up((size_t) kv_len, 64);

  for (int h = 0; h < n_heads; ++h) {
    const int h_kv = h / gqa_factor;

    for (int iq = 0; iq < qo_len; ++iq) {
      float *restrict o_row       = O + ((size_t) iq * n_heads + h) * head_dim;
      const float *restrict q_row = Q + ((size_t) iq * n_heads + h) * head_dim;

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
          float score = hvx_dot_qf32_khf(q_row, k_row, head_dim) * qk_scale;

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
            hvx_vec_axpy_vhf(o_row, v_row, head_dim, p);
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

  return 0;
}
