#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "dsp/hvx_convert.h"
#include "dsp/hvx_internal.h"
#include "dsp/op_parallel.h"
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

typedef struct {
  float       *O;
  const float *Q;
  const __fp16 *K;
  const __fp16 *V;
  const __fp16 *mask;
  int          qo_len;
  int          kv_len;
  int          n_heads;
  int          n_kv_heads;
  int          head_dim;
  int          head_dim_pad;
  int          gqa_factor;
  float        qk_scale;
  size_t       kv_pad_len;
  volatile int status;
} flash_attn_hvx_parallel_ctx_t;

static inline void flash_attn_hvx_compute_one_row(const flash_attn_hvx_parallel_ctx_t *p, int iq, int h,
                                                   __fp16 *restrict q_row_hf) {
  const int h_kv = h / p->gqa_factor;

  float *restrict       o_row = p->O + ((size_t) iq * p->n_heads + h) * p->head_dim;
  const float *restrict q_row = p->Q + ((size_t) iq * p->n_heads + h) * p->head_dim;
  hvx_copy_f16_f32_row(q_row_hf, q_row, p->head_dim);

  hvx_vec_zero_f32(o_row, p->head_dim);

  float m = -INFINITY;
  float l = 0.0f;

  for (int jb = 0; jb < p->kv_len; jb += FLASH_ATTN_HVX_BLOCK_SIZE) {
    const int block_size = Q6_R_min_RR(FLASH_ATTN_HVX_BLOCK_SIZE, p->kv_len - jb);

    const __fp16 *k_base = p->K + ((size_t) jb * p->n_kv_heads + h_kv) * p->head_dim;
    const __fp16 *v_base = p->V + ((size_t) jb * p->n_kv_heads + h_kv) * p->head_dim;

    l2fetch(k_base, p->n_kv_heads * p->head_dim * (int) sizeof(__fp16), p->head_dim * (int) sizeof(__fp16), block_size,
            0);
    l2fetch(v_base, p->n_kv_heads * p->head_dim * (int) sizeof(__fp16), p->head_dim * (int) sizeof(__fp16), block_size,
            0);

    float scores[FLASH_ATTN_HVX_BLOCK_SIZE];
    float block_max = -INFINITY;

    for (int t = 0; t < block_size; ++t) {
      const __fp16 *k_row = k_base + (size_t) t * p->n_kv_heads * p->head_dim;
      float score          = hvx_dot_f16_f16_aa(q_row_hf, k_row, p->head_dim, p->qk_scale);

      if (p->mask && (float) p->mask[(size_t) iq * p->kv_pad_len + (jb + t)] < -128.0f) {
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
        hvx_vec_scale_f32(o_row, p->head_dim, scale_old);
        l *= scale_old;
      } else {
        hvx_vec_zero_f32(o_row, p->head_dim);
        l = 0.0f;
      }

      for (int t = 0; t < block_size; ++t) {
        if (!isfinite(scores[t])) {
          continue;
        }

        const float p_attn = exp2f(scores[t] - m_new);
        l += p_attn;

        const __fp16 *v_row = v_base + (size_t) t * p->n_kv_heads * p->head_dim;
        hvx_mad_f32_f16_aa(o_row, v_row, p_attn, p->head_dim);
      }

      m = m_new;
    }
  }

  if (l > 0.0f) {
    hvx_vec_scale_f32(o_row, p->head_dim, 1.0f / l);
  } else {
    hvx_vec_zero_f32(o_row, p->head_dim);
  }
}

static void flash_attn_hvx_rows_fn(void *ctx, int row_begin, int row_end) {
  flash_attn_hvx_parallel_ctx_t *p = (flash_attn_hvx_parallel_ctx_t *) ctx;

  if (p->status != 0) {
    return;
  }

  __fp16 *q_row_hf = NULL;
  if (posix_memalign((void **) &q_row_hf, VLEN, (size_t) p->head_dim_pad * sizeof(__fp16)) != 0) {
    __sync_lock_test_and_set((int *) &p->status, -1);
    return;
  }

  for (int row = row_begin; row < row_end; ++row) {
    const int iq = row / p->n_heads;
    const int h  = row - iq * p->n_heads;
    flash_attn_hvx_compute_one_row(p, iq, h, q_row_hf);
  }

  free(q_row_hf);
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

  const int   gqa_factor  = n_heads / n_kv_heads;
  const float qk_scale    = 1.0f / sqrtf((float) head_dim) * 1.4426950408889634f;
  const size_t kv_pad_len = align_up((size_t) kv_len, 64);
  const int   head_dim_pad = (int) align_up((size_t) head_dim, 64);

  flash_attn_hvx_parallel_ctx_t ctx = {
    .O           = O,
    .Q           = Q,
    .K           = K,
    .V           = V,
    .mask        = mask,
    .qo_len      = qo_len,
    .kv_len      = kv_len,
    .n_heads     = n_heads,
    .n_kv_heads  = n_kv_heads,
    .head_dim    = head_dim,
    .head_dim_pad = head_dim_pad,
    .gqa_factor  = gqa_factor,
    .qk_scale    = qk_scale,
    .kv_pad_len  = kv_pad_len,
    .status      = 0,
  };

  const int n_rows = qo_len * n_heads;
  const int min_rows_per_task = 2;
  if (op_parallel_for_rows(n_rows, min_rows_per_task, flash_attn_hvx_rows_fn, &ctx) != 0) {
    return -1;
  }

  return ctx.status;
}
