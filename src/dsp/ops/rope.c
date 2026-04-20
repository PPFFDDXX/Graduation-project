#include <math.h>
#include <stdlib.h>
#include "dsp/hvx_internal.h"

static inline void rope_build_inv_freq(float *inv_freq, int half, float theta_base) {
  for (int pair_idx = 0; pair_idx < half; ++pair_idx) {
    inv_freq[pair_idx] = powf(theta_base, -((float) pair_idx / (float) half));
  }
}

static inline void rope_build_trig_table(float *cos_table, float *sin_table, const float *inv_freq, int half, int ne1) {
  for (int pos = 0; pos < ne1; ++pos) {
    float *row_cos = cos_table + (size_t) pos * half;
    float *row_sin = sin_table + (size_t) pos * half;
    for (int pair_idx = 0; pair_idx < half; ++pair_idx) {
      const float theta = (float) pos * inv_freq[pair_idx];
      row_cos[pair_idx] = cosf(theta);
      row_sin[pair_idx] = sinf(theta);
    }
  }
}

static inline void hvx_rope_f32_inner(float *restrict dst, const float *restrict src, int ne0, const float *row_cos,
                                      const float *row_sin) {
  for (int i = 0; i + 1 < ne0; i += 2) {
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
  const size_t table_elems = (size_t) half * (size_t) ne1;

  if (!dst || !src || !ne0 || !ne1) {
    return -1;
  }
  if (ne0 < 2) {
    return -1;
  }
  if (!is_aligned(dst, VLEN) || !is_aligned(src, VLEN)) {
    return -1;
  }

  float *inv_freq = (float *) malloc((size_t) half * sizeof(float));
  if (!inv_freq) {
    return -1;
  }
  rope_build_inv_freq(inv_freq, half, theta_base);

  float *cos_table = (float *) malloc(table_elems * sizeof(float));
  float *sin_table = (float *) malloc(table_elems * sizeof(float));
  if (!cos_table || !sin_table) {
    free(inv_freq);
    free(cos_table);
    free(sin_table);
    return -1;
  }
  rope_build_trig_table(cos_table, sin_table, inv_freq, half, ne1);

  for (int j = 0; j < ne1; ++j) {
    float       *out_row = dst + j * ne0;
    const float *in_row  = src + j * ne0;
    const float *row_cos = cos_table + (size_t) j * half;
    const float *row_sin = sin_table + (size_t) j * half;
    hvx_rope_f32_inner(out_row, in_row, ne0, row_cos, row_sin);
  }

  free(cos_table);
  free(sin_table);
  free(inv_freq);
  return 0;
}
