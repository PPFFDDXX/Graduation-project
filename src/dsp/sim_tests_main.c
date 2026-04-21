#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp/hvx_internal.h"
#include "dsp/ops.h"

#define LOGF(...)         \
  do {                    \
    printf(__VA_ARGS__);  \
    printf("\n");         \
  } while (0)

typedef int (*binary_op_fn)(float *restrict dst, const float *restrict src0, const float *restrict src1, int ne0,
                            int ne1);
typedef int (*unary_op_fn)(float *restrict dst, const float *restrict src, int ne0, int ne1);

static inline float my_fabsf(float x) {
  return x >= 0.0f ? x : -x;
}

static uint32_t g_rng_state = 1u;

static inline uint32_t rng_u32(void) {
  g_rng_state = g_rng_state * 1664525u + 1013904223u;
  return g_rng_state;
}

static inline float rng_f32(float lo, float hi) {
  const float t = (float) (rng_u32() & 0x00FFFFFFu) / (float) 0x01000000u;
  return lo + (hi - lo) * t;
}

static void ref_add(float *dst, const float *a, const float *b, int n) {
  for (int i = 0; i < n; ++i) {
    dst[i] = a[i] + b[i];
  }
}

static void ref_sub(float *dst, const float *a, const float *b, int n) {
  for (int i = 0; i < n; ++i) {
    dst[i] = a[i] - b[i];
  }
}

static void ref_mpy(float *dst, const float *a, const float *b, int n) {
  for (int i = 0; i < n; ++i) {
    dst[i] = a[i] * b[i];
  }
}

static void ref_div(float *dst, const float *a, const float *b, int n) {
  for (int i = 0; i < n; ++i) {
    // Keep reference consistent with DSP guard path for zero denominator.
    dst[i] = (b[i] == 0.0f) ? 0.0f : (a[i] / b[i]);
  }
}

static void ref_relu(float *dst, const float *src, int n) {
  for (int i = 0; i < n; ++i) {
    dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
  }
}

static void ref_leaky_relu(float *dst, const float *src, int n) {
  const float alpha = 0.01f;
  for (int i = 0; i < n; ++i) {
    dst[i] = src[i] > 0.0f ? src[i] : (alpha * src[i]);
  }
}

static void ref_sigmoid(float *dst, const float *src, int n) {
  for (int i = 0; i < n; ++i) {
    dst[i] = 1.0f / (1.0f + expf(-src[i]));
  }
}

static void ref_silu(float *dst, const float *src, int n) {
  for (int i = 0; i < n; ++i) {
    dst[i] = src[i] / (1.0f + expf(-src[i]));
  }
}

static void ref_softmax(float *dst, const float *src, int ne0, int ne1) {
  for (int j = 0; j < ne1; ++j) {
    float       *y = dst + j * ne0;
    const float *x = src + j * ne0;

    float row_max = x[0];
    for (int i = 1; i < ne0; ++i) {
      if (x[i] > row_max) {
        row_max = x[i];
      }
    }

    float sum = 0.0f;
    for (int i = 0; i < ne0; ++i) {
      y[i] = expf(x[i] - row_max);
      sum += y[i];
    }

    float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    for (int i = 0; i < ne0; ++i) {
      y[i] *= inv_sum;
    }
  }
}

static void ref_gelu(float *dst, const float *src, int n) {
  for (int i = 0; i < n; ++i) {
    float x = src[i];
    dst[i]  = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
  }
}

static void ref_layer_norm(float *dst, const float *src, int ne0, int ne1) {
  const float eps = 1e-5f;
  for (int j = 0; j < ne1; ++j) {
    float       *y = dst + j * ne0;
    const float *x = src + j * ne0;

    float sum = 0.0f;
    float sumsq = 0.0f;
    for (int i = 0; i < ne0; ++i) {
      sum += x[i];
      sumsq += x[i] * x[i];
    }
    float mean = sum / ne0;
    float var  = sumsq / ne0 - mean * mean;
    float inv_std = 1.0f / sqrtf(var > 0.0f ? (var + eps) : eps);
    for (int i = 0; i < ne0; ++i) {
      y[i] = (x[i] - mean) * inv_std;
    }
  }
}

static void ref_rope(float *dst, const float *src, int ne0, int ne1) {
  const double theta_base = 10000.0;
  const int   half       = ne0 / 2;
  const size_t table_elems = (size_t) half * (size_t) ne1;

  if (half <= 0) {
    memcpy(dst, src, (size_t) ne0 * ne1 * sizeof(float));
    return;
  }

  double *inv_freq = (double *) malloc((size_t) half * sizeof(double));
  double *cos_table = (double *) malloc(table_elems * sizeof(double));
  double *sin_table = (double *) malloc(table_elems * sizeof(double));
  if (!inv_freq || !cos_table || !sin_table) {
    free(inv_freq);
    free(cos_table);
    free(sin_table);
    for (int i = 0; i < ne0 * ne1; ++i) {
      dst[i] = src[i];
    }
    return;
  }

  for (int pair_idx = 0; pair_idx < half; ++pair_idx) {
    inv_freq[pair_idx] = pow(theta_base, -((double) pair_idx / (double) half));
  }
  for (int j = 0; j < ne1; ++j) {
    double *row_cos = cos_table + (size_t) j * half;
    double *row_sin = sin_table + (size_t) j * half;
    for (int pair_idx = 0; pair_idx < half; ++pair_idx) {
      const double theta = (double) j * inv_freq[pair_idx];
      row_cos[pair_idx] = cos(theta);
      row_sin[pair_idx] = sin(theta);
    }
  }

  for (int j = 0; j < ne1; ++j) {
    float       *y = dst + j * ne0;
    const float *x = src + j * ne0;
    const double *row_cos = cos_table + (size_t) j * half;
    const double *row_sin = sin_table + (size_t) j * half;

    for (int i = 0; i + 1 < ne0; i += 2) {
      const int   pair_idx = i / 2;
      const double c        = row_cos[pair_idx];
      const double s        = row_sin[pair_idx];

      const double x0 = (double) x[i + 0];
      const double x1 = (double) x[i + 1];
      y[i + 0]        = (float) (x0 * c - x1 * s);
      y[i + 1]        = (float) (x0 * s + x1 * c);
    }

    if (ne0 & 1) {
      y[ne0 - 1] = x[ne0 - 1];
    }
  }

  free(inv_freq);
  free(cos_table);
  free(sin_table);
}

static int run_one_binary_test(const char *name, binary_op_fn fn, void (*ref_fn)(float *, const float *, const float *, int),
                               int ne0, int ne1, int make_zero_den, float tol) {
  const int n = ne0 * ne1;

  float *src0 = NULL;
  float *src1 = NULL;
  float *dst  = NULL;
  float *ref  = NULL;

  if (posix_memalign((void **) &src0, VLEN, n * sizeof(float)) != 0 ||
      posix_memalign((void **) &src1, VLEN, n * sizeof(float)) != 0 ||
      posix_memalign((void **) &dst, VLEN, n * sizeof(float)) != 0 || posix_memalign((void **) &ref, VLEN, n * sizeof(float)) != 0) {
    LOGF("%s: allocation failed", name);
    free(src0);
    free(src1);
    free(dst);
    free(ref);
    return -1;
  }

  for (int i = 0; i < n; ++i) {
    src0[i] = rng_f32(-10.0f, 10.0f);
    if (make_zero_den) {
      float d = rng_f32(-10.0f, 10.0f);
      if (my_fabsf(d) < 0.1f) {
        d = d < 0.0f ? -0.1f : 0.1f;
      }
      src1[i] = d;
      if (i % 97 == 0) {
        src1[i] = 0.0f;
      }
    } else {
      src1[i] = rng_f32(-10.0f, 10.0f);
    }
  }

  const int warmup = 10;
  const int repeat = 20;

  int rc = 0;
  for (int t = 0; t < warmup; ++t) {
    rc = fn(dst, src0, src1, ne0, ne1);
    if (rc != 0) {
      break;
    }
  }

  for (int t = 0; t < repeat; ++t) {
    rc = fn(dst, src0, src1, ne0, ne1);
    if (rc != 0) {
      break;
    }
  }

  if (rc != 0) {
    LOGF("%s: kernel returned %d", name, rc);
    free(src0);
    free(src1);
    free(dst);
    free(ref);
    return -1;
  }

  ref_fn(ref, src0, src1, n);

  int   n_fail   = 0;
  float max_diff = 0.0f;
  for (int i = 0; i < n; ++i) {
    const float d = my_fabsf(dst[i] - ref[i]);
    if (d > max_diff) {
      max_diff = d;
    }
    if (d > tol) {
      ++n_fail;
      if (n_fail <= 8) {
        LOGF("%s mismatch @%d: dst=%g ref=%g diff=%g", name, i, dst[i], ref[i], d);
      }
    }
  }

  LOGF("%s: ne0=%d ne1=%d, repeats=%d, max_diff=%g, failed=%d", name, ne0, ne1, repeat, max_diff, n_fail);

  free(src0);
  free(src1);
  free(dst);
  free(ref);
  return n_fail == 0 ? 0 : -1;
}

static int run_one_unary_test(const char *name, unary_op_fn fn, void (*ref_fn)(float *, const float *, int), int ne0,
                              int ne1, float tol) {
  const int n = ne0 * ne1;

  float *src = NULL;
  float *dst = NULL;
  float *ref = NULL;

  if (posix_memalign((void **) &src, VLEN, n * sizeof(float)) != 0 ||
      posix_memalign((void **) &dst, VLEN, n * sizeof(float)) != 0 || posix_memalign((void **) &ref, VLEN, n * sizeof(float)) != 0) {
    LOGF("%s: allocation failed", name);
    free(src);
    free(dst);
    free(ref);
    return -1;
  }

  for (int i = 0; i < n; ++i) {
    src[i] = rng_f32(-10.0f, 10.0f);
  }

  const int warmup = 10;
  const int repeat = 20;

  int rc = 0;
  for (int t = 0; t < warmup; ++t) {
    rc = fn(dst, src, ne0, ne1);
    if (rc != 0) {
      break;
    }
  }
  for (int t = 0; t < repeat; ++t) {
    rc = fn(dst, src, ne0, ne1);
    if (rc != 0) {
      break;
    }
  }

  if (rc != 0) {
    LOGF("%s: kernel returned %d", name, rc);
    free(src);
    free(dst);
    free(ref);
    return -1;
  }

  ref_fn(ref, src, n);

  int   n_fail   = 0;
  float max_diff = 0.0f;
  for (int i = 0; i < n; ++i) {
    const float d = my_fabsf(dst[i] - ref[i]);
    if (d > max_diff) {
      max_diff = d;
    }
    if (d > tol) {
      ++n_fail;
      if (n_fail <= 8) {
        LOGF("%s mismatch @%d: dst=%g ref=%g diff=%g", name, i, dst[i], ref[i], d);
      }
    }
  }

  LOGF("%s: ne0=%d ne1=%d, repeats=%d, max_diff=%g, failed=%d", name, ne0, ne1, repeat, max_diff, n_fail);

  free(src);
  free(dst);
  free(ref);
  return n_fail == 0 ? 0 : -1;
}

static int run_one_softmax_test(const char *name, int ne0, int ne1, float tol) {
  const int n = ne0 * ne1;

  float *src = NULL;
  float *dst = NULL;
  float *ref = NULL;

  if (posix_memalign((void **) &src, VLEN, n * sizeof(float)) != 0 ||
      posix_memalign((void **) &dst, VLEN, n * sizeof(float)) != 0 || posix_memalign((void **) &ref, VLEN, n * sizeof(float)) != 0) {
    LOGF("%s: allocation failed", name);
    free(src);
    free(dst);
    free(ref);
    return -1;
  }

  for (int i = 0; i < n; ++i) {
    src[i] = rng_f32(-10.0f, 10.0f);
  }

  const int warmup = 10;
  const int repeat = 20;

  int rc = 0;
  for (int t = 0; t < warmup; ++t) {
    rc = hvx_softmax_f32(dst, src, ne0, ne1);
    if (rc != 0) {
      break;
    }
  }
  for (int t = 0; t < repeat; ++t) {
    rc = hvx_softmax_f32(dst, src, ne0, ne1);
    if (rc != 0) {
      break;
    }
  }

  if (rc != 0) {
    LOGF("%s: kernel returned %d", name, rc);
    free(src);
    free(dst);
    free(ref);
    return -1;
  }

  ref_softmax(ref, src, ne0, ne1);

  int   n_fail   = 0;
  float max_diff = 0.0f;
  for (int i = 0; i < n; ++i) {
    const float d = my_fabsf(dst[i] - ref[i]);
    if (d > max_diff) {
      max_diff = d;
    }
    if (d > tol) {
      ++n_fail;
      if (n_fail <= 8) {
        LOGF("%s mismatch @%d: dst=%g ref=%g diff=%g", name, i, dst[i], ref[i], d);
      }
    }
  }

  LOGF("%s: ne0=%d ne1=%d, repeats=%d, max_diff=%g, failed=%d", name, ne0, ne1, repeat, max_diff, n_fail);

  free(src);
  free(dst);
  free(ref);
  return n_fail == 0 ? 0 : -1;
}

static int run_one_layer_norm_test(const char *name, int ne0, int ne1, float tol) {
  const int n = ne0 * ne1;

  float *src = NULL;
  float *dst = NULL;
  float *ref = NULL;

  if (posix_memalign((void **) &src, VLEN, n * sizeof(float)) != 0 ||
      posix_memalign((void **) &dst, VLEN, n * sizeof(float)) != 0 || posix_memalign((void **) &ref, VLEN, n * sizeof(float)) != 0) {
    LOGF("%s: allocation failed", name);
    free(src);
    free(dst);
    free(ref);
    return -1;
  }

  for (int i = 0; i < n; ++i) {
    src[i] = rng_f32(-10.0f, 10.0f);
  }

  const int warmup = 10;
  const int repeat = 20;

  int rc = 0;
  for (int t = 0; t < warmup; ++t) {
    rc = hvx_layer_norm_f32(dst, src, ne0, ne1);
    if (rc != 0) {
      break;
    }
  }
  for (int t = 0; t < repeat; ++t) {
    rc = hvx_layer_norm_f32(dst, src, ne0, ne1);
    if (rc != 0) {
      break;
    }
  }

  if (rc != 0) {
    LOGF("%s: kernel returned %d", name, rc);
    free(src);
    free(dst);
    free(ref);
    return -1;
  }

  ref_layer_norm(ref, src, ne0, ne1);

  int   n_fail   = 0;
  float max_diff = 0.0f;
  for (int i = 0; i < n; ++i) {
    const float d = my_fabsf(dst[i] - ref[i]);
    if (d > max_diff) {
      max_diff = d;
    }
    if (d > tol) {
      ++n_fail;
      if (n_fail <= 8) {
        LOGF("%s mismatch @%d: dst=%g ref=%g diff=%g", name, i, dst[i], ref[i], d);
      }
    }
  }

  LOGF("%s: ne0=%d ne1=%d, repeats=%d, max_diff=%g, failed=%d", name, ne0, ne1, repeat, max_diff, n_fail);

  free(src);
  free(dst);
  free(ref);
  return n_fail == 0 ? 0 : -1;
}

static int run_one_rope_test(const char *name, int ne0, int ne1, float tol) {
  const int n = ne0 * ne1;

  float *src = NULL;
  float *dst = NULL;
  float *ref = NULL;

  if (posix_memalign((void **) &src, VLEN, n * sizeof(float)) != 0 ||
      posix_memalign((void **) &dst, VLEN, n * sizeof(float)) != 0 || posix_memalign((void **) &ref, VLEN, n * sizeof(float)) != 0) {
    LOGF("%s: allocation failed", name);
    free(src);
    free(dst);
    free(ref);
    return -1;
  }

  for (int i = 0; i < n; ++i) {
    src[i] = rng_f32(-10.0f, 10.0f);
  }

  const int warmup = 10;
  const int repeat = 20;
  int rc = 0;
  for (int t = 0; t < warmup; ++t) {
    rc = hvx_rope_f32(dst, src, ne0, ne1);
    if (rc != 0) {
      break;
    }
  }
  for (int t = 0; t < repeat; ++t) {
    rc = hvx_rope_f32(dst, src, ne0, ne1);
    if (rc != 0) {
      break;
    }
  }

  if (rc != 0) {
    LOGF("%s: kernel returned %d", name, rc);
    free(src);
    free(dst);
    free(ref);
    return -1;
  }

  ref_rope(ref, src, ne0, ne1);
  
  int   n_fail   = 0;
  float max_diff = 0.0f;
  for (int i = 0; i < n; ++i) {
    const float d = my_fabsf(dst[i] - ref[i]);
    if (d > max_diff) {
      max_diff = d;
    }
    if (d > tol) {
      ++n_fail;
      if (n_fail <= 8) {
        LOGF("%s mismatch @%d: dst=%g ref=%g diff=%g", name, i, dst[i], ref[i], d);
      }
    }
  }

  LOGF("%s: ne0=%d ne1=%d, repeats=%d, max_diff=%g, failed=%d", name, ne0, ne1, repeat, max_diff, n_fail);

  free(src);
  free(dst);
  free(ref);
  return n_fail == 0 ? 0 : -1;
}

int main(int argc, char **argv) {
  int status = 0;
  const char *mode = (argc > 1) ? argv[1] : "all";
  const int   run_all = (strcmp(mode, "all") == 0);

  // Keep build green when some test blocks are intentionally commented out.
  (void) ref_add;
  (void) ref_sub;
  (void) ref_mpy;
  (void) ref_div;
  (void) ref_relu;
  (void) ref_leaky_relu;
  (void) ref_sigmoid;
  (void) ref_silu;
  (void) ref_gelu;
  (void) ref_rope;
  (void) run_one_binary_test;
  (void) run_one_unary_test;
  (void) run_one_softmax_test;
  (void) run_one_layer_norm_test;

  // Case 1: tail path (ne0 not divisible by 32), keep ne1=1 to preserve row alignment.
  if (run_all || strcmp(mode, "add") == 0) {
    status |= run_one_binary_test("add_f32_tail", hvx_add_f32, ref_add, 4099, 1, 0, 1e-4f);
  }
  if (run_all || strcmp(mode, "sub") == 0) {
    status |= run_one_binary_test("sub_f32_tail", hvx_sub_f32, ref_sub, 4099, 1, 0, 1e-4f);
  }
  if (run_all || strcmp(mode, "mpy") == 0) {
    status |= run_one_binary_test("mpy_f32_tail", hvx_mpy_f32, ref_mpy, 4099, 1, 0, 1e-4f);
  }
  if (run_all || strcmp(mode, "div") == 0) {
    status |= run_one_binary_test("div_f32_tail", hvx_div_f32, ref_div, 4099, 1, 1, 2e-3f);
  }

  // Case 2: multi-row path with per-row 128B alignment (ne0 divisible by 32).
  if (run_all || strcmp(mode, "add") == 0) {
    status |= run_one_binary_test("add_f32_rows", hvx_add_f32, ref_add, 4096, 8, 0, 1e-4f);
  }
  if (run_all || strcmp(mode, "sub") == 0) {
    status |= run_one_binary_test("sub_f32_rows", hvx_sub_f32, ref_sub, 4096, 8, 0, 1e-4f);
  }
  if (run_all || strcmp(mode, "mpy") == 0) {
    status |= run_one_binary_test("mpy_f32_rows", hvx_mpy_f32, ref_mpy, 4096, 8, 0, 1e-4f);
  }
  if (run_all || strcmp(mode, "div") == 0) {
    status |= run_one_binary_test("div_f32_rows", hvx_div_f32, ref_div, 4096, 8, 1, 2e-3f);
  }

  // ReLU unary op
  if (run_all || strcmp(mode, "relu") == 0) {
    status |= run_one_unary_test("relu_f32_tail", hvx_relu_f32, ref_relu, 4099, 1, 1e-6f);
    status |= run_one_unary_test("relu_f32_rows", hvx_relu_f32, ref_relu, 4096, 8, 1e-6f);
  }
  if (run_all || strcmp(mode, "leaky_relu") == 0) {
    status |= run_one_unary_test("leaky_relu_f32_tail", hvx_leaky_relu_f32, ref_leaky_relu, 4099, 1, 1e-6f);
    status |= run_one_unary_test("leaky_relu_f32_rows", hvx_leaky_relu_f32, ref_leaky_relu, 4096, 8, 1e-6f);
  }
  if (run_all || strcmp(mode, "sigmoid") == 0) {
    status |= run_one_unary_test("sigmoid_f32_tail", hvx_sigmoid_f32, ref_sigmoid, 4099, 1, 3e-5f);
    status |= run_one_unary_test("sigmoid_f32_rows", hvx_sigmoid_f32, ref_sigmoid, 4096, 8, 3e-5f);
  }
  if (run_all || strcmp(mode, "silu") == 0) {
    status |= run_one_unary_test("silu_f32_tail", hvx_silu_f32, ref_silu, 4099, 1, 3e-5f);
    status |= run_one_unary_test("silu_f32_rows", hvx_silu_f32, ref_silu, 4096, 8, 3e-5f);
  }
  if (run_all || strcmp(mode, "softmax") == 0) {
    status |= run_one_softmax_test("softmax_f32_tail", 4099, 1, 6e-5f);
    status |= run_one_softmax_test("softmax_f32_rows", 4096, 8, 6e-5f);
  }
  if (run_all || strcmp(mode, "gelu") == 0) {
    status |= run_one_unary_test("gelu_f32_tail", hvx_gelu_f32, ref_gelu, 4099, 1, 2e-4f);
    status |= run_one_unary_test("gelu_f32_rows", hvx_gelu_f32, ref_gelu, 4096, 8, 2e-4f);
  }
  if (run_all || strcmp(mode, "layer_norm") == 0) {
    status |= run_one_layer_norm_test("layer_norm_f32_tail", 4099, 1, 2e-4f);
    status |= run_one_layer_norm_test("layer_norm_f32_rows", 4096, 8, 2e-4f);
  }
  if (run_all || strcmp(mode, "rope") == 0) {
    status |= run_one_rope_test("rope_f32_tail", 4099, 1, 1e-5f);
    status |= run_one_rope_test("rope_f32_rows", 4096, 8, 1e-5f);
  }

  LOGF("%s", status == 0 ? "SIM binary tests passed" : "SIM binary tests failed");
  return status == 0 ? 0 : 1;
}
