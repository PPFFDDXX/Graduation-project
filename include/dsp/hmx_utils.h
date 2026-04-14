#pragma once

#include <hexagon_types.h>
#include <stddef.h>

#define HMX_FP16_TILE_N_ROWS 32
#define HMX_FP16_TILE_N_COLS 32
#define HMX_FP16_TILE_N_ELMS 1024
#define HMX_FP16_TILE_SIZE   2048

#define HMX_INLINE_ALWAYS inline __attribute__((unused, always_inline))

static HVX_INLINE_ALWAYS HVX_Vector hvx_vec_splat_f32(float v) {
  union {float    f;uint32_t i;} u = { .f = v };
  return Q6_V_vsplat_R(u.i);
}

static HVX_INLINE_ALWAYS HVX_Vector hvx_vec_neg_f32(HVX_Vector v_sf) {
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(Q6_V_vzero(), v_sf));
}

static HVX_INLINE_ALWAYS HVX_Vector hvx_vec_floor_f32(HVX_Vector v_sf) {
  HVX_Vector floor_sf;
  (void) Q6_Vw_vfloor_VsfVsf(v_sf, &floor_sf);
  return floor_sf;
}

static HVX_INLINE_ALWAYS HVX_Vector hvx_vec_truncate_f32(HVX_Vector v_sf) {
  return qhmath_hvx_vw_truncate_vsf(v_sf);
}

static HMX_INLINE_ALWAYS void hmx_set_output_scales(const void *scales) {
  asm volatile("bias = mxmem2(%0)" ::"r"(scales));
}

// set aligned 256 bytes area
static HMX_INLINE_ALWAYS void hmx_init_column_scales(void *out_scales, HVX_Vector v_scale) {
  HVX_Vector *pv = (HVX_Vector *) out_scales;

  *pv++ = v_scale;
  *pv   = Q6_V_vzero();
}

static HMX_INLINE_ALWAYS void hmx_load_tiles_fp16(const __fp16 *row_tiles, const __fp16 *col_tiles, size_t n_tiles) {
  size_t limit = n_tiles * HMX_FP16_TILE_SIZE - 1;
  asm volatile(
    "{ activation.hf = mxmem(%0, %1):deep\n"
    "weight.hf = mxmem(%2, %3) }\n" ::"r"(row_tiles),
    "r"(limit), "r"(col_tiles), "r"(limit)
    : "memory");
}

static HMX_INLINE_ALWAYS void hmx_consume_accumulator_fp16(__fp16 *out) {
  asm volatile(
    "cvt.hf = acc(%0)\n"
    "mxmem(%1, %2) = cvt\n" ::"r"(2),
    "r"(out), "r"(0)
    : "memory");
}

// compute inner product of two vectors of tiles
static HMX_INLINE_ALWAYS void hmx_dot_fp16(__fp16 *out, const __fp16 *row_tiles, const __fp16 *col_tiles,
                                           size_t n_tiles) {
  hmx_load_tiles_fp16(row_tiles, col_tiles, n_tiles);
  hmx_consume_accumulator_fp16(out);
}
