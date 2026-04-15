#pragma once

#include <hexagon_types.h>
#include <stddef.h>
#include <stdint.h>

#include "dsp/hvx_internal.h"

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
