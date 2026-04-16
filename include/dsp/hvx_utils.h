#pragma once

#include <hexagon_types.h>
#include <stddef.h>
#include <stdint.h>

#include "dsp/hvx_internal.h"
#include "dsp/hvx_math.h"

static HVX_INLINE_ALWAYS HVX_Vector hvx_vec_splat_f32(float v) {
  union {
    float    f;
    uint32_t i;
  } u = { .f = v };
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

static HVX_INLINE_ALWAYS HVX_Vector hvx_vec_add_f32(HVX_Vector a_sf, HVX_Vector b_sf) {
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a_sf, b_sf));
}

static HVX_INLINE_ALWAYS HVX_Vector hvx_vec_sub_f32(HVX_Vector a_sf, HVX_Vector b_sf) {
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a_sf, b_sf));
}

static HVX_INLINE_ALWAYS HVX_Vector hvx_vec_mul_f32(HVX_Vector a_sf, HVX_Vector b_sf) {
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a_sf, b_sf));
}

static HVX_INLINE_ALWAYS HVX_Vector hvx_vec_fma_f32(HVX_Vector a_sf, HVX_Vector b_sf, HVX_Vector c_sf) {
  HVX_Vector ab_qf32 = Q6_Vqf32_vmpy_VsfVsf(a_sf, b_sf);
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(ab_qf32, c_sf));
}

static HVX_INLINE_ALWAYS HVX_Vector hvx_vec_square_f32(HVX_Vector x_sf) {
  return hvx_vec_mul_f32(x_sf, x_sf);
}

static HVX_INLINE_ALWAYS HVX_Vector hvx_vec_cube_f32(HVX_Vector x_sf) {
  HVX_Vector x2_sf = hvx_vec_square_f32(x_sf);
  return hvx_vec_mul_f32(x2_sf, x_sf);
}

static HVX_INLINE_ALWAYS HVX_Vector hvx_vec_clamp_f32(HVX_Vector x_sf, HVX_Vector lo_sf, HVX_Vector hi_sf) {
  HVX_Vector y_sf = Q6_Vsf_vmax_VsfVsf(x_sf, lo_sf);
  y_sf            = Q6_Vsf_vmin_VsfVsf(y_sf, hi_sf);
  return y_sf;
}

static HVX_INLINE_ALWAYS HVX_Vector hvx_vec_clamp_f32_scalar(HVX_Vector x_sf, float lo, float hi) {
  return hvx_vec_clamp_f32(x_sf, hvx_vec_splat_f32(lo), hvx_vec_splat_f32(hi));
}

static HVX_INLINE_ALWAYS HVX_VectorPred hvx_vec_nan_inf_pred_f32(HVX_Vector x_sf) {
  const HVX_Vector nan_inf_mask = Q6_V_vsplat_R(0x7F800000);
  HVX_Vector       exp_bits     = Q6_V_vand_VV(x_sf, nan_inf_mask);
  return Q6_Q_vcmp_eq_VwVw(exp_bits, nan_inf_mask);
}

static HVX_INLINE_ALWAYS HVX_Vector hvx_vec_zero_if_nan_or_inf_f32(HVX_Vector x_sf) {
  return Q6_V_vmux_QVV(hvx_vec_nan_inf_pred_f32(x_sf), Q6_V_vzero(), x_sf);
}

static HVX_INLINE_ALWAYS HVX_Vector hvx_vec_inv_f32_approx_guard(HVX_Vector x_sf) {
  HVX_Vector inv_sf = Q6_Vsf_equals_Vqf32(hvx_my_inv_vqf32_vsf(x_sf));
  return hvx_vec_zero_if_nan_or_inf_f32(inv_sf);
}

static HVX_INLINE_ALWAYS HVX_Vector hvx_vec_exp_helper_f32(HVX_Vector x_sf) {
  const HVX_Vector v_log2e = hvx_vec_splat_f32(1.4426950408889634f);
  HVX_Vector       x_clip  = hvx_vec_clamp_f32_scalar(x_sf, -80.0f, 80.0f);
  HVX_Vector       x2_arg  = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x_clip, v_log2e));
  return hvx_my_exp2_vsf(x2_arg);
}

static HVX_INLINE_ALWAYS HVX_Vector hvx_vec_sigmoid_f32(HVX_Vector x_sf) {
  const HVX_Vector v_one = hvx_vec_splat_f32(1.0f);
  HVX_Vector       neg_x = hvx_vec_neg_f32(x_sf);
  HVX_Vector       exp_n = hvx_vec_exp_helper_f32(neg_x);
  HVX_Vector       den   = hvx_vec_add_f32(v_one, exp_n);
  return hvx_vec_inv_f32_approx_guard(den);
}

static HVX_INLINE_ALWAYS HVX_Vector hvx_vec_tanh_fast_arg_f32(HVX_Vector u_sf) {
  const HVX_Vector v_3  = hvx_vec_splat_f32(3.0f);
  const HVX_Vector v_n3 = hvx_vec_splat_f32(-3.0f);
  const HVX_Vector v_9  = hvx_vec_splat_f32(9.0f);
  const HVX_Vector v_27 = hvx_vec_splat_f32(27.0f);

  HVX_Vector u_clip = hvx_vec_clamp_f32(u_sf, v_n3, v_3);
  HVX_Vector u2     = hvx_vec_square_f32(u_clip);
  HVX_Vector num    = hvx_vec_mul_f32(u_clip, hvx_vec_add_f32(v_27, u2));
  HVX_Vector den    = hvx_vec_add_f32(v_27, hvx_vec_mul_f32(v_9, u2));
  HVX_Vector inv    = hvx_vec_inv_f32_approx_guard(den);
  return hvx_vec_mul_f32(num, inv);
}

// GELU用
static HVX_INLINE_ALWAYS HVX_Vector hvx_vec_tanh_gelu_f32_fast(HVX_Vector x_sf) {
  const HVX_Vector v_c0 = hvx_vec_splat_f32(0.7978845608f);  // sqrt(2/pi)
  const HVX_Vector v_c1 = hvx_vec_splat_f32(0.044715f);

  HVX_Vector x2    = hvx_vec_square_f32(x_sf);
  HVX_Vector x3    = hvx_vec_mul_f32(x2, x_sf);
  HVX_Vector inner = hvx_vec_fma_f32(v_c1, x3, x_sf);        // x + 0.044715*x^3
  HVX_Vector u     = hvx_vec_mul_f32(v_c0, inner);
  return hvx_vec_tanh_fast_arg_f32(u);
}

static HVX_INLINE_ALWAYS HVX_Vector hvx_vec_tanh_gelu_f32(HVX_Vector x_sf) {
  const HVX_Vector v_c0  = hvx_vec_splat_f32(0.7978845608f);  // sqrt(2/pi)
  const HVX_Vector v_c1  = hvx_vec_splat_f32(0.044715f);
  const HVX_Vector v_two = hvx_vec_splat_f32(2.0f);
  const HVX_Vector v_one = hvx_vec_splat_f32(1.0f);

  HVX_Vector x2    = hvx_vec_square_f32(x_sf);
  HVX_Vector x3    = hvx_vec_mul_f32(x2, x_sf);
  HVX_Vector inner = hvx_vec_fma_f32(v_c1, x3, x_sf);
  HVX_Vector u     = hvx_vec_mul_f32(v_c0, inner);

  HVX_Vector two_u = hvx_vec_mul_f32(v_two, u);
  HVX_Vector sig   = hvx_vec_sigmoid_f32(two_u);
  HVX_Vector tanh_u = hvx_vec_sub_f32(hvx_vec_mul_f32(v_two, sig), v_one);
  return hvx_vec_zero_if_nan_or_inf_f32(tanh_u);
}
