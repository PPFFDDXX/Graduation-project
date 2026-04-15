#pragma once

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include "dsp/hvx_internal.h"
#include "dsp/hvx_utils.h"

#define HVX_EXP_COEFF_5 (0x39506967)  // 1/(7!)
#define HVX_EXP_COEFF_4 (0x3AB743CE)  // 1/(6!)
#define HVX_EXP_COEFF_3 (0x3C088908)  // 1/(5!)
#define HVX_EXP_COEFF_2 (0x3D2AA9C1)  // 1/(4!)
#define HVX_EXP_COEFF_1 (0x3E2AAAAA)  // 1/(3!)
#define HVX_EXP_COEFF_0 (0x3F000000)  // 1/(2!)
#define HVX_EXP_LOGN2   (0x3F317218)  // ln(2)
#define HVX_EXP_LOG2E   (0x3FB8AA3B)  // 1/ln(2)
#define HVX_EXP_ONE     (0x3f800000)  // 1.0
#define HVX_EXP_RANGE_R (0x42B16666)  // 88.7
#define HVX_EXP_RANGE_L (0xC2B00000)  // -88.0

static HVX_INLINE_ALWAYS HVX_Vector hvx_vec_exp_f32(HVX_Vector in_vec) {
  HVX_Vector x_qf32_v, x_v, z_qf32_v, y_v, k_v, f_v, t_v, zero_v = Q6_V_vzero();
  HVX_Vector log2e = Q6_V_vsplat_R(HVX_EXP_LOG2E);
  HVX_Vector logn2 = Q6_V_vsplat_R(HVX_EXP_LOGN2);

  // Clamp inputs to avoid overflow/underflow in fp32 exp.
  HVX_VectorPred pred_cap_right = Q6_Q_vcmp_gt_VsfVsf(in_vec, Q6_V_vsplat_R(HVX_EXP_RANGE_R));
  HVX_VectorPred pred_cap_left  = Q6_Q_vcmp_gt_VsfVsf(Q6_V_vsplat_R(HVX_EXP_RANGE_L), in_vec);
  in_vec                        = Q6_V_vmux_QVV(pred_cap_right, Q6_V_vsplat_R(HVX_EXP_RANGE_R), in_vec);
  in_vec                        = Q6_V_vmux_QVV(pred_cap_left, Q6_V_vsplat_R(HVX_EXP_RANGE_L), in_vec);

  // x = f*ln(2) + epsilon, exp(x)=exp(epsilon)*2^f
  t_v = Q6_Vqf32_vmpy_VsfVsf(log2e, in_vec);
  t_v = Q6_Vsf_equals_Vqf32(t_v);
  f_v = hvx_vec_floor_f32(t_v);
  k_v = hvx_vec_truncate_f32(f_v);

  x_qf32_v = Q6_Vqf32_vadd_VsfVsf(in_vec, zero_v);
  t_v      = Q6_Vqf32_vmpy_VsfVsf(f_v, logn2);
  x_qf32_v = Q6_Vqf32_vsub_Vqf32Vqf32(x_qf32_v, t_v);
  x_qf32_v = Q6_Vqf32_vadd_Vqf32Vsf(x_qf32_v, zero_v);  // normalize qf32
  x_v      = Q6_Vsf_equals_Vqf32(x_qf32_v);

  // z = x*x
  z_qf32_v = Q6_Vqf32_vmpy_Vqf32Vqf32(x_qf32_v, x_qf32_v);
  z_qf32_v = Q6_Vqf32_vadd_Vqf32Vsf(z_qf32_v, zero_v);

  // 
  y_v = Q6_Vqf32_vmpy_VsfVsf(Q6_V_vsplat_R(HVX_EXP_COEFF_5), x_v);
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, Q6_V_vsplat_R(HVX_EXP_COEFF_4));
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

  y_v = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, Q6_V_vsplat_R(HVX_EXP_COEFF_3));
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

  y_v = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, Q6_V_vsplat_R(HVX_EXP_COEFF_2));
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

  y_v = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, Q6_V_vsplat_R(HVX_EXP_COEFF_1));
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

  y_v = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, Q6_V_vsplat_R(HVX_EXP_COEFF_0));
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

  y_v = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, z_qf32_v);
  y_v = Q6_Vqf32_vadd_Vqf32Vqf32(y_v, x_qf32_v);
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, Q6_V_vsplat_R(HVX_EXP_ONE));

  // Reconstruct exp(x) by modifying exponent: y * 2^k
  y_v = Q6_Vsf_equals_Vqf32(y_v);
  HVX_Vector y_v_exponent = Q6_Vw_vasl_VwR(y_v, 1);
  y_v_exponent            = Q6_Vuw_vlsr_VuwR(y_v_exponent, IEEE_VSF_MANTLEN + 1);
  y_v_exponent            = Q6_Vw_vadd_VwVw(k_v, y_v_exponent);

  HVX_VectorPred qy_v_negative_exponent = Q6_Q_vcmp_gt_VwVw(zero_v, y_v_exponent);
  y_v                                = Q6_Vw_vaslacc_VwVwR(y_v, k_v, IEEE_VSF_MANTLEN);
  y_v                                = Q6_V_vmux_QVV(qy_v_negative_exponent, zero_v, y_v);

  return y_v;
}

static HVX_INLINE_ALWAYS HVX_Vector hvx_vec_exp_f32_guard(HVX_Vector in_vec, HVX_Vector max_exp, HVX_Vector inf_v) {
  HVX_VectorPred pred = Q6_Q_vcmp_gt_VsfVsf(in_vec, max_exp);
  HVX_Vector     out  = hvx_vec_exp_f32(in_vec);
  return Q6_V_vmux_QVV(pred, inf_v, out);
}

static HVX_INLINE_ALWAYS void hvx_exp_f32(uint8_t *restrict dst, const uint8_t *restrict src, int n_elems, bool negate) {
  const int n_vecs   = n_elems / 32;
  const int leftover = n_elems & 31;

  const HVX_Vector max_exp = hvx_vec_splat_f32(HVX_EXP_RANGE_R);
  const HVX_Vector inf_v   = hvx_vec_splat_f32(INFINITY);

  if (!is_aligned(dst, VLEN) || !is_aligned(src, VLEN)) {
    //如果没对齐
    for (int i = 0; i < n_vecs; ++i) {
      HVX_Vector v_in = *(const HVX_UVector *) (src + i * VLEN);
      if (negate) {
        v_in = hvx_vec_neg_f32(v_in);
      }
      *(HVX_UVector *) (dst + i * VLEN) = hvx_vec_exp_f32_guard(v_in, max_exp, inf_v);
    }
  } else {
    const HVX_Vector *pv_in  = (const HVX_Vector *) src;
    HVX_Vector       *pv_out = (HVX_Vector *) dst;
    for (int i = 0; i < n_vecs; ++i) {
      HVX_Vector v_in = *pv_in++;
      if (negate) {
        v_in = hvx_vec_neg_f32(v_in);
      }
      *pv_out++ = hvx_vec_exp_f32_guard(v_in, max_exp, inf_v);
    }
  }

  if (leftover > 0) {
    const float *src_tail = (const float *) src + n_vecs * 32;
    float       *dst_tail = (float *) dst + n_vecs * 32;
    HVX_Vector   v_in     = *(const HVX_UVector *) src_tail;
    if (negate) {
      v_in = hvx_vec_neg_f32(v_in);
    }
    HVX_Vector v_out = hvx_vec_exp_f32_guard(v_in, max_exp, inf_v);
    vstu_variable(dst_tail, leftover * sizeof(float), v_out);
  }
}
