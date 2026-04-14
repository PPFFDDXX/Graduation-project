#pragma once

#include "hvx_internal.h"

static inline HVX_Vector hvx_vec_inverse_f32(HVX_Vector v_sf) {

    HVX_Vector inv_aprox_sf = Q6_V_vsplat_R(0x7EEEEBB3);
    
    HVX_Vector two_sf = hvx_vec_splat_f32(2.0);//得到2.0的向量广播形式

    HVX_Vector i_sf = Q6_Vw_vsub_VwVw(inv_aprox_sf, v_sf);// 得到初始近似向量

    HVX_Vector r_qf;
    // 进行三次迭代
    r_qf = Q6_Vqf32_vmpy_VsfVsf(i_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(i_sf, v_sf)))));//x_(n+1)​=x_n*​(2−a*x_n​)
    r_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(r_qf, Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r_qf), v_sf))));
    r_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(r_qf, Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r_qf), v_sf))));
    //返回近似值
    return Q6_Vsf_equals_Vqf32(r_qf);
}

static inline HVX_Vector hvx_vec_inverse_f32_guard(HVX_Vector v_sf) {
    HVX_Vector out = hvx_vec_inverse_f32(v_sf);
    HVX_Vector nan_inf_mask = Q6_V_vsplat_R(0x7f800000);
    HVX_Vector masked_out = Q6_V_vand_VV(out, nan_inf_mask);
    const HVX_VectorPred pred = Q6_Q_vcmp_eq_VwVw(nan_inf_mask, masked_out);

    return Q6_V_vmux_QVV(pred, Q6_V_vzero(), out);
}
