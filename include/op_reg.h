#pragma once

#include <stdint.h>

enum HtpOpsIndex {
  HTP_OPS_RMS_NORM_F32,
  HTP_OPS_LAYER_NORM_F32,
  HTP_OPS_ADD_F32,
  HTP_OPS_SUB_F32,
  HTP_OPS_MPY_F32,
  HTP_OPS_DIV_F32,
  HTP_OPS_RELU_F32,
  HTP_OPS_LEAKY_RELU_F32,
  HTP_OPS_SIGMOID_F32,
  HTP_OPS_SILU_F32,
  HTP_OPS_BIAS_ADD_SILU_MUL_F32,
  HTP_OPS_SOFTMAX_F32,
  HTP_OPS_GELU_F32,
  HTP_OPS_ROPE_F32,
  HTP_OPS_MAT_MUL_PERMUTED_W16A32,
  HTP_OPS_MAT_MUL_PERMUTED_W4D16A32,
  HTP_OPS_MAT_MUL_PERMUTED_W8D16A32,
  HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL,
  HTP_OPS_FLASH_ATTN_QO_F32_KV_F16,
  HTP_OPS_COUNT,
};

struct RpcmemBufAddr {
  int32_t fd;
  int32_t offset;
} __attribute__((packed));

struct RmsNormF32Params {
  struct RpcmemBufAddr dst;
  struct RpcmemBufAddr src;
  int32_t       ne0;
  int32_t       ne1;
} __attribute__((packed));

struct MatMulParams {
  struct RpcmemBufAddr output;
  struct RpcmemBufAddr activation; // m * k
  struct RpcmemBufAddr weight; // k * n
  int32_t m;
  int32_t k;
  int32_t n;
} __attribute__((packed));

struct FlashAttnParams {
  struct RpcmemBufAddr o;
  struct RpcmemBufAddr q;
  struct RpcmemBufAddr k;
  struct RpcmemBufAddr v;
  struct RpcmemBufAddr mask;
  int32_t qo_len;
  int32_t kv_len;
  int32_t n_heads;
  int32_t n_kv_heads;
  int32_t head_dim;
} __attribute__((packed));

struct BinaryElemwiseF32Params{
  struct RpcmemBufAddr dst;
  struct RpcmemBufAddr src0;
  struct RpcmemBufAddr src1;
  int32_t       ne0;
  int32_t       ne1;
} __attribute__((packed));

struct BiasAddSiluMulF32Params {
  struct RpcmemBufAddr dst;
  struct RpcmemBufAddr src;
  struct RpcmemBufAddr bias;
  struct RpcmemBufAddr mul;
  int32_t              ne0;
  int32_t              ne1;
} __attribute__((packed));

struct UnaryElemwiseF32Params {
  struct RpcmemBufAddr dst;
  struct RpcmemBufAddr src;
  int32_t       ne0;
  int32_t       ne1;
} __attribute__((packed));
