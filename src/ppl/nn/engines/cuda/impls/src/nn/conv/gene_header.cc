// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "gene_header.h"

#include <fstream>
#include <sstream>

std::string GeneHeader::Find(const std::string& path) {
    auto header_ref = header_code_.find(path);
    if (header_ref != header_code_.end()) {
        return header_ref->second;
    }
    return "";
}

GeneHeader::GeneHeader() {
    header_code_.emplace("/mnt/hpc/xusi/ppl.jit/src/ppl/nn/engines/cuda/impls/src/nn/conv/2spk/common/const_macros.h", "// Licensed to the Apache Software Foundation (ASF) under one\n\
// or more contributor license agreements.  See the NOTICE file\n\
// distributed with this work for additional information\n\
// regarding copyright ownership.  The ASF licenses this file\n\
// to you under the Apache License, Version 2.0 (the\n\
// \"License\"); you may not use this file except in compliance\n\
// with the License.  You may obtain a copy of the License at\n\
//\n\
//   http://www.apache.org/licenses/LICENSE-2.0\n\
//\n\
// Unless required by applicable law or agreed to in writing,\n\
// software distributed under the License is distributed on an\n\
// \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n\
// KIND, either express or implied.  See the License for the\n\
// specific language governing permissions and limitations\n\
// under the License.\n\
\n\
////////////////////////////////////////\n\
// kernel list macros\n\
////////////////////////////////////////\n\
\n\
#define SPK_KPARAM_LIST \\\n\
        int4* dA,                                                 \\\n\
        int4* dB,                                                 \\\n\
        int4* dC,                                                 \\\n\
        int kloop_num,                                            \\\n\
        struct lut_t in_lut,          int in_lut_size,            \\\n\
        struct lut_t flt_lut,         int flt_lut_size,           \\\n\
        struct chl_lut_t chl_lut,     int chl_lut_size,           \\\n\
        struct kloop_lut_t kloop_lut, int kloop_lut_size,         \\\n\
        int in_hw,                    int out_hw,                 \\\n\
        int flt_hw,                   int splitk,                 \\\n\
        int in_height,                int in_width,               \\\n\
        int in_num,                   int num_grp,                \\\n\
        int num_chl_per_grp,          int num_chl_per_grp_pad,    \\\n\
        int flt_height,               int flt_width,              \\\n\
        int num_flt_per_grp,          int num_flt_per_grp_pad,    \\\n\
        int out_height,               int out_width,              \\\n\
        int stride_height,            int stride_width,           \\\n\
        int pad_height,               int pad_width,              \\\n\
        int hole_height,              int hole_width,             \\\n\
        int has_bias,                 int* bias\n\
\n\
#define TOTAL_KPARAM_LIST \\\n\
        int4* dA,                                                 \\\n\
        int4* dB,                                                 \\\n\
        int4* dC,                                                 \\\n\
        int kloop_num,                                            \\\n\
        struct lut_t in_lut,          int in_lut_size,            \\\n\
        struct lut_t flt_lut,         int flt_lut_size,           \\\n\
        int in_hw,                    int out_hw,                 \\\n\
        int flt_hw,                   int splitk,                 \\\n\
        int in_height,                int in_width,               \\\n\
        int in_num,                   int num_grp,                \\\n\
        int num_chl_per_grp,          int num_chl_per_grp_pad,    \\\n\
        int flt_height,               int flt_width,              \\\n\
        int num_flt_per_grp,          int num_flt_per_grp_pad,    \\\n\
        int out_height,               int out_width,              \\\n\
        int stride_height,            int stride_width,           \\\n\
        int pad_height,               int pad_width,              \\\n\
        int hole_height,              int hole_width,             \\\n\
        int  has_bias,                const int4* bias,           \\\n\
        int  has_relu,                const __half2 clip_min,     \\\n\
	    bool has_clip,                const __half2 clip_max,     \\\n\
        int  has_prelu,               const void* prelu,          \\\n\
        bool has_elt,                 const int4* pre_data,       \\\n\
        int  has_elt_relu,            const __half2 elt_clip_min, \\\n\
	    bool has_elt_clip,            const __half2 elt_clip_max, \\\n\
        int has_elt_prelu,            const void* elt_prelu,      \\\n\
        const __half leaky,           const __half elt_leaky,     \\\n\
        bool has_concat,              int concat_offset_v8,       \\\n\
        int concat_stride_v8\n\
\n\
////////////////////////////////////////\n\
// align functions\n\
////////////////////////////////////////\n\
\n\
#define Align(x, y)   (((x) + (y) - 1) / (y) * (y))\n\
#define DivUp(x, y)   (((x) + (y) - 1) / (y))\n\
\n\
#define Min(x, y)     (((x) < (y)) ? (x) : (y))\n\
#define Max(x, y)     (((x) > (y)) ? (x) : (y))\n\
\n\
////////////////////////////////////////\n\
// boundary check\n\
////////////////////////////////////////\n\
\n\
#define WidthInRange(_w)     ( (_w < in_width)  && (_w >= 0) )\n\
#define HeightInRange(_h)    ( (_h < in_height) && (_h >= 0) )\n\
\n\
////////////////////////////////////////\n\
// constant cta size macros\n\
////////////////////////////////////////\n\
\n\
#define _4CHAR_TO_INT_          4\n\
#define _4INT_TO_INT4_          4\n\
#define _2INT_TO_INT2_          2\n\
\n\
#define _2HALF_TO_INT_          2\n\
#define _2INT2_TO_INT4_         2\n\
\n\
#define _C1_                    1\n\
#define _C2_                    2\n\
#define _C4_                    4\n\
#define _C8_                    8\n\
#define _C16_                   16\n\
#define _C32_                   32\n\
\n\
#define _1INT_                  1\n\
#define _2INT_                  2\n\
#define _4INT_                  4\n\
#define _8INT_                  8\n\
\n\
#define _1INT4_                 1\n\
#define _2INT4_                 2\n\
#define _4INT4_                 4\n\
#define _8INT4_                 8\n\
\n\
#define _1INT8_                 1\n\
#define _2INT8_                 2\n\
#define _4INT8_                 4\n\
#define _8INT8_                 8\n\
\n\
#define _1HALF_                 1\n\
#define _2HALF_                 2\n\
#define _4HALF_                 4\n\
#define _8HALF_                 8\n\
\n\
#define _1HALF2_                1\n\
#define _2HALF2_                2\n\
#define _4HALF2_                4\n\
#define _8HALF2_                8\n\
\n\
#define _1MMA_                  1\n\
#define _2MMA_                  2\n\
#define _4MMA_                  4\n\
#define _8MMA_                  8\n\
\n\
#define _HALF_ZERO_             0.0\n\
\n\
\n\
#define _INT_TO_BYTE_           4\n\
#define _INT_TO_2HALF_          2\n\
#define _INT2_TO_2HALF2_        2\n\
#define _INT2_TO_2INT_          2\n\
\n\
#define _INT4_TO_INT4_          1\n\
#define _INT4_TO_2INT2_         2\n\
#define _INT4_TO_4INT_          4\n\
#define _INT4_TO_4HALF2_        4\n\
#define _INT4_TO_8HALF_         8\n\
\n\
#define SMEM_ROW_V4_SIZE        8\n\
#define SMEM_ROW_V1_SIZE        32\n\
#define SMEM_ROW_BYTE_SIZE      128\n\
#define SMEM_ROW_BIT_SIZE       1024\n\
\n\
////////////////////////////////////////\n\
// mma size macros\n\
////////////////////////////////////////\n\
\n\
#define TILE_M_PER_MMA          16\n\
#define TILE_K_PER_MMA          8\n\
#define TILE_N_PER_MMA          8\n\
#define TILE_M_PER_MMA_HALF     ((TILE_M_PER_MMA) / 2)\n\
\n\
#define MMA_SIZE_X_IN_THD       4\n\
#define MMA_SIZE_Y_IN_THD       8\n\
\n\
////////////////////////////////////////\n\
// thread / warp / cta size macros\n\
////////////////////////////////////////\n\
\n\
#define WARP_SIZE_IN_THD        32\n\
#define WARP_SIZE_IN_BITS       5\n\
\n\
#define WARP_SIZE_X_IN_THD      4\n\
#define WARP_SIZE_Y_IN_THD      8\n\
\n\
#define SET_SIZE_X_IN_WARP      ((TILE_N_PER_CTA) / (TILE_N_PER_WARP))\n\
#define SET_SIZE_Y_IN_WARP      ((TILE_M_PER_CTA) / (TILE_M_PER_WARP))\n\
\n\
#define SET_SIZE_IN_WARP        ((SET_SIZE_X_IN_WARP) * (SET_SIZE_Y_IN_WARP))\n\
#define SET_SIZE_IN_THD         ((SET_SIZE_IN_WARP)   * (WARP_SIZE_IN_THD))\n\
\n\
#define CTA_SIZE_IN_WARP        ((SET_SIZE_IN_WARP)   * (INTER_SET_REDUCE_RATIO))\n\
#define CTA_SIZE_IN_THD         ((CTA_SIZE_IN_WARP)   * (WARP_SIZE_IN_THD))\n\
\n\
#define WARP_SIZE_IN_THD_HALF   (WARP_SIZE_IN_THD / 2)\n\
#define WARP_SIZE_IN_THD_QTR    (WARP_SIZE_IN_THD / 4)\n\
////////////////////////////////////////\n\
// tiling size macros\n\
////////////////////////////////////////\n\
\n\
#define TILE_M_PER_THD          ((TILE_M_PER_WARP) / (WARP_SIZE_Y_IN_THD))\n\
#define TILE_N_PER_THD          ((TILE_N_PER_WARP) / (WARP_SIZE_X_IN_THD))\n\
\n\
/////////////////////\n\
// tile m\n\
\n\
#define TILE_M_V1_PER_CTA       ((TILE_M_PER_CTA)  / 1)\n\
#define TILE_M_V2_PER_CTA       ((TILE_M_PER_CTA)  / 2)\n\
#define TILE_M_V4_PER_CTA       ((TILE_M_PER_CTA)  / 4)\n\
#define TILE_M_V8_PER_CTA       ((TILE_M_PER_CTA)  / 8)\n\
\n\
#define TILE_M_V1_PER_WARP      ((TILE_M_PER_WARP) / 1)\n\
#define TILE_M_V2_PER_WARP      ((TILE_M_PER_WARP) / 2)\n\
#define TILE_M_V4_PER_WARP      ((TILE_M_PER_WARP) / 4)\n\
#define TILE_M_V8_PER_WARP      ((TILE_M_PER_WARP) / 8)\n\
\n\
#define TILE_M_V1_PER_THD       ((TILE_M_PER_THD)  / 1)\n\
#define TILE_M_V2_PER_THD       ((TILE_M_PER_THD)  / 2)\n\
#define TILE_M_V4_PER_THD       ((TILE_M_PER_THD)  / 4)\n\
#define TILE_M_V8_PER_THD       ((TILE_M_PER_THD)  / 8)\n\
\n\
#define TILE_M_V1_PER_MMA       ((TILE_M_PER_MMA)  / 1)\n\
#define TILE_M_V2_PER_MMA       ((TILE_M_PER_MMA)  / 2)\n\
#define TILE_M_V4_PER_MMA       ((TILE_M_PER_MMA)  / 4)\n\
#define TILE_M_V8_PER_MMA       ((TILE_M_PER_MMA)  / 8)\n\
\n\
/////////////////////\n\
// tile k\n\
\n\
#define TILE_K_V1_PER_CTA       ((TILE_K_PER_CTA)  / 1)\n\
#define TILE_K_V2_PER_CTA       ((TILE_K_PER_CTA)  / 2)\n\
#define TILE_K_V4_PER_CTA       ((TILE_K_PER_CTA)  / 4)\n\
#define TILE_K_V8_PER_CTA       ((TILE_K_PER_CTA)  / 8)\n\
\n\
#define TILE_K_V1_PER_SET       ((TILE_K_PER_SET)  / 1)\n\
#define TILE_K_V2_PER_SET       ((TILE_K_PER_SET)  / 2)\n\
#define TILE_K_V4_PER_SET       ((TILE_K_PER_SET)  / 4)\n\
#define TILE_K_V8_PER_SET       ((TILE_K_PER_SET)  / 8)\n\
\n\
#define TILE_K_V1_PER_WARP      ((TILE_K_PER_WARP) / 1)\n\
#define TILE_K_V2_PER_WARP      ((TILE_K_PER_WARP) / 2)\n\
#define TILE_K_V4_PER_WARP      ((TILE_K_PER_WARP) / 4)\n\
#define TILE_K_V8_PER_WARP      ((TILE_K_PER_WARP) / 8)\n\
\n\
#define TILE_K_V1_PER_MMA       ((TILE_K_PER_MMA) / 1)\n\
#define TILE_K_V2_PER_MMA       ((TILE_K_PER_MMA) / 2)\n\
#define TILE_K_V4_PER_MMA       ((TILE_K_PER_MMA) / 4)\n\
#define TILE_K_V8_PER_MMA       ((TILE_K_PER_MMA) / 8)\n\
\n\
/////////////////////\n\
// tile n\n\
\n\
#define TILE_N_V1_PER_CTA       ((TILE_N_PER_CTA)  / 1)\n\
#define TILE_N_V2_PER_CTA       ((TILE_N_PER_CTA)  / 2)\n\
#define TILE_N_V4_PER_CTA       ((TILE_N_PER_CTA)  / 4)\n\
#define TILE_N_V8_PER_CTA       ((TILE_N_PER_CTA)  / 8)\n\
\n\
#define TILE_N_V1_PER_WARP      ((TILE_N_PER_WARP) / 1)\n\
#define TILE_N_V2_PER_WARP      ((TILE_N_PER_WARP) / 2)\n\
#define TILE_N_V4_PER_WARP      ((TILE_N_PER_WARP) / 4)\n\
#define TILE_N_V8_PER_WARP      ((TILE_N_PER_WARP) / 8)\n\
\n\
#define TILE_N_V1_PER_THD       ((TILE_N_PER_THD)  / 1)\n\
#define TILE_N_V2_PER_THD       ((TILE_N_PER_THD)  / 2)\n\
#define TILE_N_V4_PER_THD       ((TILE_N_PER_THD)  / 4)\n\
#define TILE_N_V8_PER_THD       ((TILE_N_PER_THD)  / 8)\n\
\n\
#define TILE_N_V1_PER_MMA       ((TILE_N_PER_MMA)  / 1)\n\
#define TILE_N_V2_PER_MMA       ((TILE_N_PER_MMA)  / 2)\n\
#define TILE_N_V4_PER_MMA       ((TILE_N_PER_MMA)  / 4)\n\
#define TILE_N_V8_PER_MMA       ((TILE_N_PER_MMA)  / 8)\n\
\n\
////////////////////////////////////////\n\
// shared memory size macros\n\
////////////////////////////////////////\n\
\n\
#define OUTPUT_STEPS            ((TILE_M_V1_PER_CTA) * (TILE_N_V8_PER_CTA) / CTA_SIZE_IN_THD)\n\
\n\
#if OUTPUT_STEPS < 1\n\
#undef  OUTPUT_STEPS\n\
#define OUTPUT_STEPS  1\n\
#endif\n\
\n\
#define N_ROWS_PER_SMEM_ROW     (SMEM_ROW_V4_SIZE / TILE_N_V8_PER_CTA)\n\
#define K_ROWS_PER_SMEM_ROW     (SMEM_ROW_V4_SIZE / TILE_K_V8_PER_CTA)\n\
\n\
#if N_ROWS_PER_SMEM_ROW < 1\n\
#undef  N_ROWS_PER_SMEM_ROW\n\
#define N_ROWS_PER_SMEM_ROW 1\n\
#endif\n\
\n\
#if K_ROWS_PER_SMEM_ROW < 1\n\
#undef  K_ROWS_PER_SMEM_ROW\n\
#define K_ROWS_PER_SMEM_ROW 1\n\
#endif\n\
\n\
#define OUTPUT_SIZE_X_IN_THD    (TILE_N_V8_PER_CTA)\n\
#define OUTPUT_SIZE_Y_IN_THD    ((CTA_SIZE_IN_THD) / (OUTPUT_SIZE_X_IN_THD))\n\
\n\
////////////////////////////////////////\n\
// k group macros\n\
////////////////////////////////////////\n\
\n\
#define SWITCH_BUFFER(_buf, _size, _base) \\\n\
        { \\\n\
            _buf = ((_buf - _base) ^ _size) + _base; \\\n\
        }\n\
\n\
#define FWD_KGROUP_ODD(_sUv1_read) \\\n\
        { \\\n\
            _sUv1_read = _sUv1_read ^ 0x4; \\\n\
        }\n\
\n\
#define FWD_KGROUP_EVEN(_sUv1_read) \\\n\
        { \\\n\
            _sUv1_read = _sUv1_read ^ 0xc; \\\n\
        }\n\
\n\
#define FWD_KGROUP_STEP1(_sUv1_read)     FWD_KGROUP_ODD(_sUv1_read)\n\
#define FWD_KGROUP_STEP2(_sUv1_read)     FWD_KGROUP_EVEN(_sUv1_read)\n\
#define FWD_KGROUP_STEP3(_sUv1_read)     FWD_KGROUP_ODD(_sUv1_read)\n\
#define FWD_KGROUP_STEP4(_sUv1_read)     FWD_KGROUP_EVEN(_sUv1_read)\n\
\n\
////////////////////////////////////////\n\
// main loop macros\n\
////////////////////////////////////////\n\
\n\
#define   C_ITEMS_PER_THD       ((TILE_M_PER_CTA) * (TILE_N_PER_CTA) / (SET_SIZE_IN_THD * _INT_TO_2HALF_))\n\
#define  HC_ITEMS_PER_THD       ((TILE_M_PER_CTA) * (TILE_N_PER_CTA) / (SET_SIZE_IN_THD))\n\
#define Cv4_ITEMS_PER_THD       ((TILE_M_PER_CTA) * (TILE_N_PER_CTA) / (SET_SIZE_IN_THD * _INT_TO_2HALF_ * _4INT_TO_INT4_))\n\
\n\
#if Cv4_ITEMS_PER_THD < 1\n\
#undef Cv4_ITEMS_PER_THD\n\
#define Cv4_ITEMS_PER_THD 1\n\
#endif\n\
\n\
////////////////////////////////////////\n\
// load A and B from device memory macros\n\
////////////////////////////////////////\n\
\n\
#define REG_dAv4_SIZE           ( ((TILE_M_PER_CTA) * (TILE_K_PER_CTA)) / ((_2HALF_TO_INT_) * (_4INT_TO_INT4_) * (CTA_SIZE_IN_THD)) )\n\
#define REG_dBv4_SIZE           ( ((TILE_N_PER_CTA) * (TILE_K_PER_CTA)) / ((_2HALF_TO_INT_) * (_4INT_TO_INT4_) * (CTA_SIZE_IN_THD)) )\n\
\n\
#if REG_dAv4_SIZE < 1\n\
#undef  REG_dAv4_SIZE\n\
#define REG_dAv4_SIZE 1\n\
#endif\n\
\n\
#if REG_dBv4_SIZE < 1\n\
#undef  REG_dBv4_SIZE\n\
#define REG_dBv4_SIZE 1\n\
#endif\n\
\n\
#define READ_dAv4_STEPS         (REG_dAv4_SIZE)\n\
#define READ_dBv4_STEPS         (REG_dBv4_SIZE)\n\
\n\
////////////////////////////////////////\n\
// shared memory size macros\n\
////////////////////////////////////////\n\
\n\
#define SM_A_SIZE               ((TILE_M_PER_CTA) * (TILE_K_PER_CTA) / (_2HALF_TO_INT_))\n\
#define SM_B_SIZE               ((TILE_K_PER_CTA) * (TILE_N_PER_CTA) / (_2HALF_TO_INT_))\n\
#define SM_C_SIZE               ((TILE_M_PER_CTA) * (TILE_N_PER_CTA) / (_2HALF_TO_INT_))\n\
\n\
#define SM_A_1BUF               (SM_A_SIZE)\n\
#define SM_B_1BUF               (SM_B_SIZE)\n\
#define SM_C_1BUF               (SM_C_SIZE)\n\
\n\
#define SM_A_2BUF               ((SM_A_SIZE) * 2)\n\
#define SM_B_2BUF               ((SM_B_SIZE) * 2)\n\
#define SM_C_2BUF               ((SM_C_SIZE) * 2)\n\
\n\
#define SM_A_V1_1BUF            (SM_A_1BUF)\n\
#define SM_B_V1_1BUF            (SM_B_1BUF)\n\
#define SM_C_V1_1BUF            (SM_C_1BUF)\n\
\n\
#define SM_A_V2_1BUF            ((SM_A_1BUF) / (_2INT_TO_INT2_))\n\
#define SM_B_V2_1BUF            ((SM_B_1BUF) / (_2INT_TO_INT2_))\n\
#define SM_C_V2_1BUF            ((SM_C_1BUF) / (_2INT_TO_INT2_))\n\
\n\
#define SM_A_V4_1BUF            ((SM_A_1BUF) / (_4INT_TO_INT4_))\n\
#define SM_B_V4_1BUF            ((SM_B_1BUF) / (_4INT_TO_INT4_))\n\
#define SM_C_V4_1BUF            ((SM_C_1BUF) / (_4INT_TO_INT4_))\n\
\n\
#define SM_A_V1_2BUF            ((SM_A_V1_1BUF) * 2)\n\
#define SM_B_V1_2BUF            ((SM_B_V1_1BUF) * 2)\n\
#define SM_C_V1_2BUF            ((SM_C_V1_1BUF) * 2)\n\
\n\
#define SM_A_V2_2BUF            ((SM_A_V2_1BUF) * 2)\n\
#define SM_B_V2_2BUF            ((SM_B_V2_1BUF) * 2)\n\
#define SM_C_V2_2BUF            ((SM_C_V2_1BUF) * 2)\n\
\n\
#define SM_A_V4_2BUF            ((SM_A_V4_1BUF) * 2)\n\
#define SM_B_V4_2BUF            ((SM_B_V4_1BUF) * 2)\n\
#define SM_C_V4_2BUF            ((SM_C_V4_1BUF) * 2)\n\
\n\
#define SM_BASE_V4_1BUF         Max((SM_A_V4_1BUF + SM_B_V4_1BUF), (SM_C_V4_1BUF * INTER_SET_REDUCE_RATIO))\n\
#define SM_BASE_V4_2BUF         Max((SM_A_V4_2BUF + SM_B_V4_2BUF), (SM_C_V4_1BUF * INTER_SET_REDUCE_RATIO))\n\
\n\
#define CVT_SM_PTR(smp_base_v1, sm_base_v1) \\\n\
    asm(\"{ .reg .u64 smp_base_v1; cvta.to.shared.u64 smp_base_v1, %1; cvt.u32.u64 %0, smp_base_v1; }\\n\" \\\n\
            : \"=r\"(smp_base_v1) : \"l\"(sm_base_v1));\n\
\n\
#define FWD_LUT(_lut_id) \\\n\
        { \\\n\
            _lut_id = (_lut_id == flt_hw) ? 1 : _lut_id + 1; \\\n\
        }\n\
\n\
////////////////////////////////////////\n\
// bit size macros\n\
////////////////////////////////////////\n\
\n\
#if SET_SIZE_X_IN_WARP == 1\n\
#define SET_SIZE_X_IN_BITS      0\n\
#elif SET_SIZE_X_IN_WARP == 2\n\
#define SET_SIZE_X_IN_BITS      1\n\
#elif SET_SIZE_X_IN_WARP == 4\n\
#define SET_SIZE_X_IN_BITS      2\n\
#elif SET_SIZE_X_IN_WARP == 8\n\
#define SET_SIZE_X_IN_BITS      3\n\
#endif\n\
\n\
#if MMA_SIZE_X_IN_THD == 1\n\
#define MMA_SIZE_X_IN_BITS      0\n\
#elif MMA_SIZE_X_IN_THD == 2\n\
#define MMA_SIZE_X_IN_BITS      1\n\
#elif MMA_SIZE_X_IN_THD == 4\n\
#define MMA_SIZE_X_IN_BITS      2\n\
#elif MMA_SIZE_X_IN_THD == 8\n\
#define MMA_SIZE_X_IN_BITS      3\n\
#endif\n\
\n\
#if SET_SIZE_IN_WARP == 1\n\
#define SET_SIZE_IN_BITS        5\n\
#elif SET_SIZE_IN_WARP == 2\n\
#define SET_SIZE_IN_BITS        6\n\
#elif SET_SIZE_IN_WARP == 4\n\
#define SET_SIZE_IN_BITS        7\n\
#elif SET_SIZE_IN_WARP == 8\n\
#define SET_SIZE_IN_BITS        8\n\
#endif\n\
");
    header_code_.emplace("/mnt/hpc/xusi/ppl.jit/src/ppl/nn/engines/cuda/impls/src/nn/conv/2spk/f1/bound_macros.h", "// Licensed to the Apache Software Foundation (ASF) under one\n\
// or more contributor license agreements.  See the NOTICE file\n\
// distributed with this work for additional information\n\
// regarding copyright ownership.  The ASF licenses this file\n\
// to you under the Apache License, Version 2.0 (the\n\
// \"License\"); you may not use this file except in compliance\n\
// with the License.  You may obtain a copy of the License at\n\
//\n\
//   http://www.apache.org/licenses/LICENSE-2.0\n\
//\n\
// Unless required by applicable law or agreed to in writing,\n\
// software distributed under the License is distributed on an\n\
// \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n\
// KIND, either express or implied.  See the License for the\n\
// specific language governing permissions and limitations\n\
// under the License.\n\
\n\
#define SET_BOUND_FLT1(_in_hw_mask, _in_n_id, _in_h_id, _in_w_id) \\\n\
        { \\\n\
            _in_hw_mask = _in_n_id <  in_num && \\\n\
                        _in_h_id >= 0 && _in_h_id < in_height && \\\n\
                        _in_w_id >= 0 && _in_w_id < in_width; \\\n\
        }\n\
\n\
#define FWD_FLT1(_flt_c_v8_id, _flt_c_v8_valid) \\\n\
        { \\\n\
            flt_c_v8_id   += TILE_K_V8_PER_CTA; \\\n\
            _flt_c_v8_valid = _flt_c_v8_id < flt_c_v8_end; \\\n\
        }\n\
\n\
#define FWD_FLT(_flt_c_v8_id, _flt_c_v8_valid)    FWD_FLT1(_flt_c_v8_id, _flt_c_v8_valid)\n\
");
    header_code_.emplace("/mnt/hpc/xusi/ppl.jit/src/ppl/nn/engines/cuda/impls/src/nn/conv/2spk/f3/bound_macros.h", "// Licensed to the Apache Software Foundation (ASF) under one\n\
// or more contributor license agreements.  See the NOTICE file\n\
// distributed with this work for additional information\n\
// regarding copyright ownership.  The ASF licenses this file\n\
// to you under the Apache License, Version 2.0 (the\n\
// \"License\"); you may not use this file except in compliance\n\
// with the License.  You may obtain a copy of the License at\n\
//\n\
//   http://www.apache.org/licenses/LICENSE-2.0\n\
//\n\
// Unless required by applicable law or agreed to in writing,\n\
// software distributed under the License is distributed on an\n\
// \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n\
// KIND, either express or implied.  See the License for the\n\
// specific language governing permissions and limitations\n\
// under the License.\n\
\n\
#define SET_BOUND_FLT3(_in_hw_mask, _in_n_id, _in_h_id, _in_w_id) \\\n\
        { \\\n\
            if(_in_n_id < in_num) \\\n\
            { \\\n\
                _in_hw_mask = 0xffffffff; \\\n\
                if(_in_h_id < 0 || _in_h_id >= in_height) _in_hw_mask = _in_hw_mask & 0xfffffff8; \\\n\
                if(_in_w_id < 0 || _in_w_id >= in_width)  _in_hw_mask = _in_hw_mask & 0xffffffb6; \\\n\
                \\\n\
                _in_h_id += hole_height; \\\n\
                _in_w_id += hole_width; \\\n\
                \\\n\
                if(_in_h_id < 0 || _in_h_id >= in_height) _in_hw_mask = _in_hw_mask & 0xffffffc7; \\\n\
                if(_in_w_id < 0 || _in_w_id >= in_width)  _in_hw_mask = _in_hw_mask & 0xffffff6d; \\\n\
                \\\n\
                _in_h_id += hole_height; \\\n\
                _in_w_id += hole_width; \\\n\
                \\\n\
                if(_in_h_id < 0 || _in_h_id >= in_height)  _in_hw_mask = _in_hw_mask & 0xfffffe3f; \\\n\
                if(_in_w_id < 0 || _in_w_id >= in_width)   _in_hw_mask = _in_hw_mask & 0xfffffedb; \\\n\
            } else { \\\n\
                _in_hw_mask = 0x0; \\\n\
            } \\\n\
        }\n\
\n\
#define FWD_FLT3(_flt_hw_id, _flt_hw_bid, _flt_c_v8_id, _flt_c_v8_valid) \\\n\
        { \\\n\
            if(_flt_hw_id == 8) \\\n\
            { \\\n\
                _flt_hw_id = 0; \\\n\
                _flt_c_v8_id += TILE_K_V8_PER_CTA; \\\n\
                \\\n\
                _flt_c_v8_valid = _flt_c_v8_id < flt_c_v8_end; \\\n\
            } else { \\\n\
                _flt_hw_id = _flt_hw_id + 1; \\\n\
            } \\\n\
            \\\n\
            _flt_hw_bid = (0x1 << _flt_hw_id); \\\n\
        }\n\
\n\
#define FWD_FLT(_flt_hw_id, _flt_hw_bid, _flt_c_v8_id, _flt_c_v8_valid)    FWD_FLT3(_flt_hw_id, _flt_hw_bid, _flt_c_v8_id, _flt_c_v8_valid)\n\
");
    header_code_.emplace("/mnt/hpc/xusi/ppl.jit/src/ppl/nn/engines/cuda/impls/src/nn/conv/2spk/fn/bound_macros.h", "// Licensed to the Apache Software Foundation (ASF) under one\n\
// or more contributor license agreements.  See the NOTICE file\n\
// distributed with this work for additional information\n\
// regarding copyright ownership.  The ASF licenses this file\n\
// to you under the Apache License, Version 2.0 (the\n\
// \"License\"); you may not use this file except in compliance\n\
// with the License.  You may obtain a copy of the License at\n\
//\n\
//   http://www.apache.org/licenses/LICENSE-2.0\n\
//\n\
// Unless required by applicable law or agreed to in writing,\n\
// software distributed under the License is distributed on an\n\
// \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n\
// KIND, either express or implied.  See the License for the\n\
// specific language governing permissions and limitations\n\
// under the License.\n\
\n\
////////////////////////////////////////\n\
// filter shifting macros\n\
////////////////////////////////////////\n\
\n\
#define FWD_FLT_SIZE1(_flt_h_id, _flt_w_id, _flt_c_v8_id, _flt_c_v8_valid) \\\n\
        { \\\n\
            _flt_w_id++; \\\n\
            in_w_id[0] += hole_width; \\\n\
            \\\n\
            if(_flt_w_id == flt_width) \\\n\
            {\\\n\
                _flt_w_id = 0; \\\n\
                in_w_id[0] = in_w_start[0]; \\\n\
                _flt_h_id++; \\\n\
                in_h_id[0] += hole_height; \\\n\
            } \\\n\
            \\\n\
            if(_flt_h_id == flt_height) \\\n\
            { \\\n\
                _flt_h_id = 0;   \\\n\
                in_h_id[0] = in_h_start[0]; \\\n\
                \\\n\
                _flt_c_v8_id += TILE_K_V8_PER_CTA; \\\n\
                \\\n\
                _flt_c_v8_valid = _flt_c_v8_id < flt_c_v8_end; \\\n\
            } \\\n\
        }\n\
\n\
#define FWD_FLT_SIZE2(_flt_h_id, _flt_w_id, _flt_c_v8_id, _flt_c_v8_valid) \\\n\
        { \\\n\
            _flt_w_id++; \\\n\
            in_w_id[0] += hole_width;        in_w_id[1] += hole_width; \\\n\
            \\\n\
            if(_flt_w_id == flt_width) \\\n\
            {\\\n\
                _flt_w_id = 0; \\\n\
                in_w_id[0] = in_w_start[0];  in_w_id[1] = in_w_start[1]; \\\n\
                _flt_h_id++; \\\n\
                in_h_id[0] += hole_height;   in_h_id[1] += hole_height; \\\n\
            } \\\n\
            \\\n\
            if(_flt_h_id == flt_height) \\\n\
            { \\\n\
                _flt_h_id = 0;   \\\n\
                in_h_id[0] = in_h_start[0];  in_h_id[1] = in_h_start[1]; \\\n\
                \\\n\
                _flt_c_v8_id += TILE_K_V8_PER_CTA; \\\n\
                \\\n\
                _flt_c_v8_valid = _flt_c_v8_id < flt_c_v8_end; \\\n\
            } \\\n\
        }\n\
\n\
#define FWD_FLT_SIZE4(_flt_h_id, _flt_w_id, _flt_c_v8_id, _flt_c_v8_valid) \\\n\
        { \\\n\
            _flt_w_id++; \\\n\
            in_w_id[0] += hole_width;        in_w_id[1] += hole_width;   in_w_id[2] += hole_width;  in_w_id[3] += hole_width; \\\n\
            \\\n\
            if(_flt_w_id == flt_width) \\\n\
            { \\\n\
                _flt_w_id = 0; \\\n\
                in_w_id[0] = in_w_start[0];  in_w_id[1] = in_w_start[1]; in_w_id[2] = in_w_start[2];  in_w_id[3] = in_w_start[3]; \\\n\
                _flt_h_id++; \\\n\
                in_h_id[0] += hole_height;   in_h_id[1] += hole_height;  in_h_id[2] += hole_height;   in_h_id[3] += hole_height; \\\n\
            } \\\n\
            \\\n\
            if(_flt_h_id == flt_height) \\\n\
            { \\\n\
                _flt_h_id = 0;   \\\n\
                in_h_id[0] = in_h_start[0];  in_h_id[1] = in_h_start[1]; in_h_id[2] = in_h_start[2];  in_h_id[3] = in_h_start[3]; \\\n\
                \\\n\
                _flt_c_v8_id += TILE_K_V8_PER_CTA; \\\n\
                \\\n\
                _flt_c_v8_valid = _flt_c_v8_id < flt_c_v8_end; \\\n\
            } \\\n\
        }\n\
\n\
#define FWD_FLT_SIZE8(_flt_h_id, _flt_w_id, _flt_c_v8_id, _flt_c_v8_valid) \\\n\
        { \\\n\
            _flt_w_id++; \\\n\
            in_w_id[0] += hole_width;        in_w_id[1] += hole_width;   in_w_id[2] += hole_width;  in_w_id[3] += hole_width; \\\n\
            in_w_id[4] += hole_width;        in_w_id[5] += hole_width;   in_w_id[6] += hole_width;  in_w_id[7] += hole_width; \\\n\
            \\\n\
            if(_flt_w_id == flt_width) \\\n\
            { \\\n\
                _flt_w_id = 0; \\\n\
                in_w_id[0] = in_w_start[0];  in_w_id[1] = in_w_start[1]; in_w_id[2] = in_w_start[2];  in_w_id[3] = in_w_start[3]; \\\n\
                in_w_id[4] = in_w_start[4];  in_w_id[5] = in_w_start[5]; in_w_id[6] = in_w_start[6];  in_w_id[7] = in_w_start[7]; \\\n\
                _flt_h_id++; \\\n\
                in_h_id[0] += hole_height;   in_h_id[1] += hole_height;  in_h_id[2] += hole_height;   in_h_id[3] += hole_height; \\\n\
                in_h_id[4] += hole_height;   in_h_id[5] += hole_height;  in_h_id[6] += hole_height;   in_h_id[7] += hole_height; \\\n\
            } \\\n\
            \\\n\
            if(_flt_h_id == flt_height) \\\n\
            { \\\n\
                _flt_h_id = 0;   \\\n\
                in_h_id[0] = in_h_start[0];  in_h_id[1] = in_h_start[1]; in_h_id[2] = in_h_start[2];  in_h_id[3] = in_h_start[3]; \\\n\
                in_h_id[4] = in_h_start[4];  in_h_id[5] = in_h_start[5]; in_h_id[6] = in_h_start[6];  in_h_id[7] = in_h_start[7]; \\\n\
                \\\n\
                _flt_c_v8_id += TILE_K_V8_PER_CTA; \\\n\
                \\\n\
                _flt_c_v8_valid = _flt_c_v8_id < flt_c_v8_end; \\\n\
            } \\\n\
        }\n\
\n\
#define FWD_FLT_SIZE16(_flt_h_id, _flt_w_id, _flt_c_v8_id, _flt_c_v8_valid) \\\n\
        { \\\n\
            _flt_w_id++; \\\n\
            in_w_id[0]  += hole_width;        in_w_id[1]  += hole_width;   in_w_id[2]  += hole_width;  in_w_id[3]  += hole_width; \\\n\
            in_w_id[4]  += hole_width;        in_w_id[5]  += hole_width;   in_w_id[6]  += hole_width;  in_w_id[7]  += hole_width; \\\n\
            in_w_id[8]  += hole_width;        in_w_id[9]  += hole_width;   in_w_id[10] += hole_width;  in_w_id[11] += hole_width; \\\n\
            in_w_id[12] += hole_width;        in_w_id[13] += hole_width;   in_w_id[14] += hole_width;  in_w_id[15] += hole_width; \\\n\
            \\\n\
            if(_flt_w_id == flt_width) \\\n\
            { \\\n\
                _flt_w_id = 0; \\\n\
                in_w_id[0]  = in_w_start[0];   in_w_id[1]  = in_w_start[1];  in_w_id[2]  = in_w_start[2];   in_w_id[3]  = in_w_start[3]; \\\n\
                in_w_id[4]  = in_w_start[4];   in_w_id[5]  = in_w_start[5];  in_w_id[6]  = in_w_start[6];   in_w_id[7]  = in_w_start[7]; \\\n\
                in_w_id[8]  = in_w_start[8];   in_w_id[9]  = in_w_start[9];  in_w_id[10] = in_w_start[10];  in_w_id[11] = in_w_start[11]; \\\n\
                in_w_id[12] = in_w_start[12];  in_w_id[13] = in_w_start[13]; in_w_id[14] = in_w_start[14];  in_w_id[15] = in_w_start[15]; \\\n\
                _flt_h_id++; \\\n\
                in_h_id[0]  += hole_height;        in_h_id[1]  += hole_height;   in_h_id[2]  += hole_height;  in_h_id[3]  += hole_height; \\\n\
                in_h_id[4]  += hole_height;        in_h_id[5]  += hole_height;   in_h_id[6]  += hole_height;  in_h_id[7]  += hole_height; \\\n\
                in_h_id[8]  += hole_height;        in_h_id[9]  += hole_height;   in_h_id[10] += hole_height;  in_h_id[11] += hole_height; \\\n\
                in_h_id[12] += hole_height;        in_h_id[13] += hole_height;   in_h_id[14] += hole_height;  in_h_id[15] += hole_height; \\\n\
            } \\\n\
            \\\n\
            if(_flt_h_id == flt_height) \\\n\
            { \\\n\
                _flt_h_id = 0;   \\\n\
                in_h_id[0]  = in_h_start[0];   in_h_id[1]  = in_h_start[1];  in_h_id[2]  = in_h_start[2];   in_h_id[3]  = in_h_start[3]; \\\n\
                in_h_id[4]  = in_h_start[4];   in_h_id[5]  = in_h_start[5];  in_h_id[6]  = in_h_start[6];   in_h_id[7]  = in_h_start[7]; \\\n\
                in_h_id[8]  = in_h_start[8];   in_h_id[9]  = in_h_start[9];  in_h_id[10] = in_h_start[10];  in_h_id[11] = in_h_start[11]; \\\n\
                in_h_id[12] = in_h_start[12];  in_h_id[13] = in_h_start[13]; in_h_id[14] = in_h_start[14];  in_h_id[15] = in_h_start[15]; \\\n\
                \\\n\
                _flt_c_v8_id += TILE_K_V8_PER_CTA; \\\n\
                \\\n\
                _flt_c_v8_valid = _flt_c_v8_id < flt_c_v8_end; \\\n\
            } \\\n\
        }\n\
");
    header_code_.emplace("/mnt/hpc/xusi/ppl.jit/src/ppl/nn/engines/cuda/impls/src/nn/conv/2spk/fs/bound_macros.h", "// Licensed to the Apache Software Foundation (ASF) under one\n\
// or more contributor license agreements.  See the NOTICE file\n\
// distributed with this work for additional information\n\
// regarding copyright ownership.  The ASF licenses this file\n\
// to you under the Apache License, Version 2.0 (the\n\
// \"License\"); you may not use this file except in compliance\n\
// with the License.  You may obtain a copy of the License at\n\
//\n\
//   http://www.apache.org/licenses/LICENSE-2.0\n\
//\n\
// Unless required by applicable law or agreed to in writing,\n\
// software distributed under the License is distributed on an\n\
// \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n\
// KIND, either express or implied.  See the License for the\n\
// specific language governing permissions and limitations\n\
// under the License.\n\
\n\
#define SET_BOUND_FLT1(_in_hw_mask, _in_n_id, _in_h_id, _in_w_id) \\\n\
        { \\\n\
            _in_hw_mask = _in_n_id <  in_num && \\\n\
                        _in_h_id >= 0 && _in_h_id < in_height && \\\n\
                        _in_w_id >= 0 && _in_w_id < in_width; \\\n\
        }\n\
\n\
#define FWD_FLT1(_flt_c_v8_id, _flt_c_v8_valid) \\\n\
        { \\\n\
            flt_c_v8_id   += TILE_K_V8_PER_CTA; \\\n\
            _flt_c_v8_valid = _flt_c_v8_id < flt_c_v8_end; \\\n\
        }\n\
\n\
#define FWD_FLT(_flt_c_v8_id, _flt_c_v8_valid)    FWD_FLT1(_flt_c_v8_id, _flt_c_v8_valid)\n\
");
    header_code_.emplace("/mnt/hpc/xusi/ppl.jit/src/ppl/nn/engines/cuda/impls/src/nn/conv/2spk/common/ldsm_macros.h", "// Licensed to the Apache Software Foundation (ASF) under one\n\
// or more contributor license agreements.  See the NOTICE file\n\
// distributed with this work for additional information\n\
// regarding copyright ownership.  The ASF licenses this file\n\
// to you under the Apache License, Version 2.0 (the\n\
// \"License\"); you may not use this file except in compliance\n\
// with the License.  You may obtain a copy of the License at\n\
//\n\
//   http://www.apache.org/licenses/LICENSE-2.0\n\
//\n\
// Unless required by applicable law or agreed to in writing,\n\
// software distributed under the License is distributed on an\n\
// \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n\
// KIND, either express or implied.  See the License for the\n\
// specific language governing permissions and limitations\n\
// under the License.\n\
\n\
////////////////////////////////////////\n\
// ldsm macros\n\
////////////////////////////////////////\n\
\n\
#define LDSM_ROW_X1_OPCODE \\\n\
        \"ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\\n\"\n\
\n\
#define LDSM_ROW_X2_OPCODE \\\n\
        \"ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];\\n\"\n\
\n\
#define LDSM_ROW_X4_OPCODE \\\n\
        \"ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\\n\"\n\
\n\
#define LDSM_ROW_X1_INST(_x0, _addr) \\\n\
        asm volatile(LDSM_ROW_X1_OPCODE:   \"=r\"(_x0)   : \"r\"(_addr));\n\
\n\
#define LDSM_ROW_X2_INST(_x0, _x1, _addr) \\\n\
        asm volatile(LDSM_ROW_X2_OPCODE:   \"=r\"(_x0),   \"=r\"(_x1): \"r\"(_addr));\n\
\n\
#define LDSM_ROW_X4_INST(_x0, _x1, _x2, _x3, _addr) \\\n\
        asm volatile(LDSM_ROW_X4_OPCODE:   \"=r\"(_x0),   \"=r\"(_x1),  \"=r\"(_x2),   \"=r\"(_x3): \"r\"(_addr));\n\
\n\
#define LDSM_COL_X1_OPCODE \\\n\
        \"ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\\n\"\n\
\n\
#define LDSM_COL_X2_OPCODE \\\n\
        \"ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\\n\"\n\
\n\
#define LDSM_COL_X4_OPCODE \\\n\
        \"ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\\n\"\n\
\n\
#define LDSM_COL_X1_INST(_x0, _addr) \\\n\
        asm volatile(LDSM_COL_X1_OPCODE:   \"=r\"(_x0)   : \"r\"(_addr));\n\
\n\
#define LDSM_COL_X2_INST(_x0, _x1, _addr) \\\n\
        asm volatile(LDSM_COL_X2_OPCODE:   \"=r\"(_x0),   \"=r\"(_x1): \"r\"(_addr));\n\
\n\
#define LDSM_COL_X4_INST(_x0, _x1, _x2, _x3, _addr) \\\n\
        asm volatile(LDSM_COL_X4_OPCODE:   \"=r\"(_x0),   \"=r\"(_x1),  \"=r\"(_x2),   \"=r\"(_x3): \"r\"(_addr));\n\
");
    header_code_.emplace("/mnt/hpc/xusi/ppl.jit/src/ppl/nn/engines/cuda/impls/src/nn/conv/2spk/f1/dmem_macros.h", "// Licensed to the Apache Software Foundation (ASF) under one\n\
// or more contributor license agreements.  See the NOTICE file\n\
// distributed with this work for additional information\n\
// regarding copyright ownership.  The ASF licenses this file\n\
// to you under the Apache License, Version 2.0 (the\n\
// \"License\"); you may not use this file except in compliance\n\
// with the License.  You may obtain a copy of the License at\n\
//\n\
//   http://www.apache.org/licenses/LICENSE-2.0\n\
//\n\
// Unless required by applicable law or agreed to in writing,\n\
// software distributed under the License is distributed on an\n\
// \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n\
// KIND, either express or implied.  See the License for the\n\
// specific language governing permissions and limitations\n\
// under the License.\n\
\n\
////////////////////////////////////////\n\
// load dB macros\n\
////////////////////////////////////////\n\
\n\
#define LOAD_dBv4_SIZE_16TH(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            if(tid < (CTA_SIZE_IN_THD / 16))  \\\n\
                _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
            \\\n\
            _dBv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE_8TH(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            if(tid < (CTA_SIZE_IN_THD / 8))  \\\n\
                _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
            \\\n\
            _dBv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE_QTR(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            if(tid < (CTA_SIZE_IN_THD / 4))  \\\n\
                _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
            \\\n\
            _dBv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE_HALF(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            if(tid < (CTA_SIZE_IN_THD / 2))  \\\n\
                _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
            \\\n\
            _dBv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE1(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
            \\\n\
            _dBv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE2(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
            _regB[1] = (_flt_n_valid[1] && _flt_c_v8_valid) ? _dB[ _dBv4_off[1] ] : ZEROv4;\\\n\
            \\\n\
            _dBv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
            _dBv4_off[1] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE4(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
            _regB[1] = (_flt_n_valid[1] && _flt_c_v8_valid) ? _dB[ _dBv4_off[1] ] : ZEROv4;\\\n\
            _regB[2] = (_flt_n_valid[2] && _flt_c_v8_valid) ? _dB[ _dBv4_off[2] ] : ZEROv4;\\\n\
            _regB[3] = (_flt_n_valid[3] && _flt_c_v8_valid) ? _dB[ _dBv4_off[3] ] : ZEROv4;\\\n\
            \\\n\
            _dBv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
            _dBv4_off[1] += TILE_K_V8_PER_CTA; \\\n\
            _dBv4_off[2] += TILE_K_V8_PER_CTA; \\\n\
            _dBv4_off[3] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE8(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
            _regB[1] = (_flt_n_valid[1] && _flt_c_v8_valid) ? _dB[ _dBv4_off[1] ] : ZEROv4;\\\n\
            _regB[2] = (_flt_n_valid[2] && _flt_c_v8_valid) ? _dB[ _dBv4_off[2] ] : ZEROv4;\\\n\
            _regB[3] = (_flt_n_valid[3] && _flt_c_v8_valid) ? _dB[ _dBv4_off[3] ] : ZEROv4;\\\n\
            _regB[4] = (_flt_n_valid[4] && _flt_c_v8_valid) ? _dB[ _dBv4_off[4] ] : ZEROv4;\\\n\
            _regB[5] = (_flt_n_valid[5] && _flt_c_v8_valid) ? _dB[ _dBv4_off[5] ] : ZEROv4;\\\n\
            _regB[6] = (_flt_n_valid[6] && _flt_c_v8_valid) ? _dB[ _dBv4_off[6] ] : ZEROv4;\\\n\
            _regB[7] = (_flt_n_valid[7] && _flt_c_v8_valid) ? _dB[ _dBv4_off[7] ] : ZEROv4;\\\n\
            \\\n\
            _dBv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
            _dBv4_off[1] += TILE_K_V8_PER_CTA; \\\n\
            _dBv4_off[2] += TILE_K_V8_PER_CTA; \\\n\
            _dBv4_off[3] += TILE_K_V8_PER_CTA; \\\n\
            _dBv4_off[4] += TILE_K_V8_PER_CTA; \\\n\
            _dBv4_off[5] += TILE_K_V8_PER_CTA; \\\n\
            _dBv4_off[6] += TILE_K_V8_PER_CTA; \\\n\
            _dBv4_off[7] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define SET_dBv4_BOUND(_step_id, _dBv4_off, _flt_n_valid) \\\n\
        { \\\n\
            int _flt_n_id  =  cta_idx *  TILE_N_PER_CTA + \\\n\
                             _step_id * (TILE_N_PER_CTA / READ_dBv4_STEPS) + \\\n\
                             ldg_idy; \\\n\
            \\\n\
            _flt_n_valid  =  _flt_n_id < num_flt_per_grp_pad; \\\n\
            \\\n\
            _dBv4_off  =   grp_id   * flt_hw * num_chl_per_grp_pad_v8 * num_flt_per_grp_pad + \\\n\
                          _flt_n_id * flt_hw * num_chl_per_grp_pad_v8 + \\\n\
                           flt_c_v8_id; \\\n\
        }\n\
\n\
////////////////////////////////////////\n\
// load dA macros\n\
////////////////////////////////////////\n\
\n\
#define SET_dAv4_BOUND(_step_id, _dAv4_off, _in_hw_valid) \\\n\
        { \\\n\
            int _out_nhw_id    =  cta_idy *  TILE_M_PER_CTA + \\\n\
                                 _step_id * (TILE_M_PER_CTA / READ_dAv4_STEPS) + \\\n\
                                 ldg_idy; \\\n\
            \\\n\
            int _out_w_id =  (_out_nhw_id % out_width); \\\n\
            int _out_h_id =  (_out_nhw_id / out_width) % out_height; \\\n\
            \\\n\
            int _in_n_id  =   _out_nhw_id / out_hw; \\\n\
            int _in_h_id  =     _out_h_id * stride_height; \\\n\
            int _in_w_id  =     _out_w_id * stride_width; \\\n\
            \\\n\
            _in_h_id =  _in_h_id - pad_height; \\\n\
            _in_w_id =  _in_w_id - pad_width;  \\\n\
            \\\n\
            _dAv4_off  =  (_in_n_id  * in_hw + _in_h_id  * in_width + _in_w_id) * num_chl_per_grp_pad_v8 * num_grp + \\\n\
                           grp_id   * num_chl_per_grp_pad_v8 + \\\n\
                           flt_c_v8_id; \\\n\
            \\\n\
            SET_BOUND_FLT1(_in_hw_valid, _in_n_id, _in_h_id, _in_w_id); \\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE_16TH(_regA, _dA, _dAv4_off, _in_c_v8_valid, _in_hw_valid) \\\n\
        { \\\n\
            if(tid < (CTA_SIZE_IN_THD / 16))  \\\n\
                _regA[0] = (_in_hw_valid[0] && _in_c_v8_valid) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
            \\\n\
            _dAv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE_8TH(_regA, _dA, _dAv4_off, _in_c_v8_valid, _in_hw_valid) \\\n\
        { \\\n\
            if(tid < (CTA_SIZE_IN_THD / 8))  \\\n\
                _regA[0] = (_in_hw_valid[0] && _in_c_v8_valid) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
            \\\n\
            _dAv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE_QTR(_regA, _dA, _dAv4_off, _in_c_v8_valid, _in_hw_valid) \\\n\
        { \\\n\
            if(tid < (CTA_SIZE_IN_THD / 4))  \\\n\
                _regA[0] = (_in_hw_valid[0] && _in_c_v8_valid) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
            \\\n\
            _dAv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE_HALF(_regA, _dA, _dAv4_off, _in_c_v8_valid, _in_hw_valid) \\\n\
        { \\\n\
            if(tid < (CTA_SIZE_IN_THD / 2))  \\\n\
                _regA[0] = (_in_hw_valid[0] && _in_c_v8_valid) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
            \\\n\
            _dAv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE1(_regA, _dA, _dAv4_off, _in_c_v8_valid, _in_hw_valid) \\\n\
        { \\\n\
            _regA[0] = (_in_hw_valid[0] && _in_c_v8_valid) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
            \\\n\
            _dAv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE2(_regA, _dA, _dAv4_off, _in_c_v8_valid, _in_hw_valid) \\\n\
        { \\\n\
            _regA[0] = (_in_hw_valid[0] && _in_c_v8_valid) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
            _regA[1] = (_in_hw_valid[1] && _in_c_v8_valid) ? _dA[ _dAv4_off[1] ] : ZEROv4;\\\n\
            \\\n\
            _dAv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[1] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE4(_regA, _dA, _dAv4_off, _in_c_v8_valid, _in_hw_valid) \\\n\
        { \\\n\
            _regA[0] = (_in_hw_valid[0] && _in_c_v8_valid) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
            _regA[1] = (_in_hw_valid[1] && _in_c_v8_valid) ? _dA[ _dAv4_off[1] ] : ZEROv4;\\\n\
            _regA[2] = (_in_hw_valid[2] && _in_c_v8_valid) ? _dA[ _dAv4_off[2] ] : ZEROv4;\\\n\
            _regA[3] = (_in_hw_valid[3] && _in_c_v8_valid) ? _dA[ _dAv4_off[3] ] : ZEROv4;\\\n\
            \\\n\
            _dAv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[1] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[2] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[3] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE8(_regA, _dA, _dAv4_off, _in_c_v8_valid, _in_hw_valid) \\\n\
        { \\\n\
            _regA[0] = (_in_hw_valid[0] && _in_c_v8_valid) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
            _regA[1] = (_in_hw_valid[1] && _in_c_v8_valid) ? _dA[ _dAv4_off[1] ] : ZEROv4;\\\n\
            _regA[2] = (_in_hw_valid[2] && _in_c_v8_valid) ? _dA[ _dAv4_off[2] ] : ZEROv4;\\\n\
            _regA[3] = (_in_hw_valid[3] && _in_c_v8_valid) ? _dA[ _dAv4_off[3] ] : ZEROv4;\\\n\
            _regA[4] = (_in_hw_valid[4] && _in_c_v8_valid) ? _dA[ _dAv4_off[4] ] : ZEROv4;\\\n\
            _regA[5] = (_in_hw_valid[5] && _in_c_v8_valid) ? _dA[ _dAv4_off[5] ] : ZEROv4;\\\n\
            _regA[6] = (_in_hw_valid[6] && _in_c_v8_valid) ? _dA[ _dAv4_off[6] ] : ZEROv4;\\\n\
            _regA[7] = (_in_hw_valid[7] && _in_c_v8_valid) ? _dA[ _dAv4_off[7] ] : ZEROv4;\\\n\
            \\\n\
            _dAv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[1] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[2] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[3] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[4] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[5] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[6] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[7] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE16(_regA, _dA, _dAv4_off, _in_c_v8_valid, _in_hw_valid) \\\n\
        { \\\n\
            _regA[0]  = (_in_hw_valid[0]  && _in_c_v8_valid) ? _dA[ _dAv4_off[0]  ] : ZEROv4;\\\n\
            _regA[1]  = (_in_hw_valid[1]  && _in_c_v8_valid) ? _dA[ _dAv4_off[1]  ] : ZEROv4;\\\n\
            _regA[2]  = (_in_hw_valid[2]  && _in_c_v8_valid) ? _dA[ _dAv4_off[2]  ] : ZEROv4;\\\n\
            _regA[3]  = (_in_hw_valid[3]  && _in_c_v8_valid) ? _dA[ _dAv4_off[3]  ] : ZEROv4;\\\n\
            _regA[4]  = (_in_hw_valid[4]  && _in_c_v8_valid) ? _dA[ _dAv4_off[4]  ] : ZEROv4;\\\n\
            _regA[5]  = (_in_hw_valid[5]  && _in_c_v8_valid) ? _dA[ _dAv4_off[5]  ] : ZEROv4;\\\n\
            _regA[6]  = (_in_hw_valid[6]  && _in_c_v8_valid) ? _dA[ _dAv4_off[6]  ] : ZEROv4;\\\n\
            _regA[7]  = (_in_hw_valid[7]  && _in_c_v8_valid) ? _dA[ _dAv4_off[7]  ] : ZEROv4;\\\n\
            _regA[8]  = (_in_hw_valid[8]  && _in_c_v8_valid) ? _dA[ _dAv4_off[8]  ] : ZEROv4;\\\n\
            _regA[9]  = (_in_hw_valid[9]  && _in_c_v8_valid) ? _dA[ _dAv4_off[9]  ] : ZEROv4;\\\n\
            _regA[10] = (_in_hw_valid[10] && _in_c_v8_valid) ? _dA[ _dAv4_off[10] ] : ZEROv4;\\\n\
            _regA[11] = (_in_hw_valid[11] && _in_c_v8_valid) ? _dA[ _dAv4_off[11] ] : ZEROv4;\\\n\
            _regA[12] = (_in_hw_valid[12] && _in_c_v8_valid) ? _dA[ _dAv4_off[12] ] : ZEROv4;\\\n\
            _regA[13] = (_in_hw_valid[13] && _in_c_v8_valid) ? _dA[ _dAv4_off[13] ] : ZEROv4;\\\n\
            _regA[14] = (_in_hw_valid[14] && _in_c_v8_valid) ? _dA[ _dAv4_off[14] ] : ZEROv4;\\\n\
            _regA[15] = (_in_hw_valid[15] && _in_c_v8_valid) ? _dA[ _dAv4_off[15] ] : ZEROv4;\\\n\
            \\\n\
            _dAv4_off[0]  += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[1]  += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[2]  += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[3]  += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[4]  += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[5]  += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[6]  += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[7]  += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[8]  += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[9]  += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[10] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[11] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[12] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[13] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[14] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[15] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
");
    header_code_.emplace("/mnt/hpc/xusi/ppl.jit/src/ppl/nn/engines/cuda/impls/src/nn/conv/2spk/f3/dmem_macros.h", "// Licensed to the Apache Software Foundation (ASF) under one\n\
// or more contributor license agreements.  See the NOTICE file\n\
// distributed with this work for additional information\n\
// regarding copyright ownership.  The ASF licenses this file\n\
// to you under the Apache License, Version 2.0 (the\n\
// \"License\"); you may not use this file except in compliance\n\
// with the License.  You may obtain a copy of the License at\n\
//\n\
//   http://www.apache.org/licenses/LICENSE-2.0\n\
//\n\
// Unless required by applicable law or agreed to in writing,\n\
// software distributed under the License is distributed on an\n\
// \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n\
// KIND, either express or implied.  See the License for the\n\
// specific language governing permissions and limitations\n\
// under the License.\n\
\n\
////////////////////////////////////////\n\
// load dB macros\n\
////////////////////////////////////////\n\
\n\
#define LOAD_dBv4_SIZE_16TH(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            _dBv4_off[0] += flt_lut.idx[lut_id]; \\\n\
            \\\n\
            if(tid < (CTA_SIZE_IN_THD / 16))  \\\n\
                _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE_8TH(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            _dBv4_off[0] += flt_lut.idx[lut_id]; \\\n\
            \\\n\
            if(tid < (CTA_SIZE_IN_THD / 8))  \\\n\
                _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE_QTR(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            _dBv4_off[0] += flt_lut.idx[lut_id]; \\\n\
            \\\n\
            if(tid < (CTA_SIZE_IN_THD / 4))  \\\n\
                _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE_HALF(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            _dBv4_off[0] += flt_lut.idx[lut_id]; \\\n\
            \\\n\
            if(tid < (CTA_SIZE_IN_THD / 2))  \\\n\
                _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE1(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            _dBv4_off[0] += flt_lut.idx[lut_id]; \\\n\
            \\\n\
            _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE2(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            _dBv4_off[0] += flt_lut.idx[lut_id]; \\\n\
            _dBv4_off[1] += flt_lut.idx[lut_id]; \\\n\
            \\\n\
            _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
            _regB[1] = (_flt_n_valid[1] && _flt_c_v8_valid) ? _dB[ _dBv4_off[1] ] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE4(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            _dBv4_off[0] += flt_lut.idx[lut_id]; \\\n\
            _dBv4_off[1] += flt_lut.idx[lut_id]; \\\n\
            _dBv4_off[2] += flt_lut.idx[lut_id]; \\\n\
            _dBv4_off[3] += flt_lut.idx[lut_id]; \\\n\
            \\\n\
            _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
            _regB[1] = (_flt_n_valid[1] && _flt_c_v8_valid) ? _dB[ _dBv4_off[1] ] : ZEROv4;\\\n\
            _regB[2] = (_flt_n_valid[2] && _flt_c_v8_valid) ? _dB[ _dBv4_off[2] ] : ZEROv4;\\\n\
            _regB[3] = (_flt_n_valid[3] && _flt_c_v8_valid) ? _dB[ _dBv4_off[3] ] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE8(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            _dBv4_off[0] += flt_lut.idx[lut_id]; \\\n\
            _dBv4_off[1] += flt_lut.idx[lut_id]; \\\n\
            _dBv4_off[2] += flt_lut.idx[lut_id]; \\\n\
            _dBv4_off[3] += flt_lut.idx[lut_id]; \\\n\
            _dBv4_off[4] += flt_lut.idx[lut_id]; \\\n\
            _dBv4_off[5] += flt_lut.idx[lut_id]; \\\n\
            _dBv4_off[6] += flt_lut.idx[lut_id]; \\\n\
            _dBv4_off[7] += flt_lut.idx[lut_id]; \\\n\
            \\\n\
            _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
            _regB[1] = (_flt_n_valid[1] && _flt_c_v8_valid) ? _dB[ _dBv4_off[1] ] : ZEROv4;\\\n\
            _regB[2] = (_flt_n_valid[2] && _flt_c_v8_valid) ? _dB[ _dBv4_off[2] ] : ZEROv4;\\\n\
            _regB[3] = (_flt_n_valid[3] && _flt_c_v8_valid) ? _dB[ _dBv4_off[3] ] : ZEROv4;\\\n\
            _regB[4] = (_flt_n_valid[4] && _flt_c_v8_valid) ? _dB[ _dBv4_off[4] ] : ZEROv4;\\\n\
            _regB[5] = (_flt_n_valid[5] && _flt_c_v8_valid) ? _dB[ _dBv4_off[5] ] : ZEROv4;\\\n\
            _regB[6] = (_flt_n_valid[6] && _flt_c_v8_valid) ? _dB[ _dBv4_off[6] ] : ZEROv4;\\\n\
            _regB[7] = (_flt_n_valid[7] && _flt_c_v8_valid) ? _dB[ _dBv4_off[7] ] : ZEROv4;\\\n\
        }\n\
\n\
#define SET_dBv4_BOUND(_step_id, _dBv4_off, _flt_n_valid) \\\n\
        { \\\n\
            int _flt_n_id  =  cta_idx *  TILE_N_PER_CTA + \\\n\
                            _step_id  * (TILE_N_PER_CTA / READ_dBv4_STEPS) + \\\n\
                             ldg_idy; \\\n\
            \\\n\
            _flt_n_valid  =  _flt_n_id < num_flt_per_grp_pad; \\\n\
            \\\n\
            _dBv4_off  =   grp_id   * flt_hw * num_chl_per_grp_pad_v8 * num_flt_per_grp_pad + \\\n\
                          _flt_n_id * flt_hw * num_chl_per_grp_pad_v8 + \\\n\
                           flt_c_v8_id; \\\n\
        }\n\
\n\
////////////////////////////////////////\n\
// load dA macros\n\
////////////////////////////////////////\n\
\n\
#define SET_dAv4_BOUND(_step_id, _dAv4_off, _in_hw_mask) \\\n\
        { \\\n\
            int _out_nhw_id =  cta_idy *  TILE_M_PER_CTA + \\\n\
                              _step_id * (TILE_M_PER_CTA / READ_dAv4_STEPS) + \\\n\
                              ldg_idy; \\\n\
            \\\n\
            int _out_w_id =  (_out_nhw_id % out_width); \\\n\
            int _out_h_id =  (_out_nhw_id / out_width) % out_height; \\\n\
            \\\n\
            int _in_n_id  =   _out_nhw_id / out_hw; \\\n\
            int _in_h_id  =     _out_h_id * stride_height; \\\n\
            int _in_w_id  =     _out_w_id * stride_width; \\\n\
            \\\n\
            _dAv4_off  =  (_in_n_id  * in_hw + _in_h_id  * in_width + _in_w_id) * num_chl_per_grp_pad_v8 * num_grp + \\\n\
                           grp_id   * num_chl_per_grp_pad_v8 + \\\n\
                           flt_c_v8_id; \\\n\
            \\\n\
            _in_h_id =  _in_h_id - pad_height; \\\n\
            _in_w_id =  _in_w_id - pad_width;  \\\n\
            \\\n\
            SET_BOUND_FLT3(_in_hw_mask, _in_n_id, _in_h_id, _in_w_id); \\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE_16TH(_regA, _dA, _dAv4_off, _in_c_v8_valid, _flt_hw_bid) \\\n\
        { \\\n\
            _dAv4_off[0] += in_lut.idx[lut_id]; \\\n\
            \\\n\
            if(tid < (CTA_SIZE_IN_THD / 16))  \\\n\
                _regA[0] = ((_flt_hw_bid & in_hw_mask[0]) && _in_c_v8_valid) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE_8TH(_regA, _dA, _dAv4_off, _in_c_v8_valid, _flt_hw_bid) \\\n\
        { \\\n\
            _dAv4_off[0] += in_lut.idx[lut_id]; \\\n\
            \\\n\
            if(tid < (CTA_SIZE_IN_THD / 8))  \\\n\
                _regA[0] = ((_flt_hw_bid & in_hw_mask[0]) && _in_c_v8_valid) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE_QTR(_regA, _dA, _dAv4_off, _in_c_v8_valid, _flt_hw_bid) \\\n\
        { \\\n\
            _dAv4_off[0] += in_lut.idx[lut_id]; \\\n\
            \\\n\
            if(tid < (CTA_SIZE_IN_THD / 4))  \\\n\
                _regA[0] = ((_flt_hw_bid & in_hw_mask[0]) && _in_c_v8_valid) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE_HALF(_regA, _dA, _dAv4_off, _in_c_v8_valid, _flt_hw_bid) \\\n\
        { \\\n\
            _dAv4_off[0] += in_lut.idx[lut_id]; \\\n\
            \\\n\
            if(tid < (CTA_SIZE_IN_THD / 2))  \\\n\
                _regA[0] = ((_flt_hw_bid & in_hw_mask[0]) && _in_c_v8_valid) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE1(_regA, _dA, _dAv4_off, _in_c_v8_valid, _flt_hw_bid) \\\n\
        { \\\n\
            _dAv4_off[0] += in_lut.idx[lut_id]; \\\n\
            \\\n\
            _regA[0] = ((_flt_hw_bid & in_hw_mask[0]) && _in_c_v8_valid) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE2(_regA, _dA, _dAv4_off, _in_c_v8_valid, _flt_hw_bid) \\\n\
        { \\\n\
            _dAv4_off[0] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[1] += in_lut.idx[lut_id]; \\\n\
            \\\n\
            _regA[0] = ((_flt_hw_bid & in_hw_mask[0]) && _in_c_v8_valid) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
            _regA[1] = ((_flt_hw_bid & in_hw_mask[1]) && _in_c_v8_valid) ? _dA[ _dAv4_off[1] ] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE4(_regA, _dA, _dAv4_off, _in_c_v8_valid, _flt_hw_bid) \\\n\
        { \\\n\
            _dAv4_off[0] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[1] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[2] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[3] += in_lut.idx[lut_id]; \\\n\
            \\\n\
            _regA[0] = ((_flt_hw_bid & in_hw_mask[0]) && _in_c_v8_valid) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
            _regA[1] = ((_flt_hw_bid & in_hw_mask[1]) && _in_c_v8_valid) ? _dA[ _dAv4_off[1] ] : ZEROv4;\\\n\
            _regA[2] = ((_flt_hw_bid & in_hw_mask[2]) && _in_c_v8_valid) ? _dA[ _dAv4_off[2] ] : ZEROv4;\\\n\
            _regA[3] = ((_flt_hw_bid & in_hw_mask[3]) && _in_c_v8_valid) ? _dA[ _dAv4_off[3] ] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE8(_regA, _dA, _dAv4_off, _in_c_v8_valid, _flt_hw_bid) \\\n\
        { \\\n\
            _dAv4_off[0] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[1] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[2] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[3] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[4] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[5] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[6] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[7] += in_lut.idx[lut_id]; \\\n\
            \\\n\
            _regA[0] = ((_flt_hw_bid & in_hw_mask[0]) && _in_c_v8_valid) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
            _regA[1] = ((_flt_hw_bid & in_hw_mask[1]) && _in_c_v8_valid) ? _dA[ _dAv4_off[1] ] : ZEROv4;\\\n\
            _regA[2] = ((_flt_hw_bid & in_hw_mask[2]) && _in_c_v8_valid) ? _dA[ _dAv4_off[2] ] : ZEROv4;\\\n\
            _regA[3] = ((_flt_hw_bid & in_hw_mask[3]) && _in_c_v8_valid) ? _dA[ _dAv4_off[3] ] : ZEROv4;\\\n\
            _regA[4] = ((_flt_hw_bid & in_hw_mask[4]) && _in_c_v8_valid) ? _dA[ _dAv4_off[4] ] : ZEROv4;\\\n\
            _regA[5] = ((_flt_hw_bid & in_hw_mask[5]) && _in_c_v8_valid) ? _dA[ _dAv4_off[5] ] : ZEROv4;\\\n\
            _regA[6] = ((_flt_hw_bid & in_hw_mask[6]) && _in_c_v8_valid) ? _dA[ _dAv4_off[6] ] : ZEROv4;\\\n\
            _regA[7] = ((_flt_hw_bid & in_hw_mask[7]) && _in_c_v8_valid) ? _dA[ _dAv4_off[7] ] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE16(_regA, _dA, _dAv4_off, _in_c_v8_valid, _flt_hw_bid) \\\n\
        { \\\n\
            _dAv4_off[0]  += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[1]  += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[2]  += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[3]  += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[4]  += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[5]  += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[6]  += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[7]  += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[8]  += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[9]  += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[10] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[11] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[12] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[13] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[14] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[15] += in_lut.idx[lut_id]; \\\n\
            \\\n\
            _regA[0]  = ((_flt_hw_bid & in_hw_mask[0])  && _in_c_v8_valid) ? _dA[ _dAv4_off[0]  ] : ZEROv4;\\\n\
            _regA[1]  = ((_flt_hw_bid & in_hw_mask[1])  && _in_c_v8_valid) ? _dA[ _dAv4_off[1]  ] : ZEROv4;\\\n\
            _regA[2]  = ((_flt_hw_bid & in_hw_mask[2])  && _in_c_v8_valid) ? _dA[ _dAv4_off[2]  ] : ZEROv4;\\\n\
            _regA[3]  = ((_flt_hw_bid & in_hw_mask[3])  && _in_c_v8_valid) ? _dA[ _dAv4_off[3]  ] : ZEROv4;\\\n\
            _regA[4]  = ((_flt_hw_bid & in_hw_mask[4])  && _in_c_v8_valid) ? _dA[ _dAv4_off[4]  ] : ZEROv4;\\\n\
            _regA[5]  = ((_flt_hw_bid & in_hw_mask[5])  && _in_c_v8_valid) ? _dA[ _dAv4_off[5]  ] : ZEROv4;\\\n\
            _regA[6]  = ((_flt_hw_bid & in_hw_mask[6])  && _in_c_v8_valid) ? _dA[ _dAv4_off[6]  ] : ZEROv4;\\\n\
            _regA[7]  = ((_flt_hw_bid & in_hw_mask[7])  && _in_c_v8_valid) ? _dA[ _dAv4_off[7]  ] : ZEROv4;\\\n\
            _regA[8]  = ((_flt_hw_bid & in_hw_mask[8])  && _in_c_v8_valid) ? _dA[ _dAv4_off[8]  ] : ZEROv4;\\\n\
            _regA[9]  = ((_flt_hw_bid & in_hw_mask[9])  && _in_c_v8_valid) ? _dA[ _dAv4_off[9]  ] : ZEROv4;\\\n\
            _regA[10] = ((_flt_hw_bid & in_hw_mask[10]) && _in_c_v8_valid) ? _dA[ _dAv4_off[10] ] : ZEROv4;\\\n\
            _regA[11] = ((_flt_hw_bid & in_hw_mask[11]) && _in_c_v8_valid) ? _dA[ _dAv4_off[11] ] : ZEROv4;\\\n\
            _regA[12] = ((_flt_hw_bid & in_hw_mask[12]) && _in_c_v8_valid) ? _dA[ _dAv4_off[12] ] : ZEROv4;\\\n\
            _regA[13] = ((_flt_hw_bid & in_hw_mask[13]) && _in_c_v8_valid) ? _dA[ _dAv4_off[13] ] : ZEROv4;\\\n\
            _regA[14] = ((_flt_hw_bid & in_hw_mask[14]) && _in_c_v8_valid) ? _dA[ _dAv4_off[14] ] : ZEROv4;\\\n\
            _regA[15] = ((_flt_hw_bid & in_hw_mask[15]) && _in_c_v8_valid) ? _dA[ _dAv4_off[15] ] : ZEROv4;\\\n\
        }\n\
\n\
");
    header_code_.emplace("/mnt/hpc/xusi/ppl.jit/src/ppl/nn/engines/cuda/impls/src/nn/conv/2spk/fn/dmem_macros.h", "// Licensed to the Apache Software Foundation (ASF) under one\n\
// or more contributor license agreements.  See the NOTICE file\n\
// distributed with this work for additional information\n\
// regarding copyright ownership.  The ASF licenses this file\n\
// to you under the Apache License, Version 2.0 (the\n\
// \"License\"); you may not use this file except in compliance\n\
// with the License.  You may obtain a copy of the License at\n\
//\n\
//   http://www.apache.org/licenses/LICENSE-2.0\n\
//\n\
// Unless required by applicable law or agreed to in writing,\n\
// software distributed under the License is distributed on an\n\
// \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n\
// KIND, either express or implied.  See the License for the\n\
// specific language governing permissions and limitations\n\
// under the License.\n\
\n\
////////////////////////////////////////\n\
// load dB macros\n\
////////////////////////////////////////\n\
\n\
#define LOAD_dBv4_SIZE_16TH(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            _dBv4_off[0] += flt_lut.idx[lut_id]; \\\n\
            \\\n\
            if(tid < (CTA_SIZE_IN_THD / 16))  \\\n\
                _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE_8TH(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            _dBv4_off[0] += flt_lut.idx[lut_id]; \\\n\
            \\\n\
            if(tid < (CTA_SIZE_IN_THD / 8))  \\\n\
                _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE_QTR(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            _dBv4_off[0] += flt_lut.idx[lut_id]; \\\n\
            \\\n\
            if(tid < (CTA_SIZE_IN_THD / 4))  \\\n\
                _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE_HALF(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            _dBv4_off[0] += flt_lut.idx[lut_id]; \\\n\
            \\\n\
            if(tid < (CTA_SIZE_IN_THD / 2))  \\\n\
                _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE1(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            _dBv4_off[0] += flt_lut.idx[lut_id]; \\\n\
            \\\n\
            _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE2(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            _dBv4_off[0] += flt_lut.idx[lut_id]; \\\n\
            _dBv4_off[1] += flt_lut.idx[lut_id]; \\\n\
            \\\n\
            _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
            _regB[1] = (_flt_n_valid[1] && _flt_c_v8_valid) ? _dB[ _dBv4_off[1] ] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE4(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            _dBv4_off[0] += flt_lut.idx[lut_id]; \\\n\
            _dBv4_off[1] += flt_lut.idx[lut_id]; \\\n\
            _dBv4_off[2] += flt_lut.idx[lut_id]; \\\n\
            _dBv4_off[3] += flt_lut.idx[lut_id]; \\\n\
            \\\n\
            _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
            _regB[1] = (_flt_n_valid[1] && _flt_c_v8_valid) ? _dB[ _dBv4_off[1] ] : ZEROv4;\\\n\
            _regB[2] = (_flt_n_valid[2] && _flt_c_v8_valid) ? _dB[ _dBv4_off[2] ] : ZEROv4;\\\n\
            _regB[3] = (_flt_n_valid[3] && _flt_c_v8_valid) ? _dB[ _dBv4_off[3] ] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE8(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            _dBv4_off[0] += flt_lut.idx[lut_id]; \\\n\
            _dBv4_off[1] += flt_lut.idx[lut_id]; \\\n\
            _dBv4_off[2] += flt_lut.idx[lut_id]; \\\n\
            _dBv4_off[3] += flt_lut.idx[lut_id]; \\\n\
            _dBv4_off[4] += flt_lut.idx[lut_id]; \\\n\
            _dBv4_off[5] += flt_lut.idx[lut_id]; \\\n\
            _dBv4_off[6] += flt_lut.idx[lut_id]; \\\n\
            _dBv4_off[7] += flt_lut.idx[lut_id]; \\\n\
            \\\n\
            _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
            _regB[1] = (_flt_n_valid[1] && _flt_c_v8_valid) ? _dB[ _dBv4_off[1] ] : ZEROv4;\\\n\
            _regB[2] = (_flt_n_valid[2] && _flt_c_v8_valid) ? _dB[ _dBv4_off[2] ] : ZEROv4;\\\n\
            _regB[3] = (_flt_n_valid[3] && _flt_c_v8_valid) ? _dB[ _dBv4_off[3] ] : ZEROv4;\\\n\
            _regB[4] = (_flt_n_valid[4] && _flt_c_v8_valid) ? _dB[ _dBv4_off[4] ] : ZEROv4;\\\n\
            _regB[5] = (_flt_n_valid[5] && _flt_c_v8_valid) ? _dB[ _dBv4_off[5] ] : ZEROv4;\\\n\
            _regB[6] = (_flt_n_valid[6] && _flt_c_v8_valid) ? _dB[ _dBv4_off[6] ] : ZEROv4;\\\n\
            _regB[7] = (_flt_n_valid[7] && _flt_c_v8_valid) ? _dB[ _dBv4_off[7] ] : ZEROv4;\\\n\
        }\n\
\n\
#define SET_dBv4_BOUND(_step_id, _dBv4_off, _flt_n_valid) \\\n\
        { \\\n\
            int _flt_n_id  =  cta_idx *  TILE_N_PER_CTA + \\\n\
                             _step_id * (TILE_N_PER_CTA / READ_dBv4_STEPS) + \\\n\
                             ldg_idy; \\\n\
            \\\n\
            _flt_n_valid  =  _flt_n_id < num_flt_per_grp_pad; \\\n\
            \\\n\
            _dBv4_off  =   grp_id   * flt_hw * num_chl_per_grp_pad_v8 * num_flt_per_grp_pad + \\\n\
                          _flt_n_id * flt_hw * num_chl_per_grp_pad_v8 + \\\n\
                           flt_c_v8_id; \\\n\
        }\n\
\n\
////////////////////////////////////////\n\
// load dA macros\n\
////////////////////////////////////////\n\
\n\
#define SET_dAv4_BOUND(_step_id, _dAv4_off, _in_n_id, _in_h_start, _in_w_start) \\\n\
        { \\\n\
            int _out_nhw_id    = cta_idy  *  TILE_M_PER_CTA + \\\n\
                                 _step_id * (TILE_M_PER_CTA / READ_dAv4_STEPS) + \\\n\
                                 ldg_idy; \\\n\
            \\\n\
            int _out_w_id =  (_out_nhw_id % out_width); \\\n\
            int _out_h_id =  (_out_nhw_id / out_width) % out_height; \\\n\
            int _in_h_id  =     _out_h_id * stride_height; \\\n\
            int _in_w_id  =     _out_w_id * stride_width; \\\n\
            \\\n\
            _in_n_id      =  _out_nhw_id / out_hw; \\\n\
            _in_h_start   =  _in_h_id - pad_height; \\\n\
            _in_w_start   =  _in_w_id - pad_width;  \\\n\
            \\\n\
            _dAv4_off  =  (_in_n_id  * in_hw + _in_h_id  * in_width + _in_w_id) * num_chl_per_grp_pad_v8 * num_grp + \\\n\
                           grp_id   * num_chl_per_grp_pad_v8 + \\\n\
                           flt_c_v8_id; \\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE_16TH(_regA, _dA, _dAv4_off, _in_n_id, _in_h_id, _in_w_id) \\\n\
        { \\\n\
            _dAv4_off[0] += in_lut.idx[lut_id]; \\\n\
            \\\n\
            if(tid < (CTA_SIZE_IN_THD / 16))  \\\n\
                _regA[0] = (flt_c_v8_valid && HeightInRange(_in_h_id[0]) && WidthInRange(_in_w_id[0]) && (_in_n_id[0] < in_num)) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE_8TH(_regA, _dA, _dAv4_off, _in_n_id, _in_h_id, _in_w_id) \\\n\
        { \\\n\
            _dAv4_off[0] += in_lut.idx[lut_id]; \\\n\
            \\\n\
            if(tid < (CTA_SIZE_IN_THD / 8))  \\\n\
                _regA[0] = (flt_c_v8_valid && HeightInRange(_in_h_id[0]) && WidthInRange(_in_w_id[0]) && (_in_n_id[0] < in_num)) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE_QTR(_regA, _dA, _dAv4_off, _in_n_id, _in_h_id, _in_w_id) \\\n\
        { \\\n\
            _dAv4_off[0] += in_lut.idx[lut_id]; \\\n\
            \\\n\
            if(tid < (CTA_SIZE_IN_THD / 4))  \\\n\
                _regA[0] = (flt_c_v8_valid && HeightInRange(_in_h_id[0]) && WidthInRange(_in_w_id[0]) && (_in_n_id[0] < in_num)) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE_HALF(_regA, _dA, _dAv4_off, _in_n_id, _in_h_id, _in_w_id) \\\n\
        { \\\n\
            _dAv4_off[0] += in_lut.idx[lut_id]; \\\n\
            \\\n\
            if(tid < (CTA_SIZE_IN_THD / 2))  \\\n\
                _regA[0] = (flt_c_v8_valid && HeightInRange(_in_h_id[0]) && WidthInRange(_in_w_id[0]) && (_in_n_id[0] < in_num)) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE1(_regA, _dA, _dAv4_off, _in_n_id, _in_h_id, _in_w_id) \\\n\
        { \\\n\
            _dAv4_off[0] += in_lut.idx[lut_id]; \\\n\
            \\\n\
            _regA[0] = (flt_c_v8_valid && HeightInRange(_in_h_id[0]) && WidthInRange(_in_w_id[0]) && (_in_n_id[0] < in_num)) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE2(_regA, _dA, _dAv4_off, _in_n_id, _in_h_id, _in_w_id) \\\n\
        { \\\n\
            _dAv4_off[0] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[1] += in_lut.idx[lut_id]; \\\n\
            \\\n\
            _regA[0] = (flt_c_v8_valid && HeightInRange(_in_h_id[0]) && WidthInRange(_in_w_id[0]) && (_in_n_id[0] < in_num)) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
            _regA[1] = (flt_c_v8_valid && HeightInRange(_in_h_id[1]) && WidthInRange(_in_w_id[1]) && (_in_n_id[1] < in_num)) ? _dA[ _dAv4_off[1] ] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE4(_regA, _dA, _dAv4_off, _in_n_id, _in_h_id, _in_w_id) \\\n\
        { \\\n\
            _dAv4_off[0] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[1] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[2] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[3] += in_lut.idx[lut_id]; \\\n\
            \\\n\
            _regA[0] = (flt_c_v8_valid && HeightInRange(_in_h_id[0]) && WidthInRange(_in_w_id[0]) && (_in_n_id[0] < in_num)) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
            _regA[1] = (flt_c_v8_valid && HeightInRange(_in_h_id[1]) && WidthInRange(_in_w_id[1]) && (_in_n_id[1] < in_num)) ? _dA[ _dAv4_off[1] ] : ZEROv4;\\\n\
            _regA[2] = (flt_c_v8_valid && HeightInRange(_in_h_id[2]) && WidthInRange(_in_w_id[2]) && (_in_n_id[2] < in_num)) ? _dA[ _dAv4_off[2] ] : ZEROv4;\\\n\
            _regA[3] = (flt_c_v8_valid && HeightInRange(_in_h_id[3]) && WidthInRange(_in_w_id[3]) && (_in_n_id[3] < in_num)) ? _dA[ _dAv4_off[3] ] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE8(_regA, _dA, _dAv4_off, _in_n_id, _in_h_id, _in_w_id) \\\n\
        { \\\n\
            _dAv4_off[0] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[1] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[2] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[3] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[4] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[5] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[6] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[7] += in_lut.idx[lut_id]; \\\n\
            \\\n\
            _regA[0] = (flt_c_v8_valid && HeightInRange(_in_h_id[0]) && WidthInRange(_in_w_id[0]) && (_in_n_id[0] < in_num)) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
            _regA[1] = (flt_c_v8_valid && HeightInRange(_in_h_id[1]) && WidthInRange(_in_w_id[1]) && (_in_n_id[1] < in_num)) ? _dA[ _dAv4_off[1] ] : ZEROv4;\\\n\
            _regA[2] = (flt_c_v8_valid && HeightInRange(_in_h_id[2]) && WidthInRange(_in_w_id[2]) && (_in_n_id[2] < in_num)) ? _dA[ _dAv4_off[2] ] : ZEROv4;\\\n\
            _regA[3] = (flt_c_v8_valid && HeightInRange(_in_h_id[3]) && WidthInRange(_in_w_id[3]) && (_in_n_id[3] < in_num)) ? _dA[ _dAv4_off[3] ] : ZEROv4;\\\n\
            _regA[4] = (flt_c_v8_valid && HeightInRange(_in_h_id[4]) && WidthInRange(_in_w_id[4]) && (_in_n_id[4] < in_num)) ? _dA[ _dAv4_off[4] ] : ZEROv4;\\\n\
            _regA[5] = (flt_c_v8_valid && HeightInRange(_in_h_id[5]) && WidthInRange(_in_w_id[5]) && (_in_n_id[5] < in_num)) ? _dA[ _dAv4_off[5] ] : ZEROv4;\\\n\
            _regA[6] = (flt_c_v8_valid && HeightInRange(_in_h_id[6]) && WidthInRange(_in_w_id[6]) && (_in_n_id[6] < in_num)) ? _dA[ _dAv4_off[6] ] : ZEROv4;\\\n\
            _regA[7] = (flt_c_v8_valid && HeightInRange(_in_h_id[7]) && WidthInRange(_in_w_id[7]) && (_in_n_id[7] < in_num)) ? _dA[ _dAv4_off[7] ] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE16(_regA, _dA, _dAv4_off, _in_n_id, _in_h_id, _in_w_id) \\\n\
        { \\\n\
            _dAv4_off[0]  += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[1]  += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[2]  += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[3]  += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[4]  += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[5]  += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[6]  += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[7]  += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[8]  += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[9]  += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[10] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[11] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[12] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[13] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[14] += in_lut.idx[lut_id]; \\\n\
            _dAv4_off[15] += in_lut.idx[lut_id]; \\\n\
            \\\n\
            _regA[0]  = (flt_c_v8_valid && HeightInRange(_in_h_id[0])  && WidthInRange(_in_w_id[0])  && (_in_n_id[0]  < in_num)) ? _dA[ _dAv4_off[0] ]  : ZEROv4;\\\n\
            _regA[1]  = (flt_c_v8_valid && HeightInRange(_in_h_id[1])  && WidthInRange(_in_w_id[1])  && (_in_n_id[1]  < in_num)) ? _dA[ _dAv4_off[1] ]  : ZEROv4;\\\n\
            _regA[2]  = (flt_c_v8_valid && HeightInRange(_in_h_id[2])  && WidthInRange(_in_w_id[2])  && (_in_n_id[2]  < in_num)) ? _dA[ _dAv4_off[2] ]  : ZEROv4;\\\n\
            _regA[3]  = (flt_c_v8_valid && HeightInRange(_in_h_id[3])  && WidthInRange(_in_w_id[3])  && (_in_n_id[3]  < in_num)) ? _dA[ _dAv4_off[3] ]  : ZEROv4;\\\n\
            _regA[4]  = (flt_c_v8_valid && HeightInRange(_in_h_id[4])  && WidthInRange(_in_w_id[4])  && (_in_n_id[4]  < in_num)) ? _dA[ _dAv4_off[4] ]  : ZEROv4;\\\n\
            _regA[5]  = (flt_c_v8_valid && HeightInRange(_in_h_id[5])  && WidthInRange(_in_w_id[5])  && (_in_n_id[5]  < in_num)) ? _dA[ _dAv4_off[5] ]  : ZEROv4;\\\n\
            _regA[6]  = (flt_c_v8_valid && HeightInRange(_in_h_id[6])  && WidthInRange(_in_w_id[6])  && (_in_n_id[6]  < in_num)) ? _dA[ _dAv4_off[6] ]  : ZEROv4;\\\n\
            _regA[7]  = (flt_c_v8_valid && HeightInRange(_in_h_id[7])  && WidthInRange(_in_w_id[7])  && (_in_n_id[7]  < in_num)) ? _dA[ _dAv4_off[7] ]  : ZEROv4;\\\n\
            _regA[8]  = (flt_c_v8_valid && HeightInRange(_in_h_id[8])  && WidthInRange(_in_w_id[8])  && (_in_n_id[8]  < in_num)) ? _dA[ _dAv4_off[8] ]  : ZEROv4;\\\n\
            _regA[9]  = (flt_c_v8_valid && HeightInRange(_in_h_id[9])  && WidthInRange(_in_w_id[9])  && (_in_n_id[9]  < in_num)) ? _dA[ _dAv4_off[9] ]  : ZEROv4;\\\n\
            _regA[10] = (flt_c_v8_valid && HeightInRange(_in_h_id[10]) && WidthInRange(_in_w_id[10]) && (_in_n_id[10] < in_num)) ? _dA[ _dAv4_off[10] ] : ZEROv4;\\\n\
            _regA[11] = (flt_c_v8_valid && HeightInRange(_in_h_id[11]) && WidthInRange(_in_w_id[11]) && (_in_n_id[11] < in_num)) ? _dA[ _dAv4_off[11] ] : ZEROv4;\\\n\
            _regA[12] = (flt_c_v8_valid && HeightInRange(_in_h_id[12]) && WidthInRange(_in_w_id[12]) && (_in_n_id[12] < in_num)) ? _dA[ _dAv4_off[12] ] : ZEROv4;\\\n\
            _regA[13] = (flt_c_v8_valid && HeightInRange(_in_h_id[13]) && WidthInRange(_in_w_id[13]) && (_in_n_id[13] < in_num)) ? _dA[ _dAv4_off[13] ] : ZEROv4;\\\n\
            _regA[14] = (flt_c_v8_valid && HeightInRange(_in_h_id[14]) && WidthInRange(_in_w_id[14]) && (_in_n_id[14] < in_num)) ? _dA[ _dAv4_off[14] ] : ZEROv4;\\\n\
            _regA[15] = (flt_c_v8_valid && HeightInRange(_in_h_id[15]) && WidthInRange(_in_w_id[15]) && (_in_n_id[15] < in_num)) ? _dA[ _dAv4_off[15] ] : ZEROv4;\\\n\
        }\n\
\n\
");
    header_code_.emplace("/mnt/hpc/xusi/ppl.jit/src/ppl/nn/engines/cuda/impls/src/nn/conv/2spk/fs/dmem_macros.h", "// Licensed to the Apache Software Foundation (ASF) under one\n\
// or more contributor license agreements.  See the NOTICE file\n\
// distributed with this work for additional information\n\
// regarding copyright ownership.  The ASF licenses this file\n\
// to you under the Apache License, Version 2.0 (the\n\
// \"License\"); you may not use this file except in compliance\n\
// with the License.  You may obtain a copy of the License at\n\
//\n\
//   http://www.apache.org/licenses/LICENSE-2.0\n\
//\n\
// Unless required by applicable law or agreed to in writing,\n\
// software distributed under the License is distributed on an\n\
// \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n\
// KIND, either express or implied.  See the License for the\n\
// specific language governing permissions and limitations\n\
// under the License.\n\
\n\
////////////////////////////////////////\n\
// load dB macros\n\
////////////////////////////////////////\n\
\n\
#define LOAD_dBv4_SIZE_16TH(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            if(tid < (CTA_SIZE_IN_THD / 16))  \\\n\
                _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
            \\\n\
            _dBv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE_8TH(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            if(tid < (CTA_SIZE_IN_THD / 8))  \\\n\
                _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
            \\\n\
            _dBv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE_QTR(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            if(tid < (CTA_SIZE_IN_THD / 4))  \\\n\
                _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
            \\\n\
            _dBv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE_HALF(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            if(tid < (CTA_SIZE_IN_THD / 2))  \\\n\
                _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
            \\\n\
            _dBv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE1(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
            \\\n\
            _dBv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE2(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
            _regB[1] = (_flt_n_valid[1] && _flt_c_v8_valid) ? _dB[ _dBv4_off[1] ] : ZEROv4;\\\n\
            \\\n\
            _dBv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
            _dBv4_off[1] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE4(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
            _regB[1] = (_flt_n_valid[1] && _flt_c_v8_valid) ? _dB[ _dBv4_off[1] ] : ZEROv4;\\\n\
            _regB[2] = (_flt_n_valid[2] && _flt_c_v8_valid) ? _dB[ _dBv4_off[2] ] : ZEROv4;\\\n\
            _regB[3] = (_flt_n_valid[3] && _flt_c_v8_valid) ? _dB[ _dBv4_off[3] ] : ZEROv4;\\\n\
            \\\n\
            _dBv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
            _dBv4_off[1] += TILE_K_V8_PER_CTA; \\\n\
            _dBv4_off[2] += TILE_K_V8_PER_CTA; \\\n\
            _dBv4_off[3] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE8(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid) \\\n\
        { \\\n\
            _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\\\n\
            _regB[1] = (_flt_n_valid[1] && _flt_c_v8_valid) ? _dB[ _dBv4_off[1] ] : ZEROv4;\\\n\
            _regB[2] = (_flt_n_valid[2] && _flt_c_v8_valid) ? _dB[ _dBv4_off[2] ] : ZEROv4;\\\n\
            _regB[3] = (_flt_n_valid[3] && _flt_c_v8_valid) ? _dB[ _dBv4_off[3] ] : ZEROv4;\\\n\
            _regB[4] = (_flt_n_valid[4] && _flt_c_v8_valid) ? _dB[ _dBv4_off[4] ] : ZEROv4;\\\n\
            _regB[5] = (_flt_n_valid[5] && _flt_c_v8_valid) ? _dB[ _dBv4_off[5] ] : ZEROv4;\\\n\
            _regB[6] = (_flt_n_valid[6] && _flt_c_v8_valid) ? _dB[ _dBv4_off[6] ] : ZEROv4;\\\n\
            _regB[7] = (_flt_n_valid[7] && _flt_c_v8_valid) ? _dB[ _dBv4_off[7] ] : ZEROv4;\\\n\
            \\\n\
            _dBv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
            _dBv4_off[1] += TILE_K_V8_PER_CTA; \\\n\
            _dBv4_off[2] += TILE_K_V8_PER_CTA; \\\n\
            _dBv4_off[3] += TILE_K_V8_PER_CTA; \\\n\
            _dBv4_off[4] += TILE_K_V8_PER_CTA; \\\n\
            _dBv4_off[5] += TILE_K_V8_PER_CTA; \\\n\
            _dBv4_off[6] += TILE_K_V8_PER_CTA; \\\n\
            _dBv4_off[7] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define SET_dBv4_BOUND(_step_id, _dBv4_off, _flt_n_valid) \\\n\
        { \\\n\
            int _flt_n_id  =  cta_idx *  TILE_N_PER_CTA + \\\n\
                             _step_id * (TILE_N_PER_CTA / READ_dBv4_STEPS) + \\\n\
                             ldg_idy; \\\n\
            \\\n\
            _flt_n_valid  =  _flt_n_id < num_flt_per_grp_pad; \\\n\
            \\\n\
            _dBv4_off  =   grp_id   * num_chl_per_grp_pad_v8 * flt_hw  * num_flt_per_grp_pad + \\\n\
                          _flt_n_id * num_chl_per_grp_pad_v8 * flt_hw  + \\\n\
                           spf_id   * num_chl_per_grp_pad_v8 + \\\n\
                           flt_c_v8_id; \\\n\
        }\n\
\n\
////////////////////////////////////////\n\
// load dA macros\n\
////////////////////////////////////////\n\
\n\
#define SET_dAv4_BOUND(_step_id, _dAv4_off, _in_hw_valid) \\\n\
        { \\\n\
            int _out_nhw_id    =  cta_idy *  TILE_M_PER_CTA + \\\n\
                                 _step_id * (TILE_M_PER_CTA / READ_dAv4_STEPS) + \\\n\
                                 ldg_idy; \\\n\
            \\\n\
            int _out_w_id =  (_out_nhw_id % out_width); \\\n\
            int _out_h_id =  (_out_nhw_id / out_width) % out_height; \\\n\
            \\\n\
            int _in_n_id  =   _out_nhw_id / out_hw; \\\n\
            int _in_h_id  =     _out_h_id * stride_height; \\\n\
            int _in_w_id  =     _out_w_id * stride_width; \\\n\
            \\\n\
	        int _flt_h_id = spf_id / flt_width; \\\n\
	        int _flt_w_id = spf_id % flt_width; \\\n\
            \\\n\
            _in_h_id =  _in_h_id + _flt_h_id * hole_height - pad_height; \\\n\
            _in_w_id =  _in_w_id + _flt_w_id * hole_width - pad_width;  \\\n\
            \\\n\
            _dAv4_off  =  (_in_n_id  * in_hw + _in_h_id  * in_width + _in_w_id) * num_chl_per_grp_pad_v8 * num_grp + \\\n\
                           grp_id   * num_chl_per_grp_pad_v8 + \\\n\
                           flt_c_v8_id; \\\n\
            \\\n\
            SET_BOUND_FLT1(_in_hw_valid, _in_n_id, _in_h_id, _in_w_id); \\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE_16TH(_regA, _dA, _dAv4_off, _in_c_v8_valid, _in_hw_valid) \\\n\
        { \\\n\
            if(tid < (CTA_SIZE_IN_THD / 16))  \\\n\
                _regA[0] = (_in_hw_valid[0] && _in_c_v8_valid) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
            \\\n\
            _dAv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE_8TH(_regA, _dA, _dAv4_off, _in_c_v8_valid, _in_hw_valid) \\\n\
        { \\\n\
            if(tid < (CTA_SIZE_IN_THD / 8))  \\\n\
                _regA[0] = (_in_hw_valid[0] && _in_c_v8_valid) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
            \\\n\
            _dAv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE_QTR(_regA, _dA, _dAv4_off, _in_c_v8_valid, _in_hw_valid) \\\n\
        { \\\n\
            if(tid < (CTA_SIZE_IN_THD / 4))  \\\n\
                _regA[0] = (_in_hw_valid[0] && _in_c_v8_valid) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
            \\\n\
            _dAv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE_HALF(_regA, _dA, _dAv4_off, _in_c_v8_valid, _in_hw_valid) \\\n\
        { \\\n\
            if(tid < (CTA_SIZE_IN_THD / 2))  \\\n\
                _regA[0] = (_in_hw_valid[0] && _in_c_v8_valid) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
            \\\n\
            _dAv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE1(_regA, _dA, _dAv4_off, _in_c_v8_valid, _in_hw_valid) \\\n\
        { \\\n\
            _regA[0] = (_in_hw_valid[0] && _in_c_v8_valid) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
            \\\n\
            _dAv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE2(_regA, _dA, _dAv4_off, _in_c_v8_valid, _in_hw_valid) \\\n\
        { \\\n\
            _regA[0] = (_in_hw_valid[0] && _in_c_v8_valid) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
            _regA[1] = (_in_hw_valid[1] && _in_c_v8_valid) ? _dA[ _dAv4_off[1] ] : ZEROv4;\\\n\
            \\\n\
            _dAv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[1] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE4(_regA, _dA, _dAv4_off, _in_c_v8_valid, _in_hw_valid) \\\n\
        { \\\n\
            _regA[0] = (_in_hw_valid[0] && _in_c_v8_valid) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
            _regA[1] = (_in_hw_valid[1] && _in_c_v8_valid) ? _dA[ _dAv4_off[1] ] : ZEROv4;\\\n\
            _regA[2] = (_in_hw_valid[2] && _in_c_v8_valid) ? _dA[ _dAv4_off[2] ] : ZEROv4;\\\n\
            _regA[3] = (_in_hw_valid[3] && _in_c_v8_valid) ? _dA[ _dAv4_off[3] ] : ZEROv4;\\\n\
            \\\n\
            _dAv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[1] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[2] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[3] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE8(_regA, _dA, _dAv4_off, _in_c_v8_valid, _in_hw_valid) \\\n\
        { \\\n\
            _regA[0] = (_in_hw_valid[0] && _in_c_v8_valid) ? _dA[ _dAv4_off[0] ] : ZEROv4;\\\n\
            _regA[1] = (_in_hw_valid[1] && _in_c_v8_valid) ? _dA[ _dAv4_off[1] ] : ZEROv4;\\\n\
            _regA[2] = (_in_hw_valid[2] && _in_c_v8_valid) ? _dA[ _dAv4_off[2] ] : ZEROv4;\\\n\
            _regA[3] = (_in_hw_valid[3] && _in_c_v8_valid) ? _dA[ _dAv4_off[3] ] : ZEROv4;\\\n\
            _regA[4] = (_in_hw_valid[4] && _in_c_v8_valid) ? _dA[ _dAv4_off[4] ] : ZEROv4;\\\n\
            _regA[5] = (_in_hw_valid[5] && _in_c_v8_valid) ? _dA[ _dAv4_off[5] ] : ZEROv4;\\\n\
            _regA[6] = (_in_hw_valid[6] && _in_c_v8_valid) ? _dA[ _dAv4_off[6] ] : ZEROv4;\\\n\
            _regA[7] = (_in_hw_valid[7] && _in_c_v8_valid) ? _dA[ _dAv4_off[7] ] : ZEROv4;\\\n\
            \\\n\
            _dAv4_off[0] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[1] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[2] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[3] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[4] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[5] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[6] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[7] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE16(_regA, _dA, _dAv4_off, _in_c_v8_valid, _in_hw_valid) \\\n\
        { \\\n\
            _regA[0]  = (_in_hw_valid[0]  && _in_c_v8_valid) ? _dA[ _dAv4_off[0]  ] : ZEROv4;\\\n\
            _regA[1]  = (_in_hw_valid[1]  && _in_c_v8_valid) ? _dA[ _dAv4_off[1]  ] : ZEROv4;\\\n\
            _regA[2]  = (_in_hw_valid[2]  && _in_c_v8_valid) ? _dA[ _dAv4_off[2]  ] : ZEROv4;\\\n\
            _regA[3]  = (_in_hw_valid[3]  && _in_c_v8_valid) ? _dA[ _dAv4_off[3]  ] : ZEROv4;\\\n\
            _regA[4]  = (_in_hw_valid[4]  && _in_c_v8_valid) ? _dA[ _dAv4_off[4]  ] : ZEROv4;\\\n\
            _regA[5]  = (_in_hw_valid[5]  && _in_c_v8_valid) ? _dA[ _dAv4_off[5]  ] : ZEROv4;\\\n\
            _regA[6]  = (_in_hw_valid[6]  && _in_c_v8_valid) ? _dA[ _dAv4_off[6]  ] : ZEROv4;\\\n\
            _regA[7]  = (_in_hw_valid[7]  && _in_c_v8_valid) ? _dA[ _dAv4_off[7]  ] : ZEROv4;\\\n\
            _regA[8]  = (_in_hw_valid[8]  && _in_c_v8_valid) ? _dA[ _dAv4_off[8]  ] : ZEROv4;\\\n\
            _regA[9]  = (_in_hw_valid[9]  && _in_c_v8_valid) ? _dA[ _dAv4_off[9]  ] : ZEROv4;\\\n\
            _regA[10] = (_in_hw_valid[10] && _in_c_v8_valid) ? _dA[ _dAv4_off[10] ] : ZEROv4;\\\n\
            _regA[11] = (_in_hw_valid[11] && _in_c_v8_valid) ? _dA[ _dAv4_off[11] ] : ZEROv4;\\\n\
            _regA[12] = (_in_hw_valid[12] && _in_c_v8_valid) ? _dA[ _dAv4_off[12] ] : ZEROv4;\\\n\
            _regA[13] = (_in_hw_valid[13] && _in_c_v8_valid) ? _dA[ _dAv4_off[13] ] : ZEROv4;\\\n\
            _regA[14] = (_in_hw_valid[14] && _in_c_v8_valid) ? _dA[ _dAv4_off[14] ] : ZEROv4;\\\n\
            _regA[15] = (_in_hw_valid[15] && _in_c_v8_valid) ? _dA[ _dAv4_off[15] ] : ZEROv4;\\\n\
            \\\n\
            _dAv4_off[0]  += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[1]  += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[2]  += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[3]  += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[4]  += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[5]  += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[6]  += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[7]  += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[8]  += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[9]  += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[10] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[11] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[12] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[13] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[14] += TILE_K_V8_PER_CTA; \\\n\
            _dAv4_off[15] += TILE_K_V8_PER_CTA; \\\n\
        }\n\
\n\
");
    header_code_.emplace("/mnt/hpc/xusi/ppl.jit/src/ppl/nn/engines/cuda/impls/src/nn/conv/2spk/common/hmma_macros.h", "// Licensed to the Apache Software Foundation (ASF) under one\n\
// or more contributor license agreements.  See the NOTICE file\n\
// distributed with this work for additional information\n\
// regarding copyright ownership.  The ASF licenses this file\n\
// to you under the Apache License, Version 2.0 (the\n\
// \"License\"); you may not use this file except in compliance\n\
// with the License.  You may obtain a copy of the License at\n\
//\n\
//   http://www.apache.org/licenses/LICENSE-2.0\n\
//\n\
// Unless required by applicable law or agreed to in writing,\n\
// software distributed under the License is distributed on an\n\
// \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n\
// KIND, either express or implied.  See the License for the\n\
// specific language governing permissions and limitations\n\
// under the License.\n\
\n\
////////////////////////////////////////\n\
// hmma macros\n\
////////////////////////////////////////\n\
\n\
#define MMA_INST_OPCODE \\\n\
        \"mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%5,%6};\\n\"\n\
        \n\
#define MMA_INST(_d0, _d1, _a0, _a1, _b) \\\n\
        asm volatile(MMA_INST_OPCODE:   \"=r\"(_d0),   \"=r\"(_d1): \"r\"(_a0), \"r\"(_a1), \"r\"(_b),  \"r\"(_d0),   \"r\"(_d1));\n\
\n\
#define MMA_INST_ASCEND1(_C, _C_off, _C_stride, _a0, _a1, _B) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _B[0]); \\\n\
        }\n\
        \n\
#define MMA_INST_ASCEND2(_C, _C_off, _C_stride, _a0, _a1, _B) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _B[0]); \\\n\
            MMA_INST(_C[_C_off + 1], _C[_C_off + _C_stride + 1], _a0, _a1, _B[1]); \\\n\
        }\n\
        \n\
#define MMA_INST_ASCEND4(_C, _C_off, _C_stride, _a0, _a1, _B) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _B[0]); \\\n\
            MMA_INST(_C[_C_off + 1], _C[_C_off + _C_stride + 1], _a0, _a1, _B[1]); \\\n\
            MMA_INST(_C[_C_off + 2], _C[_C_off + _C_stride + 2], _a0, _a1, _B[2]); \\\n\
            MMA_INST(_C[_C_off + 3], _C[_C_off + _C_stride + 3], _a0, _a1, _B[3]); \\\n\
        }\n\
        \n\
#define MMA_INST_ASCEND8(_C, _C_off, _C_stride, _a0, _a1, _B) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _B[0]); \\\n\
            MMA_INST(_C[_C_off + 1], _C[_C_off + _C_stride + 1], _a0, _a1, _B[1]); \\\n\
            MMA_INST(_C[_C_off + 2], _C[_C_off + _C_stride + 2], _a0, _a1, _B[2]); \\\n\
            MMA_INST(_C[_C_off + 3], _C[_C_off + _C_stride + 3], _a0, _a1, _B[3]); \\\n\
            MMA_INST(_C[_C_off + 4], _C[_C_off + _C_stride + 4], _a0, _a1, _B[4]); \\\n\
            MMA_INST(_C[_C_off + 5], _C[_C_off + _C_stride + 5], _a0, _a1, _B[5]); \\\n\
            MMA_INST(_C[_C_off + 6], _C[_C_off + _C_stride + 6], _a0, _a1, _B[6]); \\\n\
            MMA_INST(_C[_C_off + 7], _C[_C_off + _C_stride + 7], _a0, _a1, _B[7]); \\\n\
        }\n\
        \n\
       \n\
#define MMA_INST_DESCEND1(_C, _C_off, _C_stride, _a0, _a1, _B) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _B[0]); \\\n\
        }\n\
\n\
#define MMA_INST_DESCEND2(_C, _C_off, _C_stride, _a0, _a1, _B) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _B[1]); \\\n\
            MMA_INST(_C[_C_off - 1], _C[_C_off + _C_stride - 1], _a0, _a1, _B[0]); \\\n\
        }\n\
\n\
#define MMA_INST_DESCEND4(_C, _C_off, _C_stride, _a0, _a1, _B) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _B[3]); \\\n\
            MMA_INST(_C[_C_off - 1], _C[_C_off + _C_stride - 1], _a0, _a1, _B[2]); \\\n\
            MMA_INST(_C[_C_off - 2], _C[_C_off + _C_stride - 2], _a0, _a1, _B[1]); \\\n\
            MMA_INST(_C[_C_off - 3], _C[_C_off + _C_stride - 3], _a0, _a1, _B[0]); \\\n\
        }\n\
\n\
#define MMA_INST_DESCEND8(_C, _C_off, _C_stride, _a0, _a1, _B) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _B[7]); \\\n\
            MMA_INST(_C[_C_off - 1], _C[_C_off + _C_stride - 1], _a0, _a1, _B[6]); \\\n\
            MMA_INST(_C[_C_off - 2], _C[_C_off + _C_stride - 2], _a0, _a1, _B[5]); \\\n\
            MMA_INST(_C[_C_off - 3], _C[_C_off + _C_stride - 3], _a0, _a1, _B[4]); \\\n\
            MMA_INST(_C[_C_off - 4], _C[_C_off + _C_stride - 4], _a0, _a1, _B[3]); \\\n\
            MMA_INST(_C[_C_off - 5], _C[_C_off + _C_stride - 5], _a0, _a1, _B[2]); \\\n\
            MMA_INST(_C[_C_off - 6], _C[_C_off + _C_stride - 6], _a0, _a1, _B[1]); \\\n\
            MMA_INST(_C[_C_off - 7], _C[_C_off + _C_stride - 7], _a0, _a1, _B[0]); \\\n\
        }\n\
\n\
#define MMA_INST_1x1(_C, _A, _B) \\\n\
        { \\\n\
            MMA_INST_ASCEND1  (_C, 0,  TILE_N_V2_PER_THD, _A[0], _A[1], _B); \\\n\
        }\n\
\n\
#define MMA_INST_1x2(_C, _A, _B) \\\n\
        { \\\n\
            MMA_INST_ASCEND2  (_C, 0,  TILE_N_V2_PER_THD, _A[0], _A[1], _B); \\\n\
        }\n\
\n\
#define MMA_INST_1x4(_C, _A, _B) \\\n\
        { \\\n\
            MMA_INST_ASCEND4  (_C, 0,  TILE_N_V2_PER_THD, _A[0], _A[1], _B); \\\n\
        }\n\
\n\
#define MMA_INST_1x8(_C, _A, _B) \\\n\
        { \\\n\
            MMA_INST_ASCEND8  (_C, 0,  TILE_N_V2_PER_THD, _A[0], _A[1], _B); \\\n\
        }\n\
\n\
#define MMA_INST_2x1(_C, _A, _B) \\\n\
        { \\\n\
            MMA_INST_ASCEND1  (_C, 0,  TILE_N_V2_PER_THD, _A[0], _A[1], _B); \\\n\
            MMA_INST_DESCEND1 (_C, 2,  TILE_N_V2_PER_THD, _A[2], _A[3], _B); \\\n\
        }\n\
\n\
#define MMA_INST_2x2(_C, _A, _B) \\\n\
        { \\\n\
            MMA_INST_ASCEND2  (_C, 0,  TILE_N_V2_PER_THD, _A[0], _A[1], _B); \\\n\
            MMA_INST_DESCEND2 (_C, 5,  TILE_N_V2_PER_THD, _A[2], _A[3], _B); \\\n\
        }\n\
\n\
#define MMA_INST_2x4(_C, _A, _B) \\\n\
        { \\\n\
            MMA_INST_ASCEND4  (_C, 0,  TILE_N_V2_PER_THD, _A[0], _A[1], _B); \\\n\
            MMA_INST_DESCEND4 (_C, 11, TILE_N_V2_PER_THD, _A[2], _A[3], _B); \\\n\
        }\n\
\n\
#define MMA_INST_2x8(_C, _A, _B) \\\n\
        { \\\n\
            MMA_INST_ASCEND8  (_C, 0,  TILE_N_V2_PER_THD, _A[0], _A[1], _B); \\\n\
            MMA_INST_DESCEND8 (_C, 23, TILE_N_V2_PER_THD, _A[2], _A[3], _B); \\\n\
        }\n\
\n\
#define MMA_INST_4x1(_C, _A, _B) \\\n\
        { \\\n\
            MMA_INST_ASCEND1  (_C, 0,  TILE_N_V2_PER_THD, _A[0], _A[1], _B); \\\n\
            MMA_INST_DESCEND1 (_C, 2,  TILE_N_V2_PER_THD, _A[2], _A[3], _B); \\\n\
            \\\n\
            MMA_INST_ASCEND1  (_C, 4,  TILE_N_V2_PER_THD, _A[4], _A[5], _B); \\\n\
            MMA_INST_DESCEND1 (_C, 6,  TILE_N_V2_PER_THD, _A[6], _A[7], _B); \\\n\
        }\n\
\n\
#define MMA_INST_4x2(_C, _A, _B) \\\n\
        { \\\n\
            MMA_INST_ASCEND2  (_C, 0,  TILE_N_V2_PER_THD, _A[0], _A[1], _B); \\\n\
            MMA_INST_DESCEND2 (_C, 5,  TILE_N_V2_PER_THD, _A[2], _A[3], _B); \\\n\
            \\\n\
            MMA_INST_ASCEND2  (_C, 8,  TILE_N_V2_PER_THD, _A[4], _A[5], _B); \\\n\
            MMA_INST_DESCEND2 (_C, 13, TILE_N_V2_PER_THD, _A[6], _A[7], _B); \\\n\
        }\n\
\n\
#define MMA_INST_4x4(_C, _A, _B) \\\n\
        { \\\n\
            MMA_INST_ASCEND4  (_C, 0,  TILE_N_V2_PER_THD, _A[0], _A[1], _B); \\\n\
            MMA_INST_DESCEND4 (_C, 11, TILE_N_V2_PER_THD, _A[2], _A[3], _B); \\\n\
            \\\n\
            MMA_INST_ASCEND4  (_C, 16, TILE_N_V2_PER_THD, _A[4], _A[5], _B); \\\n\
            MMA_INST_DESCEND4 (_C, 27, TILE_N_V2_PER_THD, _A[6], _A[7], _B); \\\n\
        }\n\
\n\
#define MMA_INST_4x8(_C, _A, _B) \\\n\
        { \\\n\
            MMA_INST_ASCEND8  (_C, 0,  TILE_N_V2_PER_THD, _A[0], _A[1], _B); \\\n\
            MMA_INST_DESCEND8 (_C, 23, TILE_N_V2_PER_THD, _A[2], _A[3], _B); \\\n\
            \\\n\
            MMA_INST_ASCEND8  (_C, 32, TILE_N_V2_PER_THD, _A[4], _A[5], _B); \\\n\
            MMA_INST_DESCEND8 (_C, 55, TILE_N_V2_PER_THD, _A[6], _A[7], _B); \\\n\
        }\n\
\n\
#define MMA_INST_8x1(_C, _A, _B) \\\n\
        { \\\n\
            MMA_INST_ASCEND1  (_C, 0,  TILE_N_V2_PER_THD, _A[0],  _A[1],  _B); \\\n\
            MMA_INST_DESCEND1 (_C, 2,  TILE_N_V2_PER_THD, _A[2],  _A[3],  _B); \\\n\
            \\\n\
            MMA_INST_ASCEND1  (_C, 4,  TILE_N_V2_PER_THD, _A[4],  _A[5],  _B); \\\n\
            MMA_INST_DESCEND1 (_C, 6,  TILE_N_V2_PER_THD, _A[6],  _A[7],  _B); \\\n\
            \\\n\
            MMA_INST_ASCEND1  (_C, 8,  TILE_N_V2_PER_THD, _A[8],  _A[9],  _B); \\\n\
            MMA_INST_DESCEND1 (_C, 10, TILE_N_V2_PER_THD, _A[10], _A[11], _B); \\\n\
            \\\n\
            MMA_INST_ASCEND1  (_C, 12, TILE_N_V2_PER_THD, _A[12], _A[13], _B); \\\n\
            MMA_INST_DESCEND1 (_C, 14, TILE_N_V2_PER_THD, _A[14], _A[15], _B); \\\n\
        }\n\
\n\
#define MMA_INST_8x2(_C, _A, _B) \\\n\
        { \\\n\
            MMA_INST_ASCEND2  (_C, 0,  TILE_N_V2_PER_THD, _A[0],  _A[1],  _B); \\\n\
            MMA_INST_DESCEND2 (_C, 5,  TILE_N_V2_PER_THD, _A[2],  _A[3],  _B); \\\n\
            \\\n\
            MMA_INST_ASCEND2  (_C, 8,  TILE_N_V2_PER_THD, _A[4],  _A[5],  _B); \\\n\
            MMA_INST_DESCEND2 (_C, 13, TILE_N_V2_PER_THD, _A[6],  _A[7],  _B); \\\n\
            \\\n\
            MMA_INST_ASCEND2  (_C, 16, TILE_N_V2_PER_THD, _A[8],  _A[9],  _B); \\\n\
            MMA_INST_DESCEND2 (_C, 21, TILE_N_V2_PER_THD, _A[10], _A[11], _B); \\\n\
            \\\n\
            MMA_INST_ASCEND2  (_C, 24, TILE_N_V2_PER_THD, _A[12], _A[13], _B); \\\n\
            MMA_INST_DESCEND2 (_C, 29, TILE_N_V2_PER_THD, _A[14], _A[15], _B); \\\n\
        }\n\
\n\
#define MMA_INST_8x4(_C, _A, _B) \\\n\
        { \\\n\
            MMA_INST_ASCEND4  (_C, 0,  TILE_N_V2_PER_THD, _A[0],  _A[1],  _B); \\\n\
            MMA_INST_DESCEND4 (_C, 11, TILE_N_V2_PER_THD, _A[2],  _A[3],  _B); \\\n\
            \\\n\
            MMA_INST_ASCEND4  (_C, 16, TILE_N_V2_PER_THD, _A[4],  _A[5],  _B); \\\n\
            MMA_INST_DESCEND4 (_C, 27, TILE_N_V2_PER_THD, _A[6],  _A[7],  _B); \\\n\
            \\\n\
            MMA_INST_ASCEND4  (_C, 32, TILE_N_V2_PER_THD, _A[8],  _A[9],  _B); \\\n\
            MMA_INST_DESCEND4 (_C, 43, TILE_N_V2_PER_THD, _A[10], _A[11], _B); \\\n\
            \\\n\
            MMA_INST_ASCEND4  (_C, 48, TILE_N_V2_PER_THD, _A[12], _A[13], _B); \\\n\
            MMA_INST_DESCEND4 (_C, 59, TILE_N_V2_PER_THD, _A[14], _A[15], _B); \\\n\
        }\n\
\n\
");
    header_code_.emplace("/mnt/hpc/xusi/ppl.jit/src/ppl/nn/engines/cuda/impls/src/nn/conv/2spk/common/reduce_macros.h", "// Licensed to the Apache Software Foundation (ASF) under one\n\
// or more contributor license agreements.  See the NOTICE file\n\
// distributed with this work for additional information\n\
// regarding copyright ownership.  The ASF licenses this file\n\
// to you under the Apache License, Version 2.0 (the\n\
// \"License\"); you may not use this file except in compliance\n\
// with the License.  You may obtain a copy of the License at\n\
//\n\
//   http://www.apache.org/licenses/LICENSE-2.0\n\
//\n\
// Unless required by applicable law or agreed to in writing,\n\
// software distributed under the License is distributed on an\n\
// \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n\
// KIND, either express or implied.  See the License for the\n\
// specific language governing permissions and limitations\n\
// under the License.\n\
\n\
/////////////////////////////////////////////////////\n\
// reduce half2 macros\n\
/////////////////////////////////////////////////////\n\
\n\
#define REDUCE_HALF2_SIZE4(_h2R, _h2R_off) \\\n\
        { \\\n\
            _h2R[0] = __hadd2(_h2R[0], _h2R[_h2R_off]); \\\n\
            _h2R[1] = __hadd2(_h2R[1], _h2R[_h2R_off + 1]); \\\n\
            _h2R[2] = __hadd2(_h2R[2], _h2R[_h2R_off + 2]); \\\n\
            _h2R[3] = __hadd2(_h2R[3], _h2R[_h2R_off + 3]); \\\n\
        }\n\
\n\
#define REDUCE_HALF2_1x4(_h2R) \\\n\
        { \\\n\
            REDUCE_HALF2_SIZE4(_h2R, _4HALF2_); \\\n\
        }\n\
\n\
#define REDUCE_HALF2_3x4(_h2R) \\\n\
        { \\\n\
            REDUCE_HALF2_SIZE4(_h2R, _4HALF2_); \\\n\
            REDUCE_HALF2_SIZE4(_h2R, _4HALF2_ * 2); \\\n\
            REDUCE_HALF2_SIZE4(_h2R, _4HALF2_ * 3); \\\n\
        }\n\
\n\
/////////////////////////////////////////////////////\n\
// read sRv4 macros\n\
/////////////////////////////////////////////////////\n\
\n\
#define READ_sRv4_SIZE1(_Rv4, _sm_base_v4, _sRv4_read) \\\n\
        { \\\n\
            if(dCv4_x_valid) \\\n\
            { \\\n\
                _Rv4[0] = _sm_base_v4[_sRv4_read]; \\\n\
            } \\\n\
            \\\n\
            _sRv4_read += CTA_SIZE_IN_THD; \\\n\
        }\n\
\n\
#define READ_sRv4_SIZE2(_Rv4, _sm_base_v4, _sRv4_read) \\\n\
        { \\\n\
            if(dCv4_x_valid) \\\n\
            { \\\n\
                _Rv4[0] = _sm_base_v4[_sRv4_read]; \\\n\
                _Rv4[1] = _sm_base_v4[_sRv4_read + TILE_M_V1_PER_CTA * TILE_N_V8_PER_CTA * 1]; \\\n\
            } \\\n\
            \\\n\
            _sRv4_read += CTA_SIZE_IN_THD; \\\n\
        }\n\
\n\
#define READ_sRv4_SIZE4(_Rv4, _sm_base_v4, _sRv4_read) \\\n\
        { \\\n\
            if(dCv4_x_valid) \\\n\
            { \\\n\
                _Rv4[0] = _sm_base_v4[_sRv4_read]; \\\n\
                _Rv4[1] = _sm_base_v4[_sRv4_read + TILE_M_V1_PER_CTA * TILE_N_V8_PER_CTA * 1]; \\\n\
                _Rv4[2] = _sm_base_v4[_sRv4_read + TILE_M_V1_PER_CTA * TILE_N_V8_PER_CTA * 2]; \\\n\
                _Rv4[3] = _sm_base_v4[_sRv4_read + TILE_M_V1_PER_CTA * TILE_N_V8_PER_CTA * 3]; \\\n\
            } \\\n\
            \\\n\
            _sRv4_read += CTA_SIZE_IN_THD; \\\n\
        }\n\
\n\
\n\
/////////////////////////////////////////////////////\n\
// write sRv1 macros\n\
/////////////////////////////////////////////////////\n\
\n\
#define WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write, _C, _C_off) \\\n\
        { \\\n\
            _sm_base_v1[_sRv1_write + (smem_row_write_off ^ 0x0) * TILE_N_V2_PER_MMA] = _C[_C_off + 0]; \\\n\
        }\n\
\n\
#define WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write, _C, _C_off) \\\n\
        { \\\n\
            _sm_base_v1[_sRv1_write + (smem_row_write_off ^ 0x0) * TILE_N_V2_PER_MMA] = _C[_C_off + 0]; \\\n\
            _sm_base_v1[_sRv1_write + (smem_row_write_off ^ 0x1) * TILE_N_V2_PER_MMA] = _C[_C_off + 1]; \\\n\
        }\n\
\n\
#define WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write, _C, _C_off) \\\n\
        { \\\n\
            _sm_base_v1[_sRv1_write + (smem_row_write_off ^ 0x0) * TILE_N_V2_PER_MMA] = _C[_C_off + 0]; \\\n\
            _sm_base_v1[_sRv1_write + (smem_row_write_off ^ 0x1) * TILE_N_V2_PER_MMA] = _C[_C_off + 1]; \\\n\
            _sm_base_v1[_sRv1_write + (smem_row_write_off ^ 0x2) * TILE_N_V2_PER_MMA] = _C[_C_off + 2]; \\\n\
            _sm_base_v1[_sRv1_write + (smem_row_write_off ^ 0x3) * TILE_N_V2_PER_MMA] = _C[_C_off + 3]; \\\n\
        }\n\
\n\
#define WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write, _C, _C_off) \\\n\
        { \\\n\
            _sm_base_v1[_sRv1_write + (smem_row_write_off ^ 0x0) * TILE_N_V2_PER_MMA] = _C[_C_off + 0]; \\\n\
            _sm_base_v1[_sRv1_write + (smem_row_write_off ^ 0x1) * TILE_N_V2_PER_MMA] = _C[_C_off + 1]; \\\n\
            _sm_base_v1[_sRv1_write + (smem_row_write_off ^ 0x2) * TILE_N_V2_PER_MMA] = _C[_C_off + 2]; \\\n\
            _sm_base_v1[_sRv1_write + (smem_row_write_off ^ 0x3) * TILE_N_V2_PER_MMA] = _C[_C_off + 3]; \\\n\
            _sm_base_v1[_sRv1_write + (smem_row_write_off ^ 0x4) * TILE_N_V2_PER_MMA] = _C[_C_off + 4]; \\\n\
            _sm_base_v1[_sRv1_write + (smem_row_write_off ^ 0x5) * TILE_N_V2_PER_MMA] = _C[_C_off + 5]; \\\n\
            _sm_base_v1[_sRv1_write + (smem_row_write_off ^ 0x6) * TILE_N_V2_PER_MMA] = _C[_C_off + 6]; \\\n\
            _sm_base_v1[_sRv1_write + (smem_row_write_off ^ 0x7) * TILE_N_V2_PER_MMA] = _C[_C_off + 7]; \\\n\
        }\n\
\n\
/////////////////////////\n\
// tile_n_per_warp = 8\n\
/////////////////////////\n\
\n\
#define WRITE_sRv1_1x1(_sm_base_v1, _sRv1_write, _C) \\\n\
        { \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write, _C, 0); \\\n\
        }\n\
\n\
#define WRITE_sRv1_2x1(_sm_base_v1, _sRv1_write, _C) \\\n\
        { \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 0, _C, _1MMA_ * 0); \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 1, _C, _1MMA_ * 1); \\\n\
        }\n\
\n\
#define WRITE_sRv1_4x1(_sm_base_v1, _sRv1_write, _C) \\\n\
        { \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 0, _C, _1MMA_ * 0); \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 1, _C, _1MMA_ * 1); \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 2, _C, _1MMA_ * 2); \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 3, _C, _1MMA_ * 3); \\\n\
        }\n\
\n\
#define WRITE_sRv1_8x1(_sm_base_v1, _sRv1_write, _C) \\\n\
        { \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 0, _C, _1MMA_ * 0); \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 1, _C, _1MMA_ * 1); \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 2, _C, _1MMA_ * 2); \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 3, _C, _1MMA_ * 3); \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 4, _C, _1MMA_ * 4); \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 5, _C, _1MMA_ * 5); \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 6, _C, _1MMA_ * 6); \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 7, _C, _1MMA_ * 7); \\\n\
        }\n\
\n\
#define WRITE_sRv1_16x1(_sm_base_v1, _sRv1_write, _C) \\\n\
        { \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 0,  _C, _1MMA_ * 0); \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 1,  _C, _1MMA_ * 1); \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 2,  _C, _1MMA_ * 2); \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 3,  _C, _1MMA_ * 3); \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 4,  _C, _1MMA_ * 4); \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 5,  _C, _1MMA_ * 5); \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 6,  _C, _1MMA_ * 6); \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 7,  _C, _1MMA_ * 7); \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 8,  _C, _1MMA_ * 8); \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 9,  _C, _1MMA_ * 9); \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 10, _C, _1MMA_ * 10); \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 11, _C, _1MMA_ * 11); \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 12, _C, _1MMA_ * 12); \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 13, _C, _1MMA_ * 13); \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 14, _C, _1MMA_ * 14); \\\n\
            WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 15, _C, _1MMA_ * 15); \\\n\
        }\n\
\n\
/////////////////////////\n\
// tile_n_per_warp = 16\n\
/////////////////////////\n\
\n\
#define WRITE_sRv1_1x2(_sm_base_v1, _sRv1_write, _C) \\\n\
        { \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write, _C, 0); \\\n\
        }\n\
\n\
#define WRITE_sRv1_2x2(_sm_base_v1, _sRv1_write, _C) \\\n\
        { \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 0, _C, _2MMA_ * 0); \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 1, _C, _2MMA_ * 1); \\\n\
        }\n\
\n\
#define WRITE_sRv1_4x2(_sm_base_v1, _sRv1_write, _C) \\\n\
        { \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 0, _C, _2MMA_ * 0); \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 1, _C, _2MMA_ * 1); \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 2, _C, _2MMA_ * 2); \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 3, _C, _2MMA_ * 3); \\\n\
        }\n\
\n\
#define WRITE_sRv1_8x2(_sm_base_v1, _sRv1_write, _C) \\\n\
        { \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 0, _C, _2MMA_ * 0); \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 1, _C, _2MMA_ * 1); \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 2, _C, _2MMA_ * 2); \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 3, _C, _2MMA_ * 3); \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 4, _C, _2MMA_ * 4); \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 5, _C, _2MMA_ * 5); \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 6, _C, _2MMA_ * 6); \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 7, _C, _2MMA_ * 7); \\\n\
        }\n\
\n\
#define WRITE_sRv1_16x2(_sm_base_v1, _sRv1_write, _C) \\\n\
        { \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 0,  _C, _2MMA_ * 0); \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 1,  _C, _2MMA_ * 1); \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 2,  _C, _2MMA_ * 2); \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 3,  _C, _2MMA_ * 3); \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 4,  _C, _2MMA_ * 4); \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 5,  _C, _2MMA_ * 5); \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 6,  _C, _2MMA_ * 6); \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 7,  _C, _2MMA_ * 7); \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 8,  _C, _2MMA_ * 8); \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 9,  _C, _2MMA_ * 9); \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 10, _C, _2MMA_ * 10); \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 11, _C, _2MMA_ * 11); \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 12, _C, _2MMA_ * 12); \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 13, _C, _2MMA_ * 13); \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 14, _C, _2MMA_ * 14); \\\n\
            WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 15, _C, _2MMA_ * 15); \\\n\
        }\n\
\n\
/////////////////////////\n\
// tile_n_per_warp = 32\n\
/////////////////////////\n\
\n\
#define WRITE_sRv1_1x4(_sm_base_v1, _sRv1_write, _C) \\\n\
        { \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write, _C, 0); \\\n\
        }\n\
\n\
#define WRITE_sRv1_2x4(_sm_base_v1, _sRv1_write, _C) \\\n\
        { \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 0, _C, _4MMA_ * 0); \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 1, _C, _4MMA_ * 1); \\\n\
        }\n\
\n\
#define WRITE_sRv1_4x4(_sm_base_v1, _sRv1_write, _C) \\\n\
        { \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 0, _C, _4MMA_ * 0); \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 1, _C, _4MMA_ * 1); \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 2, _C, _4MMA_ * 2); \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 3, _C, _4MMA_ * 3); \\\n\
        }\n\
\n\
#define WRITE_sRv1_8x4(_sm_base_v1, _sRv1_write, _C) \\\n\
        { \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 0, _C, _4MMA_ * 0); \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 1, _C, _4MMA_ * 1); \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 2, _C, _4MMA_ * 2); \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 3, _C, _4MMA_ * 3); \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 4, _C, _4MMA_ * 4); \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 5, _C, _4MMA_ * 5); \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 6, _C, _4MMA_ * 6); \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 7, _C, _4MMA_ * 7); \\\n\
        }\n\
\n\
#define WRITE_sRv1_16x4(_sm_base_v1, _sRv1_write, _C) \\\n\
        { \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 0,  _C, _4MMA_ * 0); \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 1,  _C, _4MMA_ * 1); \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 2,  _C, _4MMA_ * 2); \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 3,  _C, _4MMA_ * 3); \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 4,  _C, _4MMA_ * 4); \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 5,  _C, _4MMA_ * 5); \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 6,  _C, _4MMA_ * 6); \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 7,  _C, _4MMA_ * 7); \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 8,  _C, _4MMA_ * 8); \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 9,  _C, _4MMA_ * 9); \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 10, _C, _4MMA_ * 10); \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 11, _C, _4MMA_ * 11); \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 12, _C, _4MMA_ * 12); \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 13, _C, _4MMA_ * 13); \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 14, _C, _4MMA_ * 14); \\\n\
            WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 15, _C, _4MMA_ * 15); \\\n\
        }\n\
\n\
/////////////////////////\n\
// tile_n_per_warp = 64\n\
/////////////////////////\n\
\n\
#define WRITE_sRv1_1x8(_sm_base_v1, _sRv1_write, _C) \\\n\
        { \\\n\
            WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write, _C, 0); \\\n\
        }\n\
\n\
#define WRITE_sRv1_2x8(_sm_base_v1, _sRv1_write, _C) \\\n\
        { \\\n\
            WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 0, _C, _8MMA_ * 0); \\\n\
            WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 1, _C, _8MMA_ * 1); \\\n\
        }\n\
\n\
#define WRITE_sRv1_4x8(_sm_base_v1, _sRv1_write, _C) \\\n\
        { \\\n\
            WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 0, _C, _8MMA_ * 0); \\\n\
            WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 1, _C, _8MMA_ * 1); \\\n\
            WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 2, _C, _8MMA_ * 2); \\\n\
            WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 3, _C, _8MMA_ * 3); \\\n\
        }\n\
\n\
#define WRITE_sRv1_8x8(_sm_base_v1, _sRv1_write, _C) \\\n\
        { \\\n\
            WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 0, _C, _8MMA_ * 0); \\\n\
            WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 1, _C, _8MMA_ * 1); \\\n\
            WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 2, _C, _8MMA_ * 2); \\\n\
            WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 3, _C, _8MMA_ * 3); \\\n\
            WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 4, _C, _8MMA_ * 4); \\\n\
            WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 5, _C, _8MMA_ * 5); \\\n\
            WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 6, _C, _8MMA_ * 6); \\\n\
            WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 7, _C, _8MMA_ * 7); \\\n\
        }\n\
\n\
");
    header_code_.emplace("/mnt/hpc/xusi/ppl.jit/src/ppl/nn/engines/cuda/impls/src/nn/conv/2spk/common/smem_macros.h", "// Licensed to the Apache Software Foundation (ASF) under one\n\
// or more contributor license agreements.  See the NOTICE file\n\
// distributed with this work for additional information\n\
// regarding copyright ownership.  The ASF licenses this file\n\
// to you under the Apache License, Version 2.0 (the\n\
// \"License\"); you may not use this file except in compliance\n\
// with the License.  You may obtain a copy of the License at\n\
//\n\
//   http://www.apache.org/licenses/LICENSE-2.0\n\
//\n\
// Unless required by applicable law or agreed to in writing,\n\
// software distributed under the License is distributed on an\n\
// \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n\
// KIND, either express or implied.  See the License for the\n\
// specific language governing permissions and limitations\n\
// under the License.\n\
\n\
/////////////////////////////////////////////////////\n\
// common write shared memory macros\n\
/////////////////////////////////////////////////////\n\
\n\
#define WRITE_sUv4_SIZE_16TH(_sm_base_v4, _sm_off, _reg) \\\n\
        { \\\n\
            if(tid < ( CTA_SIZE_IN_THD / 16 ))  \\\n\
                _sm_base_v4[_sm_off] = _reg[0]; \\\n\
        }\n\
\n\
#define WRITE_sUv4_SIZE_8TH(_sm_base_v4, _sm_off, _reg) \\\n\
        { \\\n\
            if(tid < ( CTA_SIZE_IN_THD / 8 ))  \\\n\
                _sm_base_v4[_sm_off] = _reg[0]; \\\n\
        }\n\
\n\
#define WRITE_sUv4_SIZE_QTR(_sm_base_v4, _sm_off, _reg) \\\n\
        { \\\n\
            if(tid < ( CTA_SIZE_IN_THD / 4 ))  \\\n\
                _sm_base_v4[_sm_off] = _reg[0]; \\\n\
        }\n\
\n\
#define WRITE_sUv4_SIZE_HALF(_sm_base_v4, _sm_off, _reg) \\\n\
        { \\\n\
            if(tid < ( CTA_SIZE_IN_THD / 2 ))  \\\n\
                _sm_base_v4[_sm_off] = _reg[0]; \\\n\
        }\n\
\n\
#define WRITE_sUv4_SIZE1(_sm_base_v4, _sm_off, _reg) \\\n\
        { \\\n\
            _sm_base_v4[_sm_off] = _reg[0]; \\\n\
        }\n\
\n\
#define WRITE_sUv4_SIZE2(_sm_base_v4, _sm_off, _reg) \\\n\
        { \\\n\
            _sm_base_v4[_sm_off] = _reg[0]; \\\n\
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 1] = _reg[1]; \\\n\
        }\n\
\n\
#define WRITE_sUv4_SIZE4(_sm_base_v4, _sm_off, _reg) \\\n\
        { \\\n\
            _sm_base_v4[_sm_off] = _reg[0]; \\\n\
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 1] = _reg[1]; \\\n\
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 2] = _reg[2]; \\\n\
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 3] = _reg[3]; \\\n\
        }\n\
\n\
#define WRITE_sUv4_SIZE8(_sm_base_v4, _sm_off, _reg) \\\n\
        { \\\n\
            _sm_base_v4[_sm_off] = _reg[0]; \\\n\
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 1] = _reg[1]; \\\n\
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 2] = _reg[2]; \\\n\
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 3] = _reg[3]; \\\n\
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 4] = _reg[4]; \\\n\
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 5] = _reg[5]; \\\n\
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 6] = _reg[6]; \\\n\
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 7] = _reg[7]; \\\n\
        }\n\
\n\
#define WRITE_sUv4_SIZE16(_sm_base_v4, _sm_off, _reg) \\\n\
        { \\\n\
            _sm_base_v4[_sm_off] = _reg[0]; \\\n\
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 1]  = _reg[1]; \\\n\
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 2]  = _reg[2]; \\\n\
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 3]  = _reg[3]; \\\n\
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 4]  = _reg[4]; \\\n\
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 5]  = _reg[5]; \\\n\
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 6]  = _reg[6]; \\\n\
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 7]  = _reg[7]; \\\n\
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 8]  = _reg[8]; \\\n\
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 9]  = _reg[9]; \\\n\
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 10] = _reg[10]; \\\n\
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 11] = _reg[11]; \\\n\
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 12] = _reg[12]; \\\n\
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 13] = _reg[13]; \\\n\
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 14] = _reg[14]; \\\n\
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 15] = _reg[15]; \\\n\
        }\n\
\n\
////////////////////////////////////////////////////\n\
// read shared memory macros\n\
////////////////////////////////////////////////////\n\
\n\
#define REG_sAv1_SIZE   (TILE_M_V1_PER_THD)\n\
#define REG_sBv1_SIZE   (TILE_N_V2_PER_THD)\n\
\n\
#define READ_sUv1_SIZE1(_reg, _reg_off, _smp_base_v1, _sUv1_read) \\\n\
        { \\\n\
            LDSM_ROW_X1_INST(_reg[_reg_off], _smp_base_v1 + _INT_TO_BYTE_ * (_sUv1_read) ); \\\n\
        }\n\
\n\
#define READ_sUv1_SIZE2(_reg, _reg_off, _smp_base_v1, _sUv1_read) \\\n\
        { \\\n\
            LDSM_ROW_X2_INST(_reg[_reg_off], _reg[_reg_off + 1], _smp_base_v1 + _INT_TO_BYTE_ * (_sUv1_read) ); \\\n\
        }\n\
\n\
#define READ_sUv1_SIZE4(_reg, _reg_off, _smp_base_v1, _sUv1_read) \\\n\
        { \\\n\
            LDSM_ROW_X4_INST(_reg[_reg_off], _reg[_reg_off + 1], _reg[_reg_off + 2], _reg[_reg_off + 3], _smp_base_v1 + _INT_TO_BYTE_ * (_sUv1_read) ); \\\n\
        }\n\
\n\
#define READ_sUv1_1x1(_reg, _smp_base_v1, _sUv1_read) \\\n\
        { \\\n\
            READ_sUv1_SIZE1(_reg, 0, _smp_base_v1, _sUv1_read); \\\n\
        }\n\
\n\
#define READ_sUv1_2x1(_reg, _smp_base_v1, _sUv1_read) \\\n\
        { \\\n\
            READ_sUv1_SIZE1(_reg, 0, _smp_base_v1, _sUv1_read); \\\n\
            READ_sUv1_SIZE1(_reg, 1, _smp_base_v1, _sUv1_read + TILE_K_V2_PER_CTA * (WARP_SIZE_IN_THD / 4) ); \\\n\
        }\n\
\n\
#define READ_sUv1_1x2(_reg, _smp_base_v1, _sUv1_read) \\\n\
        { \\\n\
            READ_sUv1_SIZE2(_reg, 0, _smp_base_v1, _sUv1_read); \\\n\
        }\n\
\n\
#define READ_sUv1_2x2(_reg, _smp_base_v1, _sUv1_read) \\\n\
        { \\\n\
            READ_sUv1_SIZE2(_reg, 0, _smp_base_v1, _sUv1_read); \\\n\
            READ_sUv1_SIZE2(_reg, 2, _smp_base_v1, _sUv1_read + TILE_K_V2_PER_CTA * (WARP_SIZE_IN_THD / 2) ); \\\n\
        }\n\
\n\
#define READ_sUv1_1x4(_reg, _smp_base_v1, _sUv1_read) \\\n\
        { \\\n\
            READ_sUv1_SIZE4(_reg, 0, _smp_base_v1, _sUv1_read); \\\n\
        }\n\
\n\
#define READ_sUv1_2x4(_reg, _smp_base_v1, _sUv1_read) \\\n\
        { \\\n\
            READ_sUv1_SIZE4(_reg, 0, _smp_base_v1, _sUv1_read); \\\n\
            READ_sUv1_SIZE4(_reg, 4, _smp_base_v1, _sUv1_read + TILE_K_V2_PER_CTA * WARP_SIZE_IN_THD); \\\n\
        }\n\
\n\
#define READ_sUv1_4x4(_reg, _smp_base_v1, _sUv1_read) \\\n\
        { \\\n\
            READ_sUv1_SIZE4(_reg, 0,  _smp_base_v1, _sUv1_read); \\\n\
            READ_sUv1_SIZE4(_reg, 4,  _smp_base_v1, _sUv1_read + TILE_K_V2_PER_CTA * WARP_SIZE_IN_THD * 1); \\\n\
            READ_sUv1_SIZE4(_reg, 8,  _smp_base_v1, _sUv1_read + TILE_K_V2_PER_CTA * WARP_SIZE_IN_THD * 2); \\\n\
            READ_sUv1_SIZE4(_reg, 12, _smp_base_v1, _sUv1_read + TILE_K_V2_PER_CTA * WARP_SIZE_IN_THD * 3); \\\n\
        }\n\
\n\
#define READ_sUv1_1x8(_reg, _smp_base_v1, _sUv1_read) \\\n\
        { \\\n\
            READ_sUv1_SIZE4(_reg, 0, _smp_base_v1, _sUv1_read); \\\n\
            READ_sUv1_SIZE4(_reg, 4, _smp_base_v1, _sUv1_read + TILE_K_V2_PER_CTA * WARP_SIZE_IN_THD); \\\n\
        }\n\
\n\
");
    header_code_.emplace("/mnt/hpc/xusi/ppl.jit/src/ppl/nn/engines/cuda/impls/src/nn/conv/2spk/common/output_macros.h", "// Licensed to the Apache Software Foundation (ASF) under one\n\
// or more contributor license agreements.  See the NOTICE file\n\
// distributed with this work for additional information\n\
// regarding copyright ownership.  The ASF licenses this file\n\
// to you under the Apache License, Version 2.0 (the\n\
// \"License\"); you may not use this file except in compliance\n\
// with the License.  You may obtain a copy of the License at\n\
//\n\
//   http://www.apache.org/licenses/LICENSE-2.0\n\
//\n\
// Unless required by applicable law or agreed to in writing,\n\
// software distributed under the License is distributed on an\n\
// \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n\
// KIND, either express or implied.  See the License for the\n\
// specific language governing permissions and limitations\n\
// under the License.\n\
\n\
#if defined(ENABLE_FUSE)\n\
\n\
#define OUTPUT_PRC_HALF(_Rv4) \\\n\
        { \\\n\
            if( dCv4_x_valid && dCv4_y_valid ) \\\n\
            { \\\n\
                dC[ concatV4_off + dCv4_off ] = _Rv4[0]; \\\n\
            } \\\n\
        }\n\
\n\
#else\n\
\n\
#define OUTPUT_PRC_HALF(_Rv4) \\\n\
        { \\\n\
            if( dCv4_x_valid && dCv4_y_valid ) \\\n\
            { \\\n\
                dC[ dCv4_off ] = _Rv4[0]; \\\n\
            } \\\n\
        }\n\
#endif\n\
\n\
#define ADD_BIAS_V4(_has_bias, _bias) \\\n\
        { \\\n\
            if( _has_bias && dCv4_x_valid && dCv4_y_valid ) \\\n\
            { \\\n\
	            int4      _biasV4 = ((int4 *) _bias) [grp_id * num_flt_per_grp_pad_v8 + dCv4_idx]; \\\n\
	            __half2 * _h2Bias = (__half2 *) &_biasV4; \\\n\
                \\\n\
                _Pragma(\"unroll\") \\\n\
	            for(int i = 0; i < _INT4_TO_4HALF2_; i++) \\\n\
                { \\\n\
	                h2R[i] = __hadd2(h2R[i], _h2Bias[i]); \\\n\
	            } \\\n\
            } \\\n\
        }\n\
\n\
#define FUSE_RELU_V4(_has_relu) \\\n\
        { \\\n\
	        if( _has_relu && dCv4_x_valid  && dCv4_y_valid ) \\\n\
            { \\\n\
		        if(_has_relu == 1) \\\n\
                { \\\n\
	                int * Rv1 = (int *) Rv4; \\\n\
                    \\\n\
                    _Pragma(\"unroll\") \\\n\
	                for(int i = 0; i < _INT4_TO_4HALF2_; i++) \\\n\
                    { \\\n\
	                    Rv1[i] = __vmaxs2(Rv1[i], 0); \\\n\
	                } \\\n\
	            } \\\n\
                else if(_has_relu == 2) \\\n\
                { \\\n\
	                __half2 * h2R = (__half2 *) Rv4; \\\n\
			        __half2 h2ONE((__half) 1.f, (__half) 1.f); \\\n\
                    \\\n\
                    _Pragma(\"unroll\") \\\n\
	                for(int i = 0; i < _INT4_TO_4HALF2_; i++) \\\n\
                    { \\\n\
				        h2R[i] = __h2div(h2exp(h2R[i]), __hadd2(h2ONE, h2exp(h2R[i]))); \\\n\
	                } \\\n\
	            } \\\n\
		    } \\\n\
        }\n\
\n\
#define FUSE_CLIP_V4(_has_clip, _clip_max, _clip_min) \\\n\
        { \\\n\
	        if( _has_clip && dCv4_x_valid  && dCv4_y_valid ) { \\\n\
                \\\n\
                _Pragma(\"unroll\") \\\n\
	            for(int i = 0; i < _INT4_TO_4HALF2_; i++) \\\n\
                { \\\n\
	    	        h2R[i].x = __hgt(h2R[i].x, _clip_max.x) ? _clip_max.x : h2R[i].x; \\\n\
	    	        h2R[i].y = __hgt(h2R[i].y, _clip_max.x) ? _clip_max.y : h2R[i].y; \\\n\
	    	        h2R[i].x = __hlt(h2R[i].x, _clip_min.x) ? _clip_min.x : h2R[i].x; \\\n\
	    	        h2R[i].y = __hlt(h2R[i].y, _clip_min.x) ? _clip_min.y : h2R[i].y; \\\n\
	            } \\\n\
	        } \\\n\
        }\n\
\n\
#define FUSE_PRELU_V4(_has_prelu, _prelu, _leaky) \\\n\
        { \\\n\
	        if( _has_prelu && dCv4_x_valid  && dCv4_y_valid ) { \\\n\
                \\\n\
       	        if(_has_prelu == 1) \\\n\
                { \\\n\
                    _Pragma(\"unroll\") \\\n\
	                for(int i = 0; i < _INT4_TO_8HALF_; i++) \\\n\
                    { \\\n\
	            	    if( __hlt(hR[i],0) )    hR[i] = __hmul(hR[i], _leaky); \\\n\
	                } \\\n\
	            } \\\n\
                \\\n\
       	        if(_has_prelu == 2) \\\n\
                { \\\n\
	                int4     _scale_v4 = ( (int4 *) _prelu) [grp_id * num_flt_per_grp_pad_v8 + dCv4_idx]; \\\n\
	                __half * _hscale  = (__half *) &_scale_v4; \\\n\
                    \\\n\
                    _Pragma(\"unroll\") \\\n\
	                for(int i = 0; i < _INT4_TO_8HALF_; i++) \\\n\
                    { \\\n\
	            	    if( __hlt(hR[i], 0) )   hR[i] = __hmul(hR[i], _hscale[i]); \\\n\
	                } \\\n\
	            } \\\n\
                \\\n\
       	        if(_has_prelu == 3) \\\n\
                { \\\n\
                    int4     _scale_v4 = ((int4  *) _prelu) [dCv4_off]; \\\n\
	                __half * _hscale  = (__half *) &_scale_v4; \\\n\
                    \\\n\
                    _Pragma(\"unroll\") \\\n\
	                for(int i = 0; i < _INT4_TO_8HALF_; i++) \\\n\
                    { \\\n\
	            	    if( __hlt(hR[i], 0) )    hR[i] = __hmul(hR[i], _hscale[i]); \\\n\
	                } \\\n\
	            } \\\n\
	        } \\\n\
        }\n\
\n\
#define FUSE_ELT_V4(_has_elt, _pre_data) \\\n\
        { \\\n\
	        if( _has_elt && dCv4_x_valid && dCv4_y_valid ) \\\n\
            { \\\n\
	            int4      _elt_v4 = ((int4 *)   _pre_data) [dCv4_off]; \\\n\
	            __half2 * _h2_elt = (__half2 *) &_elt_v4; \\\n\
                \\\n\
                _Pragma(\"unroll\") \\\n\
	            for(int i = 0; i < _INT4_TO_4HALF2_; i++){ \\\n\
	                h2R[i] = __hadd2(h2R[i], _h2_elt[i]); \\\n\
	            } \\\n\
	        } \\\n\
        }\n\
\n\
#define SET_CONCAT_OFF_V4(_has_concat, _concatV4_off) \\\n\
        { \\\n\
	        if( _has_concat && dCv4_x_valid && dCv4_y_valid ) \\\n\
            { \\\n\
	            dCv4_off = concat_offset_v8 + dCv4_idy * concat_stride_v8 + dCv4_base + dCv4_idx; \\\n\
	        } \\\n\
        }\n\
        \n\
");
    header_code_.emplace("/mnt/hpc/xusi/ppl.jit/src/ppl/nn/engines/cuda/impls/src/nn/conv/2spk/common/main_body.h", "// Licensed to the Apache Software Foundation (ASF) under one\n\
// or more contributor license agreements.  See the NOTICE file\n\
// distributed with this work for additional information\n\
// regarding copyright ownership.  The ASF licenses this file\n\
// to you under the Apache License, Version 2.0 (the\n\
// \"License\"); you may not use this file except in compliance\n\
// with the License.  You may obtain a copy of the License at\n\
//\n\
//   http://www.apache.org/licenses/LICENSE-2.0\n\
//\n\
// Unless required by applicable law or agreed to in writing,\n\
// software distributed under the License is distributed on an\n\
// \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n\
// KIND, either express or implied.  See the License for the\n\
// specific language governing permissions and limitations\n\
// under the License.\n\
\n\
#if defined(ENABLE_SPLITK)\n\
__global__ void __launch_bounds__(CTA_SIZE_IN_THD) KERNEL_NAME(SPK_KPARAM_LIST)\n\
#elif defined(ENABLE_FUSE) || defined(ENABLE_SPLITF)\n\
__global__ void __launch_bounds__(CTA_SIZE_IN_THD) KERNEL_NAME(TOTAL_KPARAM_LIST)\n\
#endif\n\
{\n\
#if (__CUDA_ARCH__ >= 750) && (__CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 >= 10020)\n\
    int4 Cv4[Cv4_ITEMS_PER_THD];\n\
\n\
    __half * hC = (__half *) Cv4;\n\
    int *     C = (int *)    Cv4;\n\
\n\
    for (int i = 0; i < HC_ITEMS_PER_THD; i++) { hC[i] = _HALF_ZERO_; }\n\
\n\
    int4  Rv4[INTER_SET_REDUCE_RATIO];\n\
\n\
#if defined(ENABLE_FUSE) || ((defined(ENABLE_SPLITF) || defined(ENABLE_SPLITK)) && (TILE_K_PER_CTA > TILE_K_PER_SET))\n\
    __half2 * h2R = (__half2 *) Rv4;\n\
#endif\n\
\n\
#if defined(ENABLE_FUSE)\n\
    __half  *  hR = (__half  *) Rv4;\n\
#endif\n\
\n\
    uint tid       =  threadIdx.x;\n\
\n\
    uint local_tid =  tid & 0x1f;\n\
\n\
    uint set_tid   =  tid & (SET_SIZE_IN_THD - 1);\n\
\n\
    uint set_id    = (tid >> SET_SIZE_IN_BITS) & 0x7;\n\
\n\
    uint set_widx  = (set_tid >>  WARP_SIZE_IN_BITS) & (SET_SIZE_X_IN_WARP - 1);\n\
    uint set_widy  =  set_tid >> (WARP_SIZE_IN_BITS  +  SET_SIZE_X_IN_BITS);\n\
 \n\
    uint ldg_idx   =  tid % TILE_K_V8_PER_CTA;\n\
    uint ldg_idy   =  tid / TILE_K_V8_PER_CTA;\n\
\n\
#if TILE_K_PER_CTA == 8\n\
    uint sts_idx   =   0;\n\
    uint sts_idy   =   tid;\n\
#elif TILE_K_PER_CTA == 16\n\
    uint sts_idx   = ((tid & 0x1) ^ ((tid & 0xf) >> 3));\n\
    uint sts_idy   =   tid >> 1;\n\
#elif TILE_K_PER_CTA == 32\n\
    uint sts_idx   = ((tid & 0x3) ^ ((tid & 0x1f) >> 3));\n\
    uint sts_idy   =   tid >> 2;\n\
#elif TILE_K_PER_CTA == 64\n\
    uint sts_idx   = ((tid & 0x7) ^ ((tid & 0x3f) >> 3));\n\
    uint sts_idy   =   tid >> 3;\n\
#elif TILE_K_PER_CTA == 128\n\
    uint sts_idx   = ((tid & 0xf) ^ ((tid & 0x7f) >> 4));\n\
    uint sts_idy   =   tid >> 4;\n\
#endif\n\
\n\
    uint cta_idx   = blockIdx.y;\n\
    uint cta_idy   = blockIdx.x;\n\
\n\
#if defined(ENABLE_SPLITK) && defined(ENABLE_SPLITF)\n\
    uint spk_id    =  blockIdx.z %  splitk;\n\
    uint spf_id    = (blockIdx.z % (splitk * flt_hw)) / splitk;\n\
    uint grp_id    =  blockIdx.z / (splitk * flt_hw);\n\
#elif defined(ENABLE_SPLITK) && !defined(ENABLE_SPLITF)\n\
    uint spk_id    =  blockIdx.z %  splitk;\n\
    uint grp_id    =  blockIdx.z /  splitk;\n\
#elif !defined(ENABLE_SPLITK) && defined(ENABLE_SPLITF)\n\
    uint spf_id    =  blockIdx.z %  flt_hw;\n\
    uint grp_id    =  blockIdx.z /  flt_hw;\n\
#elif defined(ENABLE_FUSE)\n\
    uint grp_id    = blockIdx.z;\n\
#endif\n\
\n\
    uint num_chl_per_grp_pad_v8 = num_chl_per_grp_pad >> 3;\n\
    uint num_flt_per_grp_pad_v8 = num_flt_per_grp_pad >> 3;\n\
\n\
    uint dCv4_idy   =  cta_idy  * TILE_M_V1_PER_CTA  +\n\
                       tid      / TILE_N_V8_PER_CTA;\n\
\n\
    uint dCv4_idx   =  cta_idx  * TILE_N_V8_PER_CTA  +\n\
                       tid      % TILE_N_V8_PER_CTA;\n\
\n\
    bool dCv4_x_valid =  (dCv4_idx < num_flt_per_grp_pad_v8) & ((tid / TILE_N_V8_PER_CTA) < TILE_M_PER_CTA);\n\
\n\
#if defined(ENABLE_SPLITK) && defined(ENABLE_SPLITF)\n\
    uint dCv4_base  = (spf_id   * splitk + spk_id)  * num_flt_per_grp_pad_v8 * num_grp * out_hw * in_num +\n\
                       grp_id   * num_flt_per_grp_pad_v8;\n\
#elif defined(ENABLE_SPLITK) && !defined(ENABLE_SPLITF)\n\
    uint dCv4_base  =  spk_id   * num_flt_per_grp_pad_v8 * num_grp * out_hw * in_num +\n\
                       grp_id   * num_flt_per_grp_pad_v8;\n\
#elif !defined(ENABLE_SPLITK) && defined(ENABLE_SPLITF)\n\
    uint dCv4_base  =  spf_id   * num_flt_per_grp_pad_v8 * num_grp * out_hw * in_num +\n\
                       grp_id   * num_flt_per_grp_pad_v8;\n\
#elif defined(ENABLE_FUSE)\n\
    uint dCv4_base  =  grp_id   * num_flt_per_grp_pad_v8;\n\
#endif\n\
\n\
    uint mma_idx    =  local_tid %  MMA_SIZE_X_IN_THD;\n\
    uint mma_idy    =  local_tid >> MMA_SIZE_X_IN_BITS;\n\
\n\
    uint smem_row_write_id  =  (set_widx * TILE_N_V8_PER_WARP) / SMEM_ROW_V4_SIZE;\n\
#if (SET_SIZE_Y_IN_WARP * INTER_SET_REDUCE_RATIO * WARP_SIZE_IN_THD / TILE_N_V8_PER_WARP) == 4\n\
    uint smem_row_write_off = ((set_widx * TILE_N_V8_PER_WARP) ^ ((mma_idy % 4) / N_ROWS_PER_SMEM_ROW)\n\
#else\n\
    uint smem_row_write_off = ((set_widx * TILE_N_V8_PER_WARP) ^ (mma_idy  / N_ROWS_PER_SMEM_ROW)\n\
#endif\n\
                       ) % SMEM_ROW_V4_SIZE;\n\
\n\
    uint sRv1_write =  set_id     * TILE_N_V2_PER_CTA    * TILE_M_V1_PER_CTA  +\n\
                       set_widy   * TILE_N_V2_PER_CTA    * TILE_M_V1_PER_WARP +\n\
                       mma_idy    * TILE_N_V2_PER_CTA    +\n\
                       smem_row_write_id  * SMEM_ROW_V1_SIZE     +\n\
                       mma_idx;\n\
\n\
    uint mma_read_idx =  tid % TILE_N_V8_PER_CTA;\n\
    uint mma_read_idy =  tid / TILE_N_V8_PER_CTA; \n\
\n\
    uint smem_row_read_id  =  mma_read_idx / SMEM_ROW_V4_SIZE;\n\
    uint smem_row_read_off =  mma_read_idx % SMEM_ROW_V4_SIZE;\n\
\n\
    uint sRv4_read  = (mma_read_idy / TILE_M_PER_MMA_HALF) * TILE_N_V8_PER_CTA    * TILE_M_PER_MMA_HALF     +\n\
                      (mma_read_idy % TILE_M_PER_MMA_HALF) * TILE_N_V8_PER_CTA    +\n\
                      smem_row_read_id  * SMEM_ROW_V4_SIZE +\n\
                    (((mma_read_idy % TILE_M_PER_MMA_HALF) / N_ROWS_PER_SMEM_ROW) ^ smem_row_read_off);\n\
\n\
    const int4 ZEROv4 = {0, 0, 0, 0};\n\
\n\
#if defined(FLT_SIZE3)\n\
    int flt_hw_id  = 0;\n\
    int flt_hw_bid = 0x1;\n\
\n\
    int lut_id     = 0;\n\
#elif defined(FLT_SIZEN)\n\
    int  flt_h_id  = 0;\n\
    int  flt_w_id  = 0;\n\
\n\
    int lut_id     = 0;\n\
#endif\n\
\n\
#if defined(ENABLE_SPLITK)\n\
    int  flt_c_v8_end = chl_lut.idx[spk_id + 1] >> 3;\n\
    int  flt_c_v8_id  = ldg_idx + (chl_lut.idx[spk_id] >> 3);\n\
#elif defined(ENABLE_SPLITF) || defined(ENABLE_FUSE)\n\
    int  flt_c_v8_end = num_chl_per_grp_pad_v8;\n\
    int  flt_c_v8_id  = ldg_idx;\n\
#endif\n\
\n\
    bool flt_c_v8_valid  = flt_c_v8_id < flt_c_v8_end;\n\
\n\
    int4 reg_dAv4[REG_dAv4_SIZE];\n\
    int4 reg_dBv4[REG_dBv4_SIZE];\n\
\n\
#if defined(FLT_SIZE1)\n\
    int     dAv4_off[READ_dAv4_STEPS];\n\
    bool in_hw_valid[READ_dAv4_STEPS];\n\
\n\
    for(int i = 0; i < READ_dAv4_STEPS; i++)\n\
    {\n\
        SET_dAv4_BOUND(i, dAv4_off[i], in_hw_valid[i]);\n\
    }\n\
#elif defined(FLT_SIZE3)\n\
    int dAv4_off[READ_dAv4_STEPS];\n\
    int in_hw_mask[READ_dAv4_STEPS];\n\
\n\
    for(int i = 0; i < READ_dAv4_STEPS; i++)\n\
    {\n\
        SET_dAv4_BOUND(i, dAv4_off[i], in_hw_mask[i]);\n\
    }\n\
#elif defined(FLT_SIZEN)\n\
    int dAv4_off[READ_dAv4_STEPS];\n\
    int  in_n_id[READ_dAv4_STEPS];\n\
    int  in_h_id[READ_dAv4_STEPS];\n\
    int  in_w_id[READ_dAv4_STEPS];\n\
\n\
    int in_h_start[READ_dAv4_STEPS];\n\
    int in_w_start[READ_dAv4_STEPS];\n\
\n\
    for(int i = 0; i < READ_dAv4_STEPS; i++)\n\
    {\n\
        SET_dAv4_BOUND(i, dAv4_off[i], in_n_id[i], in_h_start[i], in_w_start[i]);\n\
        in_h_id[i] = in_h_start[i];\n\
        in_w_id[i] = in_w_start[i];\n\
    }\n\
#endif\n\
\n\
    int     dBv4_off[READ_dBv4_STEPS];\n\
    bool flt_n_valid[READ_dBv4_STEPS];\n\
\n\
    for(int i = 0; i < READ_dBv4_STEPS; i++)\n\
    {\n\
        SET_dBv4_BOUND(i, dBv4_off[i], flt_n_valid[i]);\n\
    }\n\
\n\
#if defined(USE_1BUF)\n\
    __shared__ int4 sm_base_v4[SM_BASE_V4_1BUF];\n\
#elif defined(USE_2BUF)\n\
    __shared__ int4 sm_base_v4[SM_BASE_V4_2BUF];\n\
#endif\n\
    int * sm_base_v1 = (int *) sm_base_v4;\n\
    \n\
    uint32_t smp_base_v1;\n\
\n\
    CVT_SM_PTR(smp_base_v1, sm_base_v1);\n\
\n\
    uint sAv4_write =  sts_idy  * TILE_K_V8_PER_CTA + sts_idx;\n\
\n\
#if defined(USE_1BUF)\n\
    uint sBv4_write =  sAv4_write + SM_A_V4_1BUF;\n\
#elif defined(USE_2BUF)\n\
    uint sBv4_write =  sAv4_write + SM_A_V4_2BUF;\n\
#endif\n\
\n\
    uint lds_idy =  local_tid;\n\
#if TILE_K_PER_CTA == 8\n\
    uint lds_idx =  0;\n\
#elif TILE_K_PER_CTA == 16\n\
    uint lds_idx = ((set_id * TILE_K_V8_PER_SET) & 0x1) ^ ((local_tid / K_ROWS_PER_SMEM_ROW) & 0x1);\n\
#elif TILE_K_PER_CTA == 32\n\
    uint lds_idx = ((set_id * TILE_K_V8_PER_SET) & 0x3) ^ ((local_tid / K_ROWS_PER_SMEM_ROW) & 0x3);\n\
#elif TILE_K_PER_CTA == 64\n\
    uint lds_idx = ((set_id * TILE_K_V8_PER_SET) & 0x7) ^ ((local_tid / K_ROWS_PER_SMEM_ROW) & 0x7);\n\
#elif TILE_K_PER_CTA == 128\n\
    uint lds_idx = ((set_id * TILE_K_V8_PER_SET) & 0xf) ^ (local_tid & 0x7);\n\
#endif\n\
\n\
    uint sAv1_read  =  set_widy   * TILE_M_PER_WARP        * TILE_K_V2_PER_CTA +\n\
#if TILE_M_PER_WARP == 16\n\
                      (lds_idy    % WARP_SIZE_IN_THD_HALF) * TILE_K_V2_PER_CTA +\n\
#elif TILE_M_PER_WARP == 32\n\
                       lds_idy    * TILE_K_V2_PER_CTA      +\n\
#elif TILE_M_PER_WARP == 64 || TILE_M_PER_WARP == 128\n\
                       lds_idy    * TILE_K_V2_PER_CTA      +\n\
#endif\n\
                       lds_idx    * _INT4_TO_4INT_;\n\
\n\
    uint sBv1_read  =  set_widx   * TILE_N_PER_WARP        * TILE_K_V2_PER_CTA +\n\
#if TILE_N_PER_WARP == 8\n\
                      (lds_idy    % WARP_SIZE_IN_THD_QTR)  * TILE_K_V2_PER_CTA +\n\
#elif TILE_N_PER_WARP == 16\n\
                      (lds_idy    % WARP_SIZE_IN_THD_HALF) * TILE_K_V2_PER_CTA +\n\
#elif TILE_N_PER_WARP == 32 || TILE_N_PER_WARP == 64\n\
                       lds_idy    * TILE_K_V2_PER_CTA      +\n\
#endif\n\
                       lds_idx    * _INT4_TO_4INT_         +\n\
#if defined(USE_1BUF)\n\
                       SM_A_V1_1BUF;\n\
#elif defined(USE_2BUF)\n\
                       SM_A_V1_2BUF;\n\
#endif\n\
\n\
    int db0_sBv1[REG_sBv1_SIZE];\n\
#if TILE_K_PER_SET == 16 || TILE_K_PER_SET == 32\n\
    int db1_sBv1[REG_sBv1_SIZE];\n\
#endif\n\
\n\
    int db0_sAv1[REG_sAv1_SIZE];\n\
#if TILE_K_PER_SET == 16 || TILE_K_PER_SET == 32\n\
    int db1_sAv1[REG_sAv1_SIZE];\n\
#endif\n\
\n\
#if defined(FLT_SIZE1)\n\
    LOAD_dAv4(reg_dAv4, dA, dAv4_off, flt_c_v8_valid, in_hw_valid);\n\
    LOAD_dBv4(reg_dBv4, dB, dBv4_off, flt_c_v8_valid, flt_n_valid);\n\
\n\
    FWD_FLT(flt_c_v8_id, flt_c_v8_valid);\n\
#elif defined(FLT_SIZE3)\n\
    LOAD_dAv4(reg_dAv4, dA, dAv4_off, flt_c_v8_valid, flt_hw_bid);\n\
    LOAD_dBv4(reg_dBv4, dB, dBv4_off, flt_c_v8_valid, flt_n_valid);\n\
\n\
    FWD_FLT(flt_hw_id, flt_hw_bid, flt_c_v8_id, flt_c_v8_valid);\n\
    FWD_LUT(lut_id);\n\
#elif defined(FLT_SIZEN)\n\
    LOAD_dAv4(reg_dAv4, dA, dAv4_off, in_n_id, in_h_id, in_w_id);\n\
    LOAD_dBv4(reg_dBv4, dB, dBv4_off, flt_c_v8_valid, flt_n_valid);\n\
\n\
    FWD_FLT(flt_h_id, flt_w_id, flt_c_v8_id, flt_c_v8_valid);\n\
    FWD_LUT(lut_id);\n\
#endif\n\
\n\
    WRITE_sAv4(sm_base_v4, sAv4_write, reg_dAv4);\n\
    WRITE_sBv4(sm_base_v4, sBv4_write, reg_dBv4);\n\
\n\
    __syncthreads();\n\
\n\
#if defined(USE_2BUF)\n\
    SWITCH_BUFFER(sAv4_write, SM_A_V4_1BUF, 0);\n\
    SWITCH_BUFFER(sBv4_write, SM_B_V4_1BUF, SM_A_V4_2BUF);\n\
#endif\n\
\n\
    READ_sAv1(db0_sAv1, smp_base_v1, sAv1_read);\n\
    READ_sBv1(db0_sBv1, smp_base_v1, sBv1_read);\n\
\n\
#if TILE_K_PER_SET == 16 || TILE_K_PER_SET == 32\n\
    FWD_KGROUP_STEP1(sAv1_read);\n\
    FWD_KGROUP_STEP1(sBv1_read);\n\
#endif\n\
\n\
#if defined(ENABLE_SPLITK)\n\
    for (uint j = 0; j < kloop_lut.idx[spk_id]; j++)\n\
#elif defined(ENABLE_SPLITF) || defined(ENABLE_FUSE)\n\
    for (uint j = 0; j < kloop_num; j++)\n\
#endif\n\
    {\n\
#if defined(FLT_SIZE1)\n\
        LOAD_dAv4(reg_dAv4, dA, dAv4_off, flt_c_v8_valid, in_hw_valid);\n\
        LOAD_dBv4(reg_dBv4, dB, dBv4_off, flt_c_v8_valid, flt_n_valid);\n\
\n\
        FWD_FLT(flt_c_v8_id, flt_c_v8_valid);\n\
#elif defined(FLT_SIZE3)\n\
        LOAD_dAv4(reg_dAv4, dA, dAv4_off, flt_c_v8_valid, flt_hw_bid);\n\
        LOAD_dBv4(reg_dBv4, dB, dBv4_off, flt_c_v8_valid, flt_n_valid);\n\
\n\
        FWD_FLT(flt_hw_id, flt_hw_bid, flt_c_v8_id, flt_c_v8_valid);\n\
        FWD_LUT(lut_id);\n\
#elif defined(FLT_SIZEN)\n\
        LOAD_dAv4(reg_dAv4, dA, dAv4_off, in_n_id, in_h_id, in_w_id);\n\
        LOAD_dBv4(reg_dBv4, dB, dBv4_off, flt_c_v8_valid, flt_n_valid);\n\
\n\
        FWD_FLT(flt_h_id, flt_w_id, flt_c_v8_id, flt_c_v8_valid);\n\
        FWD_LUT(lut_id);\n\
#endif\n\
\n\
#if TILE_K_PER_SET == 16 || TILE_K_PER_SET == 32\n\
        READ_sAv1(db1_sAv1, smp_base_v1, sAv1_read);\n\
        READ_sBv1(db1_sBv1, smp_base_v1, sBv1_read);\n\
#endif\n\
\n\
#if TILE_K_PER_SET == 16\n\
        FWD_KGROUP_STEP1(sAv1_read);\n\
        FWD_KGROUP_STEP1(sBv1_read);\n\
#elif TILE_K_PER_SET == 32\n\
        FWD_KGROUP_STEP2(sAv1_read);\n\
        FWD_KGROUP_STEP2(sBv1_read);\n\
#endif\n\
\n\
        MMA_INSTS(C, db0_sAv1, db0_sBv1);\n\
\n\
#if TILE_K_PER_SET == 32\n\
        READ_sAv1(db0_sAv1, smp_base_v1, sAv1_read);\n\
        READ_sBv1(db0_sBv1, smp_base_v1, sBv1_read);\n\
\n\
        FWD_KGROUP_STEP3(sAv1_read);\n\
        FWD_KGROUP_STEP3(sBv1_read);\n\
\n\
        MMA_INSTS(C, db1_sAv1, db1_sBv1);\n\
\n\
        READ_sAv1(db1_sAv1, smp_base_v1, sAv1_read);\n\
        READ_sBv1(db1_sBv1, smp_base_v1, sBv1_read);\n\
\n\
        FWD_KGROUP_STEP4(sAv1_read);\n\
        FWD_KGROUP_STEP4(sBv1_read);\n\
#endif\n\
\n\
#if defined(USE_1BUF)\n\
        __syncthreads();\n\
#endif\n\
\n\
        WRITE_sAv4(sm_base_v4, sAv4_write, reg_dAv4);\n\
        WRITE_sBv4(sm_base_v4, sBv4_write, reg_dBv4);\n\
\n\
#if TILE_K_PER_SET == 16\n\
        MMA_INSTS(C, db1_sAv1, db1_sBv1);\n\
#elif TILE_K_PER_SET == 32\n\
        MMA_INSTS(C, db0_sAv1, db0_sBv1);\n\
#endif\n\
\n\
#if defined(USE_2BUF)\n\
        SWITCH_BUFFER(sAv4_write, SM_A_V4_1BUF, 0);\n\
        SWITCH_BUFFER(sBv4_write, SM_B_V4_1BUF, SM_A_V4_2BUF);\n\
\n\
        SWITCH_BUFFER(sAv1_read,  SM_A_V1_1BUF, 0);\n\
        SWITCH_BUFFER(sBv1_read,  SM_B_V1_1BUF, SM_A_V1_2BUF);\n\
#endif\n\
\n\
        __syncthreads();\n\
\n\
        READ_sAv1(db0_sAv1, smp_base_v1, sAv1_read);\n\
        READ_sBv1(db0_sBv1, smp_base_v1, sBv1_read);\n\
\n\
#if TILE_K_PER_SET == 16 || TILE_K_PER_SET == 32\n\
        FWD_KGROUP_STEP1(sAv1_read);\n\
        FWD_KGROUP_STEP1(sBv1_read);\n\
#endif\n\
\n\
#if TILE_K_PER_SET == 32\n\
        MMA_INSTS(C, db1_sAv1, db1_sBv1);\n\
#endif\n\
    }\n\
\n\
    __syncthreads();\n\
\n\
    WRITE_sRv1(sm_base_v1, sRv1_write, C);\n\
\n\
    __syncthreads();\n\
\n\
    for(int s = 0; s < OUTPUT_STEPS; s++)\n\
    {\n\
        READ_sRv4(Rv4, sm_base_v4, sRv4_read);\n\
\n\
#if TILE_K_PER_CTA > TILE_K_PER_SET\n\
        REDUCE(h2R);\n\
#endif\n\
\n\
        bool dCv4_y_valid = (dCv4_idy  / out_hw) < in_num;\n\
        uint dCv4_off     =  dCv4_base +\n\
                             dCv4_idy  * num_flt_per_grp_pad_v8 * num_grp +\n\
                             dCv4_idx;\n\
\n\
#if defined(ENABLE_FUSE)\n\
        ADD_BIAS_V4(has_bias, bias);\n\
#endif\n\
\n\
#if defined(ENABLE_FUSE)\n\
        uint concatV4_off = 0;\n\
\n\
        FUSE_RELU_V4(has_relu);\n\
        FUSE_CLIP_V4(has_clip, clip_max, clip_min);\n\
        FUSE_PRELU_V4(has_prelu, prelu, leaky);\n\
\n\
        FUSE_ELT_V4(has_elt, pre_data);\n\
        FUSE_RELU_V4(has_elt_relu);\n\
        FUSE_CLIP_V4(has_elt_clip, elt_clip_max, elt_clip_min);\n\
        FUSE_PRELU_V4(has_elt_prelu, elt_prelu, elt_leaky);\n\
\n\
        SET_CONCAT_OFF_V4(has_concat, concatV4_off);\n\
#endif\n\
\n\
        OUTPUT_PRC_HALF(Rv4);\n\
\n\
        dCv4_idy += OUTPUT_SIZE_Y_IN_THD;\n\
    }\n\
\n\
#endif // __CUDA_ARCH__\n\
}\n\
");
    header_code_.emplace("/mnt/hpc/xusi/ppl.jit/src/ppl/nn/engines/cuda/impls/src/nn/conv/2spk/common/uni_undefs.h", "// Licensed to the Apache Software Foundation (ASF) under one\n\
// or more contributor license agreements.  See the NOTICE file\n\
// distributed with this work for additional information\n\
// regarding copyright ownership.  The ASF licenses this file\n\
// to you under the Apache License, Version 2.0 (the\n\
// \"License\"); you may not use this file except in compliance\n\
// with the License.  You may obtain a copy of the License at\n\
//\n\
//   http://www.apache.org/licenses/LICENSE-2.0\n\
//\n\
// Unless required by applicable law or agreed to in writing,\n\
// software distributed under the License is distributed on an\n\
// \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n\
// KIND, either express or implied.  See the License for the\n\
// specific language governing permissions and limitations\n\
// under the License.\n\
\n\
////////////////////////////////////////\n\
// kernel list macros\n\
////////////////////////////////////////\n\
\n\
#undef SPK_KPARAM_LIST\n\
#undef TOTAL_KPARAM_LIST\n\
\n\
////////////////////////////////////////\n\
// customized macros\n\
////////////////////////////////////////\n\
\n\
#undef TILE_N_PER_CTA\n\
#undef TILE_M_PER_CTA\n\
\n\
#undef TILE_K_PER_CTA\n\
#undef TILE_K_PER_SET\n\
#undef TILE_K_PER_WARP\n\
\n\
#undef TILE_N_PER_WARP\n\
#undef TILE_M_PER_WARP\n\
\n\
#undef KERNEL_NAME\n\
\n\
////////////////////////////////////////\n\
// align functions\n\
////////////////////////////////////////\n\
\n\
#undef Align\n\
#undef DivUp\n\
\n\
#undef Min\n\
#undef Max\n\
\n\
////////////////////////////////////////\n\
// boundary check\n\
////////////////////////////////////////\n\
\n\
#undef WidthInRange\n\
#undef HeightInRange\n\
\n\
////////////////////////////////////////\n\
// constant cta size macros\n\
////////////////////////////////////////\n\
\n\
#undef _4CHAR_TO_INT_\n\
#undef _4INT_TO_INT4_\n\
#undef _2INT_TO_INT2_\n\
\n\
#undef _2HALF_TO_INT_\n\
#undef _2INT2_TO_INT4_\n\
\n\
#undef _C1_\n\
#undef _C2_\n\
#undef _C4_\n\
#undef _C8_\n\
#undef _C16_\n\
#undef _C32_\n\
\n\
#undef _1INT_\n\
#undef _2INT_\n\
#undef _4INT_\n\
#undef _8INT_\n\
\n\
#undef _1INT4_\n\
#undef _2INT4_\n\
#undef _4INT4_\n\
#undef _8INT4_\n\
\n\
#undef _1INT8_\n\
#undef _2INT8_\n\
#undef _4INT8_\n\
#undef _8INT8_\n\
\n\
#undef _1HALF_\n\
#undef _2HALF_\n\
#undef _4HALF_\n\
#undef _8HALF_\n\
\n\
#undef _1HALF2_\n\
#undef _2HALF2_\n\
#undef _4HALF2_\n\
#undef _8HALF2_\n\
\n\
#undef _1MMA_\n\
#undef _2MMA_\n\
#undef _4MMA_\n\
#undef _8MMA_\n\
\n\
#undef _HALF_ZERO_\n\
\n\
#undef _INT_TO_BYTE_\n\
#undef _INT_TO_2HALF_\n\
#undef _INT2_TO_2HALF2_\n\
#undef _INT2_TO_2INT_\n\
\n\
#undef _INT4_TO_INT4_\n\
#undef _INT4_TO_2INT2_\n\
#undef _INT4_TO_4INT_\n\
#undef _INT4_TO_4HALF2_\n\
#undef _INT4_TO_8HALF_\n\
\n\
#undef SMEM_ROW_V1_SIZE\n\
#undef SMEM_ROW_V4_SIZE\n\
#undef SMEM_ROW_BYTE_SIZE\n\
#undef SMEM_ROW_BIT_SIZE\n\
\n\
\n\
////////////////////////////////////////\n\
// mma size macros\n\
////////////////////////////////////////\n\
\n\
#undef TILE_M_PER_MMA\n\
#undef TILE_K_PER_MMA\n\
#undef TILE_N_PER_MMA\n\
#undef TILE_M_PER_MMA_HALF\n\
\n\
#undef MMA_SIZE_Y_IN_THD\n\
#undef MMA_SIZE_Y_IN_THD\n\
\n\
#undef MMA_SIZE_X_IN_BITS\n\
\n\
////////////////////////////////////////\n\
// thread / warp / cta size macros\n\
////////////////////////////////////////\n\
\n\
#undef WARP_SIZE_IN_THD\n\
#undef WARP_SIZE_IN_BITS\n\
\n\
#undef WARP_SIZE_X_IN_THD\n\
#undef WARP_SIZE_Y_IN_THD\n\
\n\
#undef SET_SIZE_IN_WARP\n\
#undef SET_SIZE_IN_THD\n\
#undef SET_SIZE_IN_BITS\n\
\n\
#undef SET_SIZE_X_IN_WARP\n\
#undef SET_SIZE_Y_IN_WARP\n\
\n\
#undef SET_SIZE_X_IN_BITS\n\
#undef SET_SIZE_Y_IN_BITS\n\
\n\
#undef CTA_SIZE_IN_WARP\n\
#undef CTA_SIZE_IN_THD\n\
#undef CTA_SIZE_IN_BITS\n\
\n\
////////////////////////////////////////\n\
// tiling size macros\n\
////////////////////////////////////////\n\
\n\
#undef TILE_M_PER_THD\n\
#undef TILE_N_PER_THD\n\
\n\
/////////////////////\n\
// tile m\n\
\n\
#undef TILE_M_V1_PER_CTA\n\
#undef TILE_M_V2_PER_CTA\n\
#undef TILE_M_V4_PER_CTA\n\
#undef TILE_M_V8_PER_CTA\n\
\n\
#undef TILE_M_V1_PER_WARP\n\
#undef TILE_M_V2_PER_WARP\n\
#undef TILE_M_V4_PER_WARP\n\
#undef TILE_M_V8_PER_WARP\n\
\n\
#undef TILE_M_V1_PER_THD\n\
#undef TILE_M_V2_PER_THD\n\
#undef TILE_M_V4_PER_THD\n\
#undef TILE_M_V8_PER_THD\n\
\n\
#undef TILE_M_V1_PER_MMA\n\
#undef TILE_M_V2_PER_MMA\n\
#undef TILE_M_V4_PER_MMA\n\
#undef TILE_M_V8_PER_MMA\n\
\n\
/////////////////////\n\
// tile k\n\
\n\
#undef TILE_K_V1_PER_CTA\n\
#undef TILE_K_V2_PER_CTA\n\
#undef TILE_K_V4_PER_CTA\n\
#undef TILE_K_V8_PER_CTA\n\
\n\
#undef TILE_K_V1_PER_WARP\n\
#undef TILE_K_V2_PER_WARP\n\
#undef TILE_K_V4_PER_WARP\n\
#undef TILE_K_V8_PER_WARP\n\
\n\
#undef TILE_K_V1_PER_THD\n\
#undef TILE_K_V2_PER_THD\n\
#undef TILE_K_V4_PER_THD\n\
#undef TILE_K_V8_PER_THD\n\
\n\
#undef TILE_K_V1_PER_KMA\n\
#undef TILE_K_V2_PER_KMA\n\
#undef TILE_K_V4_PER_KMA\n\
#undef TILE_K_V8_PER_KMA\n\
\n\
\n\
/////////////////////\n\
// tile n\n\
\n\
#undef TILE_N_V1_PER_CTA\n\
#undef TILE_N_V2_PER_CTA\n\
#undef TILE_N_V4_PER_CTA\n\
#undef TILE_N_V8_PER_CTA\n\
\n\
#undef TILE_N_V1_PER_WARP\n\
#undef TILE_N_V2_PER_WARP\n\
#undef TILE_N_V4_PER_WARP\n\
#undef TILE_N_V8_PER_WARP\n\
\n\
#undef TILE_N_V1_PER_THD\n\
#undef TILE_N_V2_PER_THD\n\
#undef TILE_N_V4_PER_THD\n\
#undef TILE_N_V8_PER_THD\n\
\n\
#undef TILE_N_V1_PER_MMA\n\
#undef TILE_N_V2_PER_MMA\n\
#undef TILE_N_V4_PER_MMA\n\
#undef TILE_N_V8_PER_MMA\n\
\n\
\n\
////////////////////////////////////////\n\
// shared memory size macros\n\
////////////////////////////////////////\n\
\n\
#undef OUTPUT_STEPS\n\
\n\
#undef N_ROWS_PER_SMEM_ROW\n\
#undef K_ROWS_PER_SMEM_ROW\n\
\n\
#undef OUTPUT_SIZE_X_IN_THD\n\
#undef OUTPUT_SIZE_Y_IN_THD\n\
\n\
////////////////////////////////////////\n\
// main loop macros\n\
////////////////////////////////////////\n\
\n\
#undef C_ITEMS_PER_THD\n\
\n\
////////////////////////////////////////\n\
// load A and B from device memory macros\n\
////////////////////////////////////////\n\
\n\
#undef REG_dAv4_SIZE\n\
\n\
#undef REG_dBv1_SIZE\n\
#undef REG_dBv2_SIZE\n\
#undef REG_dBv4_SIZE\n\
\n\
#undef READ_dBv1_STEPS\n\
#undef READ_dBv4_STEPS\n\
\n\
#undef SET_dBv1_BOUND\n\
#undef SET_dBv4_BOUND\n\
\n\
////////////////////////////////////////\n\
// shared memory size macros\n\
////////////////////////////////////////\n\
\n\
#undef USE_1BUF\n\
#undef USE_2BUF\n\
\n\
#undef SM_A_SIZE\n\
#undef SM_B_SIZE\n\
#undef SM_C_SIZE\n\
\n\
#undef SM_A_1BUF\n\
#undef SM_B_1BUF\n\
#undef SM_C_1BUF\n\
\n\
#undef SM_A_2BUF\n\
#undef SM_B_2BUF\n\
#undef SM_C_2BUF\n\
\n\
#undef SM_A_V1_1BUF\n\
#undef SM_B_V1_1BUF\n\
#undef SM_C_V1_1BUF\n\
\n\
#undef SM_A_V2_1BUF\n\
#undef SM_B_V2_1BUF\n\
#undef SM_C_V2_1BUF\n\
\n\
#undef SM_A_V4_1BUF\n\
#undef SM_B_V4_1BUF\n\
#undef SM_C_V4_1BUF\n\
\n\
#undef SM_A_V1_2BUF\n\
#undef SM_B_V1_2BUF\n\
#undef SM_C_V1_2BUF\n\
\n\
#undef SM_A_V2_2BUF\n\
#undef SM_B_V2_2BUF\n\
#undef SM_C_V2_2BUF\n\
\n\
#undef SM_A_V4_2BUF\n\
#undef SM_B_V4_2BUF\n\
#undef SM_C_V4_2BUF\n\
\n\
#undef SM_BASE_V4_1BUF\n\
#undef SM_BASE_V4_2BUF\n\
\n\
#undef CVT_SM_PTR\n\
\n\
#undef FWD_LUT\n\
\n\
#undef FWD_FLT\n\
#undef FWD_FLT1\n\
#undef FLT_SIZE1\n\
#undef FWD_FLT3\n\
#undef FLT_SIZE3\n\
#undef FWD_FLTN\n\
#undef FLT_SIZEN\n\
\n\
#undef FWD_FLT_SIZE1\n\
#undef FWD_FLT_SIZE2\n\
#undef FWD_FLT_SIZE4\n\
#undef FWD_FLT_SIZE8\n\
#undef FWD_FLT_SIZE16\n\
\n\
////////////////////////////////////////\n\
// mma macros\n\
////////////////////////////////////////\n\
\n\
#undef MMA_INST_OPCODE\n\
#undef MMA_INST\n\
\n\
#undef MMA_INST_ASCEND1\n\
#undef MMA_INST_ASCEND2\n\
#undef MMA_INST_ASCEND4\n\
#undef MMA_INST_ASCEND8\n\
\n\
#undef MMA_INST_DESCEND1\n\
#undef MMA_INST_DESCEND2\n\
#undef MMA_INST_DESCEND4\n\
#undef MMA_INST_DESCEND8\n\
\n\
#undef MMA_INST_1x1\n\
#undef MMA_INST_1x2\n\
#undef MMA_INST_1x4\n\
#undef MMA_INST_1x8\n\
\n\
#undef MMA_INST_2x1\n\
#undef MMA_INST_2x2\n\
#undef MMA_INST_2x4\n\
#undef MMA_INST_2x8\n\
\n\
#undef MMA_INST_4x1\n\
#undef MMA_INST_4x2\n\
#undef MMA_INST_4x4\n\
#undef MMA_INST_4x8\n\
\n\
#undef MMA_INST_8x1\n\
#undef MMA_INST_8x2\n\
#undef MMA_INST_8x4\n\
\n\
#undef MMA_INSTS\n\
\n\
/////////////////////////////////////////////////////\n\
// reduce half2 macros\n\
/////////////////////////////////////////////////////\n\
\n\
#undef REDUCE_HALF2_SIZE4\n\
\n\
#undef REDUCE_HALF2_1x4\n\
#undef REDUCE_HALF2_3x4\n\
\n\
#undef REDUCE\n\
\n\
/////////////////////////////////////////////////////\n\
// read sRv4 macros\n\
/////////////////////////////////////////////////////\n\
\n\
#undef READ_sRv4_SIZE1\n\
#undef READ_sRv4_SIZE2\n\
#undef READ_sRv4_SIZE4\n\
\n\
#undef READ_sRv4\n\
\n\
/////////////////////////////////////////////////////\n\
// write sRv1 macros\n\
/////////////////////////////////////////////////////\n\
\n\
#undef WRITE_sRv1_SIZE1\n\
#undef WRITE_sRv1_SIZE2\n\
#undef WRITE_sRv1_SIZE4\n\
#undef WRITE_sRv1_SIZE8\n\
\n\
#undef WRITE_sRv1_1x1\n\
#undef WRITE_sRv1_2x1\n\
#undef WRITE_sRv1_4x1\n\
#undef WRITE_sRv1_8x1\n\
#undef WRITE_sRv1_16x1\n\
\n\
#undef WRITE_sRv1_1x2\n\
#undef WRITE_sRv1_2x2\n\
#undef WRITE_sRv1_4x2\n\
#undef WRITE_sRv1_8x2\n\
#undef WRITE_sRv1_16x2\n\
\n\
#undef WRITE_sRv1_1x4\n\
#undef WRITE_sRv1_2x4\n\
#undef WRITE_sRv1_4x4\n\
#undef WRITE_sRv1_8x4\n\
#undef WRITE_sRv1_16x4\n\
\n\
#undef WRITE_sRv1_1x8\n\
#undef WRITE_sRv1_2x8\n\
#undef WRITE_sRv1_4x8\n\
#undef WRITE_sRv1_8x8\n\
\n\
#undef WRITE_sRv1\n\
\n\
/////////////////////////////////////////////////////\n\
// common load global memory macros\n\
/////////////////////////////////////////////////////\n\
\n\
//////////////////////////\n\
// load dA\n\
//////////////////////////\n\
\n\
#undef LOAD_dAv4_SIZE_16TH\n\
#undef LOAD_dAv4_SIZE_8TH\n\
#undef LOAD_dAv4_SIZE_QTR\n\
#undef LOAD_dAv4_SIZE_HALF\n\
#undef LOAD_dAv4_SIZE1\n\
#undef LOAD_dAv4_SIZE2\n\
#undef LOAD_dAv4_SIZE4\n\
#undef LOAD_dAv4_SIZE8\n\
#undef LOAD_dAv4_SIZE16\n\
\n\
#undef LOAD_dAv4\n\
\n\
#undef SET_dAv4_BOUND \n\
\n\
//////////////////////////\n\
// load dB\n\
//////////////////////////\n\
\n\
#undef LOAD_dBv4_SIZE_16TH\n\
#undef LOAD_dBv4_SIZE_8TH\n\
#undef LOAD_dBv4_SIZE_QTR\n\
#undef LOAD_dBv4_SIZE_HALF\n\
#undef LOAD_dBv4_SIZE1\n\
#undef LOAD_dBv4_SIZE2\n\
#undef LOAD_dBv4_SIZE4\n\
#undef LOAD_dBv4_SIZE8\n\
#undef LOAD_dBv4_SIZE16\n\
\n\
#undef LOAD_dBv4\n\
\n\
#undef SET_dBv4_BOUND \n\
\n\
/////////////////////////////////////////////////////\n\
// common write shared memory macros\n\
/////////////////////////////////////////////////////\n\
\n\
#undef SWITCH_BUFFER\n\
\n\
#undef FWD_KGROUP_ODD\n\
#undef FWD_KGROUP_EVEN\n\
\n\
#undef FWD_KGROUP_STEP1\n\
#undef FWD_KGROUP_STEP2\n\
#undef FWD_KGROUP_STEP3\n\
#undef FWD_KGROUP_STEP4\n\
\n\
#undef C_ITEMS_PER_THD\n\
#undef HC_ITEMS_PER_THD\n\
#undef Cv4_ITEMS_PER_THD\n\
\n\
//////////////////////////\n\
// write sA & sB\n\
//////////////////////////\n\
\n\
#undef WRITE_sUv4_SIZE_16TH\n\
#undef WRITE_sUv4_SIZE_8TH\n\
#undef WRITE_sUv4_SIZE_QTR\n\
#undef WRITE_sUv4_SIZE_HALF\n\
#undef WRITE_sUv4_SIZE1\n\
#undef WRITE_sUv4_SIZE2\n\
#undef WRITE_sUv4_SIZE4\n\
#undef WRITE_sUv4_SIZE8\n\
#undef WRITE_sUv4_SIZE16\n\
\n\
#undef WRITE_sAv4\n\
#undef WRITE_sBv4\n\
\n\
/////////////////////////////////////////////////////\n\
// read shared memory macros\n\
/////////////////////////////////////////////////////\n\
\n\
//////////////////////////\n\
// read sA & sB\n\
//////////////////////////\n\
\n\
#undef REG_sAv1_SIZE\n\
#undef REG_sBv1_SIZE\n\
\n\
#undef READ_sUv1_SIZE1\n\
#undef READ_sUv1_SIZE2\n\
#undef READ_sUv1_SIZE4\n\
\n\
#undef READ_sUv1_1x1\n\
#undef READ_sUv1_2x1\n\
\n\
#undef READ_sUv1_1x2\n\
#undef READ_sUv1_2x2\n\
\n\
#undef READ_sUv1_1x4\n\
#undef READ_sUv1_2x4\n\
\n\
#undef READ_sAv1\n\
#undef READ_sBv1\n\
\n\
/////////////////////////////////////////////////////\n\
// precision half output\n\
/////////////////////////////////////////////////////\n\
\n\
#undef OUTPUT_PRC_HALF\n\
\n\
#undef ADD_BIAS_V4\n\
\n\
#undef FUSE_RELU_V4\n\
#undef FUSE_CLIP_V4\n\
#undef FUSE_PRELU_V4\n\
#undef FUSE_ELT_V4\n\
\n\
#undef SET_CONCAT_OFF_V4\n\
");
    header_code_.emplace("/mnt/hpc/xusi/ppl.jit/src/ppl/nn/engines/cuda/impls/src/nn/conv/idxn/common/const_macros.h", "// Licensed to the Apache Software Foundation (ASF) under one\n\
// or more contributor license agreements.  See the NOTICE file\n\
// distributed with this work for additional information\n\
// regarding copyright ownership.  The ASF licenses this file\n\
// to you under the Apache License, Version 2.0 (the\n\
// \"License\"); you may not use this file except in compliance\n\
// with the License.  You may obtain a copy of the License at\n\
//\n\
//   http://www.apache.org/licenses/LICENSE-2.0\n\
//\n\
// Unless required by applicable law or agreed to in writing,\n\
// software distributed under the License is distributed on an\n\
// \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n\
// KIND, either express or implied.  See the License for the\n\
// specific language governing permissions and limitations\n\
// under the License.\n\
\n\
////////////////////////////////////////\n\
// kernel list macros\n\
////////////////////////////////////////\n\
\n\
#define TOTAL_KPARAM_LIST \\\n\
        int4* dA,                                                 \\\n\
        int4* dB,                                                 \\\n\
        int4* dC,                                                 \\\n\
        int kloop_num,                int koff_num_pad,           \\\n\
        int in_hw,                    int out_hw,                 \\\n\
        int flt_hw,                   int out_nhw,                \\\n\
        int in_height,                int in_width,               \\\n\
        int in_num,                   int num_grp,                \\\n\
        int num_chl,                  int num_chl_per_grp,        \\\n\
        int in_chl_per_grp_pad,       int flt_chl_per_grp_pad,    \\\n\
        int flt_height,               int flt_width,              \\\n\
        int num_flt_per_grp,          int num_flt_per_grp_pad,    \\\n\
        int out_height,               int out_width,              \\\n\
        int stride_height,            int stride_width,           \\\n\
        int pad_height,               int pad_width,              \\\n\
        int hole_height,              int hole_width,             \\\n\
        int  has_bias,                const int4* bias,           \\\n\
        int  has_relu,                const __half2 clip_min,     \\\n\
	    bool has_clip,                const __half2 clip_max,     \\\n\
        int  has_prelu,               const void* prelu,          \\\n\
        bool has_elt,                 const int4* pre_data,       \\\n\
        int  has_elt_relu,            const __half2 elt_clip_min, \\\n\
	    bool has_elt_clip,            const __half2 elt_clip_max, \\\n\
        int has_elt_prelu,            const void* elt_prelu,      \\\n\
        const __half leaky,           const __half elt_leaky,     \\\n\
        bool has_concat,              int concat_offset_v8,       \\\n\
        int concat_stride_v8\n\
\n\
////////////////////////////////////////\n\
// align functions\n\
////////////////////////////////////////\n\
\n\
#define Align(x, y)   (((x) + (y) - 1) / (y) * (y))\n\
#define DivUp(x, y)   (((x) + (y) - 1) / (y))\n\
\n\
#define Min(x, y)     (((x) < (y)) ? (x) : (y))\n\
#define Max(x, y)     (((x) > (y)) ? (x) : (y))\n\
\n\
////////////////////////////////////////\n\
// boundary check\n\
////////////////////////////////////////\n\
\n\
#define WidthInRange(_w)     ( (_w < in_width)  && (_w >= 0) )\n\
#define HeightInRange(_h)    ( (_h < in_height) && (_h >= 0) )\n\
#define BatchInRange(_b)     ( (_b < in_num) )\n\
\n\
////////////////////////////////////////\n\
// constant cta size macros\n\
////////////////////////////////////////\n\
\n\
#define _4CHAR_TO_INT_          4\n\
#define _4INT_TO_INT4_          4\n\
#define _2INT_TO_INT2_          2\n\
\n\
#define _2HALF_TO_INT_          2\n\
#define _2INT2_TO_INT4_         2\n\
\n\
#define _C1_                    1\n\
#define _C2_                    2\n\
#define _C4_                    4\n\
#define _C8_                    8\n\
#define _C16_                   16\n\
#define _C32_                   32\n\
\n\
#define _1INT_                  1\n\
#define _2INT_                  2\n\
#define _4INT_                  4\n\
#define _8INT_                  8\n\
\n\
#define _1INT4_                 1\n\
#define _2INT4_                 2\n\
#define _4INT4_                 4\n\
#define _8INT4_                 8\n\
\n\
#define _1INT8_                 1\n\
#define _2INT8_                 2\n\
#define _4INT8_                 4\n\
#define _8INT8_                 8\n\
\n\
#define _1HALF_                 1\n\
#define _2HALF_                 2\n\
#define _4HALF_                 4\n\
#define _8HALF_                 8\n\
\n\
#define _1HALF2_                1\n\
#define _2HALF2_                2\n\
#define _4HALF2_                4\n\
#define _8HALF2_                8\n\
\n\
#define _1MMA_                  1\n\
#define _2MMA_                  2\n\
#define _4MMA_                  4\n\
#define _8MMA_                  8\n\
\n\
#define _HALF_ZERO_             0.0\n\
\n\
#define _1INT_X1_               (_1INT_ * 1)\n\
#define _1INT_X2_               (_1INT_ * 2)\n\
#define _1INT_X4_               (_1INT_ * 4)\n\
\n\
#define _2INT_X1_               (_2INT_ * 1)\n\
#define _2INT_X2_               (_2INT_ * 2)\n\
#define _2INT_X4_               (_2INT_ * 4)\n\
\n\
#define _4INT_X1_               (_4INT_ * 1)\n\
#define _4INT_X2_               (_4INT_ * 2)\n\
#define _4INT_X4_               (_4INT_ * 4)\n\
\n\
#define _INT_TO_BYTE_           4\n\
#define _INT_TO_2HALF_          2\n\
#define _INT2_TO_2HALF2_        2\n\
#define _INT2_TO_2INT_          2\n\
\n\
#define _INT4_TO_INT4_          1\n\
#define _INT4_TO_2INT2_         2\n\
#define _INT4_TO_4INT_          4\n\
#define _INT4_TO_4HALF2_        4\n\
#define _INT4_TO_8HALF_         8\n\
\n\
////////////////////////////////////////\n\
// mma size macros\n\
////////////////////////////////////////\n\
\n\
#define TILE_M_PER_MMA          16\n\
#define TILE_K_PER_MMA          8\n\
#define TILE_N_PER_MMA          8\n\
#define TILE_M_PER_MMA_HALF     ((TILE_M_PER_MMA) / 2)\n\
\n\
#define MMA_SIZE_X_IN_THD       4\n\
#define MMA_SIZE_Y_IN_THD       8\n\
\n\
#define BLK_M_PER_MMA           2\n\
#define BLK_N_PER_MMA           1\n\
\n\
////////////////////////////////////////\n\
// thread / warp / cta size macros\n\
////////////////////////////////////////\n\
\n\
#define WARP_SIZE_IN_THD        32\n\
#define WARP_SIZE_IN_BITS       5\n\
\n\
#define WARP_SIZE_X_IN_THD      4\n\
#define WARP_SIZE_Y_IN_THD      8\n\
\n\
#define CTA_SIZE_X_IN_WARP      ((TILE_N_PER_CTA) / (TILE_N_PER_WARP))\n\
#define CTA_SIZE_Y_IN_WARP      ((TILE_M_PER_CTA) / (TILE_M_PER_WARP))\n\
\n\
#define CTA_SIZE_IN_WARP        ((CTA_SIZE_X_IN_WARP) * (CTA_SIZE_Y_IN_WARP))\n\
#define CTA_SIZE_IN_THD         ((CTA_SIZE_IN_WARP)   * (WARP_SIZE_IN_THD))\n\
\n\
#define WARP_SIZE_IN_THD_HALF   (WARP_SIZE_IN_THD / 2)\n\
#define WARP_SIZE_IN_THD_QTR    (WARP_SIZE_IN_THD / 4)\n\
\n\
#define NUM_M_STEPS             (TILE_M_PER_WARP / TILE_M_PER_MMA)\n\
#define NUM_N_STEPS             (TILE_N_PER_WARP / TILE_N_PER_MMA)\n\
\n\
////////////////////////////////////////\n\
// tiling size macros\n\
////////////////////////////////////////\n\
\n\
#define TILE_M_PER_STEP         ((TILE_M_PER_MMA)  * (CTA_SIZE_Y_IN_WARP))\n\
#define TILE_N_PER_STEP         ((TILE_N_PER_MMA)  * (CTA_SIZE_X_IN_WARP))\n\
\n\
#define TILE_M_PER_THD          ((TILE_M_PER_WARP) / (WARP_SIZE_Y_IN_THD))\n\
#define TILE_N_PER_THD          ((TILE_N_PER_WARP) / (WARP_SIZE_X_IN_THD))\n\
\n\
/////////////////////\n\
// tile m\n\
\n\
#define TILE_M_V1_PER_CTA       ((TILE_M_PER_CTA)  / 1)\n\
#define TILE_M_V2_PER_CTA       ((TILE_M_PER_CTA)  / 2)\n\
#define TILE_M_V4_PER_CTA       ((TILE_M_PER_CTA)  / 4)\n\
#define TILE_M_V8_PER_CTA       ((TILE_M_PER_CTA)  / 8)\n\
\n\
#define TILE_M_V1_PER_WARP      ((TILE_M_PER_WARP) / 1)\n\
#define TILE_M_V2_PER_WARP      ((TILE_M_PER_WARP) / 2)\n\
#define TILE_M_V4_PER_WARP      ((TILE_M_PER_WARP) / 4)\n\
#define TILE_M_V8_PER_WARP      ((TILE_M_PER_WARP) / 8)\n\
\n\
#define TILE_M_V1_PER_THD       ((TILE_M_PER_THD)  / 1)\n\
#define TILE_M_V2_PER_THD       ((TILE_M_PER_THD)  / 2)\n\
#define TILE_M_V4_PER_THD       ((TILE_M_PER_THD)  / 4)\n\
#define TILE_M_V8_PER_THD       ((TILE_M_PER_THD)  / 8)\n\
\n\
#define TILE_M_V1_PER_MMA       ((TILE_M_PER_MMA)  / 1)\n\
#define TILE_M_V2_PER_MMA       ((TILE_M_PER_MMA)  / 2)\n\
#define TILE_M_V4_PER_MMA       ((TILE_M_PER_MMA)  / 4)\n\
#define TILE_M_V8_PER_MMA       ((TILE_M_PER_MMA)  / 8)\n\
#define TILE_M_V1_PER_MMA_HALF  ((TILE_M_PER_MMA)  / 2)\n\
\n\
/////////////////////\n\
// tile k\n\
\n\
#define TILE_K_V1_PER_CTA       ((TILE_K_PER_CTA)  / 1)\n\
#define TILE_K_V2_PER_CTA       ((TILE_K_PER_CTA)  / 2)\n\
#define TILE_K_V4_PER_CTA       ((TILE_K_PER_CTA)  / 4)\n\
#define TILE_K_V8_PER_CTA       ((TILE_K_PER_CTA)  / 8)\n\
\n\
#define TILE_K_V1_PER_STEP      ((TILE_K_PER_STEP) / 1)\n\
#define TILE_K_V2_PER_STEP      ((TILE_K_PER_STEP) / 2)\n\
#define TILE_K_V4_PER_STEP      ((TILE_K_PER_STEP) / 4)\n\
#define TILE_K_V8_PER_STEP      ((TILE_K_PER_STEP) / 8)\n\
\n\
#define TILE_K_V1_PER_MMA       ((TILE_K_PER_MMA)  / 1)\n\
#define TILE_K_V2_PER_MMA       ((TILE_K_PER_MMA)  / 2)\n\
#define TILE_K_V4_PER_MMA       ((TILE_K_PER_MMA)  / 4)\n\
#define TILE_K_V8_PER_MMA       ((TILE_K_PER_MMA)  / 8)\n\
\n\
/////////////////////\n\
// tile n\n\
\n\
#define TILE_N_V1_PER_CTA       ((TILE_N_PER_CTA)  / 1)\n\
#define TILE_N_V2_PER_CTA       ((TILE_N_PER_CTA)  / 2)\n\
#define TILE_N_V4_PER_CTA       ((TILE_N_PER_CTA)  / 4)\n\
#define TILE_N_V8_PER_CTA       ((TILE_N_PER_CTA)  / 8)\n\
\n\
#define TILE_N_V1_PER_WARP      ((TILE_N_PER_WARP) / 1)\n\
#define TILE_N_V2_PER_WARP      ((TILE_N_PER_WARP) / 2)\n\
#define TILE_N_V4_PER_WARP      ((TILE_N_PER_WARP) / 4)\n\
#define TILE_N_V8_PER_WARP      ((TILE_N_PER_WARP) / 8)\n\
\n\
#define TILE_N_V1_PER_THD       ((TILE_N_PER_THD)  / 1)\n\
#define TILE_N_V2_PER_THD       ((TILE_N_PER_THD)  / 2)\n\
#define TILE_N_V4_PER_THD       ((TILE_N_PER_THD)  / 4)\n\
#define TILE_N_V8_PER_THD       ((TILE_N_PER_THD)  / 8)\n\
\n\
#define TILE_N_V1_PER_MMA       ((TILE_N_PER_MMA)  / 1)\n\
#define TILE_N_V2_PER_MMA       ((TILE_N_PER_MMA)  / 2)\n\
#define TILE_N_V4_PER_MMA       ((TILE_N_PER_MMA)  / 4)\n\
#define TILE_N_V8_PER_MMA       ((TILE_N_PER_MMA)  / 8)\n\
\n\
#define TILE_N_V1_PER_STEP      ((TILE_N_PER_STEP) / 1)\n\
#define TILE_N_V2_PER_STEP      ((TILE_N_PER_STEP) / 2)\n\
#define TILE_N_V4_PER_STEP      ((TILE_N_PER_STEP) / 4)\n\
#define TILE_N_V8_PER_STEP      ((TILE_N_PER_STEP) / 8)\n\
\n\
////////////////////////////////////////\n\
// main loop macros\n\
////////////////////////////////////////\n\
\n\
#define   C_ITEMS_PER_THD       ((TILE_M_PER_CTA) * (TILE_N_PER_CTA) / (CTA_SIZE_IN_THD * _INT_TO_2HALF_))\n\
#define  HC_ITEMS_PER_THD       ((TILE_M_PER_CTA) * (TILE_N_PER_CTA) / (CTA_SIZE_IN_THD))\n\
#define Cv4_ITEMS_PER_THD       ((TILE_M_PER_CTA) * (TILE_N_PER_CTA) / (CTA_SIZE_IN_THD * _INT_TO_2HALF_ * _4INT_TO_INT4_))\n\
\n\
////////////////////////////////////////\n\
// load A and B from device memory macros\n\
////////////////////////////////////////\n\
\n\
#define REG_dAv1_SIZE           (NUM_M_STEPS * BLK_M_PER_MMA)\n\
#define REG_dBv1_SIZE           (NUM_N_STEPS * BLK_N_PER_MMA)\n\
\n\
#define REG_dAv2_SIZE           (NUM_M_STEPS * BLK_M_PER_MMA)\n\
#define REG_dBv2_SIZE           (NUM_N_STEPS * BLK_N_PER_MMA)\n\
\n\
#define REG_dAv4_SIZE           (NUM_M_STEPS * BLK_M_PER_MMA)\n\
#define REG_dBv4_SIZE           (NUM_N_STEPS * BLK_N_PER_MMA)\n\
\n\
#define READ_dAv1_STEPS         (REG_dAv1_SIZE)\n\
#define READ_dBv1_STEPS         (REG_dBv1_SIZE)\n\
\n\
#define READ_dAv2_STEPS         (REG_dAv2_SIZE)\n\
#define READ_dBv2_STEPS         (REG_dBv2_SIZE)\n\
\n\
#define READ_dAv4_STEPS         (REG_dAv4_SIZE)\n\
#define READ_dBv4_STEPS         (REG_dBv4_SIZE)\n\
\n\
////////////////////////////////////////\n\
// shared memory size macros\n\
////////////////////////////////////////\n\
\n\
#define SM_IN_ID_SIZE           (TILE_M_PER_CTA)\n\
#define SM_IN_OFF_SIZE          (CTA_SIZE_IN_THD)\n\
\n\
////////////////////////////////////////\n\
// bit size macros\n\
////////////////////////////////////////\n\
\n\
#if MMA_SIZE_X_IN_THD == 1\n\
#define MMA_SIZE_X_IN_BITS      0\n\
#elif MMA_SIZE_X_IN_THD == 2\n\
#define MMA_SIZE_X_IN_BITS      1\n\
#elif MMA_SIZE_X_IN_THD == 4\n\
#define MMA_SIZE_X_IN_BITS      2\n\
#elif MMA_SIZE_X_IN_THD == 8\n\
#define MMA_SIZE_X_IN_BITS      3\n\
#endif\n\
\n\
#if CTA_SIZE_X_IN_WARP == 1\n\
#define CTA_SIZE_X_IN_BITS      0\n\
#elif CTA_SIZE_X_IN_WARP == 2\n\
#define CTA_SIZE_X_IN_BITS      1\n\
#elif CTA_SIZE_X_IN_WARP == 4\n\
#define CTA_SIZE_X_IN_BITS      2\n\
#endif\n\
\n\
");
    header_code_.emplace("/mnt/hpc/xusi/ppl.jit/src/ppl/nn/engines/cuda/impls/src/nn/conv/idxn/common/dmem_i1_macros.h", "// Licensed to the Apache Software Foundation (ASF) under one\n\
// or more contributor license agreements.  See the NOTICE file\n\
// distributed with this work for additional information\n\
// regarding copyright ownership.  The ASF licenses this file\n\
// to you under the Apache License, Version 2.0 (the\n\
// \"License\"); you may not use this file except in compliance\n\
// with the License.  You may obtain a copy of the License at\n\
//\n\
//   http://www.apache.org/licenses/LICENSE-2.0\n\
//\n\
// Unless required by applicable law or agreed to in writing,\n\
// software distributed under the License is distributed on an\n\
// \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n\
// KIND, either express or implied.  See the License for the\n\
// specific language governing permissions and limitations\n\
// under the License.\n\
\n\
/////////////////////////////////////////////////////\n\
// common load global memory macros\n\
/////////////////////////////////////////////////////\n\
\n\
////////////////////////////////////////\n\
// load dB macros\n\
////////////////////////////////////////\n\
\n\
#define LOAD_dBv1_SIZE1(_regB, _dBv1, _dBv1_off) \\\n\
        { \\\n\
            _regB[0] = (flt_n_valid[0] && (flt_hwc_v2_off < flt_hwc_v2)) ? _dBv1[ _dBv1_off[0] * _INT4_TO_4INT_ ] : ZEROv1;\\\n\
            \\\n\
            _dBv1_off[0] += TILE_K_V2_PER_STEP; \\\n\
            \\\n\
            flt_hwc_v2_off  += TILE_K_V2_PER_STEP; \\\n\
        }\n\
\n\
#define LOAD_dBv1_SIZE2(_regB, _dBv1, _dBv1_off) \\\n\
        { \\\n\
            _regB[0] = (flt_n_valid[0] && (flt_hwc_v2_off < flt_hwc_v2)) ? _dBv1[ _dBv1_off[0] * _INT4_TO_4INT_ ] : ZEROv1;\\\n\
            _regB[1] = (flt_n_valid[1] && (flt_hwc_v2_off < flt_hwc_v2)) ? _dBv1[ _dBv1_off[1] * _INT4_TO_4INT_ ] : ZEROv1;\\\n\
            \\\n\
            _dBv1_off[0] += TILE_K_V2_PER_STEP; \\\n\
            _dBv1_off[1] += TILE_K_V2_PER_STEP; \\\n\
            \\\n\
            flt_hwc_v2_off  += TILE_K_V2_PER_STEP; \\\n\
        }\n\
\n\
#define LOAD_dBv1_SIZE4(_regB, _dBv1, _dBv1_off) \\\n\
        { \\\n\
            _regB[0] = (flt_n_valid[0] && (flt_hwc_v2_off < flt_hwc_v2)) ? _dBv1[ _dBv1_off[0] * _INT4_TO_4INT_ ] : ZEROv1;\\\n\
            _regB[1] = (flt_n_valid[1] && (flt_hwc_v2_off < flt_hwc_v2)) ? _dBv1[ _dBv1_off[1] * _INT4_TO_4INT_ ] : ZEROv1;\\\n\
            _regB[2] = (flt_n_valid[2] && (flt_hwc_v2_off < flt_hwc_v2)) ? _dBv1[ _dBv1_off[2] * _INT4_TO_4INT_ ] : ZEROv1;\\\n\
            _regB[3] = (flt_n_valid[3] && (flt_hwc_v2_off < flt_hwc_v2)) ? _dBv1[ _dBv1_off[3] * _INT4_TO_4INT_ ] : ZEROv1;\\\n\
            \\\n\
            _dBv1_off[0] += TILE_K_V2_PER_STEP; \\\n\
            _dBv1_off[1] += TILE_K_V2_PER_STEP; \\\n\
            _dBv1_off[2] += TILE_K_V2_PER_STEP; \\\n\
            _dBv1_off[3] += TILE_K_V2_PER_STEP; \\\n\
            \\\n\
            flt_hwc_v2_off  += TILE_K_V2_PER_STEP; \\\n\
        }\n\
\n\
#define LOAD_dBv1_SIZE8(_regB, _dBv1, _dBv1_off) \\\n\
        { \\\n\
            _regB[0] = (flt_n_valid[0] && (flt_hwc_v2_off < flt_hwc_v2)) ? _dBv1[ _dBv1_off[0] * _INT4_TO_4INT_ ] : ZEROv1;\\\n\
            _regB[1] = (flt_n_valid[1] && (flt_hwc_v2_off < flt_hwc_v2)) ? _dBv1[ _dBv1_off[1] * _INT4_TO_4INT_ ] : ZEROv1;\\\n\
            _regB[2] = (flt_n_valid[2] && (flt_hwc_v2_off < flt_hwc_v2)) ? _dBv1[ _dBv1_off[2] * _INT4_TO_4INT_ ] : ZEROv1;\\\n\
            _regB[3] = (flt_n_valid[3] && (flt_hwc_v2_off < flt_hwc_v2)) ? _dBv1[ _dBv1_off[3] * _INT4_TO_4INT_ ] : ZEROv1;\\\n\
            _regB[4] = (flt_n_valid[4] && (flt_hwc_v2_off < flt_hwc_v2)) ? _dBv1[ _dBv1_off[4] * _INT4_TO_4INT_ ] : ZEROv1;\\\n\
            _regB[5] = (flt_n_valid[5] && (flt_hwc_v2_off < flt_hwc_v2)) ? _dBv1[ _dBv1_off[5] * _INT4_TO_4INT_ ] : ZEROv1;\\\n\
            _regB[6] = (flt_n_valid[6] && (flt_hwc_v2_off < flt_hwc_v2)) ? _dBv1[ _dBv1_off[6] * _INT4_TO_4INT_ ] : ZEROv1;\\\n\
            _regB[7] = (flt_n_valid[7] && (flt_hwc_v2_off < flt_hwc_v2)) ? _dBv1[ _dBv1_off[7] * _INT4_TO_4INT_ ] : ZEROv1;\\\n\
            \\\n\
            _dBv1_off[0] += TILE_K_V2_PER_STEP; \\\n\
            _dBv1_off[1] += TILE_K_V2_PER_STEP; \\\n\
            _dBv1_off[2] += TILE_K_V2_PER_STEP; \\\n\
            _dBv1_off[3] += TILE_K_V2_PER_STEP; \\\n\
            _dBv1_off[4] += TILE_K_V2_PER_STEP; \\\n\
            _dBv1_off[5] += TILE_K_V2_PER_STEP; \\\n\
            _dBv1_off[6] += TILE_K_V2_PER_STEP; \\\n\
            _dBv1_off[7] += TILE_K_V2_PER_STEP; \\\n\
            \\\n\
            flt_hwc_v2_off  += TILE_K_V2_PER_STEP; \\\n\
        }\n\
\n\
#define SET_dBv1_BOUND(_step_id, _dBv1_off, _flt_n_valid) \\\n\
        { \\\n\
            int _flt_n_id  =  cta_idx * TILE_N_PER_CTA  + \\\n\
                             _step_id * TILE_N_PER_STEP + \\\n\
                             warp_idx * TILE_N_PER_MMA  + \\\n\
                             tid_y; \\\n\
            \\\n\
            _flt_n_valid  =  _flt_n_id < num_flt_per_grp_pad; \\\n\
            \\\n\
            _dBv1_off  =   grp_id   * flt_hwc_v2 * num_flt_per_grp_pad + \\\n\
                          _flt_n_id * flt_hwc_v2 + \\\n\
                           tid_x; \\\n\
        }\n\
\n\
////////////////////////////////////////\n\
// load dA macros\n\
////////////////////////////////////////\n\
\n\
#define SET_IN_Mv1_ID(_tid, _sm_base_v4) \\\n\
        { \\\n\
            int _out_nhw_id =  cta_idy * TILE_M_PER_CTA + _tid; \\\n\
            \\\n\
            int _out_w_id   = (_out_nhw_id % out_width); \\\n\
            int _out_h_id   = (_out_nhw_id / out_width) % out_height; \\\n\
            \\\n\
            int4 _in_id; \\\n\
            \\\n\
            _in_id.y = _out_w_id * stride_width  - pad_width; \\\n\
            _in_id.z = _out_h_id * stride_height - pad_height; \\\n\
            _in_id.w = _out_nhw_id / out_hw; \\\n\
            \\\n\
            _in_id.x = (_in_id.w * in_hw + _in_id.z * in_width + _in_id.y) * in_chl_per_grp_pad_v8 * num_grp + \\\n\
                         grp_id  * in_chl_per_grp_pad_v8; \\\n\
            \\\n\
            _sm_base_v4[_tid] = _in_id; \\\n\
        }\n\
\n\
#define SET_IN_Kv8_OFF(_tid, _sm_base_v4) \\\n\
        { \\\n\
            int _inNHWC8_id =  _tid; \\\n\
            \\\n\
            int4 _in_off; \\\n\
            \\\n\
            _in_off.y = ((_inNHWC8_id /  in_chl_per_grp_pad_v8) % flt_width)  * hole_width; \\\n\
            _in_off.z = ((_inNHWC8_id / (in_chl_per_grp_pad_v8  * flt_width)) % flt_height) * hole_height; \\\n\
            _in_off.w =   _inNHWC8_id / (in_chl_per_grp_pad_v8  * flt_width   * flt_height); \\\n\
            \\\n\
            _in_off.x = (_in_off.w  * in_hw + _in_off.z * in_width + _in_off.y) * in_chl_per_grp_pad_v8 * num_grp + \\\n\
                        (_inNHWC8_id %  in_chl_per_grp_pad_v8); \\\n\
            \\\n\
            _sm_base_v4[SM_IN_ID_SIZE + _tid] = _in_off; \\\n\
         }\n\
\n\
#define LOAD_dAv1_SIZE2(_regA, _dAv1, _in_id, _in_off) \\\n\
        { \\\n\
            int4 _in; \\\n\
            \\\n\
            _in.x = (_in_id[0].x + _in_off.x) * _INT4_TO_4INT_; \\\n\
            _in.y =  _in_id[0].y + _in_off.y; \\\n\
            _in.z =  _in_id[0].z + _in_off.z; \\\n\
            _in.w =  _in_id[0].w + _in_off.w; \\\n\
            _regA[0] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv1[_in.x] : ZEROv1;\\\n\
            \\\n\
            _in.x = (_in_id[1].x + _in_off.x) * _INT4_TO_4INT_; \\\n\
            _in.y =  _in_id[1].y + _in_off.y; \\\n\
            _in.z =  _in_id[1].z + _in_off.z; \\\n\
            _in.w =  _in_id[1].w + _in_off.w; \\\n\
            _regA[1] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv1[_in.x] : ZEROv1;\\\n\
        }\n\
\n\
#define LOAD_dAv1_SIZE4(_regA, _dAv1, _in_id, _in_off) \\\n\
        { \\\n\
            int4 _in; \\\n\
            \\\n\
            _in.x = (_in_id[0].x + _in_off.x) * _INT4_TO_4INT_; \\\n\
            _in.y =  _in_id[0].y + _in_off.y; \\\n\
            _in.z =  _in_id[0].z + _in_off.z; \\\n\
            _in.w =  _in_id[0].w + _in_off.w; \\\n\
            _regA[0] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv1[_in.x] : ZEROv1;\\\n\
            \\\n\
            _in.x = (_in_id[1].x + _in_off.x) * _INT4_TO_4INT_; \\\n\
            _in.y =  _in_id[1].y + _in_off.y; \\\n\
            _in.z =  _in_id[1].z + _in_off.z; \\\n\
            _in.w =  _in_id[1].w + _in_off.w; \\\n\
            _regA[1] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv1[_in.x] : ZEROv1;\\\n\
            \\\n\
            _in.x = (_in_id[2].x + _in_off.x) * _INT4_TO_4INT_; \\\n\
            _in.y =  _in_id[2].y + _in_off.y; \\\n\
            _in.z =  _in_id[2].z + _in_off.z; \\\n\
            _in.w =  _in_id[2].w + _in_off.w; \\\n\
            _regA[2] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv1[_in.x] : ZEROv1;\\\n\
            \\\n\
            _in.x = (_in_id[3].x + _in_off.x) * _INT4_TO_4INT_; \\\n\
            _in.y =  _in_id[3].y + _in_off.y; \\\n\
            _in.z =  _in_id[3].z + _in_off.z; \\\n\
            _in.w =  _in_id[3].w + _in_off.w; \\\n\
            _regA[3] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv1[_in.x] : ZEROv1;\\\n\
        }\n\
\n\
#define LOAD_dAv1_SIZE8(_regA, _dAv1, _in_id, _in_off) \\\n\
        { \\\n\
            int4 _in; \\\n\
            \\\n\
            _in.x = (_in_id[0].x + _in_off.x) * _INT4_TO_4INT_; \\\n\
            _in.y =  _in_id[0].y + _in_off.y; \\\n\
            _in.z =  _in_id[0].z + _in_off.z; \\\n\
            _in.w =  _in_id[0].w + _in_off.w; \\\n\
            _regA[0] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv1[_in.x] : ZEROv1;\\\n\
            \\\n\
            _in.x = (_in_id[1].x + _in_off.x) * _INT4_TO_4INT_; \\\n\
            _in.y =  _in_id[1].y + _in_off.y; \\\n\
            _in.z =  _in_id[1].z + _in_off.z; \\\n\
            _in.w =  _in_id[1].w + _in_off.w; \\\n\
            _regA[1] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv1[_in.x] : ZEROv1;\\\n\
            \\\n\
            _in.x = (_in_id[2].x + _in_off.x) * _INT4_TO_4INT_; \\\n\
            _in.y =  _in_id[2].y + _in_off.y; \\\n\
            _in.z =  _in_id[2].z + _in_off.z; \\\n\
            _in.w =  _in_id[2].w + _in_off.w; \\\n\
            _regA[2] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv1[_in.x] : ZEROv1;\\\n\
            \\\n\
            _in.x = (_in_id[3].x + _in_off.x) * _INT4_TO_4INT_; \\\n\
            _in.y =  _in_id[3].y + _in_off.y; \\\n\
            _in.z =  _in_id[3].z + _in_off.z; \\\n\
            _in.w =  _in_id[3].w + _in_off.w; \\\n\
            _regA[3] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv1[_in.x] : ZEROv1;\\\n\
            \\\n\
            _in.x = (_in_id[4].x + _in_off.x) * _INT4_TO_4INT_; \\\n\
            _in.y =  _in_id[4].y + _in_off.y; \\\n\
            _in.z =  _in_id[4].z + _in_off.z; \\\n\
            _in.w =  _in_id[4].w + _in_off.w; \\\n\
            _regA[4] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv1[_in.x] : ZEROv1;\\\n\
            \\\n\
            _in.x = (_in_id[5].x + _in_off.x) * _INT4_TO_4INT_; \\\n\
            _in.y =  _in_id[5].y + _in_off.y; \\\n\
            _in.z =  _in_id[5].z + _in_off.z; \\\n\
            _in.w =  _in_id[5].w + _in_off.w; \\\n\
            _regA[5] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv1[_in.x] : ZEROv1;\\\n\
            \\\n\
            _in.x = (_in_id[6].x + _in_off.x) * _INT4_TO_4INT_; \\\n\
            _in.y =  _in_id[6].y + _in_off.y; \\\n\
            _in.z =  _in_id[6].z + _in_off.z; \\\n\
            _in.w =  _in_id[6].w + _in_off.w; \\\n\
            _regA[6] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv1[_in.x] : ZEROv1;\\\n\
            \\\n\
            _in.x = (_in_id[7].x + _in_off.x) * _INT4_TO_4INT_; \\\n\
            _in.y =  _in_id[7].y + _in_off.y; \\\n\
            _in.z =  _in_id[7].z + _in_off.z; \\\n\
            _in.w =  _in_id[7].w + _in_off.w; \\\n\
            _regA[7] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv1[_in.x] : ZEROv1;\\\n\
        }\n\
");
    header_code_.emplace("/mnt/hpc/xusi/ppl.jit/src/ppl/nn/engines/cuda/impls/src/nn/conv/idxn/common/hmma_i1_macros.h", "// Licensed to the Apache Software Foundation (ASF) under one\n\
// or more contributor license agreements.  See the NOTICE file\n\
// distributed with this work for additional information\n\
// regarding copyright ownership.  The ASF licenses this file\n\
// to you under the Apache License, Version 2.0 (the\n\
// \"License\"); you may not use this file except in compliance\n\
// with the License.  You may obtain a copy of the License at\n\
//\n\
//   http://www.apache.org/licenses/LICENSE-2.0\n\
//\n\
// Unless required by applicable law or agreed to in writing,\n\
// software distributed under the License is distributed on an\n\
// \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n\
// KIND, either express or implied.  See the License for the\n\
// specific language governing permissions and limitations\n\
// under the License.\n\
\n\
////////////////////////////////////////\n\
// hmma macros\n\
////////////////////////////////////////\n\
\n\
#define MMA_INST_OPCODE \\\n\
        \"mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%5,%6};\\n\"\n\
        \n\
#define MMA_INST(_d0, _d1, _a0, _a1, _b) \\\n\
        asm volatile(MMA_INST_OPCODE:   \"=r\"(_d0),   \"=r\"(_d1): \"r\"(_a0), \"r\"(_a1), \"r\"(_b),  \"r\"(_d0),   \"r\"(_d1));\n\
\n\
\n\
#define MMA_INST_1INT_ASCEND1(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _Bv1[_Bv1_off]); \\\n\
        }\n\
        \n\
#define MMA_INST_1INT_ASCEND2(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _Bv1[_Bv1_off]); \\\n\
            MMA_INST(_C[_C_off + 1], _C[_C_off + _C_stride + 1], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 1]); \\\n\
        }\n\
        \n\
#define MMA_INST_1INT_ASCEND4(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _Bv1[_Bv1_off]); \\\n\
            MMA_INST(_C[_C_off + 1], _C[_C_off + _C_stride + 1], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 1]); \\\n\
            MMA_INST(_C[_C_off + 2], _C[_C_off + _C_stride + 2], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 2]); \\\n\
            MMA_INST(_C[_C_off + 3], _C[_C_off + _C_stride + 3], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 3]); \\\n\
        }\n\
        \n\
#define MMA_INST_1INT_ASCEND8(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _Bv1[_Bv1_off]); \\\n\
            MMA_INST(_C[_C_off + 1], _C[_C_off + _C_stride + 1], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 1]); \\\n\
            MMA_INST(_C[_C_off + 2], _C[_C_off + _C_stride + 2], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 2]); \\\n\
            MMA_INST(_C[_C_off + 3], _C[_C_off + _C_stride + 3], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 3]); \\\n\
            MMA_INST(_C[_C_off + 4], _C[_C_off + _C_stride + 4], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 4]); \\\n\
            MMA_INST(_C[_C_off + 5], _C[_C_off + _C_stride + 5], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 5]); \\\n\
            MMA_INST(_C[_C_off + 6], _C[_C_off + _C_stride + 6], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 6]); \\\n\
            MMA_INST(_C[_C_off + 7], _C[_C_off + _C_stride + 7], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 7]); \\\n\
        }\n\
        \n\
#define MMA_INST_1INT_DESCEND1(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _Bv1[_Bv1_off]); \\\n\
        }\n\
\n\
#define MMA_INST_1INT_DESCEND2(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 1]); \\\n\
            MMA_INST(_C[_C_off - 1], _C[_C_off + _C_stride - 1], _a0, _a1, _Bv1[_Bv1_off]); \\\n\
        }\n\
\n\
#define MMA_INST_1INT_DESCEND4(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 3]); \\\n\
            MMA_INST(_C[_C_off - 1], _C[_C_off + _C_stride - 1], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 2]); \\\n\
            MMA_INST(_C[_C_off - 2], _C[_C_off + _C_stride - 2], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 1]); \\\n\
            MMA_INST(_C[_C_off - 3], _C[_C_off + _C_stride - 3], _a0, _a1, _Bv1[_Bv1_off]); \\\n\
        }\n\
\n\
#define MMA_INST_1INT_DESCEND8(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 7]); \\\n\
            MMA_INST(_C[_C_off - 1], _C[_C_off + _C_stride - 1], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 6]); \\\n\
            MMA_INST(_C[_C_off - 2], _C[_C_off + _C_stride - 2], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 5]); \\\n\
            MMA_INST(_C[_C_off - 3], _C[_C_off + _C_stride - 3], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 4]); \\\n\
            MMA_INST(_C[_C_off - 4], _C[_C_off + _C_stride - 4], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 3]); \\\n\
            MMA_INST(_C[_C_off - 5], _C[_C_off + _C_stride - 5], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 2]); \\\n\
            MMA_INST(_C[_C_off - 6], _C[_C_off + _C_stride - 6], _a0, _a1, _Bv1[_Bv1_off + _1INT_ * 1]); \\\n\
            MMA_INST(_C[_C_off - 7], _C[_C_off + _C_stride - 7], _a0, _a1, _Bv1[_Bv1_off]); \\\n\
        }\n\
\n\
#define MMA_INST_1INT_1x1(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_1INT_ASCEND1 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0], _Av1[0 + _1INT_], _Bv1, 0); \\\n\
        }\n\
\n\
#define MMA_INST_1INT_1x2(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_1INT_ASCEND2 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0], _Av1[0 + _1INT_], _Bv1, 0); \\\n\
        }\n\
\n\
#define MMA_INST_1INT_1x4(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_1INT_ASCEND4 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0], _Av1[0 + _1INT_], _Bv1, 0); \\\n\
        }\n\
\n\
#define MMA_INST_1INT_1x8(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_1INT_ASCEND8 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0], _Av1[0 + _1INT_], _Bv1, 0); \\\n\
        }\n\
\n\
#define MMA_INST_1INT_2x1(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_1INT_ASCEND1 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],             _Av1[0 + _1INT_],             _Bv1, 0); \\\n\
            MMA_INST_1INT_DESCEND1(_C, 2,  TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_], _Av1[0 + _1INT_ + _1INT_X2_], _Bv1, 0); \\\n\
        }\n\
\n\
#define MMA_INST_1INT_2x2(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_1INT_ASCEND2 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],             _Av1[0 + _1INT_],             _Bv1, 0); \\\n\
            MMA_INST_1INT_DESCEND2(_C, 5,  TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_], _Av1[0 + _1INT_ + _1INT_X2_], _Bv1, 0); \\\n\
        }\n\
\n\
#define MMA_INST_1INT_2x4(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_1INT_ASCEND4 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],             _Av1[0 + _1INT_],             _Bv1, 0); \\\n\
            MMA_INST_1INT_DESCEND4(_C, 11, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_], _Av1[0 + _1INT_ + _1INT_X2_], _Bv1, 0); \\\n\
        }\n\
\n\
#define MMA_INST_1INT_2x8(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_1INT_ASCEND8 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],             _Av1[0 + _1INT_],             _Bv1, 0); \\\n\
            MMA_INST_1INT_DESCEND8(_C, 23, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_], _Av1[0 + _1INT_ + _1INT_X2_], _Bv1, 0); \\\n\
        }\n\
\n\
#define MMA_INST_1INT_4x1(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_1INT_ASCEND1 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],                 _Av1[0 + _1INT_],                 _Bv1, 0); \\\n\
            MMA_INST_1INT_DESCEND1(_C, 2,  TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 1], _Av1[0 + _1INT_ + _1INT_X2_ * 1], _Bv1, 0); \\\n\
            MMA_INST_1INT_ASCEND1 (_C, 4,  TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 2], _Av1[0 + _1INT_ + _1INT_X2_ * 2], _Bv1, 0); \\\n\
            MMA_INST_1INT_DESCEND1(_C, 6,  TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 3], _Av1[0 + _1INT_ + _1INT_X2_ * 3], _Bv1, 0); \\\n\
        }\n\
\n\
#define MMA_INST_1INT_4x2(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_1INT_ASCEND2 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],                 _Av1[0 + _1INT_],                 _Bv1, 0); \\\n\
            MMA_INST_1INT_DESCEND2(_C, 5,  TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 1], _Av1[0 + _1INT_ + _1INT_X2_ * 1], _Bv1, 0); \\\n\
            MMA_INST_1INT_ASCEND2 (_C, 8,  TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 2], _Av1[0 + _1INT_ + _1INT_X2_ * 2], _Bv1, 0); \\\n\
            MMA_INST_1INT_DESCEND2(_C, 13, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 3], _Av1[0 + _1INT_ + _1INT_X2_ * 3], _Bv1, 0); \\\n\
        }\n\
\n\
#define MMA_INST_1INT_4x4(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_1INT_ASCEND4 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],                 _Av1[0 + _1INT_],                 _Bv1, 0); \\\n\
            MMA_INST_1INT_DESCEND4(_C, 11, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 1], _Av1[0 + _1INT_ + _1INT_X2_ * 1], _Bv1, 0); \\\n\
            MMA_INST_1INT_ASCEND4 (_C, 16, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 2], _Av1[0 + _1INT_ + _1INT_X2_ * 2], _Bv1, 0); \\\n\
            MMA_INST_1INT_DESCEND4(_C, 27, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 3], _Av1[0 + _1INT_ + _1INT_X2_ * 3], _Bv1, 0); \\\n\
        }\n\
\n\
#define MMA_INST_1INT_4x8(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_1INT_ASCEND8 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],                 _Av1[0 + _1INT_],                 _Bv1, 0); \\\n\
            MMA_INST_1INT_DESCEND8(_C, 23, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 1], _Av1[0 + _1INT_ + _1INT_X2_ * 1], _Bv1, 0); \\\n\
            MMA_INST_1INT_ASCEND8 (_C, 32, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 2], _Av1[0 + _1INT_ + _1INT_X2_ * 2], _Bv1, 0); \\\n\
            MMA_INST_1INT_DESCEND8(_C, 55, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 3], _Av1[0 + _1INT_ + _1INT_X2_ * 3], _Bv1, 0); \\\n\
        }\n\
\n\
#define MMA_INST_1INT_8x1(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_1INT_ASCEND1 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],                 _Av1[0 + _1INT_],                 _Bv1, 0); \\\n\
            MMA_INST_1INT_DESCEND1(_C, 2,  TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 1], _Av1[0 + _1INT_ + _1INT_X2_ * 1], _Bv1, 0); \\\n\
            MMA_INST_1INT_ASCEND1 (_C, 4,  TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 2], _Av1[0 + _1INT_ + _1INT_X2_ * 2], _Bv1, 0); \\\n\
            MMA_INST_1INT_DESCEND1(_C, 6,  TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 3], _Av1[0 + _1INT_ + _1INT_X2_ * 3], _Bv1, 0); \\\n\
            MMA_INST_1INT_ASCEND1 (_C, 8,  TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 4], _Av1[0 + _1INT_ + _1INT_X2_ * 4], _Bv1, 0); \\\n\
            MMA_INST_1INT_DESCEND1(_C, 10, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 5], _Av1[0 + _1INT_ + _1INT_X2_ * 5], _Bv1, 0); \\\n\
            MMA_INST_1INT_ASCEND1 (_C, 12, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 6], _Av1[0 + _1INT_ + _1INT_X2_ * 6], _Bv1, 0); \\\n\
            MMA_INST_1INT_DESCEND1(_C, 14, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 7], _Av1[0 + _1INT_ + _1INT_X2_ * 7], _Bv1, 0); \\\n\
        }\n\
\n\
#define MMA_INST_1INT_8x2(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_1INT_ASCEND2 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],                 _Av1[0 + _1INT_],                 _Bv1, 0); \\\n\
            MMA_INST_1INT_DESCEND2(_C, 5,  TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 1], _Av1[0 + _1INT_ + _1INT_X2_ * 1], _Bv1, 0); \\\n\
            MMA_INST_1INT_ASCEND2 (_C, 8,  TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 2], _Av1[0 + _1INT_ + _1INT_X2_ * 2], _Bv1, 0); \\\n\
            MMA_INST_1INT_DESCEND2(_C, 13, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 3], _Av1[0 + _1INT_ + _1INT_X2_ * 3], _Bv1, 0); \\\n\
            MMA_INST_1INT_ASCEND2 (_C, 16, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 4], _Av1[0 + _1INT_ + _1INT_X2_ * 4], _Bv1, 0); \\\n\
            MMA_INST_1INT_DESCEND2(_C, 21, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 5], _Av1[0 + _1INT_ + _1INT_X2_ * 5], _Bv1, 0); \\\n\
            MMA_INST_1INT_ASCEND2 (_C, 24, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 6], _Av1[0 + _1INT_ + _1INT_X2_ * 6], _Bv1, 0); \\\n\
            MMA_INST_1INT_DESCEND2(_C, 29, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 7], _Av1[0 + _1INT_ + _1INT_X2_ * 7], _Bv1, 0); \\\n\
        }\n\
\n\
#define MMA_INST_1INT_8x4(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_1INT_ASCEND4 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],                 _Av1[0 + _1INT_],                 _Bv1, 0); \\\n\
            MMA_INST_1INT_DESCEND4(_C, 11, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 1], _Av1[0 + _1INT_ + _1INT_X2_ * 1], _Bv1, 0); \\\n\
            MMA_INST_1INT_ASCEND4 (_C, 16, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 2], _Av1[0 + _1INT_ + _1INT_X2_ * 2], _Bv1, 0); \\\n\
            MMA_INST_1INT_DESCEND4(_C, 27, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 3], _Av1[0 + _1INT_ + _1INT_X2_ * 3], _Bv1, 0); \\\n\
            MMA_INST_1INT_ASCEND4 (_C, 32, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 4], _Av1[0 + _1INT_ + _1INT_X2_ * 4], _Bv1, 0); \\\n\
            MMA_INST_1INT_DESCEND4(_C, 43, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 5], _Av1[0 + _1INT_ + _1INT_X2_ * 5], _Bv1, 0); \\\n\
            MMA_INST_1INT_ASCEND4 (_C, 48, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 6], _Av1[0 + _1INT_ + _1INT_X2_ * 6], _Bv1, 0); \\\n\
            MMA_INST_1INT_DESCEND4(_C, 59, TILE_N_V2_PER_THD, _Av1[0 + _1INT_X2_ * 7], _Av1[0 + _1INT_ + _1INT_X2_ * 7], _Bv1, 0); \\\n\
        }\n\
\n\
");
    header_code_.emplace("/mnt/hpc/xusi/ppl.jit/src/ppl/nn/engines/cuda/impls/src/nn/conv/idxn/common/dmem_i2_macros.h", "// Licensed to the Apache Software Foundation (ASF) under one\n\
// or more contributor license agreements.  See the NOTICE file\n\
// distributed with this work for additional information\n\
// regarding copyright ownership.  The ASF licenses this file\n\
// to you under the Apache License, Version 2.0 (the\n\
// \"License\"); you may not use this file except in compliance\n\
// with the License.  You may obtain a copy of the License at\n\
//\n\
//   http://www.apache.org/licenses/LICENSE-2.0\n\
//\n\
// Unless required by applicable law or agreed to in writing,\n\
// software distributed under the License is distributed on an\n\
// \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n\
// KIND, either express or implied.  See the License for the\n\
// specific language governing permissions and limitations\n\
// under the License.\n\
\n\
/////////////////////////////////////////////////////\n\
// common load global memory macros\n\
/////////////////////////////////////////////////////\n\
\n\
////////////////////////////////////////\n\
// load dB macros\n\
////////////////////////////////////////\n\
\n\
#define LOAD_dBv2_SIZE1(_regB, _dBv2, _dBv2_off) \\\n\
        { \\\n\
            _regB[0] = (flt_n_valid[0] && (flt_hwc_v4_off < flt_hwc_v4)) ? _dBv2[ _dBv2_off[0] * _INT4_TO_2INT2_ ] : ZEROv2;\\\n\
            \\\n\
            _dBv2_off[0] += TILE_K_V4_PER_STEP; \\\n\
            \\\n\
            flt_hwc_v4_off  += TILE_K_V4_PER_STEP; \\\n\
        }\n\
\n\
#define LOAD_dBv2_SIZE2(_regB, _dBv2, _dBv2_off) \\\n\
        { \\\n\
            _regB[0] = (flt_n_valid[0] && (flt_hwc_v4_off < flt_hwc_v4)) ? _dBv2[ _dBv2_off[0] * _INT4_TO_2INT2_ ] : ZEROv2;\\\n\
            _regB[1] = (flt_n_valid[1] && (flt_hwc_v4_off < flt_hwc_v4)) ? _dBv2[ _dBv2_off[1] * _INT4_TO_2INT2_ ] : ZEROv2;\\\n\
            \\\n\
            _dBv2_off[0] += TILE_K_V4_PER_STEP; \\\n\
            _dBv2_off[1] += TILE_K_V4_PER_STEP; \\\n\
            \\\n\
            flt_hwc_v4_off  += TILE_K_V4_PER_STEP; \\\n\
        }\n\
\n\
#define LOAD_dBv2_SIZE4(_regB, _dBv2, _dBv2_off) \\\n\
        { \\\n\
            _regB[0] = (flt_n_valid[0] && (flt_hwc_v4_off < flt_hwc_v4)) ? _dBv2[ _dBv2_off[0] * _INT4_TO_2INT2_ ] : ZEROv2;\\\n\
            _regB[1] = (flt_n_valid[1] && (flt_hwc_v4_off < flt_hwc_v4)) ? _dBv2[ _dBv2_off[1] * _INT4_TO_2INT2_ ] : ZEROv2;\\\n\
            _regB[2] = (flt_n_valid[2] && (flt_hwc_v4_off < flt_hwc_v4)) ? _dBv2[ _dBv2_off[2] * _INT4_TO_2INT2_ ] : ZEROv2;\\\n\
            _regB[3] = (flt_n_valid[3] && (flt_hwc_v4_off < flt_hwc_v4)) ? _dBv2[ _dBv2_off[3] * _INT4_TO_2INT2_ ] : ZEROv2;\\\n\
            \\\n\
            _dBv2_off[0] += TILE_K_V4_PER_STEP; \\\n\
            _dBv2_off[1] += TILE_K_V4_PER_STEP; \\\n\
            _dBv2_off[2] += TILE_K_V4_PER_STEP; \\\n\
            _dBv2_off[3] += TILE_K_V4_PER_STEP; \\\n\
            \\\n\
            flt_hwc_v4_off  += TILE_K_V4_PER_STEP; \\\n\
        }\n\
\n\
#define LOAD_dBv2_SIZE8(_regB, _dBv2, _dBv2_off) \\\n\
        { \\\n\
            _regB[0] = (flt_n_valid[0] && (flt_hwc_v4_off < flt_hwc_v4)) ? _dBv2[ _dBv2_off[0] * _INT4_TO_2INT2_ ] : ZEROv2;\\\n\
            _regB[1] = (flt_n_valid[1] && (flt_hwc_v4_off < flt_hwc_v4)) ? _dBv2[ _dBv2_off[1] * _INT4_TO_2INT2_ ] : ZEROv2;\\\n\
            _regB[2] = (flt_n_valid[2] && (flt_hwc_v4_off < flt_hwc_v4)) ? _dBv2[ _dBv2_off[2] * _INT4_TO_2INT2_ ] : ZEROv2;\\\n\
            _regB[3] = (flt_n_valid[3] && (flt_hwc_v4_off < flt_hwc_v4)) ? _dBv2[ _dBv2_off[3] * _INT4_TO_2INT2_ ] : ZEROv2;\\\n\
            _regB[4] = (flt_n_valid[4] && (flt_hwc_v4_off < flt_hwc_v4)) ? _dBv2[ _dBv2_off[4] * _INT4_TO_2INT2_ ] : ZEROv2;\\\n\
            _regB[5] = (flt_n_valid[5] && (flt_hwc_v4_off < flt_hwc_v4)) ? _dBv2[ _dBv2_off[5] * _INT4_TO_2INT2_ ] : ZEROv2;\\\n\
            _regB[6] = (flt_n_valid[6] && (flt_hwc_v4_off < flt_hwc_v4)) ? _dBv2[ _dBv2_off[6] * _INT4_TO_2INT2_ ] : ZEROv2;\\\n\
            _regB[7] = (flt_n_valid[7] && (flt_hwc_v4_off < flt_hwc_v4)) ? _dBv2[ _dBv2_off[7] * _INT4_TO_2INT2_ ] : ZEROv2;\\\n\
            \\\n\
            _dBv2_off[0] += TILE_K_V4_PER_STEP; \\\n\
            _dBv2_off[1] += TILE_K_V4_PER_STEP; \\\n\
            _dBv2_off[2] += TILE_K_V4_PER_STEP; \\\n\
            _dBv2_off[3] += TILE_K_V4_PER_STEP; \\\n\
            _dBv2_off[4] += TILE_K_V4_PER_STEP; \\\n\
            _dBv2_off[5] += TILE_K_V4_PER_STEP; \\\n\
            _dBv2_off[6] += TILE_K_V4_PER_STEP; \\\n\
            _dBv2_off[7] += TILE_K_V4_PER_STEP; \\\n\
            \\\n\
            flt_hwc_v4_off  += TILE_K_V4_PER_STEP; \\\n\
        }\n\
\n\
#define SET_dBv2_BOUND(_step_id, _dBv2_off, _flt_n_valid) \\\n\
        { \\\n\
            int _flt_n_id  =  cta_idx * TILE_N_PER_CTA  + \\\n\
                             _step_id * TILE_N_PER_STEP + \\\n\
                             warp_idx * TILE_N_PER_MMA  + \\\n\
                             tid_y; \\\n\
            \\\n\
            _flt_n_valid  =  _flt_n_id < num_flt_per_grp_pad; \\\n\
            \\\n\
            _dBv2_off  =   grp_id   * flt_hwc_v4 * num_flt_per_grp_pad + \\\n\
                          _flt_n_id * flt_hwc_v4 + \\\n\
                           tid_x; \\\n\
        }\n\
\n\
////////////////////////////////////////\n\
// load dA macros\n\
////////////////////////////////////////\n\
\n\
#define SET_IN_Mv1_ID(_tid, _sm_base_v4) \\\n\
        { \\\n\
            int _out_nhw_id =  cta_idy * TILE_M_PER_CTA + _tid; \\\n\
            \\\n\
            int _out_w_id   = (_out_nhw_id % out_width); \\\n\
            int _out_h_id   = (_out_nhw_id / out_width) % out_height; \\\n\
            \\\n\
            int4 _in_id; \\\n\
            \\\n\
            _in_id.y = _out_w_id * stride_width  - pad_width; \\\n\
            _in_id.z = _out_h_id * stride_height - pad_height; \\\n\
            _in_id.w = _out_nhw_id / out_hw; \\\n\
            \\\n\
            _in_id.x = (_in_id.w * in_hw + _in_id.z * in_width + _in_id.y) * in_chl_per_grp_pad_v8 * num_grp + \\\n\
                         grp_id  * in_chl_per_grp_pad_v8; \\\n\
            \\\n\
            _sm_base_v4[_tid] = _in_id; \\\n\
        }\n\
\n\
#define SET_IN_Kv8_OFF(_tid, _sm_base_v4) \\\n\
        { \\\n\
            int _inNHWC8_id =  _tid; \\\n\
            \\\n\
            int4 _in_off; \\\n\
            \\\n\
            _in_off.y = ((_inNHWC8_id /  in_chl_per_grp_pad_v8) % flt_width)  * hole_width; \\\n\
            _in_off.z = ((_inNHWC8_id / (in_chl_per_grp_pad_v8  * flt_width)) % flt_height) * hole_height; \\\n\
            _in_off.w =   _inNHWC8_id / (in_chl_per_grp_pad_v8  * flt_width   * flt_height); \\\n\
            \\\n\
            _in_off.x = (_in_off.w  * in_hw + _in_off.z * in_width + _in_off.y) * in_chl_per_grp_pad_v8 * num_grp + \\\n\
                        (_inNHWC8_id %  in_chl_per_grp_pad_v8); \\\n\
            \\\n\
            _sm_base_v4[SM_IN_ID_SIZE + _tid] = _in_off; \\\n\
         }\n\
\n\
#define LOAD_dAv2_SIZE2(_regA, _dAv2, _in_id, _in_off) \\\n\
        { \\\n\
            int4 _in; \\\n\
            \\\n\
            _in.x = (_in_id[0].x + _in_off.x) * _INT4_TO_2INT2_; \\\n\
            _in.y =  _in_id[0].y + _in_off.y; \\\n\
            _in.z =  _in_id[0].z + _in_off.z; \\\n\
            _in.w =  _in_id[0].w + _in_off.w; \\\n\
            _regA[0] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv2[_in.x] : ZEROv2;\\\n\
            \\\n\
            _in.x = (_in_id[1].x + _in_off.x) * _INT4_TO_2INT2_; \\\n\
            _in.y =  _in_id[1].y + _in_off.y; \\\n\
            _in.z =  _in_id[1].z + _in_off.z; \\\n\
            _in.w =  _in_id[1].w + _in_off.w; \\\n\
            _regA[1] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv2[_in.x] : ZEROv2;\\\n\
        }\n\
\n\
#define LOAD_dAv2_SIZE4(_regA, _dAv2, _in_id, _in_off) \\\n\
        { \\\n\
            int4 _in; \\\n\
            \\\n\
            _in.x = (_in_id[0].x + _in_off.x) * _INT4_TO_2INT2_; \\\n\
            _in.y =  _in_id[0].y + _in_off.y; \\\n\
            _in.z =  _in_id[0].z + _in_off.z; \\\n\
            _in.w =  _in_id[0].w + _in_off.w; \\\n\
            _regA[0] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv2[_in.x] : ZEROv2;\\\n\
            \\\n\
            _in.x = (_in_id[1].x + _in_off.x) * _INT4_TO_2INT2_; \\\n\
            _in.y =  _in_id[1].y + _in_off.y; \\\n\
            _in.z =  _in_id[1].z + _in_off.z; \\\n\
            _in.w =  _in_id[1].w + _in_off.w; \\\n\
            _regA[1] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv2[_in.x] : ZEROv2;\\\n\
            \\\n\
            _in.x = (_in_id[2].x + _in_off.x) * _INT4_TO_2INT2_; \\\n\
            _in.y =  _in_id[2].y + _in_off.y; \\\n\
            _in.z =  _in_id[2].z + _in_off.z; \\\n\
            _in.w =  _in_id[2].w + _in_off.w; \\\n\
            _regA[2] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv2[_in.x] : ZEROv2;\\\n\
            \\\n\
            _in.x = (_in_id[3].x + _in_off.x) * _INT4_TO_2INT2_; \\\n\
            _in.y =  _in_id[3].y + _in_off.y; \\\n\
            _in.z =  _in_id[3].z + _in_off.z; \\\n\
            _in.w =  _in_id[3].w + _in_off.w; \\\n\
            _regA[3] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv2[_in.x] : ZEROv2;\\\n\
        }\n\
\n\
#define LOAD_dAv2_SIZE8(_regA, _dAv2, _in_id, _in_off) \\\n\
        { \\\n\
            int4 _in; \\\n\
            \\\n\
            _in.x = (_in_id[0].x + _in_off.x) * _INT4_TO_2INT2_; \\\n\
            _in.y =  _in_id[0].y + _in_off.y; \\\n\
            _in.z =  _in_id[0].z + _in_off.z; \\\n\
            _in.w =  _in_id[0].w + _in_off.w; \\\n\
            _regA[0] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv2[_in.x] : ZEROv2;\\\n\
            \\\n\
            _in.x = (_in_id[1].x + _in_off.x) * _INT4_TO_2INT2_; \\\n\
            _in.y =  _in_id[1].y + _in_off.y; \\\n\
            _in.z =  _in_id[1].z + _in_off.z; \\\n\
            _in.w =  _in_id[1].w + _in_off.w; \\\n\
            _regA[1] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv2[_in.x] : ZEROv2;\\\n\
            \\\n\
            _in.x = (_in_id[2].x + _in_off.x) * _INT4_TO_2INT2_; \\\n\
            _in.y =  _in_id[2].y + _in_off.y; \\\n\
            _in.z =  _in_id[2].z + _in_off.z; \\\n\
            _in.w =  _in_id[2].w + _in_off.w; \\\n\
            _regA[2] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv2[_in.x] : ZEROv2;\\\n\
            \\\n\
            _in.x = (_in_id[3].x + _in_off.x) * _INT4_TO_2INT2_; \\\n\
            _in.y =  _in_id[3].y + _in_off.y; \\\n\
            _in.z =  _in_id[3].z + _in_off.z; \\\n\
            _in.w =  _in_id[3].w + _in_off.w; \\\n\
            _regA[3] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv2[_in.x] : ZEROv2;\\\n\
            \\\n\
            _in.x = (_in_id[4].x + _in_off.x) * _INT4_TO_2INT2_; \\\n\
            _in.y =  _in_id[4].y + _in_off.y; \\\n\
            _in.z =  _in_id[4].z + _in_off.z; \\\n\
            _in.w =  _in_id[4].w + _in_off.w; \\\n\
            _regA[4] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv2[_in.x] : ZEROv2;\\\n\
            \\\n\
            _in.x = (_in_id[5].x + _in_off.x) * _INT4_TO_2INT2_; \\\n\
            _in.y =  _in_id[5].y + _in_off.y; \\\n\
            _in.z =  _in_id[5].z + _in_off.z; \\\n\
            _in.w =  _in_id[5].w + _in_off.w; \\\n\
            _regA[5] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv2[_in.x] : ZEROv2;\\\n\
            \\\n\
            _in.x = (_in_id[6].x + _in_off.x) * _INT4_TO_2INT2_; \\\n\
            _in.y =  _in_id[6].y + _in_off.y; \\\n\
            _in.z =  _in_id[6].z + _in_off.z; \\\n\
            _in.w =  _in_id[6].w + _in_off.w; \\\n\
            _regA[6] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv2[_in.x] : ZEROv2;\\\n\
            \\\n\
            _in.x = (_in_id[7].x + _in_off.x) * _INT4_TO_2INT2_; \\\n\
            _in.y =  _in_id[7].y + _in_off.y; \\\n\
            _in.z =  _in_id[7].z + _in_off.z; \\\n\
            _in.w =  _in_id[7].w + _in_off.w; \\\n\
            _regA[7] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv2[_in.x] : ZEROv2;\\\n\
        }\n\
");
    header_code_.emplace("/mnt/hpc/xusi/ppl.jit/src/ppl/nn/engines/cuda/impls/src/nn/conv/idxn/common/hmma_i2_macros.h", "// Licensed to the Apache Software Foundation (ASF) under one\n\
// or more contributor license agreements.  See the NOTICE file\n\
// distributed with this work for additional information\n\
// regarding copyright ownership.  The ASF licenses this file\n\
// to you under the Apache License, Version 2.0 (the\n\
// \"License\"); you may not use this file except in compliance\n\
// with the License.  You may obtain a copy of the License at\n\
//\n\
//   http://www.apache.org/licenses/LICENSE-2.0\n\
//\n\
// Unless required by applicable law or agreed to in writing,\n\
// software distributed under the License is distributed on an\n\
// \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n\
// KIND, either express or implied.  See the License for the\n\
// specific language governing permissions and limitations\n\
// under the License.\n\
\n\
////////////////////////////////////////\n\
// hmma macros\n\
////////////////////////////////////////\n\
\n\
#define MMA_INST_OPCODE \\\n\
        \"mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%5,%6};\\n\"\n\
        \n\
#define MMA_INST(_d0, _d1, _a0, _a1, _b) \\\n\
        asm volatile(MMA_INST_OPCODE:   \"=r\"(_d0),   \"=r\"(_d1): \"r\"(_a0), \"r\"(_a1), \"r\"(_b),  \"r\"(_d0),   \"r\"(_d1));\n\
\n\
\n\
#define MMA_INST_2INT_ASCEND1(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _Bv1[_Bv1_off]); \\\n\
        }\n\
        \n\
#define MMA_INST_2INT_ASCEND2(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _Bv1[_Bv1_off]); \\\n\
            MMA_INST(_C[_C_off + 1], _C[_C_off + _C_stride + 1], _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 1]); \\\n\
        }\n\
        \n\
#define MMA_INST_2INT_ASCEND4(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _Bv1[_Bv1_off]); \\\n\
            MMA_INST(_C[_C_off + 1], _C[_C_off + _C_stride + 1], _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 1]); \\\n\
            MMA_INST(_C[_C_off + 2], _C[_C_off + _C_stride + 2], _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 2]); \\\n\
            MMA_INST(_C[_C_off + 3], _C[_C_off + _C_stride + 3], _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 3]); \\\n\
        }\n\
        \n\
#define MMA_INST_2INT_ASCEND8(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _Bv1[_Bv1_off]); \\\n\
            MMA_INST(_C[_C_off + 1], _C[_C_off + _C_stride + 1], _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 1]); \\\n\
            MMA_INST(_C[_C_off + 2], _C[_C_off + _C_stride + 2], _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 2]); \\\n\
            MMA_INST(_C[_C_off + 3], _C[_C_off + _C_stride + 3], _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 3]); \\\n\
            MMA_INST(_C[_C_off + 4], _C[_C_off + _C_stride + 4], _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 4]); \\\n\
            MMA_INST(_C[_C_off + 5], _C[_C_off + _C_stride + 5], _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 5]); \\\n\
            MMA_INST(_C[_C_off + 6], _C[_C_off + _C_stride + 6], _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 6]); \\\n\
            MMA_INST(_C[_C_off + 7], _C[_C_off + _C_stride + 7], _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 7]); \\\n\
        }\n\
        \n\
#define MMA_INST_2INT_DESCEND1(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _Bv1[_Bv1_off]); \\\n\
        }\n\
\n\
#define MMA_INST_2INT_DESCEND2(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 1]); \\\n\
            MMA_INST(_C[_C_off - 1], _C[_C_off + _C_stride - 1], _a0, _a1, _Bv1[_Bv1_off]); \\\n\
        }\n\
\n\
#define MMA_INST_2INT_DESCEND4(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 3]); \\\n\
            MMA_INST(_C[_C_off - 1], _C[_C_off + _C_stride - 1], _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 2]); \\\n\
            MMA_INST(_C[_C_off - 2], _C[_C_off + _C_stride - 2], _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 1]); \\\n\
            MMA_INST(_C[_C_off - 3], _C[_C_off + _C_stride - 3], _a0, _a1, _Bv1[_Bv1_off]); \\\n\
        }\n\
\n\
#define MMA_INST_2INT_DESCEND8(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 7]); \\\n\
            MMA_INST(_C[_C_off - 1], _C[_C_off + _C_stride - 1], _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 6]); \\\n\
            MMA_INST(_C[_C_off - 2], _C[_C_off + _C_stride - 2], _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 5]); \\\n\
            MMA_INST(_C[_C_off - 3], _C[_C_off + _C_stride - 3], _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 4]); \\\n\
            MMA_INST(_C[_C_off - 4], _C[_C_off + _C_stride - 4], _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 3]); \\\n\
            MMA_INST(_C[_C_off - 5], _C[_C_off + _C_stride - 5], _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 2]); \\\n\
            MMA_INST(_C[_C_off - 6], _C[_C_off + _C_stride - 6], _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 1]); \\\n\
            MMA_INST(_C[_C_off - 7], _C[_C_off + _C_stride - 7], _a0, _a1, _Bv1[_Bv1_off]); \\\n\
        }\n\
\n\
#define MMA_INST_2INT_1x1(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_2INT_ASCEND1 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0], _Av1[0 + _2INT_], _Bv1, 0); \\\n\
            \\\n\
            MMA_INST_2INT_ASCEND1 (_C, 0,  TILE_N_V2_PER_THD, _Av1[1], _Av1[1 + _2INT_], _Bv1, 1); \\\n\
        }\n\
\n\
#define MMA_INST_2INT_1x2(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_2INT_ASCEND2 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0], _Av1[0 + _2INT_], _Bv1, 0); \\\n\
            \\\n\
            MMA_INST_2INT_ASCEND2 (_C, 0,  TILE_N_V2_PER_THD, _Av1[1], _Av1[1 + _2INT_], _Bv1, 1); \\\n\
        }\n\
\n\
#define MMA_INST_2INT_1x4(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_2INT_ASCEND4 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0], _Av1[0 + _2INT_], _Bv1, 0); \\\n\
            \\\n\
            MMA_INST_2INT_ASCEND4 (_C, 0,  TILE_N_V2_PER_THD, _Av1[1], _Av1[1 + _2INT_], _Bv1, 1); \\\n\
        }\n\
\n\
#define MMA_INST_2INT_1x8(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_2INT_ASCEND8 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0], _Av1[0 + _2INT_], _Bv1, 0); \\\n\
            \\\n\
            MMA_INST_2INT_ASCEND8 (_C, 0,  TILE_N_V2_PER_THD, _Av1[1], _Av1[1 + _2INT_], _Bv1, 1); \\\n\
        }\n\
\n\
#define MMA_INST_2INT_2x1(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_2INT_ASCEND1 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],             _Av1[0 + _2INT_],             _Bv1, 0); \\\n\
            MMA_INST_2INT_DESCEND1(_C, 2,  TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_], _Av1[0 + _2INT_ + _2INT_X2_], _Bv1, 0); \\\n\
            \\\n\
            MMA_INST_2INT_ASCEND1 (_C, 0,  TILE_N_V2_PER_THD, _Av1[1],             _Av1[1 + _2INT_],             _Bv1, 1); \\\n\
            MMA_INST_2INT_DESCEND1(_C, 2,  TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_], _Av1[1 + _2INT_ + _2INT_X2_], _Bv1, 1); \\\n\
        }\n\
\n\
#define MMA_INST_2INT_2x2(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_2INT_ASCEND2 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],             _Av1[0 + _2INT_],             _Bv1, 0); \\\n\
            MMA_INST_2INT_DESCEND2(_C, 5,  TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_], _Av1[0 + _2INT_ + _2INT_X2_], _Bv1, 0); \\\n\
            \\\n\
            MMA_INST_2INT_ASCEND2 (_C, 0,  TILE_N_V2_PER_THD, _Av1[1],             _Av1[1 + _2INT_],             _Bv1, 1); \\\n\
            MMA_INST_2INT_DESCEND2(_C, 5,  TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_], _Av1[1 + _2INT_ + _2INT_X2_], _Bv1, 1); \\\n\
        }\n\
\n\
#define MMA_INST_2INT_2x4(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_2INT_ASCEND4 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],             _Av1[0 + _2INT_],             _Bv1, 0); \\\n\
            MMA_INST_2INT_DESCEND4(_C, 11, TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_], _Av1[0 + _2INT_ + _2INT_X2_], _Bv1, 0); \\\n\
            \\\n\
            MMA_INST_2INT_ASCEND4 (_C, 0,  TILE_N_V2_PER_THD, _Av1[1],             _Av1[1 + _2INT_],             _Bv1, 1); \\\n\
            MMA_INST_2INT_DESCEND4(_C, 11, TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_], _Av1[1 + _2INT_ + _2INT_X2_], _Bv1, 1); \\\n\
        }\n\
\n\
#define MMA_INST_2INT_2x8(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_2INT_ASCEND8 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],             _Av1[0 + _2INT_],             _Bv1, 0); \\\n\
            MMA_INST_2INT_DESCEND8(_C, 23, TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_], _Av1[0 + _2INT_ + _2INT_X2_], _Bv1, 0); \\\n\
            \\\n\
            MMA_INST_2INT_ASCEND8 (_C, 0,  TILE_N_V2_PER_THD, _Av1[1],             _Av1[1 + _2INT_],             _Bv1, 1); \\\n\
            MMA_INST_2INT_DESCEND8(_C, 23, TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_], _Av1[1 + _2INT_ + _2INT_X2_], _Bv1, 1); \\\n\
        }\n\
\n\
#define MMA_INST_2INT_4x1(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_2INT_ASCEND1 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],                 _Av1[0 + _2INT_],                 _Bv1, 0); \\\n\
            MMA_INST_2INT_DESCEND1(_C, 2,  TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 1], _Av1[0 + _2INT_ + _2INT_X2_ * 1], _Bv1, 0); \\\n\
            MMA_INST_2INT_ASCEND1 (_C, 4,  TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 2], _Av1[0 + _2INT_ + _2INT_X2_ * 2], _Bv1, 0); \\\n\
            MMA_INST_2INT_DESCEND1(_C, 6,  TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 3], _Av1[0 + _2INT_ + _2INT_X2_ * 3], _Bv1, 0); \\\n\
            \\\n\
            MMA_INST_2INT_ASCEND1 (_C, 0,  TILE_N_V2_PER_THD, _Av1[1],                 _Av1[1 + _2INT_],                 _Bv1, 1); \\\n\
            MMA_INST_2INT_DESCEND1(_C, 2,  TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 1], _Av1[1 + _2INT_ + _2INT_X2_ * 1], _Bv1, 1); \\\n\
            MMA_INST_2INT_ASCEND1 (_C, 4,  TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 2], _Av1[1 + _2INT_ + _2INT_X2_ * 2], _Bv1, 1); \\\n\
            MMA_INST_2INT_DESCEND1(_C, 6,  TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 3], _Av1[1 + _2INT_ + _2INT_X2_ * 3], _Bv1, 1); \\\n\
        }\n\
\n\
#define MMA_INST_2INT_4x2(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_2INT_ASCEND2 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],                 _Av1[0 + _2INT_],                 _Bv1, 0); \\\n\
            MMA_INST_2INT_DESCEND2(_C, 5,  TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 1], _Av1[0 + _2INT_ + _2INT_X2_ * 1], _Bv1, 0); \\\n\
            MMA_INST_2INT_ASCEND2 (_C, 8,  TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 2], _Av1[0 + _2INT_ + _2INT_X2_ * 2], _Bv1, 0); \\\n\
            MMA_INST_2INT_DESCEND2(_C, 13, TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 3], _Av1[0 + _2INT_ + _2INT_X2_ * 3], _Bv1, 0); \\\n\
            \\\n\
            MMA_INST_2INT_ASCEND2 (_C, 0,  TILE_N_V2_PER_THD, _Av1[1],                 _Av1[1 + _2INT_],                 _Bv1, 1); \\\n\
            MMA_INST_2INT_DESCEND2(_C, 5,  TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 1], _Av1[1 + _2INT_ + _2INT_X2_ * 1], _Bv1, 1); \\\n\
            MMA_INST_2INT_ASCEND2 (_C, 8,  TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 2], _Av1[1 + _2INT_ + _2INT_X2_ * 2], _Bv1, 1); \\\n\
            MMA_INST_2INT_DESCEND2(_C, 13, TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 3], _Av1[1 + _2INT_ + _2INT_X2_ * 3], _Bv1, 1); \\\n\
        }\n\
\n\
#define MMA_INST_2INT_4x4(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_2INT_ASCEND4 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],                 _Av1[0 + _2INT_],                 _Bv1, 0); \\\n\
            MMA_INST_2INT_DESCEND4(_C, 11, TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 1], _Av1[0 + _2INT_ + _2INT_X2_ * 1], _Bv1, 0); \\\n\
            MMA_INST_2INT_ASCEND4 (_C, 16, TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 2], _Av1[0 + _2INT_ + _2INT_X2_ * 2], _Bv1, 0); \\\n\
            MMA_INST_2INT_DESCEND4(_C, 27, TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 3], _Av1[0 + _2INT_ + _2INT_X2_ * 3], _Bv1, 0); \\\n\
            \\\n\
            MMA_INST_2INT_ASCEND4 (_C, 0,  TILE_N_V2_PER_THD, _Av1[1],                 _Av1[1 + _2INT_],                 _Bv1, 1); \\\n\
            MMA_INST_2INT_DESCEND4(_C, 11, TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 1], _Av1[1 + _2INT_ + _2INT_X2_ * 1], _Bv1, 1); \\\n\
            MMA_INST_2INT_ASCEND4 (_C, 16, TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 2], _Av1[1 + _2INT_ + _2INT_X2_ * 2], _Bv1, 1); \\\n\
            MMA_INST_2INT_DESCEND4(_C, 27, TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 3], _Av1[1 + _2INT_ + _2INT_X2_ * 3], _Bv1, 1); \\\n\
        }\n\
\n\
#define MMA_INST_2INT_4x8(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_2INT_ASCEND8 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],                 _Av1[0 + _2INT_],                 _Bv1, 0); \\\n\
            MMA_INST_2INT_DESCEND8(_C, 23, TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 1], _Av1[0 + _2INT_ + _2INT_X2_ * 1], _Bv1, 0); \\\n\
            MMA_INST_2INT_ASCEND8 (_C, 32, TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 2], _Av1[0 + _2INT_ + _2INT_X2_ * 2], _Bv1, 0); \\\n\
            MMA_INST_2INT_DESCEND8(_C, 55, TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 3], _Av1[0 + _2INT_ + _2INT_X2_ * 3], _Bv1, 0); \\\n\
            \\\n\
            MMA_INST_2INT_ASCEND8 (_C, 0,  TILE_N_V2_PER_THD, _Av1[1],                 _Av1[1 + _2INT_],                 _Bv1, 1); \\\n\
            MMA_INST_2INT_DESCEND8(_C, 23, TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 1], _Av1[1 + _2INT_ + _2INT_X2_ * 1], _Bv1, 1); \\\n\
            MMA_INST_2INT_ASCEND8 (_C, 32, TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 2], _Av1[1 + _2INT_ + _2INT_X2_ * 2], _Bv1, 1); \\\n\
            MMA_INST_2INT_DESCEND8(_C, 55, TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 3], _Av1[1 + _2INT_ + _2INT_X2_ * 3], _Bv1, 1); \\\n\
        }\n\
\n\
#define MMA_INST_2INT_8x1(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_2INT_ASCEND1 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],                 _Av1[0 + _2INT_],                 _Bv1, 0); \\\n\
            MMA_INST_2INT_DESCEND1(_C, 2,  TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 1], _Av1[0 + _2INT_ + _2INT_X2_ * 1], _Bv1, 0); \\\n\
            MMA_INST_2INT_ASCEND1 (_C, 4,  TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 2], _Av1[0 + _2INT_ + _2INT_X2_ * 2], _Bv1, 0); \\\n\
            MMA_INST_2INT_DESCEND1(_C, 6,  TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 3], _Av1[0 + _2INT_ + _2INT_X2_ * 3], _Bv1, 0); \\\n\
            MMA_INST_2INT_ASCEND1 (_C, 8,  TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 4], _Av1[0 + _2INT_ + _2INT_X2_ * 4], _Bv1, 0); \\\n\
            MMA_INST_2INT_DESCEND1(_C, 10, TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 5], _Av1[0 + _2INT_ + _2INT_X2_ * 5], _Bv1, 0); \\\n\
            MMA_INST_2INT_ASCEND1 (_C, 12, TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 6], _Av1[0 + _2INT_ + _2INT_X2_ * 6], _Bv1, 0); \\\n\
            MMA_INST_2INT_DESCEND1(_C, 14, TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 7], _Av1[0 + _2INT_ + _2INT_X2_ * 7], _Bv1, 0); \\\n\
            \\\n\
            MMA_INST_2INT_ASCEND1 (_C, 0,  TILE_N_V2_PER_THD, _Av1[1],                 _Av1[1 + _2INT_],                 _Bv1, 1); \\\n\
            MMA_INST_2INT_DESCEND1(_C, 2,  TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 1], _Av1[1 + _2INT_ + _2INT_X2_ * 1], _Bv1, 1); \\\n\
            MMA_INST_2INT_ASCEND1 (_C, 4,  TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 2], _Av1[1 + _2INT_ + _2INT_X2_ * 2], _Bv1, 1); \\\n\
            MMA_INST_2INT_DESCEND1(_C, 6,  TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 3], _Av1[1 + _2INT_ + _2INT_X2_ * 3], _Bv1, 1); \\\n\
            MMA_INST_2INT_ASCEND1 (_C, 8,  TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 4], _Av1[1 + _2INT_ + _2INT_X2_ * 4], _Bv1, 1); \\\n\
            MMA_INST_2INT_DESCEND1(_C, 10, TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 5], _Av1[1 + _2INT_ + _2INT_X2_ * 5], _Bv1, 1); \\\n\
            MMA_INST_2INT_ASCEND1 (_C, 12, TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 6], _Av1[1 + _2INT_ + _2INT_X2_ * 6], _Bv1, 1); \\\n\
            MMA_INST_2INT_DESCEND1(_C, 14, TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 7], _Av1[1 + _2INT_ + _2INT_X2_ * 7], _Bv1, 1); \\\n\
        }\n\
\n\
#define MMA_INST_2INT_8x2(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_2INT_ASCEND2 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],                 _Av1[0 + _2INT_],                 _Bv1, 0); \\\n\
            MMA_INST_2INT_DESCEND2(_C, 5,  TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 1], _Av1[0 + _2INT_ + _2INT_X2_ * 1], _Bv1, 0); \\\n\
            MMA_INST_2INT_ASCEND2 (_C, 8,  TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 2], _Av1[0 + _2INT_ + _2INT_X2_ * 2], _Bv1, 0); \\\n\
            MMA_INST_2INT_DESCEND2(_C, 13, TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 3], _Av1[0 + _2INT_ + _2INT_X2_ * 3], _Bv1, 0); \\\n\
            MMA_INST_2INT_ASCEND2 (_C, 16, TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 4], _Av1[0 + _2INT_ + _2INT_X2_ * 4], _Bv1, 0); \\\n\
            MMA_INST_2INT_DESCEND2(_C, 21, TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 5], _Av1[0 + _2INT_ + _2INT_X2_ * 5], _Bv1, 0); \\\n\
            MMA_INST_2INT_ASCEND2 (_C, 24, TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 6], _Av1[0 + _2INT_ + _2INT_X2_ * 6], _Bv1, 0); \\\n\
            MMA_INST_2INT_DESCEND2(_C, 29, TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 7], _Av1[0 + _2INT_ + _2INT_X2_ * 7], _Bv1, 0); \\\n\
            \\\n\
            MMA_INST_2INT_ASCEND2 (_C, 0,  TILE_N_V2_PER_THD, _Av1[1],                 _Av1[1 + _2INT_],                 _Bv1, 1); \\\n\
            MMA_INST_2INT_DESCEND2(_C, 5,  TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 1], _Av1[1 + _2INT_ + _2INT_X2_ * 1], _Bv1, 1); \\\n\
            MMA_INST_2INT_ASCEND2 (_C, 8,  TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 2], _Av1[1 + _2INT_ + _2INT_X2_ * 2], _Bv1, 1); \\\n\
            MMA_INST_2INT_DESCEND2(_C, 13, TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 3], _Av1[1 + _2INT_ + _2INT_X2_ * 3], _Bv1, 1); \\\n\
            MMA_INST_2INT_ASCEND2 (_C, 16, TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 4], _Av1[1 + _2INT_ + _2INT_X2_ * 4], _Bv1, 1); \\\n\
            MMA_INST_2INT_DESCEND2(_C, 21, TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 5], _Av1[1 + _2INT_ + _2INT_X2_ * 5], _Bv1, 1); \\\n\
            MMA_INST_2INT_ASCEND2 (_C, 24, TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 6], _Av1[1 + _2INT_ + _2INT_X2_ * 6], _Bv1, 1); \\\n\
            MMA_INST_2INT_DESCEND2(_C, 29, TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 7], _Av1[1 + _2INT_ + _2INT_X2_ * 7], _Bv1, 1); \\\n\
        }\n\
\n\
#define MMA_INST_2INT_8x4(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_2INT_ASCEND4 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],                 _Av1[0 + _2INT_],                 _Bv1, 0); \\\n\
            MMA_INST_2INT_DESCEND4(_C, 11, TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 1], _Av1[0 + _2INT_ + _2INT_X2_ * 1], _Bv1, 0); \\\n\
            MMA_INST_2INT_ASCEND4 (_C, 16, TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 2], _Av1[0 + _2INT_ + _2INT_X2_ * 2], _Bv1, 0); \\\n\
            MMA_INST_2INT_DESCEND4(_C, 27, TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 3], _Av1[0 + _2INT_ + _2INT_X2_ * 3], _Bv1, 0); \\\n\
            MMA_INST_2INT_ASCEND4 (_C, 32, TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 4], _Av1[0 + _2INT_ + _2INT_X2_ * 4], _Bv1, 0); \\\n\
            MMA_INST_2INT_DESCEND4(_C, 43, TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 5], _Av1[0 + _2INT_ + _2INT_X2_ * 5], _Bv1, 0); \\\n\
            MMA_INST_2INT_ASCEND4 (_C, 48, TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 6], _Av1[0 + _2INT_ + _2INT_X2_ * 6], _Bv1, 0); \\\n\
            MMA_INST_2INT_DESCEND4(_C, 59, TILE_N_V2_PER_THD, _Av1[0 + _2INT_X2_ * 7], _Av1[0 + _2INT_ + _2INT_X2_ * 7], _Bv1, 0); \\\n\
            \\\n\
            MMA_INST_2INT_ASCEND4 (_C, 0,  TILE_N_V2_PER_THD, _Av1[1],                 _Av1[1 + _2INT_],                 _Bv1, 1); \\\n\
            MMA_INST_2INT_DESCEND4(_C, 11, TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 1], _Av1[1 + _2INT_ + _2INT_X2_ * 1], _Bv1, 1); \\\n\
            MMA_INST_2INT_ASCEND4 (_C, 16, TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 2], _Av1[1 + _2INT_ + _2INT_X2_ * 2], _Bv1, 1); \\\n\
            MMA_INST_2INT_DESCEND4(_C, 27, TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 3], _Av1[1 + _2INT_ + _2INT_X2_ * 3], _Bv1, 1); \\\n\
            MMA_INST_2INT_ASCEND4 (_C, 32, TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 4], _Av1[1 + _2INT_ + _2INT_X2_ * 4], _Bv1, 1); \\\n\
            MMA_INST_2INT_DESCEND4(_C, 43, TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 5], _Av1[1 + _2INT_ + _2INT_X2_ * 5], _Bv1, 1); \\\n\
            MMA_INST_2INT_ASCEND4 (_C, 48, TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 6], _Av1[1 + _2INT_ + _2INT_X2_ * 6], _Bv1, 1); \\\n\
            MMA_INST_2INT_DESCEND4(_C, 59, TILE_N_V2_PER_THD, _Av1[1 + _2INT_X2_ * 7], _Av1[1 + _2INT_ + _2INT_X2_ * 7], _Bv1, 1); \\\n\
        }\n\
\n\
");
    header_code_.emplace("/mnt/hpc/xusi/ppl.jit/src/ppl/nn/engines/cuda/impls/src/nn/conv/idxn/common/dmem_i4_macros.h", "// Licensed to the Apache Software Foundation (ASF) under one\n\
// or more contributor license agreements.  See the NOTICE file\n\
// distributed with this work for additional information\n\
// regarding copyright ownership.  The ASF licenses this file\n\
// to you under the Apache License, Version 2.0 (the\n\
// \"License\"); you may not use this file except in compliance\n\
// with the License.  You may obtain a copy of the License at\n\
//\n\
//   http://www.apache.org/licenses/LICENSE-2.0\n\
//\n\
// Unless required by applicable law or agreed to in writing,\n\
// software distributed under the License is distributed on an\n\
// \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n\
// KIND, either express or implied.  See the License for the\n\
// specific language governing permissions and limitations\n\
// under the License.\n\
\n\
/////////////////////////////////////////////////////\n\
// common load global memory macros\n\
/////////////////////////////////////////////////////\n\
\n\
////////////////////////////////////////\n\
// load dB macros\n\
////////////////////////////////////////\n\
\n\
#define LOAD_dBv4_SIZE1(_regB, _dBv4, _dBv4_off) \\\n\
        { \\\n\
            _regB[0] = (flt_n_valid[0] && (flt_hwc_v8_off < flt_hwc_v8)) ? _dBv4[ _dBv4_off[0] ] : ZEROv4;\\\n\
            \\\n\
            _dBv4_off[0] += TILE_K_V8_PER_STEP; \\\n\
            \\\n\
            flt_hwc_v8_off  += TILE_K_V8_PER_STEP; \\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE2(_regB, _dBv4, _dBv4_off) \\\n\
        { \\\n\
            _regB[0] = (flt_n_valid[0] && (flt_hwc_v8_off < flt_hwc_v8)) ? _dBv4[ _dBv4_off[0] ] : ZEROv4;\\\n\
            _regB[1] = (flt_n_valid[1] && (flt_hwc_v8_off < flt_hwc_v8)) ? _dBv4[ _dBv4_off[1] ] : ZEROv4;\\\n\
            \\\n\
            _dBv4_off[0] += TILE_K_V8_PER_STEP; \\\n\
            _dBv4_off[1] += TILE_K_V8_PER_STEP; \\\n\
            \\\n\
            flt_hwc_v8_off  += TILE_K_V8_PER_STEP; \\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE4(_regB, _dBv4, _dBv4_off) \\\n\
        { \\\n\
            _regB[0] = (flt_n_valid[0] && (flt_hwc_v8_off < flt_hwc_v8)) ? _dBv4[ _dBv4_off[0] ] : ZEROv4;\\\n\
            _regB[1] = (flt_n_valid[1] && (flt_hwc_v8_off < flt_hwc_v8)) ? _dBv4[ _dBv4_off[1] ] : ZEROv4;\\\n\
            _regB[2] = (flt_n_valid[2] && (flt_hwc_v8_off < flt_hwc_v8)) ? _dBv4[ _dBv4_off[2] ] : ZEROv4;\\\n\
            _regB[3] = (flt_n_valid[3] && (flt_hwc_v8_off < flt_hwc_v8)) ? _dBv4[ _dBv4_off[3] ] : ZEROv4;\\\n\
            \\\n\
            _dBv4_off[0] += TILE_K_V8_PER_STEP; \\\n\
            _dBv4_off[1] += TILE_K_V8_PER_STEP; \\\n\
            _dBv4_off[2] += TILE_K_V8_PER_STEP; \\\n\
            _dBv4_off[3] += TILE_K_V8_PER_STEP; \\\n\
            \\\n\
            flt_hwc_v8_off  += TILE_K_V8_PER_STEP; \\\n\
        }\n\
\n\
#define LOAD_dBv4_SIZE8(_regB, _dBv4, _dBv4_off) \\\n\
        { \\\n\
            _regB[0] = (flt_n_valid[0] && (flt_hwc_v8_off < flt_hwc_v8)) ? _dBv4[ _dBv4_off[0] ] : ZEROv4;\\\n\
            _regB[1] = (flt_n_valid[1] && (flt_hwc_v8_off < flt_hwc_v8)) ? _dBv4[ _dBv4_off[1] ] : ZEROv4;\\\n\
            _regB[2] = (flt_n_valid[2] && (flt_hwc_v8_off < flt_hwc_v8)) ? _dBv4[ _dBv4_off[2] ] : ZEROv4;\\\n\
            _regB[3] = (flt_n_valid[3] && (flt_hwc_v8_off < flt_hwc_v8)) ? _dBv4[ _dBv4_off[3] ] : ZEROv4;\\\n\
            _regB[4] = (flt_n_valid[4] && (flt_hwc_v8_off < flt_hwc_v8)) ? _dBv4[ _dBv4_off[4] ] : ZEROv4;\\\n\
            _regB[5] = (flt_n_valid[5] && (flt_hwc_v8_off < flt_hwc_v8)) ? _dBv4[ _dBv4_off[5] ] : ZEROv4;\\\n\
            _regB[6] = (flt_n_valid[6] && (flt_hwc_v8_off < flt_hwc_v8)) ? _dBv4[ _dBv4_off[6] ] : ZEROv4;\\\n\
            _regB[7] = (flt_n_valid[7] && (flt_hwc_v8_off < flt_hwc_v8)) ? _dBv4[ _dBv4_off[7] ] : ZEROv4;\\\n\
            \\\n\
            _dBv4_off[0] += TILE_K_V8_PER_STEP; \\\n\
            _dBv4_off[1] += TILE_K_V8_PER_STEP; \\\n\
            _dBv4_off[2] += TILE_K_V8_PER_STEP; \\\n\
            _dBv4_off[3] += TILE_K_V8_PER_STEP; \\\n\
            _dBv4_off[4] += TILE_K_V8_PER_STEP; \\\n\
            _dBv4_off[5] += TILE_K_V8_PER_STEP; \\\n\
            _dBv4_off[6] += TILE_K_V8_PER_STEP; \\\n\
            _dBv4_off[7] += TILE_K_V8_PER_STEP; \\\n\
            \\\n\
            flt_hwc_v8_off  += TILE_K_V8_PER_STEP; \\\n\
        }\n\
\n\
#define SET_dBv4_BOUND(_step_id, _dBv4_off, _flt_n_valid) \\\n\
        { \\\n\
            int _flt_n_id  =  cta_idx * TILE_N_PER_CTA  + \\\n\
                             _step_id * TILE_N_PER_STEP + \\\n\
                             warp_idx * TILE_N_PER_MMA  + \\\n\
                             tid_y; \\\n\
            \\\n\
            _flt_n_valid  =  _flt_n_id < num_flt_per_grp_pad; \\\n\
            \\\n\
            _dBv4_off  =   grp_id   * flt_hwc_v8 * num_flt_per_grp_pad + \\\n\
                          _flt_n_id * flt_hwc_v8 + \\\n\
                           tid_x; \\\n\
        }\n\
\n\
////////////////////////////////////////\n\
// load dA macros\n\
////////////////////////////////////////\n\
\n\
#define SET_IN_Mv1_ID(_tid, _sm_base_v4) \\\n\
        { \\\n\
            int _out_nhw_id =  cta_idy * TILE_M_PER_CTA + _tid; \\\n\
            \\\n\
            int _out_w_id   = (_out_nhw_id % out_width); \\\n\
            int _out_h_id   = (_out_nhw_id / out_width) % out_height; \\\n\
            \\\n\
            int4 _in_id; \\\n\
            \\\n\
            _in_id.y = _out_w_id * stride_width  - pad_width; \\\n\
            _in_id.z = _out_h_id * stride_height - pad_height; \\\n\
            _in_id.w = _out_nhw_id / out_hw; \\\n\
            \\\n\
            _in_id.x = (_in_id.w * in_hw + _in_id.z * in_width + _in_id.y) * in_chl_per_grp_pad_v8 * num_grp + \\\n\
                         grp_id  * in_chl_per_grp_pad_v8; \\\n\
            \\\n\
            _sm_base_v4[_tid] = _in_id; \\\n\
        }\n\
\n\
#define SET_IN_Kv8_OFF(_tid, _sm_base_v4) \\\n\
        { \\\n\
            int _inNHWC8_id =  _tid; \\\n\
            \\\n\
            int4 _in_off; \\\n\
            \\\n\
            _in_off.y = ((_inNHWC8_id /  in_chl_per_grp_pad_v8) % flt_width)  * hole_width; \\\n\
            _in_off.z = ((_inNHWC8_id / (in_chl_per_grp_pad_v8  * flt_width)) % flt_height) * hole_height; \\\n\
            _in_off.w =   _inNHWC8_id / (in_chl_per_grp_pad_v8  * flt_width   * flt_height); \\\n\
            \\\n\
            _in_off.x = (_in_off.w  * in_hw + _in_off.z * in_width + _in_off.y) * in_chl_per_grp_pad_v8 * num_grp + \\\n\
                        (_inNHWC8_id %  in_chl_per_grp_pad_v8); \\\n\
            \\\n\
            _sm_base_v4[SM_IN_ID_SIZE + _tid] = _in_off; \\\n\
         }\n\
\n\
#define LOAD_dAv4_SIZE2(_regA, _dAv4, _in_id, _in_off) \\\n\
        { \\\n\
            int4 _in; \\\n\
            \\\n\
            _in.x =  _in_id[0].x + _in_off.x; \\\n\
            _in.y =  _in_id[0].y + _in_off.y; \\\n\
            _in.z =  _in_id[0].z + _in_off.z; \\\n\
            _in.w =  _in_id[0].w + _in_off.w; \\\n\
            _regA[0] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv4[_in.x] : ZEROv4;\\\n\
            \\\n\
            _in.x =  _in_id[1].x + _in_off.x; \\\n\
            _in.y =  _in_id[1].y + _in_off.y; \\\n\
            _in.z =  _in_id[1].z + _in_off.z; \\\n\
            _in.w =  _in_id[1].w + _in_off.w; \\\n\
            _regA[1] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv4[_in.x] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE4(_regA, _dAv4, _in_id, _in_off) \\\n\
        { \\\n\
            int4 _in; \\\n\
            \\\n\
            _in.x =  _in_id[0].x + _in_off.x; \\\n\
            _in.y =  _in_id[0].y + _in_off.y; \\\n\
            _in.z =  _in_id[0].z + _in_off.z; \\\n\
            _in.w =  _in_id[0].w + _in_off.w; \\\n\
            _regA[0] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv4[_in.x] : ZEROv4;\\\n\
            \\\n\
            _in.x =  _in_id[1].x + _in_off.x; \\\n\
            _in.y =  _in_id[1].y + _in_off.y; \\\n\
            _in.z =  _in_id[1].z + _in_off.z; \\\n\
            _in.w =  _in_id[1].w + _in_off.w; \\\n\
            _regA[1] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv4[_in.x] : ZEROv4;\\\n\
            \\\n\
            _in.x =  _in_id[2].x + _in_off.x; \\\n\
            _in.y =  _in_id[2].y + _in_off.y; \\\n\
            _in.z =  _in_id[2].z + _in_off.z; \\\n\
            _in.w =  _in_id[2].w + _in_off.w; \\\n\
            _regA[2] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv4[_in.x] : ZEROv4;\\\n\
            \\\n\
            _in.x =  _in_id[3].x + _in_off.x; \\\n\
            _in.y =  _in_id[3].y + _in_off.y; \\\n\
            _in.z =  _in_id[3].z + _in_off.z; \\\n\
            _in.w =  _in_id[3].w + _in_off.w; \\\n\
            _regA[3] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv4[_in.x] : ZEROv4;\\\n\
        }\n\
\n\
#define LOAD_dAv4_SIZE8(_regA, _dAv4, _in_id, _in_off) \\\n\
        { \\\n\
            int4 _in; \\\n\
            \\\n\
            _in.x =  _in_id[0].x + _in_off.x; \\\n\
            _in.y =  _in_id[0].y + _in_off.y; \\\n\
            _in.z =  _in_id[0].z + _in_off.z; \\\n\
            _in.w =  _in_id[0].w + _in_off.w; \\\n\
            _regA[0] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv4[_in.x] : ZEROv4;\\\n\
            \\\n\
            _in.x =  _in_id[1].x + _in_off.x; \\\n\
            _in.y =  _in_id[1].y + _in_off.y; \\\n\
            _in.z =  _in_id[1].z + _in_off.z; \\\n\
            _in.w =  _in_id[1].w + _in_off.w; \\\n\
            _regA[1] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv4[_in.x] : ZEROv4;\\\n\
            \\\n\
            _in.x =  _in_id[2].x + _in_off.x; \\\n\
            _in.y =  _in_id[2].y + _in_off.y; \\\n\
            _in.z =  _in_id[2].z + _in_off.z; \\\n\
            _in.w =  _in_id[2].w + _in_off.w; \\\n\
            _regA[2] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv4[_in.x] : ZEROv4;\\\n\
            \\\n\
            _in.x =  _in_id[3].x + _in_off.x; \\\n\
            _in.y =  _in_id[3].y + _in_off.y; \\\n\
            _in.z =  _in_id[3].z + _in_off.z; \\\n\
            _in.w =  _in_id[3].w + _in_off.w; \\\n\
            _regA[3] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv4[_in.x] : ZEROv4;\\\n\
            \\\n\
            _in.x =  _in_id[4].x + _in_off.x; \\\n\
            _in.y =  _in_id[4].y + _in_off.y; \\\n\
            _in.z =  _in_id[4].z + _in_off.z; \\\n\
            _in.w =  _in_id[4].w + _in_off.w; \\\n\
            _regA[4] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv4[_in.x] : ZEROv4;\\\n\
            \\\n\
            _in.x =  _in_id[5].x + _in_off.x; \\\n\
            _in.y =  _in_id[5].y + _in_off.y; \\\n\
            _in.z =  _in_id[5].z + _in_off.z; \\\n\
            _in.w =  _in_id[5].w + _in_off.w; \\\n\
            _regA[5] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv4[_in.x] : ZEROv4;\\\n\
            \\\n\
            _in.x =  _in_id[6].x + _in_off.x; \\\n\
            _in.y =  _in_id[6].y + _in_off.y; \\\n\
            _in.z =  _in_id[6].z + _in_off.z; \\\n\
            _in.w =  _in_id[6].w + _in_off.w; \\\n\
            _regA[6] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv4[_in.x] : ZEROv4;\\\n\
            \\\n\
            _in.x =  _in_id[7].x + _in_off.x; \\\n\
            _in.y =  _in_id[7].y + _in_off.y; \\\n\
            _in.z =  _in_id[7].z + _in_off.z; \\\n\
            _in.w =  _in_id[7].w + _in_off.w; \\\n\
            _regA[7] = (BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z)) ? _dAv4[_in.x] : ZEROv4;\\\n\
        }\n\
");
    header_code_.emplace("/mnt/hpc/xusi/ppl.jit/src/ppl/nn/engines/cuda/impls/src/nn/conv/idxn/common/hmma_i4_macros.h", "// Licensed to the Apache Software Foundation (ASF) under one\n\
// or more contributor license agreements.  See the NOTICE file\n\
// distributed with this work for additional information\n\
// regarding copyright ownership.  The ASF licenses this file\n\
// to you under the Apache License, Version 2.0 (the\n\
// \"License\"); you may not use this file except in compliance\n\
// with the License.  You may obtain a copy of the License at\n\
//\n\
//   http://www.apache.org/licenses/LICENSE-2.0\n\
//\n\
// Unless required by applicable law or agreed to in writing,\n\
// software distributed under the License is distributed on an\n\
// \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n\
// KIND, either express or implied.  See the License for the\n\
// specific language governing permissions and limitations\n\
// under the License.\n\
\n\
////////////////////////////////////////\n\
// hmma macros\n\
////////////////////////////////////////\n\
\n\
#define MMA_INST_OPCODE \\\n\
        \"mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%5,%6};\\n\"\n\
        \n\
#define MMA_INST(_d0, _d1, _a0, _a1, _b) \\\n\
        asm volatile(MMA_INST_OPCODE:   \"=r\"(_d0),   \"=r\"(_d1): \"r\"(_a0), \"r\"(_a1), \"r\"(_b),  \"r\"(_d0),   \"r\"(_d1));\n\
\n\
\n\
#define MMA_INST_4INT_ASCEND1(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _Bv1[_Bv1_off]); \\\n\
        }\n\
        \n\
#define MMA_INST_4INT_ASCEND2(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _Bv1[_Bv1_off]); \\\n\
            MMA_INST(_C[_C_off + 1], _C[_C_off + _C_stride + 1], _a0, _a1, _Bv1[_Bv1_off + _4INT_ * 1]); \\\n\
        }\n\
        \n\
#define MMA_INST_4INT_ASCEND4(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _Bv1[_Bv1_off]); \\\n\
            MMA_INST(_C[_C_off + 1], _C[_C_off + _C_stride + 1], _a0, _a1, _Bv1[_Bv1_off + _4INT_ * 1]); \\\n\
            MMA_INST(_C[_C_off + 2], _C[_C_off + _C_stride + 2], _a0, _a1, _Bv1[_Bv1_off + _4INT_ * 2]); \\\n\
            MMA_INST(_C[_C_off + 3], _C[_C_off + _C_stride + 3], _a0, _a1, _Bv1[_Bv1_off + _4INT_ * 3]); \\\n\
        }\n\
        \n\
#define MMA_INST_4INT_ASCEND8(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _Bv1[_Bv1_off]); \\\n\
            MMA_INST(_C[_C_off + 1], _C[_C_off + _C_stride + 1], _a0, _a1, _Bv1[_Bv1_off + _4INT_ * 1]); \\\n\
            MMA_INST(_C[_C_off + 2], _C[_C_off + _C_stride + 2], _a0, _a1, _Bv1[_Bv1_off + _4INT_ * 2]); \\\n\
            MMA_INST(_C[_C_off + 3], _C[_C_off + _C_stride + 3], _a0, _a1, _Bv1[_Bv1_off + _4INT_ * 3]); \\\n\
            MMA_INST(_C[_C_off + 4], _C[_C_off + _C_stride + 4], _a0, _a1, _Bv1[_Bv1_off + _4INT_ * 4]); \\\n\
            MMA_INST(_C[_C_off + 5], _C[_C_off + _C_stride + 5], _a0, _a1, _Bv1[_Bv1_off + _4INT_ * 5]); \\\n\
            MMA_INST(_C[_C_off + 6], _C[_C_off + _C_stride + 6], _a0, _a1, _Bv1[_Bv1_off + _4INT_ * 6]); \\\n\
            MMA_INST(_C[_C_off + 7], _C[_C_off + _C_stride + 7], _a0, _a1, _Bv1[_Bv1_off + _4INT_ * 7]); \\\n\
        }\n\
        \n\
#define MMA_INST_4INT_DESCEND1(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _Bv1[_Bv1_off]); \\\n\
        }\n\
\n\
#define MMA_INST_4INT_DESCEND2(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _Bv1[_Bv1_off + _4INT_ * 1]); \\\n\
            MMA_INST(_C[_C_off - 1], _C[_C_off + _C_stride - 1], _a0, _a1, _Bv1[_Bv1_off]); \\\n\
        }\n\
\n\
#define MMA_INST_4INT_DESCEND4(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _Bv1[_Bv1_off + _4INT_ * 3]); \\\n\
            MMA_INST(_C[_C_off - 1], _C[_C_off + _C_stride - 1], _a0, _a1, _Bv1[_Bv1_off + _4INT_ * 2]); \\\n\
            MMA_INST(_C[_C_off - 2], _C[_C_off + _C_stride - 2], _a0, _a1, _Bv1[_Bv1_off + _4INT_ * 1]); \\\n\
            MMA_INST(_C[_C_off - 3], _C[_C_off + _C_stride - 3], _a0, _a1, _Bv1[_Bv1_off]); \\\n\
        }\n\
\n\
#define MMA_INST_4INT_DESCEND8(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \\\n\
        { \\\n\
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _a0, _a1, _Bv1[_Bv1_off + _4INT_ * 7]); \\\n\
            MMA_INST(_C[_C_off - 1], _C[_C_off + _C_stride - 1], _a0, _a1, _Bv1[_Bv1_off + _4INT_ * 6]); \\\n\
            MMA_INST(_C[_C_off - 2], _C[_C_off + _C_stride - 2], _a0, _a1, _Bv1[_Bv1_off + _4INT_ * 5]); \\\n\
            MMA_INST(_C[_C_off - 3], _C[_C_off + _C_stride - 3], _a0, _a1, _Bv1[_Bv1_off + _4INT_ * 4]); \\\n\
            MMA_INST(_C[_C_off - 4], _C[_C_off + _C_stride - 4], _a0, _a1, _Bv1[_Bv1_off + _4INT_ * 3]); \\\n\
            MMA_INST(_C[_C_off - 5], _C[_C_off + _C_stride - 5], _a0, _a1, _Bv1[_Bv1_off + _4INT_ * 2]); \\\n\
            MMA_INST(_C[_C_off - 6], _C[_C_off + _C_stride - 6], _a0, _a1, _Bv1[_Bv1_off + _4INT_ * 1]); \\\n\
            MMA_INST(_C[_C_off - 7], _C[_C_off + _C_stride - 7], _a0, _a1, _Bv1[_Bv1_off]); \\\n\
        }\n\
\n\
#define MMA_INST_4INT_1x1(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0], _Av1[0 + _4INT_], _Bv1, 0); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 0,  TILE_N_V2_PER_THD, _Av1[1], _Av1[1 + _4INT_], _Bv1, 1); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 0,  TILE_N_V2_PER_THD, _Av1[2], _Av1[2 + _4INT_], _Bv1, 2); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 0,  TILE_N_V2_PER_THD, _Av1[3], _Av1[3 + _4INT_], _Bv1, 3); \\\n\
        }\n\
\n\
#define MMA_INST_4INT_1x2(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0], _Av1[0 + _4INT_], _Bv1, 0); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 0,  TILE_N_V2_PER_THD, _Av1[1], _Av1[1 + _4INT_], _Bv1, 1); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 0,  TILE_N_V2_PER_THD, _Av1[2], _Av1[2 + _4INT_], _Bv1, 2); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 0,  TILE_N_V2_PER_THD, _Av1[3], _Av1[3 + _4INT_], _Bv1, 3); \\\n\
        }\n\
\n\
#define MMA_INST_4INT_1x4(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0], _Av1[0 + _4INT_], _Bv1, 0); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 0,  TILE_N_V2_PER_THD, _Av1[1], _Av1[1 + _4INT_], _Bv1, 1); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 0,  TILE_N_V2_PER_THD, _Av1[2], _Av1[2 + _4INT_], _Bv1, 2); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 0,  TILE_N_V2_PER_THD, _Av1[3], _Av1[3 + _4INT_], _Bv1, 3); \\\n\
        }\n\
\n\
#define MMA_INST_4INT_1x8(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_4INT_ASCEND8 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0], _Av1[0 + _4INT_], _Bv1, 0); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND8 (_C, 0,  TILE_N_V2_PER_THD, _Av1[1], _Av1[1 + _4INT_], _Bv1, 1); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND8 (_C, 0,  TILE_N_V2_PER_THD, _Av1[2], _Av1[2 + _4INT_], _Bv1, 2); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND8 (_C, 0,  TILE_N_V2_PER_THD, _Av1[3], _Av1[3 + _4INT_], _Bv1, 3); \\\n\
        }\n\
\n\
#define MMA_INST_4INT_2x1(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],             _Av1[0 + _4INT_],             _Bv1, 0); \\\n\
            MMA_INST_4INT_DESCEND1(_C, 2,  TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_], _Av1[0 + _4INT_ + _4INT_X2_], _Bv1, 0); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 0,  TILE_N_V2_PER_THD, _Av1[1],             _Av1[1 + _4INT_],             _Bv1, 1); \\\n\
            MMA_INST_4INT_DESCEND1(_C, 2,  TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_], _Av1[1 + _4INT_ + _4INT_X2_], _Bv1, 1); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 0,  TILE_N_V2_PER_THD, _Av1[2],             _Av1[2 + _4INT_],             _Bv1, 2); \\\n\
            MMA_INST_4INT_DESCEND1(_C, 2,  TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_], _Av1[2 + _4INT_ + _4INT_X2_], _Bv1, 2); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 0,  TILE_N_V2_PER_THD, _Av1[3],             _Av1[3 + _4INT_],             _Bv1, 3); \\\n\
            MMA_INST_4INT_DESCEND1(_C, 2,  TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_], _Av1[3 + _4INT_ + _4INT_X2_], _Bv1, 3); \\\n\
        }\n\
\n\
#define MMA_INST_4INT_2x2(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],             _Av1[0 + _4INT_],             _Bv1, 0); \\\n\
            MMA_INST_4INT_DESCEND2(_C, 5,  TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_], _Av1[0 + _4INT_ + _4INT_X2_], _Bv1, 0); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 0,  TILE_N_V2_PER_THD, _Av1[1],             _Av1[1 + _4INT_],             _Bv1, 1); \\\n\
            MMA_INST_4INT_DESCEND2(_C, 5,  TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_], _Av1[1 + _4INT_ + _4INT_X2_], _Bv1, 1); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 0,  TILE_N_V2_PER_THD, _Av1[2],             _Av1[2 + _4INT_],             _Bv1, 2); \\\n\
            MMA_INST_4INT_DESCEND2(_C, 5,  TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_], _Av1[2 + _4INT_ + _4INT_X2_], _Bv1, 2); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 0,  TILE_N_V2_PER_THD, _Av1[3],             _Av1[3 + _4INT_],             _Bv1, 3); \\\n\
            MMA_INST_4INT_DESCEND2(_C, 5,  TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_], _Av1[3 + _4INT_ + _4INT_X2_], _Bv1, 3); \\\n\
        }\n\
\n\
#define MMA_INST_4INT_2x4(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],             _Av1[0 + _4INT_],             _Bv1, 0); \\\n\
            MMA_INST_4INT_DESCEND4(_C, 11, TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_], _Av1[0 + _4INT_ + _4INT_X2_], _Bv1, 0); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 0,  TILE_N_V2_PER_THD, _Av1[1],             _Av1[1 + _4INT_],             _Bv1, 1); \\\n\
            MMA_INST_4INT_DESCEND4(_C, 11, TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_], _Av1[1 + _4INT_ + _4INT_X2_], _Bv1, 1); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 0,  TILE_N_V2_PER_THD, _Av1[2],             _Av1[2 + _4INT_],             _Bv1, 2); \\\n\
            MMA_INST_4INT_DESCEND4(_C, 11, TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_], _Av1[2 + _4INT_ + _4INT_X2_], _Bv1, 2); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 0,  TILE_N_V2_PER_THD, _Av1[3],             _Av1[3 + _4INT_],             _Bv1, 3); \\\n\
            MMA_INST_4INT_DESCEND4(_C, 11, TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_], _Av1[3 + _4INT_ + _4INT_X2_], _Bv1, 3); \\\n\
        }\n\
\n\
#define MMA_INST_4INT_2x8(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_4INT_ASCEND8 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],             _Av1[0 + _4INT_],             _Bv1, 0); \\\n\
            MMA_INST_4INT_DESCEND8(_C, 23, TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_], _Av1[0 + _4INT_ + _4INT_X2_], _Bv1, 0); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND8 (_C, 0,  TILE_N_V2_PER_THD, _Av1[1],             _Av1[1 + _4INT_],             _Bv1, 1); \\\n\
            MMA_INST_4INT_DESCEND8(_C, 23, TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_], _Av1[1 + _4INT_ + _4INT_X2_], _Bv1, 1); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND8 (_C, 0,  TILE_N_V2_PER_THD, _Av1[2],             _Av1[2 + _4INT_],             _Bv1, 2); \\\n\
            MMA_INST_4INT_DESCEND8(_C, 23, TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_], _Av1[2 + _4INT_ + _4INT_X2_], _Bv1, 2); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND8 (_C, 0,  TILE_N_V2_PER_THD, _Av1[3],             _Av1[3 + _4INT_],             _Bv1, 3); \\\n\
            MMA_INST_4INT_DESCEND8(_C, 23, TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_], _Av1[3 + _4INT_ + _4INT_X2_], _Bv1, 3); \\\n\
        }\n\
\n\
#define MMA_INST_4INT_4x1(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],                 _Av1[0 + _4INT_],                 _Bv1, 0); \\\n\
            MMA_INST_4INT_DESCEND1(_C, 2,  TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 1], _Av1[0 + _4INT_ + _4INT_X2_ * 1], _Bv1, 0); \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 4,  TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 2], _Av1[0 + _4INT_ + _4INT_X2_ * 2], _Bv1, 0); \\\n\
            MMA_INST_4INT_DESCEND1(_C, 6,  TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 3], _Av1[0 + _4INT_ + _4INT_X2_ * 3], _Bv1, 0); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 0,  TILE_N_V2_PER_THD, _Av1[1],                 _Av1[1 + _4INT_],                 _Bv1, 1); \\\n\
            MMA_INST_4INT_DESCEND1(_C, 2,  TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 1], _Av1[1 + _4INT_ + _4INT_X2_ * 1], _Bv1, 1); \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 4,  TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 2], _Av1[1 + _4INT_ + _4INT_X2_ * 2], _Bv1, 1); \\\n\
            MMA_INST_4INT_DESCEND1(_C, 6,  TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 3], _Av1[1 + _4INT_ + _4INT_X2_ * 3], _Bv1, 1); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 0,  TILE_N_V2_PER_THD, _Av1[2],                 _Av1[2 + _4INT_],                 _Bv1, 2); \\\n\
            MMA_INST_4INT_DESCEND1(_C, 2,  TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 1], _Av1[2 + _4INT_ + _4INT_X2_ * 1], _Bv1, 2); \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 4,  TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 2], _Av1[2 + _4INT_ + _4INT_X2_ * 2], _Bv1, 2); \\\n\
            MMA_INST_4INT_DESCEND1(_C, 6,  TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 3], _Av1[2 + _4INT_ + _4INT_X2_ * 3], _Bv1, 2); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 0,  TILE_N_V2_PER_THD, _Av1[3],                 _Av1[3 + _4INT_],                 _Bv1, 3); \\\n\
            MMA_INST_4INT_DESCEND1(_C, 2,  TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 1], _Av1[3 + _4INT_ + _4INT_X2_ * 1], _Bv1, 3); \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 4,  TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 2], _Av1[3 + _4INT_ + _4INT_X2_ * 2], _Bv1, 3); \\\n\
            MMA_INST_4INT_DESCEND1(_C, 6,  TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 3], _Av1[3 + _4INT_ + _4INT_X2_ * 3], _Bv1, 3); \\\n\
        }\n\
\n\
#define MMA_INST_4INT_4x2(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],                 _Av1[0 + _4INT_],                 _Bv1, 0); \\\n\
            MMA_INST_4INT_DESCEND2(_C, 5,  TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 1], _Av1[0 + _4INT_ + _4INT_X2_ * 1], _Bv1, 0); \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 8,  TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 2], _Av1[0 + _4INT_ + _4INT_X2_ * 2], _Bv1, 0); \\\n\
            MMA_INST_4INT_DESCEND2(_C, 13, TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 3], _Av1[0 + _4INT_ + _4INT_X2_ * 3], _Bv1, 0); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 0,  TILE_N_V2_PER_THD, _Av1[1],                 _Av1[1 + _4INT_],                 _Bv1, 1); \\\n\
            MMA_INST_4INT_DESCEND2(_C, 5,  TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 1], _Av1[1 + _4INT_ + _4INT_X2_ * 1], _Bv1, 1); \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 8,  TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 2], _Av1[1 + _4INT_ + _4INT_X2_ * 2], _Bv1, 1); \\\n\
            MMA_INST_4INT_DESCEND2(_C, 13, TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 3], _Av1[1 + _4INT_ + _4INT_X2_ * 3], _Bv1, 1); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 0,  TILE_N_V2_PER_THD, _Av1[2],                 _Av1[2 + _4INT_],                 _Bv1, 2); \\\n\
            MMA_INST_4INT_DESCEND2(_C, 5,  TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 1], _Av1[2 + _4INT_ + _4INT_X2_ * 1], _Bv1, 2); \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 8,  TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 2], _Av1[2 + _4INT_ + _4INT_X2_ * 2], _Bv1, 2); \\\n\
            MMA_INST_4INT_DESCEND2(_C, 13, TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 3], _Av1[2 + _4INT_ + _4INT_X2_ * 3], _Bv1, 2); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 0,  TILE_N_V2_PER_THD, _Av1[3],                 _Av1[3 + _4INT_],                 _Bv1, 3); \\\n\
            MMA_INST_4INT_DESCEND2(_C, 5,  TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 1], _Av1[3 + _4INT_ + _4INT_X2_ * 1], _Bv1, 3); \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 8,  TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 2], _Av1[3 + _4INT_ + _4INT_X2_ * 2], _Bv1, 3); \\\n\
            MMA_INST_4INT_DESCEND2(_C, 13, TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 3], _Av1[3 + _4INT_ + _4INT_X2_ * 3], _Bv1, 3); \\\n\
        }\n\
\n\
#define MMA_INST_4INT_4x4(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],                 _Av1[0 + _4INT_],                 _Bv1, 0); \\\n\
            MMA_INST_4INT_DESCEND4(_C, 11, TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 1], _Av1[0 + _4INT_ + _4INT_X2_ * 1], _Bv1, 0); \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 16, TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 2], _Av1[0 + _4INT_ + _4INT_X2_ * 2], _Bv1, 0); \\\n\
            MMA_INST_4INT_DESCEND4(_C, 27, TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 3], _Av1[0 + _4INT_ + _4INT_X2_ * 3], _Bv1, 0); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 0,  TILE_N_V2_PER_THD, _Av1[1],                 _Av1[1 + _4INT_],                 _Bv1, 1); \\\n\
            MMA_INST_4INT_DESCEND4(_C, 11, TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 1], _Av1[1 + _4INT_ + _4INT_X2_ * 1], _Bv1, 1); \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 16, TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 2], _Av1[1 + _4INT_ + _4INT_X2_ * 2], _Bv1, 1); \\\n\
            MMA_INST_4INT_DESCEND4(_C, 27, TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 3], _Av1[1 + _4INT_ + _4INT_X2_ * 3], _Bv1, 1); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 0,  TILE_N_V2_PER_THD, _Av1[2],                 _Av1[2 + _4INT_],                 _Bv1, 2); \\\n\
            MMA_INST_4INT_DESCEND4(_C, 11, TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 1], _Av1[2 + _4INT_ + _4INT_X2_ * 1], _Bv1, 2); \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 16, TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 2], _Av1[2 + _4INT_ + _4INT_X2_ * 2], _Bv1, 2); \\\n\
            MMA_INST_4INT_DESCEND4(_C, 27, TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 3], _Av1[2 + _4INT_ + _4INT_X2_ * 3], _Bv1, 2); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 0,  TILE_N_V2_PER_THD, _Av1[3],                 _Av1[3 + _4INT_],                 _Bv1, 3); \\\n\
            MMA_INST_4INT_DESCEND4(_C, 11, TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 1], _Av1[3 + _4INT_ + _4INT_X2_ * 1], _Bv1, 3); \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 16, TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 2], _Av1[3 + _4INT_ + _4INT_X2_ * 2], _Bv1, 3); \\\n\
            MMA_INST_4INT_DESCEND4(_C, 27, TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 3], _Av1[3 + _4INT_ + _4INT_X2_ * 3], _Bv1, 3); \\\n\
        }\n\
\n\
#define MMA_INST_4INT_4x8(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_4INT_ASCEND8 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],                 _Av1[0 + _4INT_],                 _Bv1, 0); \\\n\
            MMA_INST_4INT_DESCEND8(_C, 23, TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 1], _Av1[0 + _4INT_ + _4INT_X2_ * 1], _Bv1, 0); \\\n\
            MMA_INST_4INT_ASCEND8 (_C, 32, TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 2], _Av1[0 + _4INT_ + _4INT_X2_ * 2], _Bv1, 0); \\\n\
            MMA_INST_4INT_DESCEND8(_C, 55, TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 3], _Av1[0 + _4INT_ + _4INT_X2_ * 3], _Bv1, 0); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND8 (_C, 0,  TILE_N_V2_PER_THD, _Av1[1],                 _Av1[1 + _4INT_],                 _Bv1, 1); \\\n\
            MMA_INST_4INT_DESCEND8(_C, 23, TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 1], _Av1[1 + _4INT_ + _4INT_X2_ * 1], _Bv1, 1); \\\n\
            MMA_INST_4INT_ASCEND8 (_C, 32, TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 2], _Av1[1 + _4INT_ + _4INT_X2_ * 2], _Bv1, 1); \\\n\
            MMA_INST_4INT_DESCEND8(_C, 55, TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 3], _Av1[1 + _4INT_ + _4INT_X2_ * 3], _Bv1, 1); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND8 (_C, 0,  TILE_N_V2_PER_THD, _Av1[2],                 _Av1[2 + _4INT_],                 _Bv1, 2); \\\n\
            MMA_INST_4INT_DESCEND8(_C, 23, TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 1], _Av1[2 + _4INT_ + _4INT_X2_ * 1], _Bv1, 2); \\\n\
            MMA_INST_4INT_ASCEND8 (_C, 32, TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 2], _Av1[2 + _4INT_ + _4INT_X2_ * 2], _Bv1, 2); \\\n\
            MMA_INST_4INT_DESCEND8(_C, 55, TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 3], _Av1[2 + _4INT_ + _4INT_X2_ * 3], _Bv1, 2); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND8 (_C, 0,  TILE_N_V2_PER_THD, _Av1[3],                 _Av1[3 + _4INT_],                 _Bv1, 3); \\\n\
            MMA_INST_4INT_DESCEND8(_C, 23, TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 1], _Av1[3 + _4INT_ + _4INT_X2_ * 1], _Bv1, 3); \\\n\
            MMA_INST_4INT_ASCEND8 (_C, 32, TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 2], _Av1[3 + _4INT_ + _4INT_X2_ * 2], _Bv1, 3); \\\n\
            MMA_INST_4INT_DESCEND8(_C, 55, TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 3], _Av1[3 + _4INT_ + _4INT_X2_ * 3], _Bv1, 3); \\\n\
        }\n\
\n\
#define MMA_INST_4INT_8x1(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],                 _Av1[0 + _4INT_],                 _Bv1, 0); \\\n\
            MMA_INST_4INT_DESCEND1(_C, 2,  TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 1], _Av1[0 + _4INT_ + _4INT_X2_ * 1], _Bv1, 0); \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 4,  TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 2], _Av1[0 + _4INT_ + _4INT_X2_ * 2], _Bv1, 0); \\\n\
            MMA_INST_4INT_DESCEND1(_C, 6,  TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 3], _Av1[0 + _4INT_ + _4INT_X2_ * 3], _Bv1, 0); \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 8,  TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 4], _Av1[0 + _4INT_ + _4INT_X2_ * 4], _Bv1, 0); \\\n\
            MMA_INST_4INT_DESCEND1(_C, 10, TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 5], _Av1[0 + _4INT_ + _4INT_X2_ * 5], _Bv1, 0); \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 12, TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 6], _Av1[0 + _4INT_ + _4INT_X2_ * 6], _Bv1, 0); \\\n\
            MMA_INST_4INT_DESCEND1(_C, 14, TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 7], _Av1[0 + _4INT_ + _4INT_X2_ * 7], _Bv1, 0); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 0,  TILE_N_V2_PER_THD, _Av1[1],                 _Av1[1 + _4INT_],                 _Bv1, 1); \\\n\
            MMA_INST_4INT_DESCEND1(_C, 2,  TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 1], _Av1[1 + _4INT_ + _4INT_X2_ * 1], _Bv1, 1); \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 4,  TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 2], _Av1[1 + _4INT_ + _4INT_X2_ * 2], _Bv1, 1); \\\n\
            MMA_INST_4INT_DESCEND1(_C, 6,  TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 3], _Av1[1 + _4INT_ + _4INT_X2_ * 3], _Bv1, 1); \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 8,  TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 4], _Av1[1 + _4INT_ + _4INT_X2_ * 4], _Bv1, 1); \\\n\
            MMA_INST_4INT_DESCEND1(_C, 10, TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 5], _Av1[1 + _4INT_ + _4INT_X2_ * 5], _Bv1, 1); \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 12, TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 6], _Av1[1 + _4INT_ + _4INT_X2_ * 6], _Bv1, 1); \\\n\
            MMA_INST_4INT_DESCEND1(_C, 14, TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 7], _Av1[1 + _4INT_ + _4INT_X2_ * 7], _Bv1, 1); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 0,  TILE_N_V2_PER_THD, _Av1[2],                 _Av1[2 + _4INT_],                 _Bv1, 2); \\\n\
            MMA_INST_4INT_DESCEND1(_C, 2,  TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 1], _Av1[2 + _4INT_ + _4INT_X2_ * 1], _Bv1, 2); \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 4,  TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 2], _Av1[2 + _4INT_ + _4INT_X2_ * 2], _Bv1, 2); \\\n\
            MMA_INST_4INT_DESCEND1(_C, 6,  TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 3], _Av1[2 + _4INT_ + _4INT_X2_ * 3], _Bv1, 2); \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 8,  TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 4], _Av1[2 + _4INT_ + _4INT_X2_ * 4], _Bv1, 2); \\\n\
            MMA_INST_4INT_DESCEND1(_C, 10, TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 5], _Av1[2 + _4INT_ + _4INT_X2_ * 5], _Bv1, 2); \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 12, TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 6], _Av1[2 + _4INT_ + _4INT_X2_ * 6], _Bv1, 2); \\\n\
            MMA_INST_4INT_DESCEND1(_C, 14, TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 7], _Av1[2 + _4INT_ + _4INT_X2_ * 7], _Bv1, 2); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 0,  TILE_N_V2_PER_THD, _Av1[3],                 _Av1[3 + _4INT_],                 _Bv1, 3); \\\n\
            MMA_INST_4INT_DESCEND1(_C, 2,  TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 1], _Av1[3 + _4INT_ + _4INT_X2_ * 1], _Bv1, 3); \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 4,  TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 2], _Av1[3 + _4INT_ + _4INT_X2_ * 2], _Bv1, 3); \\\n\
            MMA_INST_4INT_DESCEND1(_C, 6,  TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 3], _Av1[3 + _4INT_ + _4INT_X2_ * 3], _Bv1, 3); \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 8,  TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 4], _Av1[3 + _4INT_ + _4INT_X2_ * 4], _Bv1, 3); \\\n\
            MMA_INST_4INT_DESCEND1(_C, 10, TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 5], _Av1[3 + _4INT_ + _4INT_X2_ * 5], _Bv1, 3); \\\n\
            MMA_INST_4INT_ASCEND1 (_C, 12, TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 6], _Av1[3 + _4INT_ + _4INT_X2_ * 6], _Bv1, 3); \\\n\
            MMA_INST_4INT_DESCEND1(_C, 14, TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 7], _Av1[3 + _4INT_ + _4INT_X2_ * 7], _Bv1, 3); \\\n\
        }\n\
\n\
#define MMA_INST_4INT_8x2(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],                 _Av1[0 + _4INT_],                 _Bv1, 0); \\\n\
            MMA_INST_4INT_DESCEND2(_C, 5,  TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 1], _Av1[0 + _4INT_ + _4INT_X2_ * 1], _Bv1, 0); \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 8,  TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 2], _Av1[0 + _4INT_ + _4INT_X2_ * 2], _Bv1, 0); \\\n\
            MMA_INST_4INT_DESCEND2(_C, 13, TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 3], _Av1[0 + _4INT_ + _4INT_X2_ * 3], _Bv1, 0); \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 16, TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 4], _Av1[0 + _4INT_ + _4INT_X2_ * 4], _Bv1, 0); \\\n\
            MMA_INST_4INT_DESCEND2(_C, 21, TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 5], _Av1[0 + _4INT_ + _4INT_X2_ * 5], _Bv1, 0); \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 24, TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 6], _Av1[0 + _4INT_ + _4INT_X2_ * 6], _Bv1, 0); \\\n\
            MMA_INST_4INT_DESCEND2(_C, 29, TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 7], _Av1[0 + _4INT_ + _4INT_X2_ * 7], _Bv1, 0); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 0,  TILE_N_V2_PER_THD, _Av1[1],                 _Av1[1 + _4INT_],                 _Bv1, 1); \\\n\
            MMA_INST_4INT_DESCEND2(_C, 5,  TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 1], _Av1[1 + _4INT_ + _4INT_X2_ * 1], _Bv1, 1); \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 8,  TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 2], _Av1[1 + _4INT_ + _4INT_X2_ * 2], _Bv1, 1); \\\n\
            MMA_INST_4INT_DESCEND2(_C, 13, TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 3], _Av1[1 + _4INT_ + _4INT_X2_ * 3], _Bv1, 1); \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 16, TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 4], _Av1[1 + _4INT_ + _4INT_X2_ * 4], _Bv1, 1); \\\n\
            MMA_INST_4INT_DESCEND2(_C, 21, TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 5], _Av1[1 + _4INT_ + _4INT_X2_ * 5], _Bv1, 1); \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 24, TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 6], _Av1[1 + _4INT_ + _4INT_X2_ * 6], _Bv1, 1); \\\n\
            MMA_INST_4INT_DESCEND2(_C, 29, TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 7], _Av1[1 + _4INT_ + _4INT_X2_ * 7], _Bv1, 1); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 0,  TILE_N_V2_PER_THD, _Av1[2],                 _Av1[2 + _4INT_],                 _Bv1, 2); \\\n\
            MMA_INST_4INT_DESCEND2(_C, 5,  TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 1], _Av1[2 + _4INT_ + _4INT_X2_ * 1], _Bv1, 2); \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 8,  TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 2], _Av1[2 + _4INT_ + _4INT_X2_ * 2], _Bv1, 2); \\\n\
            MMA_INST_4INT_DESCEND2(_C, 13, TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 3], _Av1[2 + _4INT_ + _4INT_X2_ * 3], _Bv1, 2); \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 16, TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 4], _Av1[2 + _4INT_ + _4INT_X2_ * 4], _Bv1, 2); \\\n\
            MMA_INST_4INT_DESCEND2(_C, 21, TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 5], _Av1[2 + _4INT_ + _4INT_X2_ * 5], _Bv1, 2); \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 24, TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 6], _Av1[2 + _4INT_ + _4INT_X2_ * 6], _Bv1, 2); \\\n\
            MMA_INST_4INT_DESCEND2(_C, 29, TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 7], _Av1[2 + _4INT_ + _4INT_X2_ * 7], _Bv1, 2); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 0,  TILE_N_V2_PER_THD, _Av1[3],                 _Av1[3 + _4INT_],                 _Bv1, 3); \\\n\
            MMA_INST_4INT_DESCEND2(_C, 5,  TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 1], _Av1[3 + _4INT_ + _4INT_X2_ * 1], _Bv1, 3); \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 8,  TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 2], _Av1[3 + _4INT_ + _4INT_X2_ * 2], _Bv1, 3); \\\n\
            MMA_INST_4INT_DESCEND2(_C, 13, TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 3], _Av1[3 + _4INT_ + _4INT_X2_ * 3], _Bv1, 3); \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 16, TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 4], _Av1[3 + _4INT_ + _4INT_X2_ * 4], _Bv1, 3); \\\n\
            MMA_INST_4INT_DESCEND2(_C, 21, TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 5], _Av1[3 + _4INT_ + _4INT_X2_ * 5], _Bv1, 3); \\\n\
            MMA_INST_4INT_ASCEND2 (_C, 24, TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 6], _Av1[3 + _4INT_ + _4INT_X2_ * 6], _Bv1, 3); \\\n\
            MMA_INST_4INT_DESCEND2(_C, 29, TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 7], _Av1[3 + _4INT_ + _4INT_X2_ * 7], _Bv1, 3); \\\n\
        }\n\
\n\
#define MMA_INST_4INT_8x4(_C, _Av1, _Bv1) \\\n\
        { \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 0,  TILE_N_V2_PER_THD, _Av1[0],                 _Av1[0 + _4INT_],                 _Bv1, 0); \\\n\
            MMA_INST_4INT_DESCEND4(_C, 11, TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 1], _Av1[0 + _4INT_ + _4INT_X2_ * 1], _Bv1, 0); \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 16, TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 2], _Av1[0 + _4INT_ + _4INT_X2_ * 2], _Bv1, 0); \\\n\
            MMA_INST_4INT_DESCEND4(_C, 27, TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 3], _Av1[0 + _4INT_ + _4INT_X2_ * 3], _Bv1, 0); \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 32, TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 4], _Av1[0 + _4INT_ + _4INT_X2_ * 4], _Bv1, 0); \\\n\
            MMA_INST_4INT_DESCEND4(_C, 43, TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 5], _Av1[0 + _4INT_ + _4INT_X2_ * 5], _Bv1, 0); \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 48, TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 6], _Av1[0 + _4INT_ + _4INT_X2_ * 6], _Bv1, 0); \\\n\
            MMA_INST_4INT_DESCEND4(_C, 59, TILE_N_V2_PER_THD, _Av1[0 + _4INT_X2_ * 7], _Av1[0 + _4INT_ + _4INT_X2_ * 7], _Bv1, 0); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 0,  TILE_N_V2_PER_THD, _Av1[1],                 _Av1[1 + _4INT_],                 _Bv1, 1); \\\n\
            MMA_INST_4INT_DESCEND4(_C, 11, TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 1], _Av1[1 + _4INT_ + _4INT_X2_ * 1], _Bv1, 1); \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 16, TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 2], _Av1[1 + _4INT_ + _4INT_X2_ * 2], _Bv1, 1); \\\n\
            MMA_INST_4INT_DESCEND4(_C, 27, TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 3], _Av1[1 + _4INT_ + _4INT_X2_ * 3], _Bv1, 1); \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 32, TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 4], _Av1[1 + _4INT_ + _4INT_X2_ * 4], _Bv1, 1); \\\n\
            MMA_INST_4INT_DESCEND4(_C, 43, TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 5], _Av1[1 + _4INT_ + _4INT_X2_ * 5], _Bv1, 1); \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 48, TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 6], _Av1[1 + _4INT_ + _4INT_X2_ * 6], _Bv1, 1); \\\n\
            MMA_INST_4INT_DESCEND4(_C, 59, TILE_N_V2_PER_THD, _Av1[1 + _4INT_X2_ * 7], _Av1[1 + _4INT_ + _4INT_X2_ * 7], _Bv1, 1); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 0,  TILE_N_V2_PER_THD, _Av1[2],                 _Av1[2 + _4INT_],                 _Bv1, 2); \\\n\
            MMA_INST_4INT_DESCEND4(_C, 11, TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 1], _Av1[2 + _4INT_ + _4INT_X2_ * 1], _Bv1, 2); \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 16, TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 2], _Av1[2 + _4INT_ + _4INT_X2_ * 2], _Bv1, 2); \\\n\
            MMA_INST_4INT_DESCEND4(_C, 27, TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 3], _Av1[2 + _4INT_ + _4INT_X2_ * 3], _Bv1, 2); \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 32, TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 4], _Av1[2 + _4INT_ + _4INT_X2_ * 4], _Bv1, 2); \\\n\
            MMA_INST_4INT_DESCEND4(_C, 43, TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 5], _Av1[2 + _4INT_ + _4INT_X2_ * 5], _Bv1, 2); \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 48, TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 6], _Av1[2 + _4INT_ + _4INT_X2_ * 6], _Bv1, 2); \\\n\
            MMA_INST_4INT_DESCEND4(_C, 59, TILE_N_V2_PER_THD, _Av1[2 + _4INT_X2_ * 7], _Av1[2 + _4INT_ + _4INT_X2_ * 7], _Bv1, 2); \\\n\
            \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 0,  TILE_N_V2_PER_THD, _Av1[3],                 _Av1[3 + _4INT_],                 _Bv1, 3); \\\n\
            MMA_INST_4INT_DESCEND4(_C, 11, TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 1], _Av1[3 + _4INT_ + _4INT_X2_ * 1], _Bv1, 3); \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 16, TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 2], _Av1[3 + _4INT_ + _4INT_X2_ * 2], _Bv1, 3); \\\n\
            MMA_INST_4INT_DESCEND4(_C, 27, TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 3], _Av1[3 + _4INT_ + _4INT_X2_ * 3], _Bv1, 3); \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 32, TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 4], _Av1[3 + _4INT_ + _4INT_X2_ * 4], _Bv1, 3); \\\n\
            MMA_INST_4INT_DESCEND4(_C, 43, TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 5], _Av1[3 + _4INT_ + _4INT_X2_ * 5], _Bv1, 3); \\\n\
            MMA_INST_4INT_ASCEND4 (_C, 48, TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 6], _Av1[3 + _4INT_ + _4INT_X2_ * 6], _Bv1, 3); \\\n\
            MMA_INST_4INT_DESCEND4(_C, 59, TILE_N_V2_PER_THD, _Av1[3 + _4INT_X2_ * 7], _Av1[3 + _4INT_ + _4INT_X2_ * 7], _Bv1, 3); \\\n\
        }\n\
\n\
");
    header_code_.emplace("/mnt/hpc/xusi/ppl.jit/src/ppl/nn/engines/cuda/impls/src/nn/conv/idxn/common/output_macros.h", "// Licensed to the Apache Software Foundation (ASF) under one\n\
// or more contributor license agreements.  See the NOTICE file\n\
// distributed with this work for additional information\n\
// regarding copyright ownership.  The ASF licenses this file\n\
// to you under the Apache License, Version 2.0 (the\n\
// \"License\"); you may not use this file except in compliance\n\
// with the License.  You may obtain a copy of the License at\n\
//\n\
//   http://www.apache.org/licenses/LICENSE-2.0\n\
//\n\
// Unless required by applicable law or agreed to in writing,\n\
// software distributed under the License is distributed on an\n\
// \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n\
// KIND, either express or implied.  See the License for the\n\
// specific language governing permissions and limitations\n\
// under the License.\n\
\n\
//////////////////////////////////////////////////////\n\
// half output interface\n\
//////////////////////////////////////////////////////\n\
\n\
#if defined(ENABLE_FUSE)\n\
\n\
#define OUTPUT_2x1_BY_INT1() \\\n\
        { \\\n\
            if(dCv1_y_valid[0] && dCv1_x_valid[0]) dCv1[dCv1_idx[0] + concat_v1_off0] = C[Cv1_off + 0]; \\\n\
            \\\n\
            if(dCv1_y_valid[1] && dCv1_x_valid[0]) dCv1[dCv1_idx[0] + concat_v1_off1] = C[Cv1_off + 1]; \\\n\
        }\n\
\n\
#define OUTPUT_2x2_BY_INT1() \\\n\
        { \\\n\
            if(dCv1_y_valid[0] && dCv1_x_valid[0]) dCv1[dCv1_idx[0] + concat_v1_off0] = C[Cv1_off + 0]; \\\n\
            if(dCv1_y_valid[0] && dCv1_x_valid[1]) dCv1[dCv1_idx[1] + concat_v1_off0] = C[Cv1_off + 1]; \\\n\
            \\\n\
            if(dCv1_y_valid[1] && dCv1_x_valid[0]) dCv1[dCv1_idx[0] + concat_v1_off1] = C[Cv1_off + 2]; \\\n\
            if(dCv1_y_valid[1] && dCv1_x_valid[1]) dCv1[dCv1_idx[1] + concat_v1_off1] = C[Cv1_off + 3]; \\\n\
        }\n\
\n\
#define OUTPUT_2x4_BY_INT1() \\\n\
        { \\\n\
            if(dCv1_y_valid[0] && dCv1_x_valid[0]) dCv1[dCv1_idx[0] + concat_v1_off0] = C[Cv1_off + 0]; \\\n\
            if(dCv1_y_valid[0] && dCv1_x_valid[1]) dCv1[dCv1_idx[1] + concat_v1_off0] = C[Cv1_off + 1]; \\\n\
            if(dCv1_y_valid[0] && dCv1_x_valid[2]) dCv1[dCv1_idx[2] + concat_v1_off0] = C[Cv1_off + 2]; \\\n\
            if(dCv1_y_valid[0] && dCv1_x_valid[3]) dCv1[dCv1_idx[3] + concat_v1_off0] = C[Cv1_off + 3]; \\\n\
            \\\n\
            if(dCv1_y_valid[1] && dCv1_x_valid[0]) dCv1[dCv1_idx[0] + concat_v1_off1] = C[Cv1_off + 4]; \\\n\
            if(dCv1_y_valid[1] && dCv1_x_valid[1]) dCv1[dCv1_idx[1] + concat_v1_off1] = C[Cv1_off + 5]; \\\n\
            if(dCv1_y_valid[1] && dCv1_x_valid[2]) dCv1[dCv1_idx[2] + concat_v1_off1] = C[Cv1_off + 6]; \\\n\
            if(dCv1_y_valid[1] && dCv1_x_valid[3]) dCv1[dCv1_idx[3] + concat_v1_off1] = C[Cv1_off + 7]; \\\n\
        }\n\
\n\
#else\n\
\n\
#define OUTPUT_2x1_BY_INT1() \\\n\
        { \\\n\
            if(dCv1_y_valid[0] && dCv1_x_valid[0]) dCv1[dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]] = C[Cv1_off + 0]; \\\n\
            \\\n\
            if(dCv1_y_valid[1] && dCv1_x_valid[0]) dCv1[dCv1_idy[1] * num_flt_v2 + dCv1_idx[0]] = C[Cv1_off + 1]; \\\n\
        }\n\
\n\
#define OUTPUT_2x2_BY_INT1() \\\n\
        { \\\n\
            if(dCv1_y_valid[0] && dCv1_x_valid[0]) dCv1[dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]] = C[Cv1_off + 0]; \\\n\
            if(dCv1_y_valid[0] && dCv1_x_valid[1]) dCv1[dCv1_idy[0] * num_flt_v2 + dCv1_idx[1]] = C[Cv1_off + 1]; \\\n\
            \\\n\
            if(dCv1_y_valid[1] && dCv1_x_valid[0]) dCv1[dCv1_idy[1] * num_flt_v2 + dCv1_idx[0]] = C[Cv1_off + 2]; \\\n\
            if(dCv1_y_valid[1] && dCv1_x_valid[1]) dCv1[dCv1_idy[1] * num_flt_v2 + dCv1_idx[1]] = C[Cv1_off + 3]; \\\n\
        }\n\
\n\
#define OUTPUT_2x4_BY_INT1() \\\n\
        { \\\n\
            if(dCv1_y_valid[0] && dCv1_x_valid[0]) dCv1[dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]] = C[Cv1_off + 0]; \\\n\
            if(dCv1_y_valid[0] && dCv1_x_valid[1]) dCv1[dCv1_idy[0] * num_flt_v2 + dCv1_idx[1]] = C[Cv1_off + 1]; \\\n\
            if(dCv1_y_valid[0] && dCv1_x_valid[2]) dCv1[dCv1_idy[0] * num_flt_v2 + dCv1_idx[2]] = C[Cv1_off + 2]; \\\n\
            if(dCv1_y_valid[0] && dCv1_x_valid[3]) dCv1[dCv1_idy[0] * num_flt_v2 + dCv1_idx[3]] = C[Cv1_off + 3]; \\\n\
            \\\n\
            if(dCv1_y_valid[1] && dCv1_x_valid[0]) dCv1[dCv1_idy[1] * num_flt_v2 + dCv1_idx[0]] = C[Cv1_off + 4]; \\\n\
            if(dCv1_y_valid[1] && dCv1_x_valid[1]) dCv1[dCv1_idy[1] * num_flt_v2 + dCv1_idx[1]] = C[Cv1_off + 5]; \\\n\
            if(dCv1_y_valid[1] && dCv1_x_valid[2]) dCv1[dCv1_idy[1] * num_flt_v2 + dCv1_idx[2]] = C[Cv1_off + 6]; \\\n\
            if(dCv1_y_valid[1] && dCv1_x_valid[3]) dCv1[dCv1_idy[1] * num_flt_v2 + dCv1_idx[3]] = C[Cv1_off + 7]; \\\n\
        }\n\
\n\
#endif\n\
\n\
//////////////////////////////////////////////////////\n\
// bias macros\n\
//////////////////////////////////////////////////////\n\
\n\
#define ADD_BIAS_2x1_V1(_has_bias, _bias, _step) \\\n\
        { \\\n\
            if(_has_bias) \\\n\
            { \\\n\
                if(dCv1_y_valid[0] && dCv1_x_valid[0]) h2C[Cv1_off + 0] = __hadd2(h2C[Cv1_off + 0], ((__half2 *) _bias) [dCv1_idx[0]]); \\\n\
                \\\n\
                if(dCv1_y_valid[1] && dCv1_x_valid[0]) h2C[Cv1_off + 1] = __hadd2(h2C[Cv1_off + 1], ((__half2 *) _bias) [dCv1_idx[0]]); \\\n\
            } \\\n\
        }\n\
\n\
#define ADD_BIAS_2x2_V1(_has_bias, _bias, _step) \\\n\
        { \\\n\
            if(_has_bias) \\\n\
            { \\\n\
                if(dCv1_y_valid[0] && dCv1_x_valid[0]) h2C[Cv1_off + 0] = __hadd2(h2C[Cv1_off + 0], ((__half2 *) _bias) [dCv1_idx[0]]); \\\n\
                if(dCv1_y_valid[0] && dCv1_x_valid[1]) h2C[Cv1_off + 1] = __hadd2(h2C[Cv1_off + 1], ((__half2 *) _bias) [dCv1_idx[1]]); \\\n\
                \\\n\
                if(dCv1_y_valid[1] && dCv1_x_valid[0]) h2C[Cv1_off + 2] = __hadd2(h2C[Cv1_off + 2], ((__half2 *) _bias) [dCv1_idx[0]]); \\\n\
                if(dCv1_y_valid[1] && dCv1_x_valid[1]) h2C[Cv1_off + 3] = __hadd2(h2C[Cv1_off + 3], ((__half2 *) _bias) [dCv1_idx[1]]); \\\n\
            } \\\n\
        }\n\
\n\
#define ADD_BIAS_2x4_V1(_has_bias, _bias, _step) \\\n\
        { \\\n\
            if(_has_bias) \\\n\
            { \\\n\
                if(dCv1_y_valid[0] && dCv1_x_valid[0]) h2C[Cv1_off + 0] = __hadd2(h2C[Cv1_off + 0], ((__half2 *) _bias) [dCv1_idx[0]]); \\\n\
                if(dCv1_y_valid[0] && dCv1_x_valid[1]) h2C[Cv1_off + 1] = __hadd2(h2C[Cv1_off + 1], ((__half2 *) _bias) [dCv1_idx[1]]); \\\n\
                if(dCv1_y_valid[0] && dCv1_x_valid[2]) h2C[Cv1_off + 2] = __hadd2(h2C[Cv1_off + 2], ((__half2 *) _bias) [dCv1_idx[2]]); \\\n\
                if(dCv1_y_valid[0] && dCv1_x_valid[3]) h2C[Cv1_off + 3] = __hadd2(h2C[Cv1_off + 3], ((__half2 *) _bias) [dCv1_idx[3]]); \\\n\
                \\\n\
                if(dCv1_y_valid[1] && dCv1_x_valid[0]) h2C[Cv1_off + 4] = __hadd2(h2C[Cv1_off + 4], ((__half2 *) _bias) [dCv1_idx[0]]); \\\n\
                if(dCv1_y_valid[1] && dCv1_x_valid[1]) h2C[Cv1_off + 5] = __hadd2(h2C[Cv1_off + 5], ((__half2 *) _bias) [dCv1_idx[1]]); \\\n\
                if(dCv1_y_valid[1] && dCv1_x_valid[2]) h2C[Cv1_off + 6] = __hadd2(h2C[Cv1_off + 6], ((__half2 *) _bias) [dCv1_idx[2]]); \\\n\
                if(dCv1_y_valid[1] && dCv1_x_valid[3]) h2C[Cv1_off + 7] = __hadd2(h2C[Cv1_off + 7], ((__half2 *) _bias) [dCv1_idx[3]]); \\\n\
            } \\\n\
        }\n\
\n\
//////////////////////////////////////////////////////\n\
// relu macros\n\
//////////////////////////////////////////////////////\n\
\n\
#define FUSE_RELU_2x1_V1(_has_relu) \\\n\
        { \\\n\
	        if(_has_relu == 1) \\\n\
            { \\\n\
                if(dCv1_y_valid[0] && dCv1_x_valid[0]) C[Cv1_off + 0] = __vmaxs2(C[Cv1_off + 0], 0); \\\n\
                if(dCv1_y_valid[1] && dCv1_x_valid[0]) C[Cv1_off + 1] = __vmaxs2(C[Cv1_off + 1], 0); \\\n\
	        } \\\n\
            else if(_has_relu == 2) \\\n\
            { \\\n\
			    __half2 h2ONE((__half) 1.f, (__half) 1.f); \\\n\
                \\\n\
                if(dCv1_y_valid[0] && dCv1_x_valid[0]) h2C[Cv1_off + 0] = __h2div(h2exp(h2C[Cv1_off + 0]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 0]))); \\\n\
                if(dCv1_y_valid[1] && dCv1_x_valid[0]) h2C[Cv1_off + 1] = __h2div(h2exp(h2C[Cv1_off + 1]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 1]))); \\\n\
	        } \\\n\
        }\n\
\n\
#define FUSE_RELU_2x2_V1(_has_relu) \\\n\
        { \\\n\
	        if(_has_relu == 1) \\\n\
            { \\\n\
                if(dCv1_y_valid[0] && dCv1_x_valid[0]) C[Cv1_off + 0] = __vmaxs2(C[Cv1_off + 0], 0); \\\n\
                if(dCv1_y_valid[0] && dCv1_x_valid[1]) C[Cv1_off + 1] = __vmaxs2(C[Cv1_off + 1], 0); \\\n\
                \\\n\
                if(dCv1_y_valid[1] && dCv1_x_valid[0]) C[Cv1_off + 2] = __vmaxs2(C[Cv1_off + 2], 0); \\\n\
                if(dCv1_y_valid[1] && dCv1_x_valid[1]) C[Cv1_off + 3] = __vmaxs2(C[Cv1_off + 3], 0); \\\n\
	        } \\\n\
            else if(_has_relu == 2) \\\n\
            { \\\n\
			    __half2 h2ONE((__half) 1.f, (__half) 1.f); \\\n\
                \\\n\
                if(dCv1_y_valid[0] && dCv1_x_valid[0]) h2C[Cv1_off + 0] = __h2div(h2exp(h2C[Cv1_off + 0]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 0]))); \\\n\
                if(dCv1_y_valid[0] && dCv1_x_valid[1]) h2C[Cv1_off + 1] = __h2div(h2exp(h2C[Cv1_off + 1]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 1]))); \\\n\
                \\\n\
                if(dCv1_y_valid[1] && dCv1_x_valid[0]) h2C[Cv1_off + 2] = __h2div(h2exp(h2C[Cv1_off + 2]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 2]))); \\\n\
                if(dCv1_y_valid[1] && dCv1_x_valid[1]) h2C[Cv1_off + 3] = __h2div(h2exp(h2C[Cv1_off + 3]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 3]))); \\\n\
	        } \\\n\
        }\n\
\n\
#define FUSE_RELU_2x4_V1(_has_relu) \\\n\
        { \\\n\
	        if(_has_relu == 1) \\\n\
            { \\\n\
                if(dCv1_y_valid[0] && dCv1_x_valid[0]) C[Cv1_off + 0] = __vmaxs2(C[Cv1_off + 0], 0); \\\n\
                if(dCv1_y_valid[0] && dCv1_x_valid[1]) C[Cv1_off + 1] = __vmaxs2(C[Cv1_off + 1], 0); \\\n\
                if(dCv1_y_valid[0] && dCv1_x_valid[2]) C[Cv1_off + 2] = __vmaxs2(C[Cv1_off + 2], 0); \\\n\
                if(dCv1_y_valid[0] && dCv1_x_valid[3]) C[Cv1_off + 3] = __vmaxs2(C[Cv1_off + 3], 0); \\\n\
                \\\n\
                if(dCv1_y_valid[1] && dCv1_x_valid[0]) C[Cv1_off + 4] = __vmaxs2(C[Cv1_off + 4], 0); \\\n\
                if(dCv1_y_valid[1] && dCv1_x_valid[1]) C[Cv1_off + 5] = __vmaxs2(C[Cv1_off + 5], 0); \\\n\
                if(dCv1_y_valid[1] && dCv1_x_valid[2]) C[Cv1_off + 6] = __vmaxs2(C[Cv1_off + 6], 0); \\\n\
                if(dCv1_y_valid[1] && dCv1_x_valid[3]) C[Cv1_off + 7] = __vmaxs2(C[Cv1_off + 7], 0); \\\n\
	        } \\\n\
            else if(_has_relu == 2) \\\n\
            { \\\n\
			    __half2 h2ONE((__half) 1.f, (__half) 1.f); \\\n\
                \\\n\
                if(dCv1_y_valid[0] && dCv1_x_valid[0]) h2C[Cv1_off + 0] = __h2div(h2exp(h2C[Cv1_off + 0]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 0]))); \\\n\
                if(dCv1_y_valid[0] && dCv1_x_valid[1]) h2C[Cv1_off + 1] = __h2div(h2exp(h2C[Cv1_off + 1]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 1]))); \\\n\
                if(dCv1_y_valid[0] && dCv1_x_valid[2]) h2C[Cv1_off + 2] = __h2div(h2exp(h2C[Cv1_off + 2]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 2]))); \\\n\
                if(dCv1_y_valid[0] && dCv1_x_valid[3]) h2C[Cv1_off + 3] = __h2div(h2exp(h2C[Cv1_off + 3]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 3]))); \\\n\
                \\\n\
                if(dCv1_y_valid[1] && dCv1_x_valid[0]) h2C[Cv1_off + 4] = __h2div(h2exp(h2C[Cv1_off + 4]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 4]))); \\\n\
                if(dCv1_y_valid[1] && dCv1_x_valid[1]) h2C[Cv1_off + 5] = __h2div(h2exp(h2C[Cv1_off + 5]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 5]))); \\\n\
                if(dCv1_y_valid[1] && dCv1_x_valid[2]) h2C[Cv1_off + 6] = __h2div(h2exp(h2C[Cv1_off + 6]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 6]))); \\\n\
                if(dCv1_y_valid[1] && dCv1_x_valid[3]) h2C[Cv1_off + 7] = __h2div(h2exp(h2C[Cv1_off + 7]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 7]))); \\\n\
	        } \\\n\
        }\n\
\n\
//////////////////////////////////////////////////////\n\
// clip macros\n\
//////////////////////////////////////////////////////\n\
\n\
#define FUSE_CLIP_2x1_V1(_has_clip, _clip_max, _clip_min) \\\n\
        { \\\n\
	        if(_has_clip) \\\n\
            { \\\n\
	    	    if(dCv1_y_valid[0] && dCv1_x_valid[0]) h2C[Cv1_off + 0].x = __hgt(h2C[Cv1_off + 0].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 0].x; \\\n\
	    	    if(dCv1_y_valid[0] && dCv1_x_valid[0]) h2C[Cv1_off + 0].y = __hgt(h2C[Cv1_off + 0].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 0].y; \\\n\
	    	    if(dCv1_y_valid[0] && dCv1_x_valid[0]) h2C[Cv1_off + 0].x = __hlt(h2C[Cv1_off + 0].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 0].x; \\\n\
	    	    if(dCv1_y_valid[0] && dCv1_x_valid[0]) h2C[Cv1_off + 0].y = __hlt(h2C[Cv1_off + 0].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 0].y; \\\n\
                \\\n\
                \\\n\
	    	    if(dCv1_y_valid[1] && dCv1_x_valid[0]) h2C[Cv1_off + 1].x = __hgt(h2C[Cv1_off + 1].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 1].x; \\\n\
	    	    if(dCv1_y_valid[1] && dCv1_x_valid[0]) h2C[Cv1_off + 1].y = __hgt(h2C[Cv1_off + 1].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 1].y; \\\n\
	    	    if(dCv1_y_valid[1] && dCv1_x_valid[0]) h2C[Cv1_off + 1].x = __hlt(h2C[Cv1_off + 1].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 1].x; \\\n\
	    	    if(dCv1_y_valid[1] && dCv1_x_valid[0]) h2C[Cv1_off + 1].y = __hlt(h2C[Cv1_off + 1].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 1].y; \\\n\
	        } \\\n\
        }\n\
\n\
#define FUSE_CLIP_2x2_V1(_has_clip, _clip_max, _clip_min) \\\n\
        { \\\n\
	        if(_has_clip) \\\n\
            { \\\n\
	    	    if(dCv1_y_valid[0] && dCv1_x_valid[0]) h2C[Cv1_off + 0].x = __hgt(h2C[Cv1_off + 0].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 0].x; \\\n\
	    	    if(dCv1_y_valid[0] && dCv1_x_valid[0]) h2C[Cv1_off + 0].y = __hgt(h2C[Cv1_off + 0].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 0].y; \\\n\
	    	    if(dCv1_y_valid[0] && dCv1_x_valid[0]) h2C[Cv1_off + 0].x = __hlt(h2C[Cv1_off + 0].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 0].x; \\\n\
	    	    if(dCv1_y_valid[0] && dCv1_x_valid[0]) h2C[Cv1_off + 0].y = __hlt(h2C[Cv1_off + 0].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 0].y; \\\n\
                \\\n\
	    	    if(dCv1_y_valid[0] && dCv1_x_valid[1]) h2C[Cv1_off + 1].x = __hgt(h2C[Cv1_off + 1].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 1].x; \\\n\
	    	    if(dCv1_y_valid[0] && dCv1_x_valid[1]) h2C[Cv1_off + 1].y = __hgt(h2C[Cv1_off + 1].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 1].y; \\\n\
	    	    if(dCv1_y_valid[0] && dCv1_x_valid[1]) h2C[Cv1_off + 1].x = __hlt(h2C[Cv1_off + 1].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 1].x; \\\n\
	    	    if(dCv1_y_valid[0] && dCv1_x_valid[1]) h2C[Cv1_off + 1].y = __hlt(h2C[Cv1_off + 1].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 1].y; \\\n\
                \\\n\
                \\\n\
	    	    if(dCv1_y_valid[1] && dCv1_x_valid[0]) h2C[Cv1_off + 2].x = __hgt(h2C[Cv1_off + 2].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 2].x; \\\n\
	    	    if(dCv1_y_valid[1] && dCv1_x_valid[0]) h2C[Cv1_off + 2].y = __hgt(h2C[Cv1_off + 2].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 2].y; \\\n\
	    	    if(dCv1_y_valid[1] && dCv1_x_valid[0]) h2C[Cv1_off + 2].x = __hlt(h2C[Cv1_off + 2].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 2].x; \\\n\
	    	    if(dCv1_y_valid[1] && dCv1_x_valid[0]) h2C[Cv1_off + 2].y = __hlt(h2C[Cv1_off + 2].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 2].y; \\\n\
                \\\n\
	    	    if(dCv1_y_valid[1] && dCv1_x_valid[1]) h2C[Cv1_off + 3].x = __hgt(h2C[Cv1_off + 3].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 3].x; \\\n\
	    	    if(dCv1_y_valid[1] && dCv1_x_valid[1]) h2C[Cv1_off + 3].y = __hgt(h2C[Cv1_off + 3].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 3].y; \\\n\
	    	    if(dCv1_y_valid[1] && dCv1_x_valid[1]) h2C[Cv1_off + 3].x = __hlt(h2C[Cv1_off + 3].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 3].x; \\\n\
	    	    if(dCv1_y_valid[1] && dCv1_x_valid[1]) h2C[Cv1_off + 3].y = __hlt(h2C[Cv1_off + 3].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 3].y; \\\n\
	        } \\\n\
        }\n\
\n\
#define FUSE_CLIP_2x4_V1(_has_clip, _clip_max, _clip_min) \\\n\
        { \\\n\
	        if(_has_clip) \\\n\
            { \\\n\
	    	    if(dCv1_y_valid[0] && dCv1_x_valid[0]) h2C[Cv1_off + 0].x = __hgt(h2C[Cv1_off + 0].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 0].x; \\\n\
	    	    if(dCv1_y_valid[0] && dCv1_x_valid[0]) h2C[Cv1_off + 0].y = __hgt(h2C[Cv1_off + 0].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 0].y; \\\n\
	    	    if(dCv1_y_valid[0] && dCv1_x_valid[0]) h2C[Cv1_off + 0].x = __hlt(h2C[Cv1_off + 0].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 0].x; \\\n\
	    	    if(dCv1_y_valid[0] && dCv1_x_valid[0]) h2C[Cv1_off + 0].y = __hlt(h2C[Cv1_off + 0].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 0].y; \\\n\
                \\\n\
	    	    if(dCv1_y_valid[0] && dCv1_x_valid[1]) h2C[Cv1_off + 1].x = __hgt(h2C[Cv1_off + 1].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 1].x; \\\n\
	    	    if(dCv1_y_valid[0] && dCv1_x_valid[1]) h2C[Cv1_off + 1].y = __hgt(h2C[Cv1_off + 1].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 1].y; \\\n\
	    	    if(dCv1_y_valid[0] && dCv1_x_valid[1]) h2C[Cv1_off + 1].x = __hlt(h2C[Cv1_off + 1].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 1].x; \\\n\
	    	    if(dCv1_y_valid[0] && dCv1_x_valid[1]) h2C[Cv1_off + 1].y = __hlt(h2C[Cv1_off + 1].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 1].y; \\\n\
                \\\n\
	    	    if(dCv1_y_valid[0] && dCv1_x_valid[2]) h2C[Cv1_off + 2].x = __hgt(h2C[Cv1_off + 2].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 2].x; \\\n\
	    	    if(dCv1_y_valid[0] && dCv1_x_valid[2]) h2C[Cv1_off + 2].y = __hgt(h2C[Cv1_off + 2].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 2].y; \\\n\
	    	    if(dCv1_y_valid[0] && dCv1_x_valid[2]) h2C[Cv1_off + 2].x = __hlt(h2C[Cv1_off + 2].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 2].x; \\\n\
	    	    if(dCv1_y_valid[0] && dCv1_x_valid[2]) h2C[Cv1_off + 2].y = __hlt(h2C[Cv1_off + 2].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 2].y; \\\n\
                \\\n\
	    	    if(dCv1_y_valid[0] && dCv1_x_valid[3]) h2C[Cv1_off + 3].x = __hgt(h2C[Cv1_off + 3].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 3].x; \\\n\
	    	    if(dCv1_y_valid[0] && dCv1_x_valid[3]) h2C[Cv1_off + 3].y = __hgt(h2C[Cv1_off + 3].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 3].y; \\\n\
	    	    if(dCv1_y_valid[0] && dCv1_x_valid[3]) h2C[Cv1_off + 3].x = __hlt(h2C[Cv1_off + 3].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 3].x; \\\n\
	    	    if(dCv1_y_valid[0] && dCv1_x_valid[3]) h2C[Cv1_off + 3].y = __hlt(h2C[Cv1_off + 3].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 3].y; \\\n\
                \\\n\
                \\\n\
	    	    if(dCv1_y_valid[1] && dCv1_x_valid[0]) h2C[Cv1_off + 4].x = __hgt(h2C[Cv1_off + 4].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 4].x; \\\n\
	    	    if(dCv1_y_valid[1] && dCv1_x_valid[0]) h2C[Cv1_off + 4].y = __hgt(h2C[Cv1_off + 4].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 4].y; \\\n\
	    	    if(dCv1_y_valid[1] && dCv1_x_valid[0]) h2C[Cv1_off + 4].x = __hlt(h2C[Cv1_off + 4].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 4].x; \\\n\
	    	    if(dCv1_y_valid[1] && dCv1_x_valid[0]) h2C[Cv1_off + 4].y = __hlt(h2C[Cv1_off + 4].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 4].y; \\\n\
                \\\n\
	    	    if(dCv1_y_valid[1] && dCv1_x_valid[1]) h2C[Cv1_off + 5].x = __hgt(h2C[Cv1_off + 5].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 5].x; \\\n\
	    	    if(dCv1_y_valid[1] && dCv1_x_valid[1]) h2C[Cv1_off + 5].y = __hgt(h2C[Cv1_off + 5].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 5].y; \\\n\
	    	    if(dCv1_y_valid[1] && dCv1_x_valid[1]) h2C[Cv1_off + 5].x = __hlt(h2C[Cv1_off + 5].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 5].x; \\\n\
	    	    if(dCv1_y_valid[1] && dCv1_x_valid[1]) h2C[Cv1_off + 5].y = __hlt(h2C[Cv1_off + 5].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 5].y; \\\n\
                \\\n\
	    	    if(dCv1_y_valid[1] && dCv1_x_valid[2]) h2C[Cv1_off + 6].x = __hgt(h2C[Cv1_off + 6].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 6].x; \\\n\
	    	    if(dCv1_y_valid[1] && dCv1_x_valid[2]) h2C[Cv1_off + 6].y = __hgt(h2C[Cv1_off + 6].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 6].y; \\\n\
	    	    if(dCv1_y_valid[1] && dCv1_x_valid[2]) h2C[Cv1_off + 6].x = __hlt(h2C[Cv1_off + 6].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 6].x; \\\n\
	    	    if(dCv1_y_valid[1] && dCv1_x_valid[2]) h2C[Cv1_off + 6].y = __hlt(h2C[Cv1_off + 6].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 6].y; \\\n\
                \\\n\
	    	    if(dCv1_y_valid[1] && dCv1_x_valid[3]) h2C[Cv1_off + 7].x = __hgt(h2C[Cv1_off + 7].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 7].x; \\\n\
	    	    if(dCv1_y_valid[1] && dCv1_x_valid[3]) h2C[Cv1_off + 7].y = __hgt(h2C[Cv1_off + 7].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 7].y; \\\n\
	    	    if(dCv1_y_valid[1] && dCv1_x_valid[3]) h2C[Cv1_off + 7].x = __hlt(h2C[Cv1_off + 7].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 7].x; \\\n\
	    	    if(dCv1_y_valid[1] && dCv1_x_valid[3]) h2C[Cv1_off + 7].y = __hlt(h2C[Cv1_off + 7].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 7].y; \\\n\
	        } \\\n\
        }\n\
\n\
//////////////////////////////////////////////////////\n\
// prelu macros\n\
//////////////////////////////////////////////////////\n\
\n\
#define FUSE_PRELU_2x1_V1(_has_prelu, _prelu, _leaky) \\\n\
        { \\\n\
       	    if(_has_prelu == 1 && dCv1_x_valid[0]) \\\n\
            { \\\n\
                _Pragma(\"unroll\") \\\n\
	            for(int i = 0; i < _INT_TO_2HALF_; i++) \\\n\
                { \\\n\
                    if(dCv1_y_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _leaky); \\\n\
                    \\\n\
                    if(dCv1_y_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _leaky); \\\n\
	            } \\\n\
	        } \\\n\
            \\\n\
       	    if(_has_prelu == 2 && dCv1_x_valid[0]) \\\n\
            { \\\n\
	            int      _scale0_v1 = ((int  *) _prelu) [dCv1_idx[0]]; \\\n\
	            __half * _hscale0  = (__half *) &_scale0_v1; \\\n\
                \\\n\
                _Pragma(\"unroll\") \\\n\
	            for(int i = 0; i < _INT_TO_2HALF_; i++) \\\n\
                { \\\n\
                    if(dCv1_y_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _hscale0[i]); \\\n\
                    \\\n\
                    if(dCv1_y_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _hscale0[i]); \\\n\
	            } \\\n\
	        } \\\n\
            \\\n\
       	    if(_has_prelu == 3 && dCv1_x_valid[0]) \\\n\
            { \\\n\
                int      _scale0_v1 = dCv1_y_valid[0] ? ((int   *) _prelu) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]] : 0; \\\n\
                int      _scale1_v1 = dCv1_y_valid[1] ? ((int   *) _prelu) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[0]] : 0; \\\n\
                \\\n\
	            __half * _hscale0  = (__half *) &_scale0_v1; \\\n\
	            __half * _hscale1  = (__half *) &_scale1_v1; \\\n\
                \\\n\
                _Pragma(\"unroll\") \\\n\
	            for(int i = 0; i < _INT_TO_2HALF_; i++) \\\n\
                { \\\n\
                    if(dCv1_y_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _hscale0[i]); \\\n\
                    \\\n\
                    if(dCv1_y_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _hscale1[i]); \\\n\
	            } \\\n\
	        } \\\n\
        }\n\
\n\
#define FUSE_PRELU_2x2_V1(_has_prelu, _prelu, _leaky) \\\n\
        { \\\n\
       	    if(_has_prelu == 1) \\\n\
            { \\\n\
                _Pragma(\"unroll\") \\\n\
	            for(int i = 0; i < _INT_TO_2HALF_; i++) \\\n\
                { \\\n\
                    if(dCv1_y_valid[0] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _leaky); \\\n\
                    if(dCv1_y_valid[0] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _leaky); \\\n\
                    \\\n\
                    if(dCv1_y_valid[1] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], _leaky); \\\n\
                    if(dCv1_y_valid[1] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], _leaky); \\\n\
	            } \\\n\
	        } \\\n\
            \\\n\
       	    if(_has_prelu == 2) \\\n\
            { \\\n\
	            int      _scale0_v1 = dCv1_x_valid[0] ? ((int  *) _prelu) [dCv1_idx[0]] : 0; \\\n\
	            int      _scale1_v1 = dCv1_x_valid[1] ? ((int  *) _prelu) [dCv1_idx[1]] : 0; \\\n\
	            __half * _hscale0  = (__half *) &_scale0_v1; \\\n\
	            __half * _hscale1  = (__half *) &_scale1_v1; \\\n\
                \\\n\
                _Pragma(\"unroll\") \\\n\
	            for(int i = 0; i < _INT_TO_2HALF_; i++) \\\n\
                { \\\n\
                    if(dCv1_y_valid[0] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _hscale0[i]); \\\n\
                    if(dCv1_y_valid[0] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _hscale1[i]); \\\n\
                    \\\n\
                    if(dCv1_y_valid[1] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], _hscale0[i]); \\\n\
                    if(dCv1_y_valid[1] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], _hscale1[i]); \\\n\
	            } \\\n\
	        } \\\n\
            \\\n\
       	    if(_has_prelu == 3) \\\n\
            { \\\n\
                int      _scale00_v1 = (dCv1_y_valid[0] && dCv1_x_valid[0]) ? ((int   *) _prelu) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]] : 0; \\\n\
                int      _scale01_v1 = (dCv1_y_valid[0] && dCv1_x_valid[1]) ? ((int   *) _prelu) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[1]] : 0; \\\n\
                \\\n\
                int      _scale10_v1 = (dCv1_y_valid[1] && dCv1_x_valid[0]) ? ((int   *) _prelu) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[0]] : 0; \\\n\
                int      _scale11_v1 = (dCv1_y_valid[1] && dCv1_x_valid[1]) ? ((int   *) _prelu) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[1]] : 0; \\\n\
                \\\n\
	            __half * _hscale00  = (__half *) &_scale00_v1; \\\n\
	            __half * _hscale01  = (__half *) &_scale01_v1; \\\n\
                \\\n\
	            __half * _hscale10  = (__half *) &_scale10_v1; \\\n\
	            __half * _hscale11  = (__half *) &_scale11_v1; \\\n\
                \\\n\
                _Pragma(\"unroll\") \\\n\
	            for(int i = 0; i < _INT_TO_2HALF_; i++) \\\n\
                { \\\n\
                    if(dCv1_y_valid[0] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _hscale00[i]); \\\n\
                    if(dCv1_y_valid[0] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _hscale01[i]); \\\n\
                    \\\n\
                    if(dCv1_y_valid[1] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], _hscale10[i]); \\\n\
                    if(dCv1_y_valid[1] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], _hscale11[i]); \\\n\
	            } \\\n\
	        } \\\n\
        }\n\
\n\
#define FUSE_PRELU_2x4_V1(_has_prelu, _prelu, _leaky) \\\n\
        { \\\n\
       	    if(_has_prelu == 1) \\\n\
            { \\\n\
                _Pragma(\"unroll\") \\\n\
	            for(int i = 0; i < _INT_TO_2HALF_; i++) \\\n\
                { \\\n\
                    if(dCv1_y_valid[0] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _leaky); \\\n\
                    if(dCv1_y_valid[0] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _leaky); \\\n\
                    if(dCv1_y_valid[0] && dCv1_x_valid[2] && __hlt(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], _leaky); \\\n\
                    if(dCv1_y_valid[0] && dCv1_x_valid[3] && __hlt(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], _leaky); \\\n\
                    \\\n\
                    if(dCv1_y_valid[1] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i], _leaky); \\\n\
                    if(dCv1_y_valid[1] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i], _leaky); \\\n\
                    if(dCv1_y_valid[1] && dCv1_x_valid[2] && __hlt(hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i], _leaky); \\\n\
                    if(dCv1_y_valid[1] && dCv1_x_valid[3] && __hlt(hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i], _leaky); \\\n\
	            } \\\n\
	        } \\\n\
            \\\n\
       	    if(_has_prelu == 2) \\\n\
            { \\\n\
	            int      _scale0_v1 = dCv1_x_valid[0] ? ((int  *) _prelu) [dCv1_idx[0]] : 0; \\\n\
	            int      _scale1_v1 = dCv1_x_valid[1] ? ((int  *) _prelu) [dCv1_idx[1]] : 0; \\\n\
	            int      _scale2_v1 = dCv1_x_valid[2] ? ((int  *) _prelu) [dCv1_idx[2]] : 0; \\\n\
	            int      _scale3_v1 = dCv1_x_valid[3] ? ((int  *) _prelu) [dCv1_idx[3]] : 0; \\\n\
	            __half * _hscale0  = (__half *) &_scale0_v1; \\\n\
	            __half * _hscale1  = (__half *) &_scale1_v1; \\\n\
	            __half * _hscale2  = (__half *) &_scale2_v1; \\\n\
	            __half * _hscale3  = (__half *) &_scale3_v1; \\\n\
                \\\n\
                _Pragma(\"unroll\") \\\n\
	            for(int i = 0; i < _INT_TO_2HALF_; i++) \\\n\
                { \\\n\
                    if(dCv1_y_valid[0] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _hscale0[i]); \\\n\
                    if(dCv1_y_valid[0] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _hscale1[i]); \\\n\
                    if(dCv1_y_valid[0] && dCv1_x_valid[2] && __hlt(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], _hscale2[i]); \\\n\
                    if(dCv1_y_valid[0] && dCv1_x_valid[3] && __hlt(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], _hscale3[i]); \\\n\
                    \\\n\
                    if(dCv1_y_valid[1] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i], _hscale0[i]); \\\n\
                    if(dCv1_y_valid[1] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i], _hscale1[i]); \\\n\
                    if(dCv1_y_valid[1] && dCv1_x_valid[2] && __hlt(hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i], _hscale2[i]); \\\n\
                    if(dCv1_y_valid[1] && dCv1_x_valid[3] && __hlt(hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i], _hscale3[i]); \\\n\
	            } \\\n\
	        } \\\n\
            \\\n\
       	    if(_has_prelu == 3) \\\n\
            { \\\n\
                int      _scale00_v1 = (dCv1_y_valid[0] && dCv1_x_valid[0]) ? ((int   *) _prelu) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]] : 0; \\\n\
                int      _scale01_v1 = (dCv1_y_valid[0] && dCv1_x_valid[1]) ? ((int   *) _prelu) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[1]] : 0; \\\n\
                int      _scale02_v1 = (dCv1_y_valid[0] && dCv1_x_valid[2]) ? ((int   *) _prelu) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[2]] : 0; \\\n\
                int      _scale03_v1 = (dCv1_y_valid[0] && dCv1_x_valid[3]) ? ((int   *) _prelu) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[3]] : 0; \\\n\
                \\\n\
                int      _scale10_v1 = (dCv1_y_valid[1] && dCv1_x_valid[0]) ? ((int   *) _prelu) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[0]] : 0; \\\n\
                int      _scale11_v1 = (dCv1_y_valid[1] && dCv1_x_valid[1]) ? ((int   *) _prelu) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[1]] : 0; \\\n\
                int      _scale12_v1 = (dCv1_y_valid[1] && dCv1_x_valid[2]) ? ((int   *) _prelu) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[2]] : 0; \\\n\
                int      _scale13_v1 = (dCv1_y_valid[1] && dCv1_x_valid[3]) ? ((int   *) _prelu) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[3]] : 0; \\\n\
                \\\n\
	            __half * _hscale00  = (__half *) &_scale00_v1; \\\n\
	            __half * _hscale01  = (__half *) &_scale01_v1; \\\n\
	            __half * _hscale02  = (__half *) &_scale02_v1; \\\n\
	            __half * _hscale03  = (__half *) &_scale03_v1; \\\n\
                \\\n\
	            __half * _hscale10  = (__half *) &_scale10_v1; \\\n\
	            __half * _hscale11  = (__half *) &_scale11_v1; \\\n\
	            __half * _hscale12  = (__half *) &_scale12_v1; \\\n\
	            __half * _hscale13  = (__half *) &_scale13_v1; \\\n\
                \\\n\
                _Pragma(\"unroll\") \\\n\
	            for(int i = 0; i < _INT_TO_2HALF_; i++) \\\n\
                { \\\n\
                    if(dCv1_y_valid[0] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _hscale00[i]); \\\n\
                    if(dCv1_y_valid[0] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _hscale01[i]); \\\n\
                    if(dCv1_y_valid[0] && dCv1_x_valid[2] && __hlt(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], _hscale02[i]); \\\n\
                    if(dCv1_y_valid[0] && dCv1_x_valid[3] && __hlt(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], _hscale03[i]); \\\n\
                    \\\n\
                    if(dCv1_y_valid[1] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i], _hscale10[i]); \\\n\
                    if(dCv1_y_valid[1] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i], _hscale11[i]); \\\n\
                    if(dCv1_y_valid[1] && dCv1_x_valid[2] && __hlt(hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i], _hscale12[i]); \\\n\
                    if(dCv1_y_valid[1] && dCv1_x_valid[3] && __hlt(hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i], 0)) \\\n\
                        hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i], _hscale13[i]); \\\n\
	            } \\\n\
	        } \\\n\
        }\n\
\n\
//////////////////////////////////////////////////////\n\
// eltwise macros\n\
//////////////////////////////////////////////////////\n\
\n\
#define FUSE_ELT_2x1_V1(_has_elt, _pre_data) \\\n\
        { \\\n\
	        if(_has_elt) \\\n\
            { \\\n\
                if(dCv1_y_valid[0] && dCv1_x_valid[0]) h2C[Cv1_off + 0] = __hadd2(h2C[Cv1_off + 0], ((__half2 *) _pre_data) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]]); \\\n\
                \\\n\
                if(dCv1_y_valid[1] && dCv1_x_valid[0]) h2C[Cv1_off + 1] = __hadd2(h2C[Cv1_off + 1], ((__half2 *) _pre_data) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[0]]); \\\n\
	        } \\\n\
        }\n\
\n\
#define FUSE_ELT_2x2_V1(_has_elt, _pre_data) \\\n\
        { \\\n\
	        if(_has_elt) \\\n\
            { \\\n\
                if(dCv1_y_valid[0] && dCv1_x_valid[0]) h2C[Cv1_off + 0] = __hadd2(h2C[Cv1_off + 0], ((__half2 *) _pre_data) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]]); \\\n\
                if(dCv1_y_valid[0] && dCv1_x_valid[1]) h2C[Cv1_off + 1] = __hadd2(h2C[Cv1_off + 1], ((__half2 *) _pre_data) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[1]]); \\\n\
                \\\n\
                if(dCv1_y_valid[1] && dCv1_x_valid[0]) h2C[Cv1_off + 2] = __hadd2(h2C[Cv1_off + 2], ((__half2 *) _pre_data) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[0]]); \\\n\
                if(dCv1_y_valid[1] && dCv1_x_valid[1]) h2C[Cv1_off + 3] = __hadd2(h2C[Cv1_off + 3], ((__half2 *) _pre_data) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[1]]); \\\n\
	        } \\\n\
        }\n\
\n\
#define FUSE_ELT_2x4_V1(_has_elt, _pre_data) \\\n\
        { \\\n\
	        if(_has_elt) \\\n\
            { \\\n\
                if(dCv1_y_valid[0] && dCv1_x_valid[0]) h2C[Cv1_off + 0] = __hadd2(h2C[Cv1_off + 0], ((__half2 *) _pre_data) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]]); \\\n\
                if(dCv1_y_valid[0] && dCv1_x_valid[1]) h2C[Cv1_off + 1] = __hadd2(h2C[Cv1_off + 1], ((__half2 *) _pre_data) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[1]]); \\\n\
                if(dCv1_y_valid[0] && dCv1_x_valid[2]) h2C[Cv1_off + 2] = __hadd2(h2C[Cv1_off + 2], ((__half2 *) _pre_data) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[2]]); \\\n\
                if(dCv1_y_valid[0] && dCv1_x_valid[3]) h2C[Cv1_off + 3] = __hadd2(h2C[Cv1_off + 3], ((__half2 *) _pre_data) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[3]]); \\\n\
                \\\n\
                if(dCv1_y_valid[1] && dCv1_x_valid[0]) h2C[Cv1_off + 4] = __hadd2(h2C[Cv1_off + 4], ((__half2 *) _pre_data) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[0]]); \\\n\
                if(dCv1_y_valid[1] && dCv1_x_valid[1]) h2C[Cv1_off + 5] = __hadd2(h2C[Cv1_off + 5], ((__half2 *) _pre_data) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[1]]); \\\n\
                if(dCv1_y_valid[1] && dCv1_x_valid[2]) h2C[Cv1_off + 6] = __hadd2(h2C[Cv1_off + 6], ((__half2 *) _pre_data) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[2]]); \\\n\
                if(dCv1_y_valid[1] && dCv1_x_valid[3]) h2C[Cv1_off + 7] = __hadd2(h2C[Cv1_off + 7], ((__half2 *) _pre_data) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[3]]); \\\n\
	        } \\\n\
        }\n\
\n\
//////////////////////////////////////////////////////\n\
// concat macros\n\
//////////////////////////////////////////////////////\n\
\n\
#define SET_CONCAT_OFF_V1(_has_concat, _concat_v1_off0, _concat_v1_off1) \\\n\
        { \\\n\
                _concat_v1_off0 = dCv1_idy[0] * num_flt_v2; \\\n\
                _concat_v1_off1 = dCv1_idy[1] * num_flt_v2; \\\n\
	        if(_has_concat) \\\n\
            { \\\n\
                if(dCv1_y_valid[0]) _concat_v1_off0 = concat_offset_v8 * _INT4_TO_4HALF2_ + dCv1_idy[0] * concat_stride_v8 * _INT4_TO_4HALF2_; \\\n\
                if(dCv1_y_valid[1]) _concat_v1_off1 = concat_offset_v8 * _INT4_TO_4HALF2_ + dCv1_idy[1] * concat_stride_v8 * _INT4_TO_4HALF2_; \\\n\
	        } \\\n\
        }\n\
");
    header_code_.emplace("/mnt/hpc/xusi/ppl.jit/src/ppl/nn/engines/cuda/impls/src/nn/conv/idxn/common/main_body.h", "// Licensed to the Apache Software Foundation (ASF) under one\n\
// or more contributor license agreements.  See the NOTICE file\n\
// distributed with this work for additional information\n\
// regarding copyright ownership.  The ASF licenses this file\n\
// to you under the Apache License, Version 2.0 (the\n\
// \"License\"); you may not use this file except in compliance\n\
// with the License.  You may obtain a copy of the License at\n\
//\n\
//   http://www.apache.org/licenses/LICENSE-2.0\n\
//\n\
// Unless required by applicable law or agreed to in writing,\n\
// software distributed under the License is distributed on an\n\
// \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n\
// KIND, either express or implied.  See the License for the\n\
// specific language governing permissions and limitations\n\
// under the License.\n\
\n\
#if defined(ENABLE_FUSE)\n\
__global__ void __launch_bounds__(CTA_SIZE_IN_THD) KERNEL_NAME(TOTAL_KPARAM_LIST)\n\
#endif\n\
{\n\
#if (__CUDA_ARCH__ >= 750) && (__CUDACC_VER_MAJOR__  * 1000  + __CUDACC_VER_MINOR__ * 10  >= 10020)\n\
    int C[C_ITEMS_PER_THD];\n\
\n\
    __half  * hC  = (__half  *) C;\n\
    __half2 * h2C = (__half2 *) C;\n\
\n\
#if TILE_K_PER_STEP == 8\n\
    int  * dAv1 = (int  *) dA;\n\
    int  * dBv1 = (int  *) dB;\n\
#elif TILE_K_PER_STEP == 16\n\
    int2 * dAv2 = (int2 *) dA;\n\
    int2 * dBv2 = (int2 *) dB;\n\
#elif TILE_K_PER_STEP == 32\n\
    int4 * dAv4 = (int4 *) dA;\n\
    int4 * dBv4 = (int4 *) dB;\n\
#endif\n\
    int  * dCv1 = (int  *) dC;\n\
\n\
    for (int i = 0; i < HC_ITEMS_PER_THD; i++) { hC[i] = _HALF_ZERO_; }\n\
\n\
    uint tid       =  threadIdx.x;\n\
    uint tid_x     =  tid &  0x3;\n\
    uint tid_y     = (tid & 0x1f) >> 2;\n\
\n\
    uint warp_idx  = (tid >>  WARP_SIZE_IN_BITS) & (CTA_SIZE_X_IN_WARP - 1);\n\
    uint warp_idy  =  tid >> (WARP_SIZE_IN_BITS  +  CTA_SIZE_X_IN_BITS);\n\
\n\
    uint cta_idx   = blockIdx.y;\n\
    uint cta_idy   = blockIdx.x;\n\
\n\
    uint grp_id    = blockIdx.z;\n\
\n\
    uint in_chl_per_grp_pad_v8 = in_chl_per_grp_pad >> 3;\n\
#if TILE_K_PER_STEP == 8\n\
    uint flt_chl_per_grp_pad_v2 = flt_chl_per_grp_pad >> 1;\n\
#elif TILE_K_PER_STEP == 16\n\
    uint flt_chl_per_grp_pad_v4 = flt_chl_per_grp_pad >> 2;\n\
#elif TILE_K_PER_STEP == 32\n\
    uint flt_chl_per_grp_pad_v8 = flt_chl_per_grp_pad >> 3;\n\
#endif\n\
    uint num_flt_per_grp_pad_v2 = num_flt_per_grp_pad >> 1;\n\
\n\
    uint num_flt_v2 = num_flt_per_grp_pad_v2 * num_grp;\n\
#if TILE_K_PER_STEP == 8\n\
    uint flt_hwc_v2 = flt_hw * flt_chl_per_grp_pad_v2;\n\
#elif TILE_K_PER_STEP == 16\n\
    uint flt_hwc_v4 = flt_hw * flt_chl_per_grp_pad_v4;\n\
#elif TILE_K_PER_STEP == 32\n\
    uint flt_hwc_v8 = flt_hw * flt_chl_per_grp_pad_v8;\n\
#endif\n\
\n\
    bool dCv1_y_valid[BLK_M_PER_MMA];\n\
    uint   dCv1_idy[BLK_M_PER_MMA];\n\
\n\
    dCv1_idy[0] =  cta_idy     * TILE_M_V1_PER_CTA  +\n\
                   warp_idy    * TILE_M_V1_PER_MMA  +\n\
                   tid_y;\n\
\n\
    dCv1_idy[1] =  dCv1_idy[0] + TILE_M_V1_PER_MMA_HALF;\n\
\n\
    bool dCv1_x_valid[NUM_N_STEPS];\n\
    uint   dCv1_idx[NUM_N_STEPS];\n\
\n\
    uint dCv1_idx_base  =  grp_id      * num_flt_per_grp_pad_v2  +\n\
                           cta_idx     * TILE_N_V2_PER_CTA  +\n\
                           warp_idx    * TILE_N_V2_PER_MMA  +\n\
                           tid_x;\n\
    uint dCv1_idx_bound = (grp_id + 1) * num_flt_per_grp_pad_v2;\n\
\n\
    for(uint i = 0; i < NUM_N_STEPS; i++)\n\
    {\n\
        dCv1_idx[i]     =  dCv1_idx_base + i * TILE_N_V2_PER_STEP;\n\
        dCv1_x_valid[i]   = (dCv1_idx[i] < dCv1_idx_bound);\n\
    }\n\
\n\
#if TILE_K_PER_STEP == 8\n\
    const int ZEROv1 = 0;\n\
#elif TILE_K_PER_STEP == 16\n\
    const int2 ZEROv2 = {0, 0};\n\
#elif TILE_K_PER_STEP == 32\n\
    const int4 ZEROv4 = {0, 0, 0, 0};\n\
#endif\n\
\n\
    __shared__ int4 sm_base_v4[SM_IN_ID_SIZE + SM_IN_OFF_SIZE];\n\
\n\
#if TILE_K_PER_STEP == 8\n\
    int  reg_dAv1[REG_dAv1_SIZE];\n\
    int  reg_dBv1[REG_dBv1_SIZE];\n\
#elif TILE_K_PER_STEP == 16\n\
    int2 reg_dAv2[REG_dAv2_SIZE];\n\
    int2 reg_dBv2[REG_dBv2_SIZE];\n\
\n\
    int * reg_dAv1 = (int *) reg_dAv2;\n\
    int * reg_dBv1 = (int *) reg_dBv2;\n\
#elif TILE_K_PER_STEP == 32\n\
    int4 reg_dAv4[REG_dAv4_SIZE];\n\
    int4 reg_dBv4[REG_dBv4_SIZE];\n\
\n\
    int * reg_dAv1 = (int *) reg_dAv4;\n\
    int * reg_dBv1 = (int *) reg_dBv4;\n\
#endif\n\
\n\
#if (TILE_M_PER_CTA > CTA_SIZE_IN_THD)\n\
    SET_IN_Mv1_ID(tid, sm_base_v4);\n\
    SET_IN_Mv1_ID(tid + CTA_SIZE_IN_THD, sm_base_v4);\n\
#elif (TILE_M_PER_CTA == CTA_SIZE_IN_THD)\n\
    SET_IN_Mv1_ID(tid, sm_base_v4);\n\
#elif (TILE_M_PER_CTA < CTA_SIZE_IN_THD)\n\
    if(tid < TILE_M_PER_CTA)\n\
        SET_IN_Mv1_ID(tid, sm_base_v4);\n\
#endif\n\
\n\
    if(tid < koff_num_pad)\n\
        SET_IN_Kv8_OFF(tid, sm_base_v4);\n\
\n\
#if TILE_K_PER_STEP == 8\n\
    int   dBv1_off[READ_dBv1_STEPS];\n\
    bool flt_n_valid[READ_dBv1_STEPS];\n\
    int  flt_hwc_v2_off = tid_x;\n\
\n\
    for(int i = 0; i < READ_dBv1_STEPS; i++)\n\
    {\n\
        SET_dBv1_BOUND(i, dBv1_off[i], flt_n_valid[i]);\n\
    }\n\
#elif TILE_K_PER_STEP == 16\n\
    int   dBv2_off[READ_dBv2_STEPS];\n\
    bool flt_n_valid[READ_dBv2_STEPS];\n\
    int  flt_hwc_v4_off = tid_x;\n\
\n\
    for(int i = 0; i < READ_dBv2_STEPS; i++)\n\
    {\n\
        SET_dBv2_BOUND(i, dBv2_off[i], flt_n_valid[i]);\n\
    }\n\
#elif TILE_K_PER_STEP == 32\n\
    int   dBv4_off[READ_dBv4_STEPS];\n\
    bool flt_n_valid[READ_dBv4_STEPS];\n\
    int  flt_hwc_v8_off = tid_x;\n\
\n\
    for(int i = 0; i < READ_dBv4_STEPS; i++)\n\
    {\n\
        SET_dBv4_BOUND(i, dBv4_off[i], flt_n_valid[i]);\n\
    }\n\
#endif\n\
    \n\
    uint in_id_read  =  warp_idy * TILE_M_PER_MMA + tid_y;\n\
\n\
    uint in_off_read =  tid_x + SM_IN_ID_SIZE;\n\
\n\
#if TILE_K_PER_STEP == 8\n\
    int4  in_id[READ_dAv1_STEPS];\n\
#elif TILE_K_PER_STEP == 16\n\
    int4  in_id[READ_dAv2_STEPS];\n\
#elif TILE_K_PER_STEP == 32\n\
    int4  in_id[READ_dAv4_STEPS];\n\
#endif\n\
    int4 in_off;\n\
\n\
    __syncthreads();\n\
\n\
    for(int i = 0; i < NUM_M_STEPS; i++)\n\
    {\n\
        in_id[i * BLK_M_PER_MMA]     = sm_base_v4[in_id_read + TILE_M_PER_STEP * i];\n\
        in_id[i * BLK_M_PER_MMA + 1] = sm_base_v4[in_id_read + TILE_M_PER_STEP * i + TILE_M_PER_MMA_HALF];\n\
    }\n\
\n\
    for(uint i = 0; i < kloop_num; i++)\n\
    {\n\
        in_off = sm_base_v4[in_off_read];\n\
\n\
#if TILE_K_PER_STEP == 8\n\
        LOAD_dBv1(reg_dBv1, dBv1, dBv1_off);\n\
        LOAD_dAv1(reg_dAv1, dAv1, in_id, in_off);\n\
#elif TILE_K_PER_STEP == 16\n\
        LOAD_dBv2(reg_dBv2, dBv2, dBv2_off);\n\
        LOAD_dAv2(reg_dAv2, dAv2, in_id, in_off);\n\
#elif TILE_K_PER_STEP == 32\n\
        LOAD_dBv4(reg_dBv4, dBv4, dBv4_off);\n\
        LOAD_dAv4(reg_dAv4, dAv4, in_id, in_off);\n\
#endif\n\
\n\
        MMA_INSTS(C, reg_dAv1, reg_dBv1);\n\
\n\
#if 2 * TILE_K_PER_STEP == TILE_K_PER_CTA\n\
\n\
#if TILE_K_PER_STEP == 8\n\
        in_off = sm_base_v4[in_off_read + TILE_K_V2_PER_STEP];\n\
#elif TILE_K_PER_STEP == 16\n\
        in_off = sm_base_v4[in_off_read + TILE_K_V4_PER_STEP];\n\
#elif TILE_K_PER_STEP == 32\n\
        in_off = sm_base_v4[in_off_read + TILE_K_V8_PER_STEP];\n\
#endif\n\
\n\
        __syncthreads();\n\
\n\
#if TILE_K_PER_STEP == 8\n\
        LOAD_dBv1(reg_dBv1, dBv1, dBv1_off);\n\
        LOAD_dAv1(reg_dAv1, dAv1, in_id, in_off);\n\
#elif TILE_K_PER_STEP == 16\n\
        LOAD_dBv2(reg_dBv2, dBv2, dBv2_off);\n\
        LOAD_dAv2(reg_dAv2, dAv2, in_id, in_off);\n\
#elif TILE_K_PER_STEP == 32\n\
        LOAD_dBv4(reg_dBv4, dBv4, dBv4_off);\n\
        LOAD_dAv4(reg_dAv4, dAv4, in_id, in_off);\n\
#endif\n\
\n\
        MMA_INSTS(C, reg_dAv1, reg_dBv1);\n\
#endif\n\
\n\
#if TILE_K_PER_STEP == 8\n\
        in_off_read += TILE_K_V2_PER_CTA;\n\
#elif TILE_K_PER_STEP == 16\n\
        in_off_read += TILE_K_V4_PER_CTA;\n\
#elif TILE_K_PER_STEP == 32\n\
        in_off_read += TILE_K_V8_PER_CTA;\n\
#endif\n\
    }\n\
\n\
    for(int step = 0; step < NUM_M_STEPS; step++)\n\
    {\n\
        dCv1_y_valid[0] = (dCv1_idy[0] < out_nhw);\n\
        dCv1_y_valid[1] = (dCv1_idy[1] < out_nhw);\n\
\n\
        uint Cv1_off  = step * TILE_N_V2_PER_THD * BLK_M_PER_MMA;\n\
\n\
#if TILE_N_PER_WARP == 8\n\
\n\
        ADD_BIAS_2x1_V1(has_bias, bias, step);\n\
\n\
#if defined(ENABLE_FUSE)\n\
        uint concat_v1_off0 = 0;\n\
        uint concat_v1_off1 = 0;\n\
\n\
        FUSE_RELU_2x1_V1(has_relu);\n\
        FUSE_CLIP_2x1_V1(has_clip, clip_max, clip_min);\n\
        FUSE_PRELU_2x1_V1(has_prelu, prelu, leaky);\n\
\n\
        FUSE_ELT_2x1_V1(has_elt, pre_data);\n\
        FUSE_RELU_2x1_V1(has_elt_relu);\n\
        FUSE_CLIP_2x1_V1(has_elt_clip, elt_clip_max, elt_clip_min);\n\
        FUSE_PRELU_2x1_V1(has_elt_prelu, elt_prelu, elt_leaky);\n\
\n\
        SET_CONCAT_OFF_V1(has_concat, concat_v1_off0, concat_v1_off1);\n\
#endif\n\
\n\
        OUTPUT_2x1_BY_INT1();\n\
#elif TILE_N_PER_WARP == 16\n\
\n\
        ADD_BIAS_2x2_V1(has_bias, bias, step);\n\
\n\
#if defined(ENABLE_FUSE)\n\
        uint concat_v1_off0 = 0;\n\
        uint concat_v1_off1 = 0;\n\
\n\
        FUSE_RELU_2x2_V1(has_relu);\n\
        FUSE_CLIP_2x2_V1(has_clip, clip_max, clip_min);\n\
        FUSE_PRELU_2x2_V1(has_prelu, prelu, leaky);\n\
\n\
        FUSE_ELT_2x2_V1(has_elt, pre_data);\n\
        FUSE_RELU_2x2_V1(has_elt_relu);\n\
        FUSE_CLIP_2x2_V1(has_elt_clip, elt_clip_max, elt_clip_min);\n\
        FUSE_PRELU_2x2_V1(has_elt_prelu, elt_prelu, elt_leaky);\n\
\n\
        SET_CONCAT_OFF_V1(has_concat, concat_v1_off0, concat_v1_off1);\n\
#endif\n\
\n\
        OUTPUT_2x2_BY_INT1();\n\
#elif TILE_N_PER_WARP == 32\n\
\n\
        ADD_BIAS_2x4_V1(has_bias, bias, step);\n\
\n\
#if defined(ENABLE_FUSE)\n\
        uint concat_v1_off0 = 0;\n\
        uint concat_v1_off1 = 0;\n\
\n\
        FUSE_RELU_2x4_V1(has_relu);\n\
        FUSE_CLIP_2x4_V1(has_clip, clip_max, clip_min);\n\
        FUSE_PRELU_2x4_V1(has_prelu, prelu, leaky);\n\
\n\
        FUSE_ELT_2x4_V1(has_elt, pre_data);\n\
        FUSE_RELU_2x4_V1(has_elt_relu);\n\
        FUSE_CLIP_2x4_V1(has_elt_clip, elt_clip_max, elt_clip_min);\n\
        FUSE_PRELU_2x4_V1(has_elt_prelu, elt_prelu, elt_leaky);\n\
\n\
        SET_CONCAT_OFF_V1(has_concat, concat_v1_off0, concat_v1_off1);\n\
#endif\n\
\n\
        OUTPUT_2x4_BY_INT1();\n\
#endif\n\
\n\
        dCv1_idy[0] += TILE_M_PER_STEP;\n\
        dCv1_idy[1] += TILE_M_PER_STEP;\n\
    }\n\
\n\
#endif // __CUDA_ARCH__\n\
}\n\
");
    header_code_.emplace("/mnt/hpc/xusi/ppl.jit/src/ppl/nn/engines/cuda/impls/src/nn/conv/idxn/common/uni_undefs.h", "// Licensed to the Apache Software Foundation (ASF) under one\n\
// or more contributor license agreements.  See the NOTICE file\n\
// distributed with this work for additional information\n\
// regarding copyright ownership.  The ASF licenses this file\n\
// to you under the Apache License, Version 2.0 (the\n\
// \"License\"); you may not use this file except in compliance\n\
// with the License.  You may obtain a copy of the License at\n\
//\n\
//   http://www.apache.org/licenses/LICENSE-2.0\n\
//\n\
// Unless required by applicable law or agreed to in writing,\n\
// software distributed under the License is distributed on an\n\
// \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n\
// KIND, either express or implied.  See the License for the\n\
// specific language governing permissions and limitations\n\
// under the License.\n\
\n\
////////////////////////////////////////\n\
// kernel list macros\n\
////////////////////////////////////////\n\
\n\
#undef TOTAL_KPARAM_LIST\n\
\n\
////////////////////////////////////////\n\
// customized macros\n\
////////////////////////////////////////\n\
\n\
#undef TILE_N_PER_CTA\n\
#undef TILE_M_PER_CTA\n\
\n\
#undef TILE_K_PER_CTA\n\
#undef TILE_K_PER_STEP\n\
\n\
#undef TILE_N_PER_WARP\n\
#undef TILE_M_PER_WARP\n\
\n\
#undef KERNEL_NAME\n\
\n\
////////////////////////////////////////\n\
// align functions\n\
////////////////////////////////////////\n\
\n\
#undef Align\n\
#undef DivUp\n\
\n\
#undef Min\n\
#undef Max\n\
\n\
////////////////////////////////////////\n\
// boundary check\n\
////////////////////////////////////////\n\
\n\
#undef WidthInRange\n\
#undef HeightInRange\n\
#undef BatchInRange\n\
\n\
////////////////////////////////////////\n\
// constant cta size macros\n\
////////////////////////////////////////\n\
\n\
#undef _4CHAR_TO_INT_\n\
#undef _4INT_TO_INT4_\n\
#undef _2INT_TO_INT2_\n\
\n\
#undef _2HALF_TO_INT_\n\
#undef _2INT2_TO_INT4_\n\
\n\
#undef _C1_\n\
#undef _C2_\n\
#undef _C4_\n\
#undef _C8_\n\
#undef _C16_\n\
#undef _C32_\n\
\n\
#undef _1INT_\n\
#undef _2INT_\n\
#undef _4INT_\n\
#undef _8INT_\n\
\n\
#undef _1INT4_\n\
#undef _2INT4_\n\
#undef _4INT4_\n\
#undef _8INT4_\n\
\n\
#undef _1INT8_\n\
#undef _2INT8_\n\
#undef _4INT8_\n\
#undef _8INT8_\n\
\n\
#undef _1HALF_\n\
#undef _2HALF_\n\
#undef _4HALF_\n\
#undef _8HALF_\n\
\n\
#undef _1HALF2_\n\
#undef _2HALF2_\n\
#undef _4HALF2_\n\
#undef _8HALF2_\n\
\n\
#undef _1MMA_\n\
#undef _2MMA_\n\
#undef _4MMA_\n\
#undef _8MMA_\n\
\n\
#undef _HALF_ZERO_\n\
\n\
#undef _1INT_X1_\n\
#undef _1INT_X2_\n\
#undef _1INT_X4_\n\
\n\
#undef _2INT_X1_\n\
#undef _2INT_X2_\n\
#undef _2INT_X4_\n\
\n\
#undef _4INT_X1_\n\
#undef _4INT_X2_\n\
#undef _4INT_X4_\n\
\n\
#undef _INT_TO_BYTE_\n\
#undef _INT_TO_2HALF_\n\
#undef _INT2_TO_2HALF2_\n\
#undef _INT2_TO_2INT_\n\
\n\
#undef _INT4_TO_INT4_\n\
#undef _INT4_TO_2INT2_\n\
#undef _INT4_TO_4INT_\n\
#undef _INT4_TO_4HALF2_\n\
#undef _INT4_TO_8HALF_\n\
\n\
////////////////////////////////////////\n\
// mma size macros\n\
////////////////////////////////////////\n\
\n\
#undef TILE_M_PER_MMA\n\
#undef TILE_K_PER_MMA\n\
#undef TILE_N_PER_MMA\n\
#undef TILE_M_PER_MMA_HALF\n\
\n\
#undef MMA_SIZE_Y_IN_THD\n\
#undef MMA_SIZE_Y_IN_THD\n\
\n\
#undef MMA_SIZE_X_IN_BITS\n\
#undef CTA_SIZE_X_IN_BITS\n\
\n\
#undef BLK_M_PER_MMA \n\
#undef BLK_N_PER_MMA\n\
\n\
////////////////////////////////////////\n\
// thread / warp / cta size macros\n\
////////////////////////////////////////\n\
\n\
#undef WARP_SIZE_IN_THD\n\
#undef WARP_SIZE_IN_BITS\n\
\n\
#undef WARP_SIZE_X_IN_THD\n\
#undef WARP_SIZE_Y_IN_THD\n\
\n\
#undef CTA_SIZE_X_IN_WARP\n\
#undef CTA_SIZE_Y_IN_WARP\n\
\n\
#undef CTA_SIZE_IN_WARP\n\
#undef CTA_SIZE_IN_THD\n\
\n\
#undef WARP_SIZE_IN_THD_HALF\n\
#undef WARP_SIZE_IN_THD_QTR\n\
\n\
#undef NUM_M_STEPS\n\
#undef NUM_N_STEPS\n\
\n\
////////////////////////////////////////\n\
// tiling size macros\n\
////////////////////////////////////////\n\
\n\
#undef TILE_M_PER_STEP\n\
#undef TILE_N_PER_STEP\n\
\n\
#undef TILE_M_PER_THD\n\
#undef TILE_N_PER_THD\n\
\n\
/////////////////////\n\
// tile m\n\
\n\
#undef TILE_M_V1_PER_CTA\n\
#undef TILE_M_V2_PER_CTA\n\
#undef TILE_M_V4_PER_CTA\n\
#undef TILE_M_V8_PER_CTA\n\
\n\
#undef TILE_M_V1_PER_WARP\n\
#undef TILE_M_V2_PER_WARP\n\
#undef TILE_M_V4_PER_WARP\n\
#undef TILE_M_V8_PER_WARP\n\
\n\
#undef TILE_M_V1_PER_THD\n\
#undef TILE_M_V2_PER_THD\n\
#undef TILE_M_V4_PER_THD\n\
#undef TILE_M_V8_PER_THD\n\
\n\
#undef TILE_M_V1_PER_MMA\n\
#undef TILE_M_V2_PER_MMA\n\
#undef TILE_M_V4_PER_MMA\n\
#undef TILE_M_V8_PER_MMA\n\
#undef TILE_M_V1_PER_MMA_HALF\n\
\n\
/////////////////////\n\
// tile k\n\
\n\
#undef TILE_K_V1_PER_CTA\n\
#undef TILE_K_V2_PER_CTA\n\
#undef TILE_K_V4_PER_CTA\n\
#undef TILE_K_V8_PER_CTA\n\
\n\
#undef TILE_K_V1_PER_STEP\n\
#undef TILE_K_V2_PER_STEP\n\
#undef TILE_K_V4_PER_STEP\n\
#undef TILE_K_V8_PER_STEP\n\
\n\
#undef TILE_K_V1_PER_MMA\n\
#undef TILE_K_V2_PER_MMA\n\
#undef TILE_K_V4_PER_MMA\n\
#undef TILE_K_V8_PER_MMA\n\
\n\
/////////////////////\n\
// tile n\n\
\n\
#undef TILE_N_V1_PER_CTA\n\
#undef TILE_N_V2_PER_CTA\n\
#undef TILE_N_V4_PER_CTA\n\
#undef TILE_N_V8_PER_CTA\n\
\n\
#undef TILE_N_V1_PER_WARP\n\
#undef TILE_N_V2_PER_WARP\n\
#undef TILE_N_V4_PER_WARP\n\
#undef TILE_N_V8_PER_WARP\n\
\n\
#undef TILE_N_V1_PER_THD\n\
#undef TILE_N_V2_PER_THD\n\
#undef TILE_N_V4_PER_THD\n\
#undef TILE_N_V8_PER_THD\n\
\n\
#undef TILE_N_V1_PER_MMA\n\
#undef TILE_N_V2_PER_MMA\n\
#undef TILE_N_V4_PER_MMA\n\
#undef TILE_N_V8_PER_MMA\n\
\n\
#undef TILE_N_V1_PER_STEP\n\
#undef TILE_N_V2_PER_STEP\n\
#undef TILE_N_V4_PER_STEP\n\
#undef TILE_N_V8_PER_STEP\n\
\n\
////////////////////////////////////////\n\
// main loop macros\n\
////////////////////////////////////////\n\
\n\
#undef C_ITEMS_PER_THD\n\
#undef HC_ITEMS_PER_THD\n\
#undef Cv4_ITEMS_PER_THD\n\
\n\
////////////////////////////////////////\n\
// load A and B from device memory macros\n\
////////////////////////////////////////\n\
\n\
#undef REG_dAv1_SIZE\n\
#undef REG_dAv2_SIZE\n\
#undef REG_dAv4_SIZE\n\
\n\
#undef REG_dBv1_SIZE\n\
#undef REG_dBv2_SIZE\n\
#undef REG_dBv4_SIZE\n\
\n\
#undef READ_dAv1_STEPS\n\
#undef READ_dAv2_STEPS\n\
#undef READ_dAv4_STEPS\n\
\n\
#undef READ_dBv1_STEPS\n\
#undef READ_dBv2_STEPS\n\
#undef READ_dBv4_STEPS\n\
\n\
////////////////////////////////////////\n\
// shared memory size macros\n\
////////////////////////////////////////\n\
\n\
#undef  SM_IN_ID_SIZE\n\
#undef SM_OFF_ID_SIZE\n\
\n\
////////////////////////////////////////\n\
// mma macros\n\
////////////////////////////////////////\n\
\n\
#undef MMA_INST_OPCODE\n\
#undef MMA_INST\n\
\n\
#undef MMA_INST_INT4_ASCEND1\n\
#undef MMA_INST_INT4_ASCEND2\n\
#undef MMA_INST_INT4_ASCEND4\n\
#undef MMA_INST_INT4_ASCEND8\n\
\n\
#undef MMA_INST_INT4_DESCEND1\n\
#undef MMA_INST_INT4_DESCEND2\n\
#undef MMA_INST_INT4_DESCEND4\n\
#undef MMA_INST_INT4_DESCEND8\n\
\n\
#undef MMA_INST_INT4_1x1\n\
#undef MMA_INST_INT4_1x2\n\
#undef MMA_INST_INT4_1x4\n\
#undef MMA_INST_INT4_1x8\n\
\n\
#undef MMA_INST_INT4_2x1\n\
#undef MMA_INST_INT4_2x2\n\
#undef MMA_INST_INT4_2x4\n\
#undef MMA_INST_INT4_2x8\n\
\n\
#undef MMA_INST_INT4_4x1\n\
#undef MMA_INST_INT4_4x2\n\
#undef MMA_INST_INT4_4x4\n\
#undef MMA_INST_INT4_4x8\n\
\n\
#undef MMA_INST_INT4_8x1\n\
#undef MMA_INST_INT4_8x2\n\
#undef MMA_INST_INT4_8x4\n\
\n\
#undef MMA_INSTS\n\
\n\
/////////////////////////////////////////////////////\n\
// common load global memory macros\n\
/////////////////////////////////////////////////////\n\
\n\
//////////////////////////\n\
// load dA\n\
//////////////////////////\n\
\n\
#undef LOAD_dAv1_SIZE_16TH\n\
#undef LOAD_dAv1_SIZE_8TH\n\
#undef LOAD_dAv1_SIZE_QTR\n\
#undef LOAD_dAv1_SIZE_HALF\n\
#undef LOAD_dAv1_SIZE1\n\
#undef LOAD_dAv1_SIZE2\n\
#undef LOAD_dAv1_SIZE4\n\
#undef LOAD_dAv1_SIZE8\n\
\n\
#undef LOAD_dAv2_SIZE_16TH\n\
#undef LOAD_dAv2_SIZE_8TH\n\
#undef LOAD_dAv2_SIZE_QTR\n\
#undef LOAD_dAv2_SIZE_HALF\n\
#undef LOAD_dAv2_SIZE1\n\
#undef LOAD_dAv2_SIZE2\n\
#undef LOAD_dAv2_SIZE4\n\
#undef LOAD_dAv2_SIZE8\n\
\n\
#undef LOAD_dAv4_SIZE_16TH\n\
#undef LOAD_dAv4_SIZE_8TH\n\
#undef LOAD_dAv4_SIZE_QTR\n\
#undef LOAD_dAv4_SIZE_HALF\n\
#undef LOAD_dAv4_SIZE1\n\
#undef LOAD_dAv4_SIZE2\n\
#undef LOAD_dAv4_SIZE4\n\
#undef LOAD_dAv4_SIZE8\n\
\n\
#undef LOAD_dAv1\n\
#undef LOAD_dAv2\n\
#undef LOAD_dAv4\n\
\n\
#undef SET_IN_Mv1_ID\n\
#undef SET_IN_Kv8_OFF\n\
\n\
//////////////////////////\n\
// load dB\n\
//////////////////////////\n\
\n\
#undef LOAD_dBv1_SIZE_16TH\n\
#undef LOAD_dBv1_SIZE_8TH\n\
#undef LOAD_dBv1_SIZE_QTR\n\
#undef LOAD_dBv1_SIZE_HALF\n\
#undef LOAD_dBv1_SIZE1\n\
#undef LOAD_dBv1_SIZE2\n\
#undef LOAD_dBv1_SIZE4\n\
#undef LOAD_dBv1_SIZE8\n\
\n\
#undef LOAD_dBv2_SIZE_16TH\n\
#undef LOAD_dBv2_SIZE_8TH\n\
#undef LOAD_dBv2_SIZE_QTR\n\
#undef LOAD_dBv2_SIZE_HALF\n\
#undef LOAD_dBv2_SIZE1\n\
#undef LOAD_dBv2_SIZE2\n\
#undef LOAD_dBv2_SIZE4\n\
#undef LOAD_dBv2_SIZE8\n\
\n\
#undef LOAD_dBv4_SIZE_16TH\n\
#undef LOAD_dBv4_SIZE_8TH\n\
#undef LOAD_dBv4_SIZE_QTR\n\
#undef LOAD_dBv4_SIZE_HALF\n\
#undef LOAD_dBv4_SIZE1\n\
#undef LOAD_dBv4_SIZE2\n\
#undef LOAD_dBv4_SIZE4\n\
#undef LOAD_dBv4_SIZE8\n\
\n\
#undef LOAD_dBv1\n\
#undef LOAD_dBv2\n\
#undef LOAD_dBv4\n\
\n\
#undef SET_dBv4_BOUND \n\
\n\
/////////////////////////////////////////////////////\n\
// precision half output\n\
/////////////////////////////////////////////////////\n\
\n\
#undef OUTPUT_2x1_BY_INT1\n\
#undef OUTPUT_2x2_BY_INT1\n\
#undef OUTPUT_2x4_BY_INT1\n\
\n\
#undef ADD_BIAS_2x1_V1\n\
#undef ADD_BIAS_2x2_V1\n\
#undef ADD_BIAS_2x4_V1\n\
\n\
#undef FUSE_RELU_2x1_V1\n\
#undef FUSE_RELU_2x2_V1\n\
#undef FUSE_RELU_2x4_V1\n\
\n\
#undef FUSE_CLIP_2x1_V1\n\
#undef FUSE_CLIP_2x2_V1\n\
#undef FUSE_CLIP_2x4_V1\n\
\n\
#undef FUSE_PRELU_2x1_V1\n\
#undef FUSE_PRELU_2x2_V1\n\
#undef FUSE_PRELU_2x4_V1\n\
\n\
#undef FUSE_ELT_2x1_V1\n\
#undef FUSE_ELT_2x2_V1\n\
#undef FUSE_ELT_2x4_V1\n\
\n\
#undef SET_CONCAT_OFF_V1\n\
");
}

