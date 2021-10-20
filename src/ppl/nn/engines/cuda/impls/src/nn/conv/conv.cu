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

#include <vector>
#include <cuda.h>
#include <assert.h>

#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <unordered_map>
#include <nvrtc.h>

#include "cudakernel/nn/conv/conv_fp16.h"
#include "cudakernel/nn/conv/gene_kernel.h"
#include "kernel_type.h"
#include "conv_common.h"
#include "common/init_lut.h"
#include "common/merge_split.h"

#include "ppl/nn/engines/cuda/module/cuda_compiler.h"
#include "ppl/nn/engines/cuda/module/cuda_module.h"

#include "float.h"

#define TIMES 4

#define SPK_KPARAM_LIST \
            pad_input,                                                                  \
            d_flt,                                                                      \
            conv_out,                                                                   \
            kloop_num,                                                                  \
    		in_lut,                        in_lut_size,                                 \
    		flt_lut,                       flt_lut_size,                                \
            chl_lut,                       chl_lut_size,                                \
            kloop_lut,                     kloop_lut_size,                              \
            in_hw,                         out_hw,                                      \
            flt_hw,                        splitk,                                      \
            conv_param.in_height,          conv_param.in_width,                         \
            conv_param.in_num,             conv_param.num_grp,                          \
            num_chl_per_grp,               num_chl_per_grp_pad,                         \
            conv_param.flt_height,         conv_param.flt_width,                        \
            num_flt_per_grp,               num_flt_per_grp_pad,                         \
            conv_param.out_height,         conv_param.out_width,                        \
            conv_param.stride_height,      conv_param.stride_width,                     \
            conv_param.pad_height,         conv_param.pad_width,                        \
            conv_param.hole_height,        conv_param.hole_width,                       \
            conv_param.has_bias,           (int *)bias

#define LUT_KPARAM_LIST \
            pad_input,                                                                  \
            d_flt,                                                                      \
            conv_out,                                                                   \
            kloop_num,                                                                  \
    		in_lut,                        in_lut_size,                                 \
    		flt_lut,                       flt_lut_size,                                \
            in_hw,                         out_hw,                                      \
            flt_hw,                        splitk,                                      \
            conv_param.in_height,          conv_param.in_width,                         \
            conv_param.in_num,             conv_param.num_grp,                          \
            num_chl_per_grp,               num_chl_per_grp_pad,                         \
            conv_param.flt_height,         conv_param.flt_width,                        \
            num_flt_per_grp,               num_flt_per_grp_pad,                         \
            conv_param.out_height,         conv_param.out_width,                        \
            conv_param.stride_height,      conv_param.stride_width,                     \
            conv_param.pad_height,         conv_param.pad_width,                        \
            conv_param.hole_height,        conv_param.hole_width,                       \
            conv_param.has_bias,           bias,                                        \
            fuse_param.has_activation,     clip_min,                                    \
            fuse_param.has_clip,           clip_max,                                    \
            fuse_param.has_prelu,          (const void *) fuse_param.prelu,             \
            fuse_param.has_elt,            (const int4 *) fuse_param.pre_data,          \
            fuse_param.has_elt_activation, elt_clip_min,                                \
            fuse_param.has_elt_clip,       elt_clip_max,                                \
            fuse_param.has_elt_prelu,      (const void *) fuse_param.elt_prelu,         \
            leaky,                         elt_leaky,                                   \
            fuse_param.has_concat,         concat_offset_v8,                            \
            concat_stride_v8


#define IDX_KPARAM_LIST \
            pad_input,                                                                  \
            d_flt,                                                                      \
            conv_out,                                                                   \
            kloop_num,                      koff_num_pad,                               \
            in_hw,                         out_hw,                                      \
            flt_hw,                        out_nhw,                                     \
            conv_param.in_height,          conv_param.in_width,                         \
            conv_param.in_num,             conv_param.num_grp,                          \
            conv_param.num_chl,            num_chl_per_grp,                             \
            in_chl_per_grp_pad,            flt_chl_per_grp_pad,                         \
            conv_param.flt_height,         conv_param.flt_width,                        \
            num_flt_per_grp,               num_flt_per_grp_pad,                         \
            conv_param.out_height,         conv_param.out_width,                        \
            conv_param.stride_height,      conv_param.stride_width,                     \
            conv_param.pad_height,         conv_param.pad_width,                        \
            conv_param.hole_height,        conv_param.hole_width,                       \
            conv_param.has_bias,           bias,                                        \
            fuse_param.has_activation,     clip_min,                                    \
            fuse_param.has_clip,           clip_max,                                    \
            fuse_param.has_prelu,          (const void *) fuse_param.prelu,             \
            fuse_param.has_elt,            (const int4 *) fuse_param.pre_data,          \
            fuse_param.has_elt_activation, elt_clip_min,                                \
            fuse_param.has_elt_clip,       elt_clip_max,                                \
            fuse_param.has_elt_prelu,      (const void *) fuse_param.elt_prelu,         \
            leaky,                         elt_leaky,                                   \
            fuse_param.has_concat,         concat_offset_v8,                            \
            concat_stride_v8

#define MERGE_KPARAM_LIST \
        	conv_out,                      final_out,                                   \
        	spk_height_v1,                 spk_width_v8,                                \
        	out_hw,                        splitk * splitf,                             \
            conv_param.has_bias,           bias,                                        \
            fuse_param.has_activation,     clip_min,                                    \
            fuse_param.has_clip,           clip_max,                                    \
            fuse_param.has_prelu,          (const void *) fuse_param.prelu,             \
            fuse_param.has_elt,            (const int4 *) fuse_param.pre_data,          \
            fuse_param.has_elt_activation, elt_clip_min,                                \
            fuse_param.has_elt_clip,       elt_clip_max,                                \
            fuse_param.has_elt_prelu,      (const void *) fuse_param.elt_prelu,         \
            leaky,                         elt_leaky,                                   \
            fuse_param.has_concat,         concat_offset_v8,                            \
            concat_stride_v8

static std::vector<kernel_info_t> g_kernel_container;
static bool is_g_kernel_container_initialized = false;

static std::unordered_map<size_t, algo_param_t> g_conv_shape_hash;

void InitializeKernelContainer(std::vector<kernel_info_t> &g_kernel_container, ppl::common::datatype_t type)
{
    if( type == ppl::common::DATATYPE_FLOAT16 ) {
        Initialize2spkConvF1KernelContainer(g_kernel_container);
        Initialize2spkConvF3KernelContainer(g_kernel_container);
        Initialize2spkConvFNKernelContainer(g_kernel_container);
        Initialize2spkConvFSKernelContainer(g_kernel_container);
                      
        InitializeIdxnConvKernelContainer(g_kernel_container);
    }
    
    is_g_kernel_container_initialized = true;
}

__inline__ std::string GetConvShapeString(conv_param_t &conv_param)
{
    return std::string("b" + std::to_string(conv_param.in_num)  + \
                       "_c" + std::to_string(conv_param.num_chl) + \
                       "_d" + std::to_string(conv_param.num_flt) + \
                       "_g" + std::to_string(conv_param.num_grp) + \
                       "_h" + std::to_string(conv_param.in_height) + \
                       "_w" + std::to_string(conv_param.in_width) + \
                       "_r" + std::to_string(conv_param.flt_height) + \
                       "_s" + std::to_string(conv_param.flt_width) + \
                       "_p" + std::to_string(conv_param.pad_height) + \
                       "_q" + std::to_string(conv_param.pad_width) + \
                       "_u" + std::to_string(conv_param.stride_height) + \
                       "_v" + std::to_string(conv_param.stride_width) + \
                       "_y" + std::to_string(conv_param.hole_height) + \
                       "_x" + std::to_string(conv_param.hole_width) + \
                       "_");
}

__inline__ size_t GetConvShapeHashKey( conv_param_t &conv_param )
{
    return std::hash<std::string>{} (GetConvShapeString(conv_param));
}

uint64_t PPLCUDAConvolutionGetCompilationBufSize(ppl::common::datatype_t type, conv_param_t &conv_param, uint64_t workspace)
{
    int pad_size = GetPadSize(type);

    uint32_t num_chl_per_grp = conv_param.num_chl / conv_param.num_grp;
    uint32_t num_flt_per_grp = conv_param.num_flt / conv_param.num_grp;

    uint32_t num_chl_per_grp_pad = Align(num_chl_per_grp, pad_size);
    uint32_t num_flt_per_grp_pad = Align(num_flt_per_grp, pad_size); 

    bool  is_in_grp_pad = num_chl_per_grp_pad != num_chl_per_grp && conv_param.num_grp != 1;
    bool is_out_grp_pad = num_flt_per_grp_pad != num_chl_per_grp && conv_param.num_grp != 1;

    uint32_t cvt_input_size = 0;
    uint32_t cvt_output_size = 0;

    if(is_in_grp_pad)
        cvt_input_size = GetCvtInputSize( type, conv_param, num_chl_per_grp_pad);

    if(is_out_grp_pad)
        cvt_output_size = getCvtOutputSize(type, conv_param, num_flt_per_grp_pad);

    uint32_t split_size = GetMaxSplitSize(type, conv_param, num_flt_per_grp_pad);

    uint64_t total_size = cvt_input_size + cvt_output_size + split_size;

    return total_size <= workspace ? total_size : workspace;
}
uint64_t PPLCUDAConvolutionGetRuntimeBufSize(
        ppl::common::datatype_t type,
        conv_param_t &conv_param,
        unsigned int splitk,
        unsigned int splitf,
        uint64_t workspace)
{
    int pad_size = GetPadSize(type);

    uint32_t num_chl_per_grp = conv_param.num_chl / conv_param.num_grp;
    uint32_t num_flt_per_grp = conv_param.num_flt / conv_param.num_grp;

    uint32_t num_chl_per_grp_pad = Align(num_chl_per_grp, pad_size);
    uint32_t num_flt_per_grp_pad = Align(num_flt_per_grp, pad_size); 

    bool  is_in_grp_pad = num_chl_per_grp_pad != num_chl_per_grp && conv_param.num_grp != 1;
    bool is_out_grp_pad = num_flt_per_grp_pad != num_chl_per_grp && conv_param.num_grp != 1;

    uint32_t cvt_input_size = 0;
    uint32_t cvt_output_size = 0;

    if(is_in_grp_pad)
        cvt_input_size = GetCvtInputSize(type, conv_param, num_chl_per_grp_pad);
    if(is_out_grp_pad)
        cvt_output_size = getCvtOutputSize(type, conv_param, num_flt_per_grp_pad);

    uint32_t split_size = 0;
    
    if(splitk > 1 || splitf > 1)
        split_size = GetSplitKFSize(type, conv_param, num_flt_per_grp_pad, splitk, splitf);

    uint64_t total_size  = cvt_input_size + cvt_output_size + split_size;

    return total_size <= workspace ? total_size : workspace;
}

std::string ToString(int v) {
    std::stringstream ss;
    ss << v;
    return ss.str();
}

ppl::common::RetCode PPLCUDAConvolutionQuickSelectKernel(
        algo_param_t &algo_param,
        conv_param_t &conv_param) {
    int in_hw = conv_param.in_num * conv_param.in_height * conv_param.in_width;
    int out_hw = conv_param.in_num * conv_param.out_height * conv_param.out_width;
    int flt_hw = conv_param.flt_height * conv_param.flt_width;
    int chl_per_group = conv_param.num_chl / conv_param.num_grp;

    if(!is_g_kernel_container_initialized) 
        InitializeKernelContainer(g_kernel_container, ppl::common::DATATYPE_FLOAT16);

    if (algo_param.kid >= 0) {
        auto kid = algo_param.kid;
        algo_param.algo_name = g_kernel_container[kid].kname;
        algo_param.tiles.m_cta = g_kernel_container[kid].tile_m_per_cta;
        algo_param.tiles.m_warp = g_kernel_container[kid].tile_m_per_warp;
        algo_param.tiles.n_cta = g_kernel_container[kid].tile_n_per_cta;
        algo_param.tiles.n_warp = g_kernel_container[kid].tile_n_per_warp;
        algo_param.tiles.k_cta = g_kernel_container[kid].tile_k_per_cta;
        algo_param.tiles.k_per_step = g_kernel_container[kid].tile_k_per_step;
        algo_param.tiles.k_per_set = g_kernel_container[kid].tile_k_per_set;
        algo_param.tiles.flt_size = g_kernel_container[kid].flt_size;
        algo_param.tiles.flt_pad_size = g_kernel_container[kid].flt_pad_size;
        algo_param.tiles.cta_size_in_thd = g_kernel_container[kid].cta_size_in_thd;
    } else if (chl_per_group < 64) { // Use non-shared memory algo for small channel
        if (flt_hw > 9) {
            algo_param.tiles.m_cta = 128;
            algo_param.tiles.m_warp = 64;
        } else {
            algo_param.tiles.m_cta = 32;
            algo_param.tiles.m_warp = 16;
        }

        if (in_hw == out_hw) {
            algo_param.tiles.n_cta = 64;
            algo_param.tiles.n_warp = 32;
        } else {
            algo_param.tiles.n_cta = 32;
            algo_param.tiles.n_warp = 16;
        }

        if (conv_param.num_chl >= 16) {
            algo_param.tiles.k_cta = 32;
            algo_param.tiles.k_per_step = 32;
        } else {
            algo_param.tiles.k_cta = 16;
            algo_param.tiles.k_per_step = 16;
        }

        algo_param.tiles.cta_size_in_thd = (algo_param.tiles.m_cta / algo_param.tiles.m_warp) * \
                    (algo_param.tiles.n_cta / algo_param.tiles.n_warp) * \
                    WARP_SIZE;

        if(algo_param.tiles.k_per_step == 8)  algo_param.tiles.flt_pad_size = 2;
        else if(algo_param.tiles.k_per_step == 16) algo_param.tiles.flt_pad_size = 4;
        else if(algo_param.tiles.k_per_step == 32) algo_param.tiles.flt_pad_size = 8;

        algo_param.algo_name = "nvIdxnConv_hmma1688_nhwc_b"+ToString(algo_param.tiles.m_cta)+"x"+ToString(algo_param.tiles.n_cta)+
                                            "_w"+ToString(algo_param.tiles.m_warp)+"x"+ToString(algo_param.tiles.n_warp)+
                                            "_k"+ToString(algo_param.tiles.k_cta)+"_s"+ToString(algo_param.tiles.k_per_step)+"_nosmem";
    } else { // Use 3spk algo for large channel
        float min_pad = 1.0;
        algo_param.tiles.m_cta = 16;
        for (int32_t i = 128; i >= 16; i = i / 2) {
            if (out_hw < i) continue;
            float pad = 1.0 * (DivUp(out_hw, i) * i - out_hw) / out_hw;
            if (pad < min_pad)  {
                min_pad = pad;
                algo_param.tiles.m_cta = i;
            }
            if (min_pad < 0.1)  break;
        }

        algo_param.tiles.n_cta = 16;
        for (int32_t i = 128; i >= 16; i = i / 2) {
            int cout = conv_param.num_flt;
            if ((cout < 64 && i / cout == 1) || (cout >= 64 && cout % i == 0)) {
                algo_param.tiles.n_cta = i;
                break;
            }
        }

        if (conv_param.num_chl >= 128) {
            algo_param.tiles.k_cta = 64;
        } else {
            algo_param.tiles.k_cta = 32;
        }

        if (algo_param.tiles.m_cta == 128 && algo_param.tiles.n_cta == 128) {
            algo_param.tiles.m_cta = 64;
        }

        if (algo_param.tiles.m_cta * 4 < algo_param.tiles.n_cta) {
            algo_param.tiles.m_cta *= 2;
            algo_param.tiles.n_cta /= 2;
        }
        if (algo_param.tiles.n_cta *4 < algo_param.tiles.m_cta) {
            algo_param.tiles.m_cta /= 2;
            algo_param.tiles.n_cta *= 2;
        }

        algo_param.tiles.m_warp = algo_param.tiles.m_cta / 2;
        algo_param.tiles.n_warp = algo_param.tiles.n_cta / 2;
        algo_param.tiles.k_per_set = algo_param.tiles.k_cta / 2;
        if (algo_param.tiles.k_per_set <= 8) {
            algo_param.tiles.k_per_set = 16;
        }
        if (algo_param.tiles.m_warp <= 8) {
            algo_param.tiles.m_warp = 16;
        }
        if (algo_param.tiles.n_warp <= 8) {
            algo_param.tiles.n_warp = 16;
        }

        algo_param.tiles.cta_size_in_thd = (algo_param.tiles.m_cta / algo_param.tiles.m_warp) *  \
                               (algo_param.tiles.n_cta / algo_param.tiles.n_warp) *  \
                               (algo_param.tiles.k_cta / algo_param.tiles.k_per_set)  * \
                               WARP_SIZE;

        std::string f_size = "f1";
        algo_param.tiles.flt_size = 1;
        if (conv_param.flt_height == 3) {
            f_size = "f3";
            algo_param.tiles.flt_size = 3;
        } else if (conv_param.flt_height > 3) {
            f_size = "fn";
            algo_param.tiles.flt_size = 0;
        }
        algo_param.algo_name = "nv2spkConv_hmma1688_nhwc_"+f_size+"_b"+ToString(algo_param.tiles.m_cta)+"x"+ToString(algo_param.tiles.n_cta)+
                                                       "_w"+ToString(algo_param.tiles.m_warp)+"x"+ToString(algo_param.tiles.n_warp)+
                                                       "_k"+ToString(algo_param.tiles.k_cta)+"_s"+ToString(algo_param.tiles.k_per_set)+"_buf1";
    }
    return ppl::common::RC_SUCCESS;
}

string PPLCUDACompile(string name, string code, std::vector<const char*> compile_params, int device, bool include) {
    string ptx = ppl::nn::cuda::CUDANVRTCCompile(pair<string,string>(name, code), compile_params, device, include);
    return ptx;
}

float AlgoForwardTime(
    cudaStream_t &stream, 
    string name,
    string code,
    std::vector<const char*> compile_params,
    int device,
    bool include,
    ppl::common::datatype_t type,
    int4* d_input,
    int4* d_flt,
    int4* d_output,
    int4* bias,
    int4* d_temp_buf, 
    algo_param_t &algo_param,
    conv_param_t &conv_param, 
    fuse_param_t &fuse_param,
    uint64_t workspace) 
{
    // printf("%s\n", name.c_str());
    string ptx = ppl::nn::cuda::CUDANVRTCCompile(pair<string,string>(name, code), compile_params, device, include);
    ppl::nn::cuda::CUDAModule* cuda_module = new ppl::nn::cuda::CUDAModule();
    cuda_module->SetSourceCode(name, ptx);
    CUfunction function = cuda_module->GetKernelFunc();

    int times = 4;
    float elapsed = 0;
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    for (int i = 0; i < times; i++) {
        PPLCUDAConvolutionForwardJITImp( 
            stream, function, type, d_input, d_flt, d_output, bias, d_temp_buf,
            algo_param, conv_param, fuse_param);
    }
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed, begin, end);

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    delete cuda_module;
    return elapsed; 
}

ppl::common::RetCode PPLCUDAConvolutionSelectKernel(
        cudaStream_t &stream, 
        ppl::common::datatype_t type,
        int4* d_input,
        int4* d_flt,
        int4* d_output,
        int4* bias,
        int4* d_temp_buf, 
        algo_param_t &algo_param,
        conv_param_t &conv_param, 
        fuse_param_t &fuse_param,
	    uint64_t workspace)
{
    if(!is_g_kernel_container_initialized)
        InitializeKernelContainer(g_kernel_container, type);

    size_t conv_shape_hash = GetConvShapeHashKey(conv_param);

    std::unordered_map<size_t, algo_param_t>::const_iterator conv_shape_hash_iterator = g_conv_shape_hash.find(conv_shape_hash);

    if(conv_shape_hash_iterator != g_conv_shape_hash.end()) {
        algo_param = conv_shape_hash_iterator->second;
        return ppl::common::RC_SUCCESS;
    }

    auto pre_algo_param = algo_param;
    int num_chl_per_grp = conv_param.num_chl / conv_param.num_grp;
    int flt_hw = conv_param.flt_height * conv_param.flt_width;
    
    int declare_times = 0;
    float minTime = FLT_MAX;
    float elapsed;

    const int SPLITK_OPTIONS[] = {1, 2, 4, 8};

    for(unsigned int spk = 0; spk < 1; spk++) {
        unsigned int splitk = SPLITK_OPTIONS[spk];

        for(unsigned int kid = 0; kid < g_kernel_container.size(); kid++) {
            unsigned int splitf = (g_kernel_container[kid].ktype == CONV_2SPK_FS) ? flt_hw : 1;
            printf("%d,%s\n", kid,g_kernel_container[kid].kname.c_str());
        
            if(!g_kernel_container[kid].CheckKernelTypeFeasible(conv_param.flt_height, conv_param.flt_width, num_chl_per_grp, splitk)) continue;

            if(!g_kernel_container[kid].CheckSplitkFeasible(num_chl_per_grp, splitk)) continue;

            if(!g_kernel_container[kid].CheckSplitfFeasible(splitf, splitk)) continue;

            algo_param_t temp_algo_param;
            temp_algo_param.kid = kid;
            temp_algo_param.splitk = splitk;
            temp_algo_param.splitf = splitf;
            temp_algo_param.algo_name = g_kernel_container[kid].kname;
            temp_algo_param.tiles.m_cta = g_kernel_container[kid].tile_m_per_cta;
            temp_algo_param.tiles.m_warp = g_kernel_container[kid].tile_m_per_warp;
            temp_algo_param.tiles.n_cta = g_kernel_container[kid].tile_n_per_cta;
            temp_algo_param.tiles.n_warp = g_kernel_container[kid].tile_n_per_warp;
            temp_algo_param.tiles.k_cta = g_kernel_container[kid].tile_k_per_cta;
            temp_algo_param.tiles.k_per_step = g_kernel_container[kid].tile_k_per_step;
            temp_algo_param.tiles.k_per_set = g_kernel_container[kid].tile_k_per_set;
            temp_algo_param.tiles.flt_size = g_kernel_container[kid].flt_size;
            temp_algo_param.tiles.flt_pad_size = g_kernel_container[kid].flt_pad_size;
            temp_algo_param.tiles.cta_size_in_thd = g_kernel_container[kid].cta_size_in_thd;

            if(!g_kernel_container[kid].CheckQuickSelectFeasible(pre_algo_param, conv_param.num_chl / conv_param.num_grp, splitk, splitf)) continue;

            std::string source = "";
            if (temp_algo_param.algo_name.find("Idxn") != std::string::npos) {
                GeneIdxnKernel(source, temp_algo_param.algo_name, 
                                       temp_algo_param.tiles.m_cta, 
                                       temp_algo_param.tiles.n_cta, 
                                       temp_algo_param.tiles.m_warp, 
                                       temp_algo_param.tiles.n_warp, 
                                       temp_algo_param.tiles.k_cta, 
                                       temp_algo_param.tiles.k_per_step, declare_times);
                declare_times++;
            } else {
                Gene2spkKernel(source, temp_algo_param.algo_name, 
                                       temp_algo_param.tiles.m_cta, 
                                       temp_algo_param.tiles.n_cta, 
                                       temp_algo_param.tiles.m_warp, 
                                       temp_algo_param.tiles.n_warp, 
                                       temp_algo_param.tiles.k_cta, 
                                       temp_algo_param.tiles.k_per_set, 
                                       temp_algo_param.splitk, 
                                       temp_algo_param.splitf, 
                                       1, declare_times);
                declare_times++;
            }

            std::vector<const char*> compile_params;
            elapsed = AlgoForwardTime(stream, 
                                      g_kernel_container[kid].kname,
                                      source,
                                      compile_params,
                                      0,
                                      true,
                                      type,
                                      d_input,
                                      d_flt,
                                      d_output,
                                      bias,
                                      d_temp_buf, 
                                      temp_algo_param,
                                      conv_param,
                                      fuse_param,
                                      workspace);
            
	        if(elapsed < minTime){
                algo_param = temp_algo_param;
	            minTime = elapsed;
	        }
        }
    }
    g_conv_shape_hash[conv_shape_hash] = algo_param;

    return ppl::common::RC_SUCCESS;
}

void PPLCUDAConvolutionForwardImp(
        cudaStream_t &stream, 
        ppl::common::datatype_t type,
        int4* d_input,
        int4* d_flt,
        int4* d_output,
        int4* bias,
        int4* d_temp_buf,
        algo_param_t& algo_param,
        conv_param_t &conv_param,
        fuse_param_t &fuse_param)
{
    if(!is_g_kernel_container_initialized)
        InitializeKernelContainer(g_kernel_container, type);

    unsigned int kid = algo_param.kid;
    unsigned int splitk = algo_param.splitk;
    unsigned int splitf = algo_param.splitf;

    int pad_size = GetPadSize(type);

    int num_chl_per_grp = conv_param.num_chl / conv_param.num_grp;
    int num_flt_per_grp = conv_param.num_flt / conv_param.num_grp;

    int num_chl_per_grp_pad = Align(num_chl_per_grp, pad_size);
    int num_flt_per_grp_pad = Align(num_flt_per_grp, pad_size);

    int in_hw  = conv_param.in_height * conv_param.in_width;
    int flt_hw = conv_param.flt_height * conv_param.flt_width;
    int out_hw = conv_param.out_height * conv_param.out_width;

    int concat_offset_v8 = fuse_param.concat_offset / pad_size;
    int concat_stride_v8 = fuse_param.concat_stride / pad_size;

    bool  is_in_grp_pad = num_chl_per_grp_pad != num_chl_per_grp && conv_param.num_grp != 1;
    bool is_out_grp_pad = num_flt_per_grp_pad != num_chl_per_grp && conv_param.num_grp != 1;

    uint64_t buf_off_v4 = 0;

    int4 *pad_input = d_input;
    int4 *pad_output = d_output;

    if(is_in_grp_pad) {
	    pad_input = d_temp_buf; 
	    buf_off_v4 += GetCvtInputSize(type, conv_param, num_chl_per_grp_pad) / (_4INT_TO_INT4_ * _INT_TO_4BYTE_);

        PPLCUDAConvolutionCvtInput(stream, pad_input, d_input, type, conv_param);
    }

    if(is_out_grp_pad) {
	    pad_output = d_temp_buf + buf_off_v4;
	    buf_off_v4 += getCvtOutputSize(type, conv_param, num_flt_per_grp_pad) / (_4INT_TO_INT4_ * _INT_TO_4BYTE_);
    } 

    int4 *final_out  = fuse_param.has_concat ? (int4 *) fuse_param.post_concat : pad_output;

    int4 *splitk_buf = d_temp_buf + buf_off_v4;
    int4 *conv_out   = (splitk > 1 || splitf > 1) ? splitk_buf : final_out;

    __half2 clip_min     = __float2half2_rn(fuse_param.clip_min);
    __half2 clip_max     = __float2half2_rn(fuse_param.clip_max);
    __half2 elt_clip_min = __float2half2_rn(fuse_param.elt_clip_min);
    __half2 elt_clip_max = __float2half2_rn(fuse_param.elt_clip_max);
    __half  leaky        = __float2half(fuse_param.leaky);
    __half  elt_leaky    = __float2half(fuse_param.elt_leaky);

    dim3 block_size, grid_size;

    block_size.x = g_kernel_container[kid].cta_size_in_thd;
    block_size.y = 1;
    block_size.z = 1;

    grid_size.x  = DivUp(conv_param.in_num * conv_param.out_height * conv_param.out_width, g_kernel_container[kid].tile_m_per_cta);
    grid_size.y  = DivUp(num_flt_per_grp_pad, g_kernel_container[kid].tile_n_per_cta);
    grid_size.z  = conv_param.num_grp * splitk * splitf;

    if(g_kernel_container[kid].ktype == CONV_IDXN_C2 || g_kernel_container[kid].ktype == CONV_IDXN_C4 || \
            g_kernel_container[kid].ktype == CONV_IDXN_C32) {
        int img_pad_size = pad_size;
        int flt_pad_size = g_kernel_container[kid].flt_pad_size;

        int out_nhw = out_hw * conv_param.in_num;

        int in_chl_per_grp_pad = Align(num_chl_per_grp, img_pad_size);
        int flt_chl_per_grp_pad = Align(num_chl_per_grp, flt_pad_size);
        int num_flt_per_grp_pad = Align(num_flt_per_grp, img_pad_size);

	    int kloop_num = DivUp(flt_hw * flt_chl_per_grp_pad, g_kernel_container[kid].tile_k_per_cta);
        int koff_num_pad = Align(kloop_num * (g_kernel_container[kid].tile_k_per_cta / flt_pad_size), WARP_SIZE);

        (g_kernel_container[kid].idx_kptr)<<<grid_size, block_size, 0, stream>>>(IDX_KPARAM_LIST);

    } else if(g_kernel_container[kid].ktype == CONV_2SPK_F1 || g_kernel_container[kid].ktype == CONV_2SPK_F3 || \
            g_kernel_container[kid].ktype == CONV_2SPK_FN || g_kernel_container[kid].ktype == CONV_2SPK_FS ) {

	    int kloop_num = (flt_hw / splitf) * DivUp(num_chl_per_grp_pad, g_kernel_container[kid].tile_k_per_cta);

        lut_t in_lut, flt_lut;
        int in_lut_size, flt_lut_size;
    
        InitializeInputLut(in_lut_size, in_lut.idx, conv_param.flt_height, conv_param.flt_width, conv_param.in_height,
                conv_param.in_width, conv_param.pad_height, conv_param.pad_width, conv_param.hole_height, conv_param.hole_width,
                num_chl_per_grp_pad, conv_param.num_grp, g_kernel_container[kid].tile_k_per_cta, pad_size);

        InitializeFilterLut(flt_lut_size, flt_lut.idx, conv_param.flt_height, conv_param.flt_width, num_chl_per_grp_pad,
                g_kernel_container[kid].tile_k_per_cta, pad_size);

        if(splitk == 1) {
            (g_kernel_container[kid].lut_kptr)<<<grid_size, block_size, 0, stream>>>(LUT_KPARAM_LIST);
        } else {
            int chl_lut_size, kloop_lut_size;
            struct chl_lut_t chl_lut;
            struct kloop_lut_t kloop_lut;

            InitializeChlLut(chl_lut_size, chl_lut.idx, conv_param.num_chl, conv_param.num_grp, pad_size,
                    g_kernel_container[kid].tile_k_per_cta, splitk);
            InitializeKloopLut(kloop_lut_size, kloop_lut.idx, conv_param.num_chl, conv_param.num_grp, pad_size,
                    g_kernel_container[kid].tile_k_per_cta, splitk, splitf, flt_hw);

            (g_kernel_container[kid].spk_kptr)<<<grid_size, block_size, 0, stream>>>(SPK_KPARAM_LIST);
        }
    }
    
    if(splitk > 1 || splitf > 1) {
        int spk_width_v8   = num_flt_per_grp_pad * conv_param.num_grp / pad_size;
        int spk_height_v1  = out_hw * conv_param.in_num;

        dim3 merge_grid_size, merge_block_size;
        merge_block_size.x = 64;
        merge_block_size.y = 1;
        merge_block_size.z = 1;

        merge_grid_size.x  = spk_height_v1;
        merge_grid_size.y  = DivUp(spk_width_v8, merge_block_size.x);
        merge_grid_size.z  = 1;

        MergeConvSplitResults<<<merge_grid_size, merge_block_size, 0, stream>>>(MERGE_KPARAM_LIST);
    }

    if(is_out_grp_pad) {
        PPLCUDAConvolutionCvtOutput(stream, d_output, final_out, type, conv_param);
    }

}

#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      std::cerr << "\nerror: " #x " failed with error "           \
                << nvrtcGetErrorString(result) << '\n';           \
      exit(1);                                                    \
    }                                                             \
  } while(0)
#define CUDA_SAFE_CALL(x)                                         \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      std::cerr << "\nerror: " #x " failed with error "           \
                << msg << '\n';                                   \
      exit(1);                                                    \
    }                                                             \
  } while(0)

#define CUDA_RUNTIME_CALL(x)                                    \
  do {                                                            \
    cudaError_t result = x;                                       \
    if (result != cudaSuccess) {                                 \
      const char *msg = cudaGetErrorName(result);                   \
      std::cerr << "\nerror: " #x " failed with error "           \
                << msg << '\n';                                   \
      exit(1);                                                    \
    }                                                             \
  } while(0)

void PPLCUDAConvolutionForwardJITImp(
    cudaStream_t &stream,
    CUfunction function,
    ppl::common::datatype_t type,
    int4* d_input,
    int4* d_flt,
    int4* d_output,
    int4* bias,
    int4* d_temp_buf,
    algo_param_t &algo_param,
    conv_param_t &conv_param,
    fuse_param_t &fuse_param)
{
    unsigned int kid = algo_param.kid;
    unsigned int splitk = algo_param.splitk;
    unsigned int splitf = algo_param.splitf;

    int pad_size = GetPadSize(type);

    int num_chl_per_grp = conv_param.num_chl / conv_param.num_grp;
    int num_flt_per_grp = conv_param.num_flt / conv_param.num_grp;

    int num_chl_per_grp_pad = Align(num_chl_per_grp, pad_size);
    int num_flt_per_grp_pad = Align(num_flt_per_grp, pad_size);

    int in_hw  = conv_param.in_height * conv_param.in_width;
    int flt_hw = conv_param.flt_height * conv_param.flt_width;
    int out_hw = conv_param.out_height * conv_param.out_width;

    int concat_offset_v8 = fuse_param.concat_offset / pad_size;
    int concat_stride_v8 = fuse_param.concat_stride / pad_size;

    bool  is_in_grp_pad = num_chl_per_grp_pad != num_chl_per_grp && conv_param.num_grp != 1;
    bool is_out_grp_pad = num_flt_per_grp_pad != num_chl_per_grp && conv_param.num_grp != 1;

    uint64_t buf_off_v4 = 0;

    int4 *pad_input = d_input;
    int4 *pad_output = d_output;

    if(is_in_grp_pad) {
	    pad_input = d_temp_buf; 
	    buf_off_v4 += GetCvtInputSize(type, conv_param, num_chl_per_grp_pad) / (_4INT_TO_INT4_ * _INT_TO_4BYTE_);

        PPLCUDAConvolutionCvtInput(stream, pad_input, d_input, type, conv_param);
    }

    if(is_out_grp_pad) {
	    pad_output = d_temp_buf + buf_off_v4;
	    buf_off_v4 += getCvtOutputSize(type, conv_param, num_flt_per_grp_pad) / (_4INT_TO_INT4_ * _INT_TO_4BYTE_);
    } 

    int4 *final_out  = fuse_param.has_concat ? (int4 *) fuse_param.post_concat : pad_output;

    int4 *splitk_buf = d_temp_buf + buf_off_v4;
    int4 *conv_out   = (splitk > 1 || splitf > 1) ? splitk_buf : final_out;

    __half2 clip_min     = __float2half2_rn(fuse_param.clip_min);
    __half2 clip_max     = __float2half2_rn(fuse_param.clip_max);
    __half2 elt_clip_min = __float2half2_rn(fuse_param.elt_clip_min);
    __half2 elt_clip_max = __float2half2_rn(fuse_param.elt_clip_max);
    __half  leaky        = __float2half(fuse_param.leaky);
    __half  elt_leaky    = __float2half(fuse_param.elt_leaky);
    
    int tile_n = algo_param.tiles.n_cta;
    int tile_m = algo_param.tiles.m_cta;
    int cta_k = algo_param.tiles.k_cta;

    dim3 block_size, grid_size;
    block_size.x = algo_param.tiles.cta_size_in_thd;;
    block_size.y = 1;
    block_size.z = 1;

    grid_size.x  = DivUp(conv_param.in_num * conv_param.out_height * conv_param.out_width, tile_m);
    grid_size.y  = DivUp(num_flt_per_grp_pad, tile_n);
    grid_size.z  = conv_param.num_grp * splitk * splitf;

    // int has_relu = fuse_param.has_activation == 1? 1:0;
    // int has_elt_relu = fuse_param.has_elt_activation == 1 ? 1 : 0;
    const int4* pre_data = (const int4*)fuse_param.pre_data;
    const void* prelu = (const void*)fuse_param.prelu;
    const void* elt_prelu = (const void*)fuse_param.elt_prelu;


    if (algo_param.algo_name.find("Idxn") != std::string::npos) {
        int img_pad_size = pad_size;
        int flt_pad_size = algo_param.tiles.flt_pad_size;

        int out_nhw = out_hw * conv_param.in_num;

        int in_chl_per_grp_pad = Align(num_chl_per_grp, img_pad_size);
        int flt_chl_per_grp_pad = Align(num_chl_per_grp, flt_pad_size);
        int num_flt_per_grp_pad = Align(num_flt_per_grp, img_pad_size);

	    int kloop_num = DivUp(flt_hw * flt_chl_per_grp_pad, cta_k);
        int koff_num_pad = Align(kloop_num * (cta_k / flt_pad_size), WARP_SIZE);
        
        void *args[] = {&pad_input, &d_flt, &conv_out, 
                        &kloop_num, &koff_num_pad, &in_hw, &out_hw,
                        &flt_hw, &out_nhw, &conv_param.in_height, &conv_param.in_width,
                        &conv_param.in_num, &conv_param.num_grp, &conv_param.num_chl, &num_chl_per_grp,
                        &in_chl_per_grp_pad, &flt_chl_per_grp_pad,
                        &conv_param.flt_height, &conv_param.flt_width, &num_flt_per_grp, &num_flt_per_grp_pad,
                        &conv_param.out_height, &conv_param.out_width, &conv_param.stride_height, &conv_param.stride_width,
                        &conv_param.pad_height, &conv_param.pad_width, &conv_param.hole_height, &conv_param.hole_width,
                        &conv_param.has_bias, &bias, &fuse_param.has_activation, &clip_min,
                        &fuse_param.has_clip, &clip_max, 
                        &fuse_param.has_prelu, &prelu,
                        &fuse_param.has_elt, &(pre_data),
                        &fuse_param.has_elt_activation, &elt_clip_min, &fuse_param.has_elt_clip, &elt_clip_max,
                        &fuse_param.has_elt_prelu, &(elt_prelu), &leaky, &elt_leaky,
                        &fuse_param.has_concat, &concat_offset_v8, &concat_stride_v8};

        CUDA_SAFE_CALL(cuLaunchKernel(function, grid_size.x, grid_size.y, grid_size.z, 
                        block_size.x, block_size.y, block_size.z,
                        0, stream, args, 0));    
    } else if (algo_param.algo_name.find("2spk") != std::string::npos) {

        // std::cout << "block size " << block_size.x << std::endl;
        // std::cout << "grid_size " << grid_size.x << " " << grid_size.y << " " << grid_size.z << std::endl;
        int kloop_num = (flt_hw / splitf) * DivUp(num_chl_per_grp_pad, cta_k);//g_kernel_container[kid].tile_k_per_cta);

        lut_t in_lut, flt_lut;
        int in_lut_size, flt_lut_size;

        InitializeInputLut(in_lut_size, in_lut.idx, conv_param.flt_height, conv_param.flt_width, conv_param.in_height,
                conv_param.in_width, conv_param.pad_height, conv_param.pad_width, conv_param.hole_height, conv_param.hole_width,
                num_chl_per_grp_pad, conv_param.num_grp, cta_k, pad_size);

        InitializeFilterLut(flt_lut_size, flt_lut.idx, conv_param.flt_height, conv_param.flt_width, num_chl_per_grp_pad,
            cta_k, pad_size);
        if (splitk == 1) {
            void *args[] = {&pad_input, &d_flt, &conv_out, &kloop_num,
                        &in_lut, &in_lut_size, &flt_lut, &flt_lut_size, &in_hw, &out_hw,
                        &flt_hw, &splitk, &conv_param.in_height, &conv_param.in_width,
                        &conv_param.in_num, &conv_param.num_grp, &num_chl_per_grp, &num_chl_per_grp_pad,
                        &conv_param.flt_height, &conv_param.flt_width, &num_flt_per_grp, &num_flt_per_grp_pad,
                        &conv_param.out_height, &conv_param.out_width, &conv_param.stride_height, &conv_param.stride_width,
                        &conv_param.pad_height, &conv_param.pad_width, &conv_param.hole_height, &conv_param.hole_width,
                        &conv_param.has_bias, &bias, &fuse_param.has_activation, &clip_min,
                        &fuse_param.has_clip, &clip_max, 
                        &fuse_param.has_prelu, &prelu,
                        &fuse_param.has_elt, &(pre_data),
                        &fuse_param.has_elt_activation, &elt_clip_min, &fuse_param.has_elt_clip, &elt_clip_max,
                        &fuse_param.has_elt_prelu, &(elt_prelu), &leaky, &elt_leaky,
                        &fuse_param.has_concat, &concat_offset_v8, &concat_stride_v8};
            CUDA_SAFE_CALL(cuLaunchKernel(function, grid_size.x, grid_size.y, grid_size.z, 
                        block_size.x, block_size.y, block_size.z,
                        0, stream, args, 0));
        } else {
            int chl_lut_size, kloop_lut_size;
            struct chl_lut_t chl_lut;
            struct kloop_lut_t kloop_lut;

            InitializeChlLut(chl_lut_size, chl_lut.idx, conv_param.num_chl, conv_param.num_grp, pad_size,
                    g_kernel_container[kid].tile_k_per_cta, splitk);
            InitializeKloopLut(kloop_lut_size, kloop_lut.idx, conv_param.num_chl, conv_param.num_grp, pad_size,
                    g_kernel_container[kid].tile_k_per_cta, splitk, splitf, flt_hw);
            
            void* args[] = {&pad_input, &d_flt, &conv_out, &kloop_num,
                &in_lut, &in_lut_size, &flt_lut, &flt_lut_size,
                &chl_lut, &chl_lut_size, &kloop_lut, &kloop_lut_size,
                &in_hw, &out_hw, &flt_hw, &splitk,
                &conv_param.in_height, &conv_param.in_width,
                &conv_param.in_num, &conv_param.num_grp, &num_chl_per_grp, &num_chl_per_grp_pad,
                &conv_param.flt_height, &conv_param.flt_width, &num_flt_per_grp, &num_flt_per_grp_pad,
                &conv_param.out_height, &conv_param.out_width, &conv_param.stride_height, &conv_param.stride_width,
                &conv_param.pad_height, &conv_param.pad_width, &conv_param.hole_height, &conv_param.hole_width,
                &conv_param.has_bias, &bias };
            CUDA_SAFE_CALL(cuLaunchKernel(function, grid_size.x, grid_size.y, grid_size.z, 
                block_size.x, block_size.y, block_size.z,
                0, stream, args, 0));
        }
    } 
    else {

    }
    
    if(splitk > 1 || splitf > 1) {
        int spk_width_v8   = num_flt_per_grp_pad * conv_param.num_grp / pad_size;
        int spk_height_v1  = out_hw * conv_param.in_num;

        dim3 merge_grid_size, merge_block_size;
        merge_block_size.x = 64;
        merge_block_size.y = 1;
        merge_block_size.z = 1;

        merge_grid_size.x  = spk_height_v1;
        merge_grid_size.y  = DivUp(spk_width_v8, merge_block_size.x);
        merge_grid_size.z  = 1;

        MergeConvSplitResults<<<merge_grid_size, merge_block_size, 0, stream>>>(MERGE_KPARAM_LIST);
    }
    if(is_out_grp_pad) {
        PPLCUDAConvolutionCvtOutput(stream, d_output, final_out, type, conv_param);
    }

}