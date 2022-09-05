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

#if defined(ENABLE_SPLITK)
__global__ void __launch_bounds__(CTA_SIZE_IN_THD) KERNEL_NAME(SPK_KPARAM_LIST)
#elif defined(ENABLE_FUSE) || defined(ENABLE_SPLITF)
__global__ void __launch_bounds__(CTA_SIZE_IN_THD) KERNEL_NAME(TOTAL_KPARAM_LIST)
#endif
{
#if (__CUDA_ARCH__ >= 700) && (__CUDA_ARCH__ <= 720) && (__CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 >= 10020)

    int4 Cv4[Cv4_ITEMS_PER_THD];

    __half *hC = (__half *)Cv4;
    int *C     = (int *)Cv4;

#pragma unroll
    for (int i = 0; i < HC_ITEMS_PER_THD; i++) {
        hC[i] = _HALF_ZERO_;
    }

    uint tid = threadIdx.x;

    uint local_tid = tid & 0x1f;

    uint set_tid = tid & (SET_SIZE_IN_THD - 1);

    uint set_id = (tid >> SET_SIZE_IN_BITS) & 0x7;

    uint set_widx = (set_tid >> WARP_SIZE_IN_BITS) & (SET_SIZE_X_IN_WARP - 1);
    uint set_widy = set_tid >> (WARP_SIZE_IN_BITS + SET_SIZE_X_IN_BITS);

    uint ldg_idx = tid % TILE_K_V8_PER_CTA;
    uint ldg_idy = tid / TILE_K_V8_PER_CTA;

#if TILE_K_PER_CTA == 8
    uint sts_idx = 0;
    uint sts_idy = tid;
#elif TILE_K_PER_CTA == 16
    uint sts_idx   = ((tid & 0x1));
    uint sts_idy   = tid >> 1;
#elif TILE_K_PER_CTA == 32
    uint sts_idx   = ((tid & 0x3));
    uint sts_idy   = tid >> 2;
#elif TILE_K_PER_CTA == 64
    uint sts_idx   = ((tid & 0x7));
    uint sts_idy   = tid >> 3;
#elif TILE_K_PER_CTA == 128
    uint sts_idx   = ((tid & 0xf));
    uint sts_idy   = tid >> 4;
#endif

    uint cta_idx = blockIdx.y;
    uint cta_idy = blockIdx.x;

#if defined(ENABLE_SPLITK) && defined(ENABLE_SPLITF)
    uint spk_id = blockIdx.z % splitk;
    uint spf_id = (blockIdx.z % (splitk * flt_hw)) / splitk;
    uint grp_id = blockIdx.z / (splitk * flt_hw);

    uint num_chl_per_spk = (spk_id != splitk - 1) ? num_chl_per_spk_head: num_chl_per_spk_tail;

    int kloop = DivUp(num_chl_per_spk, TILE_K_PER_CTA);
#elif defined(ENABLE_SPLITK) && !defined(ENABLE_SPLITF)
    uint spk_id    = blockIdx.z % splitk;
    uint grp_id    = blockIdx.z / splitk;

    uint num_chl_per_spk = (spk_id != splitk - 1) ? num_chl_per_spk_head: num_chl_per_spk_tail;

    int kloop = flt_hw * DivUp(num_chl_per_spk, TILE_K_PER_CTA);
#elif !defined(ENABLE_SPLITK) && defined(ENABLE_SPLITF)
    uint spf_id    = blockIdx.z % flt_hw;
    uint grp_id    = blockIdx.z / flt_hw;

    int kloop = kloop_num;
#elif defined(ENABLE_FUSE)
    uint grp_id    = blockIdx.z % num_grp;
    // only for batch gemm, can also work in parallel multi-convs fusion.
    uint batch_id  = blockIdx.z / num_grp;

    int kloop = kloop_num;
#endif

    uint num_chl_per_grp_pad_v8 = num_chl_per_grp_pad >> 3;
    uint num_flt_per_grp_pad_v8 = num_flt_per_grp_pad >> 3;

    uint dCv4_idy = cta_idy * TILE_M_V1_PER_CTA +
                    tid / TILE_N_V8_PER_CTA;

    uint dCv4_idx = cta_idx * TILE_N_V8_PER_CTA +
                    tid % TILE_N_V8_PER_CTA;

    bool dCv4_x_valid = (dCv4_idx < num_flt_per_grp_pad_v8) & ((tid / TILE_N_V8_PER_CTA) < TILE_M_PER_CTA);

#if defined(ENABLE_SPLITK) && defined(ENABLE_SPLITF)
    uint dCv4_base = (spf_id * splitk + spk_id) * num_flt_per_grp_pad_v8 * num_grp * out_hw * in_num +
                     grp_id * num_flt_per_grp_pad_v8;
#elif defined(ENABLE_SPLITK) && !defined(ENABLE_SPLITF)
    uint dCv4_base = spk_id * num_flt_per_grp_pad_v8 * num_grp * out_hw * in_num +
                     grp_id * num_flt_per_grp_pad_v8;
#elif !defined(ENABLE_SPLITK) && defined(ENABLE_SPLITF)
    uint dCv4_base = spf_id * num_flt_per_grp_pad_v8 * num_grp * out_hw * in_num +
                     grp_id * num_flt_per_grp_pad_v8;
#elif defined(ENABLE_FUSE)
    uint dCv4_base   = grp_id * num_flt_per_grp_pad_v8 +
                     batch_id * num_grp * num_flt_per_grp_pad_v8 * out_hw * in_num;
    dA += batch_id * num_grp * num_chl_per_grp_pad_v8 * in_hw * in_num;
    dB += batch_id * num_chl_per_grp_pad_v8 * flt_hw * num_grp * num_flt_per_grp;
#endif

    uint mma_idx = (local_tid >> 3) & 0x1;
    uint mma_idy = ((local_tid & 0x07) + ((local_tid >> 4) << 3));

    uint smem_row_write_id = (set_widx * TILE_N_V8_PER_WARP); 

    uint sRv4_write = set_id * TILE_N_V8_PER_CTA * TILE_M_V1_PER_CTA +
                      set_widy * TILE_N_V8_PER_CTA * TILE_M_V1_PER_WARP +
                      mma_idy * TILE_N_V8_PER_CTA +
                      smem_row_write_id +   // SMEM_ROW_V1_SIZE = 32
                      mma_idx;

    uint sRv4_read = threadIdx.x;

#if defined(FLT_SIZE3)
    int flt_hw_id  = 0;
    int flt_hw_bid = 0x1;

    int lut_id = 0;
#elif defined(FLT_SIZEN)
    int flt_h_id = 0;
    int flt_w_id = 0;

    int lut_id = 0;
#endif

#if defined(ENABLE_SPLITK)
    int flt_c_v8_end = (spk_id * num_chl_per_spk_head + num_chl_per_spk) >> 3;
    int flt_c_v8_id  = ldg_idx + ((spk_id * num_chl_per_spk_head) >> 3);
#elif defined(ENABLE_SPLITF) || defined(ENABLE_FUSE)
    int flt_c_v8_end = num_chl_per_grp_pad_v8;
    int flt_c_v8_id  = ldg_idx;
#endif

    bool flt_c_v8_valid = flt_c_v8_id < flt_c_v8_end;

    int4 Rv4[Rv4_SIZE];
#if BUF_NUM <= 2
    const int4 ZEROv4 = {0, 0, 0, 0};

    int4 * reg_dAv4 = (int4 *) Rv4;
    int4 * reg_dBv4 = (int4 *) Rv4 + REG_dAv4_SIZE;
#endif

#if defined(ENABLE_FUSE) || ((defined(ENABLE_SPLITF) || defined(ENABLE_SPLITK)) && (TILE_K_PER_CTA > TILE_K_PER_SET))
    __half2 * h2R = (__half2 *) Rv4;
#endif

#if defined(FLT_SIZE1)
    int dAv4_off[READ_dAv4_STEPS];
    bool in_hw_valid[READ_dAv4_STEPS];

#pragma unroll
    for (int i = 0; i < READ_dAv4_STEPS; i++) {
        SET_dAv4_BOUND(i, dAv4_off[i], in_hw_valid[i]);
    }
#elif defined(FLT_SIZE3)
    int dAv4_off[READ_dAv4_STEPS];
    int in_hw_mask[READ_dAv4_STEPS];

#pragma unroll
    for (int i = 0; i < READ_dAv4_STEPS; i++) {
        SET_dAv4_BOUND(i, dAv4_off[i], in_hw_mask[i]);
    }
#elif defined(FLT_SIZEN)
    int dAv4_off[READ_dAv4_STEPS];
    int in_n_id[READ_dAv4_STEPS];
    int in_h_id[READ_dAv4_STEPS];
    int in_w_id[READ_dAv4_STEPS];

    int in_h_start[READ_dAv4_STEPS];
    int in_w_start[READ_dAv4_STEPS];

#pragma unroll
    for (int i = 0; i < READ_dAv4_STEPS; i++) {
        SET_dAv4_BOUND(i, dAv4_off[i], in_n_id[i], in_h_start[i], in_w_start[i]);
        in_h_id[i] = in_h_start[i];
        in_w_id[i] = in_w_start[i];
    }
#endif

    int dBv4_off[READ_dBv4_STEPS];
    bool flt_n_valid[READ_dBv4_STEPS];

#pragma unroll
    for (int i = 0; i < READ_dBv4_STEPS; i++) {
        SET_dBv4_BOUND(i, dBv4_off[i], flt_n_valid[i]);
    }

    extern __shared__ char sm_base[];

    // int  * sm_base_v1 = (int  *) sm_base;
    int4 * sm_base_v4 = (int4 *) sm_base;

    uint sAv4_write = sts_idy * TILE_K_V8_PER_CTA + sts_idx;

    uint sBv4_write = sAv4_write + SM_A_V4_1BUF * BUF_NUM;

    uint lds_idy = local_tid;

    uint sAv4_read = set_widy * TILE_M_PER_WARP * TILE_K_V8_PER_CTA +
                     ((lds_idy & 0x07) + ((lds_idy >> 4) << 3)) * TILE_K_V8_PER_CTA;

    uint sBv4_read = set_widx * TILE_N_PER_WARP * TILE_K_V8_PER_CTA +
                     ((lds_idy & 0x03) + ((lds_idy >> 4) & 0x01) * 4 + ((lds_idy >> 3) & 0x01) * 8) * TILE_K_V8_PER_CTA +
                     SM_A_V4_1BUF * BUF_NUM;

    int4 db0_sBv4[REG_sBv1_SIZE];
#if TILE_K_PER_SET == 16 || TILE_K_PER_SET == 32
    int4 db1_sBv4[REG_sBv1_SIZE];
#endif

    int4 db0_sAv4[REG_sAv1_SIZE];
#if TILE_K_PER_SET == 16 || TILE_K_PER_SET == 32
    int4 db1_sAv4[REG_sAv1_SIZE];
#endif


#if defined(FLT_SIZE1)
#if BUF_NUM <=2
        LOAD_dAv4(reg_dAv4, dA, dAv4_off, flt_c_v8_valid, in_hw_valid);
        LOAD_dBv4(reg_dBv4, dB, dBv4_off, flt_c_v8_valid, flt_n_valid);
#endif

        FWD_FLT(flt_c_v8_id, flt_c_v8_valid);
#elif defined(FLT_SIZE3)
#if BUF_NUM <=2
        LOAD_dAv4(reg_dAv4, dA, dAv4_off, flt_c_v8_valid, flt_hw_bid);
        LOAD_dBv4(reg_dBv4, dB, dBv4_off, flt_c_v8_valid, flt_n_valid);
#endif

        FWD_FLT(flt_hw_id, flt_hw_bid, flt_c_v8_id, flt_c_v8_valid);
        FWD_LUT(lut_id);

#elif defined(FLT_SIZEN)
#if BUF_NUM <=2
        LOAD_dAv4(reg_dAv4, dA, dAv4_off, in_n_id, in_h_id, in_w_id);
        LOAD_dBv4(reg_dBv4, dB, dBv4_off, flt_c_v8_valid, flt_n_valid);

#endif

        FWD_FLT(flt_h_id, flt_w_id, flt_c_v8_id, flt_c_v8_valid);
        FWD_LUT(lut_id);
#endif


#if BUF_NUM <= 2
    WRITE_sAv4(sm_base_v4, sAv4_write, reg_dAv4);
    WRITE_sBv4(sm_base_v4, sBv4_write, reg_dBv4);
#endif
    __syncthreads();

#if BUF_NUM == 2
    FWD_BUF(sAv4_write, SM_A_V4_1BUF, 0, sBv4_write, SM_B_V4_1BUF, SM_A_V4_2BUF);
#endif

    READ_sAv4(db0_sAv4, sm_base_v4, sAv4_read);
    READ_sBv4(db0_sBv4, sm_base_v4, sBv4_read);

#if TILE_K_PER_SET == 16 || TILE_K_PER_SET == 32
    FWD_KGROUP_STEP1(sAv4_read);
    FWD_KGROUP_STEP1(sBv4_read);
#endif
#if BUF_NUM <= 2
    for (; kloop > 0; --kloop)
#endif
    {
#if defined(FLT_SIZE1)
#if BUF_NUM <= 2
        LOAD_dAv4(reg_dAv4, dA, dAv4_off, flt_c_v8_valid, in_hw_valid);
        LOAD_dBv4(reg_dBv4, dB, dBv4_off, flt_c_v8_valid, flt_n_valid);
#endif

        FWD_FLT(flt_c_v8_id, flt_c_v8_valid);
#elif defined(FLT_SIZE3)
#if BUF_NUM <= 2
        LOAD_dAv4(reg_dAv4, dA, dAv4_off, flt_c_v8_valid, flt_hw_bid);
        LOAD_dBv4(reg_dBv4, dB, dBv4_off, flt_c_v8_valid, flt_n_valid);
#endif

        FWD_FLT(flt_hw_id, flt_hw_bid, flt_c_v8_id, flt_c_v8_valid);
        FWD_LUT(lut_id);
#elif defined(FLT_SIZEN)
#if BUF_NUM <= 2
        LOAD_dAv4(reg_dAv4, dA, dAv4_off, in_n_id, in_h_id, in_w_id);
        LOAD_dBv4(reg_dBv4, dB, dBv4_off, flt_c_v8_valid, flt_n_valid);
#endif

        FWD_FLT(flt_h_id, flt_w_id, flt_c_v8_id, flt_c_v8_valid);
        FWD_LUT(lut_id);
#endif


#if TILE_K_PER_SET == 16 || TILE_K_PER_SET == 32
        READ_sAv4(db1_sAv4, sm_base_v4, sAv4_read);
        READ_sBv4(db1_sBv4, sm_base_v4, sBv4_read);

        FWD_KGROUP_STEP2(sAv4_read);
        FWD_KGROUP_STEP2(sBv4_read);
#endif

        MMA_INSTS(C, db0_sAv4, db0_sBv4);

#if TILE_K_PER_SET == 32
        READ_sAv4(db0_sAv4, sm_base_v4, sAv4_read);
        READ_sBv4(db0_sBv4, sm_base_v4, sBv4_read);

        FWD_KGROUP_STEP3(sAv4_read);
        FWD_KGROUP_STEP3(sBv4_read);

        MMA_INSTS(C, db1_sAv4, db1_sBv4);

        READ_sAv4(db1_sAv4, sm_base_v4, sAv4_read);
        READ_sBv4(db1_sBv4, sm_base_v4, sBv4_read);

        FWD_KGROUP_STEP4(sAv4_read);
        FWD_KGROUP_STEP4(sBv4_read);
#endif

#if BUF_NUM == 1
        __syncthreads();
#endif

#if BUF_NUM <= 2
        WRITE_sAv4(sm_base_v4, sAv4_write, reg_dAv4);
        WRITE_sBv4(sm_base_v4, sBv4_write, reg_dBv4);
#endif

#if TILE_K_PER_SET == 16
        MMA_INSTS(C, db1_sAv4, db1_sBv4);
#elif TILE_K_PER_SET == 32
        MMA_INSTS(C, db0_sAv4, db0_sBv4);
#endif

#if BUF_NUM == 2
        FWD_BUF(sAv4_write, SM_A_V4_1BUF, 0, sBv4_write, SM_B_V4_1BUF, SM_A_V4_2BUF);

        FWD_BUF(sAv4_read,  SM_A_V4_1BUF, 0, sBv4_read,  SM_B_V4_1BUF, SM_A_V4_2BUF);

#endif

        __syncthreads();

        READ_sAv4(db0_sAv4, sm_base_v4, sAv4_read);
        READ_sBv4(db0_sBv4, sm_base_v4, sBv4_read);

#if TILE_K_PER_SET == 16 || TILE_K_PER_SET == 32
        FWD_KGROUP_STEP1(sAv4_read);
        FWD_KGROUP_STEP1(sBv4_read);
#endif

#if TILE_K_PER_SET == 32
        MMA_INSTS(C  , db1_sAv4, db1_sBv4);
#endif
    }

    __syncthreads();

    WRITE_sRv4(sm_base_v4, sRv4_write, Cv4);

    __syncthreads();

#pragma unroll
    for(int s = 0; s < OUTPUT_STEPS; s++) {
        READ_sRv4(Rv4, sm_base_v4, sRv4_read);

#if TILE_K_PER_CTA > TILE_K_PER_SET
        REDUCE(h2R);
#endif

        bool dCv4_y_valid = (dCv4_idy / out_hw) < in_num;
        uint dCv4_off     = dCv4_base +
            dCv4_idy * num_flt_per_grp_pad_v8 * num_grp +
            dCv4_idx;

#if defined(ENABLE_FUSE)
        ADD_BIAS_V4(has_bias, bias);

        uint concat_v4_off = 0;

        FUSE_RELU_V4(has_relu);
        FUSE_CLIP_V4(has_clip, clip_max, clip_min);
        // FUSE_PRELU_V4(has_prelu, prelu, leaky);

        FUSE_ELT_V4(has_elt, pre_data);
        FUSE_RELU_V4(has_elt_relu);
        FUSE_CLIP_V4(has_elt_clip, elt_clip_max, elt_clip_min);
        // FUSE_PRELU_V4(has_elt_prelu, elt_prelu, elt_leaky);

        SET_CONCAT_OFF_V4(has_concat, concat_v4_off);
#endif

        OUTPUT_PRC_HALF(Rv4);

        dCv4_idy += OUTPUT_SIZE_Y_IN_THD;
    }

#endif // __CUDA_ARCH__
}
